"""Bounded CUDA training and development evaluation; importing never launches work."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import (
    STATE_CONFIDENCE,
    estimator_loss,
    geometry_errors,
    geometry_tolerances,
    make_estimator,
    usable_geometry,
)
from .run import EarlyStopping, save_json


def move(batch, device):
    inputs, target, episodes, ends = batch
    return (
        {k: v.to(device) for k, v in inputs.items()},
        {k: v.to(device) for k, v in target.items()},
        episodes,
        ends,
    )


@torch.no_grad()
def evaluate(model, dataset, config, device):
    if not dataset.episodes:
        raise ValueError("Evaluation requires nonempty observed episodes")
    model.eval()
    gates = config["gates"]
    tolerances = geometry_tolerances(gates)
    keys = list(tolerances)
    scales = np.asarray(list(tolerances.values()))
    confidence_scope = getattr(model, "confidence_scope", "contact-only")
    rows = {e["asset_id"]: [] for e in dataset.episodes}
    for batch in DataLoader(dataset, batch_size=config["batch_size"], shuffle=False):
        inputs, target, episodes, ends = move(batch, device)
        prediction = model(**inputs)
        errors = geometry_errors(prediction, target)
        finite = torch.stack(
            [
                v.flatten(1).isfinite().all(1) if v.ndim > 1 else v.isfinite()
                for v in prediction.values()
            ]
        ).all(0)
        support = inputs["geometry"][:, :, 3].flatten(2).mean(-1)
        available = (support > config.get("min_valid_depth_fraction", 0)).all(-1)
        confident = (
            finite & available & (prediction["confidence"] >= config["confidence_threshold"])
        )
        values = torch.stack([*(errors[k] for k in keys), confident.float()], -1).cpu().numpy()
        for i, episode in enumerate(episodes.tolist()):
            entry = dataset.episodes[episode]
            # Phase is scoring metadata, never an estimator input.
            phase = dataset.window_phases[(episode, int(ends[i]))]
            rows[entry["asset_id"]].append((*values[i], phase))
    result = {}
    for asset, values in rows.items():
        array = np.asarray(values)
        manipulation = array[np.isin(array[:, -1], [1, 2, 3])]
        if not len(manipulation):
            raise ValueError(f"No manipulation samples: {asset}")
        errors = manipulation[:, : len(keys)]
        p95 = np.percentile(errors, 95, axis=0)
        good = (np.isfinite(errors) & (errors <= scales)).all(1)
        coverage = float(good.mean())
        confident = manipulation[:, -2].astype(bool)
        valid = confident & (confidence_scope == STATE_CONFIDENCE)
        valid_coverage = float(valid.mean())
        physical = dict(zip(keys, map(float, p95), strict=True))
        pe, re = physical["contact_position_m"], physical["contact_rotation_deg"]
        contact_passed = bool(
            np.isfinite([pe, re]).all()
            and pe <= gates["contact_position_p95_m"]
            and re <= gates["contact_orientation_p95_deg"]
        )
        geometry_passed = bool(
            np.isfinite(p95).all() and (p95 <= scales).all() and coverage >= gates["valid_coverage"]
        )
        entry = next(e for e in dataset.episodes if e["asset_id"] == asset)
        result[asset] = dict(
            position_p95_m=pe,
            orientation_p95_deg=re,
            errors_p95=physical,
            valid_coverage=valid_coverage,
            confidence_coverage=float(confident.mean()),
            valid_precision=float(good[valid].mean()) if valid.any() else None,
            handedness=entry["handedness"],
            frames=len(values),
            manipulation_frames=len(manipulation),
            geometric_coverage=coverage,
            contact_geometry_passed=contact_passed,
            geometry_passed=geometry_passed,
            geometry_score=float(np.max(p95 / scales)),
            passed=(
                geometry_passed
                and valid_coverage >= gates["valid_coverage"]
                and valid.any()
                and float(good[valid].mean()) >= gates["valid_coverage"]
            ),
        )
    geometry_score = float(np.mean([r["geometry_score"] for r in result.values()]))
    score = geometry_score + float(np.mean([1 - r["valid_coverage"] for r in result.values()]))
    return dict(
        per_door=result,
        confidence_scope=confidence_scope,
        confidence_qualified=bool(getattr(model, "confidence_qualified", False)),
        geometry_tolerances=tolerances,
        selection_score=score,
        contact_geometry_passed=all(r["contact_geometry_passed"] for r in result.values()),
        geometry_passed=all(r["geometry_passed"] for r in result.values()),
        geometry_score=geometry_score,
        offline_passed=all(r["passed"] for r in result.values()),
        dynamic_validation="pending",
        subphase_complete=False,
        by_handedness={
            side: dict(
                doors=sum(r["handedness"] == side for r in result.values()),
                max_position_p95_m=max(
                    r["position_p95_m"] for r in result.values() if r["handedness"] == side
                ),
                max_orientation_p95_deg=max(
                    r["orientation_p95_deg"] for r in result.values() if r["handedness"] == side
                ),
                coverage=float(
                    np.mean(
                        [r["valid_coverage"] for r in result.values() if r["handedness"] == side]
                    )
                ),
            )
            for side in ("left", "right")
            if any(r["handedness"] == side for r in result.values())
        },
    )


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, config, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload["schema"]
        not in (
            "b1.perception.checkpoint.v1",
            "b1.perception.checkpoint.v2",
            "b1.perception.checkpoint.v3",
        )
        or payload["config"] != config
    ):
        raise ValueError("Incompatible perception checkpoint")
    model = make_estimator(config).to(device)
    model.load_state_dict(payload["model"])
    if payload["schema"] != "b1.perception.checkpoint.v1":
        if payload.get("confidence_scope") != STATE_CONFIDENCE:
            raise ValueError("Checkpoint has an incompatible confidence contract")
    else:
        model.confidence_scope = "contact-only"
    model.confidence_qualified = (
        payload.get("confidence_qualification", {}).get("qualified") is True
    )
    return model, payload


def training_scope(train_data, evaluation_data, fit_check):
    if not train_data.episodes or any(e["split"] != "train" for e in train_data.episodes):
        raise ValueError("Training requires train episodes only")
    if not evaluation_data.episodes:
        raise ValueError("Evaluation requires nonempty observed episodes")
    doors = {e["asset_id"]: e["handedness"] for e in train_data.episodes}
    if fit_check:
        if evaluation_data is not train_data or sorted(doors.values()) != ["left", "right"]:
            raise ValueError(
                "Fit check requires the same train subset with one door per handedness"
            )
    elif any(e["split"] != "development" for e in evaluation_data.episodes):
        raise ValueError("Full training requires development evaluation")
    return dict(
        kind="train_fit_check" if fit_check else "full_training",
        train_doors=sorted(doors),
        evaluation_split="train" if fit_check else "development",
    )


def validate_resume_recipe(state, loss_config, scope):
    if state.get("loss_config") != loss_config or state.get("scope") != scope:
        raise ValueError("Resume requires the same loss recipe and train/evaluation scope")


def train(
    train_data,
    development,
    config,
    output,
    *,
    stopping,
    loss_config,
    hours=1,
    resume=None,
    device="cuda:0",
    fit_check=False,
    refine_from=None,
    refinement_learning_rate=3e-5,
    learning_rate_schedule=None,
):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Training requires the actual CUDA device; no CPU fallback")
    if not 0 < hours <= 10:
        raise ValueError("Training budget must be in (0, 10] hours")
    if (fit_check or refine_from is not None) and hours > 1 / 6:
        raise ValueError("A diagnostic fitting run is limited to ten minutes")
    if loss_config != {"recipe": "articulated-state-v3"}:
        raise ValueError("Unknown perception loss recipe")
    if refine_from is not None:
        if fit_check or resume is not None or development is not train_data:
            raise ValueError("Refinement requires fresh output and train-only evaluation")
        if not 0 < refinement_learning_rate < config["learning_rate"]:
            raise ValueError("Refinement requires a smaller positive learning rate")
        if any(e["split"] != "train" for e in train_data.episodes):
            raise ValueError("Refinement requires train episodes only")
        scope = dict(
            kind="train_refinement",
            train_doors=sorted({e["asset_id"] for e in train_data.episodes}),
            evaluation_split="train",
            source_checkpoint=str(Path(refine_from).resolve()),
            learning_rate=refinement_learning_rate,
        )
    else:
        scope = training_scope(train_data, development, fit_check)
    scope["confidence_training"] = "frozen_geometry_postfit"
    scope["learning_rate_schedule"] = learning_rate_schedule
    evaluation_split = scope["evaluation_split"]
    stopper = EarlyStopping(stopping)
    output = Path(output)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Resume requires the existing run directory")
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    manifest = json.loads((train_data.root / "index.json").read_text())
    model = make_estimator(config).to(device)
    mean, std = train_data.normalization()
    model.proprio_mean.copy_(torch.as_tensor(mean, device=device))
    model.proprio_std.copy_(torch.as_tensor(std, device=device))
    if refine_from is not None:
        model, source = load_checkpoint(refine_from, config, device)
        if source.get("loss_config") != loss_config or source["dataset"] != manifest:
            raise ValueError("Refinement requires unchanged geometry recipe and feature corpus")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=refinement_learning_rate if refine_from is not None else config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **learning_rate_schedule)
        if learning_rate_schedule
        else None
    )
    epoch, best, total_steps = 0, float("inf"), 0
    best_metrics = None
    if resume is not None:
        model, state = load_checkpoint(resume, config, device)
        validate_resume_recipe(state, loss_config, scope)
        if "optimizer" not in state:
            raise ValueError("Calibrated inference artifacts cannot resume geometry training")
        if state["dataset"] != manifest:
            raise ValueError("Cannot resume against a different feature dataset")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        optimizer.load_state_dict(state["optimizer"])
        if learning_rate_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **learning_rate_schedule
            )
            scheduler.load_state_dict(state["lr_scheduler"])
        epoch, best, total_steps = state["epoch"], state["best_score"], state["steps"]
        stopper = EarlyStopping(**state["early_stopping"])
        if stopper.settings != stopping:
            raise ValueError("Resume requires the same early-stopping settings")
        best_metrics = state["best_metrics"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        generator.set_state(state["sampler_rng"].cpu())
    sampler = WeightedRandomSampler(
        train_data.weights(), len(train_data), replacement=True, generator=generator
    )
    loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=sampler)
    started = time.monotonic()
    deadline = started + hours * 3600
    status = dict(
        state="running",
        device=device,
        budget_hours=hours,
        training_started=True,
        scope=scope,
        loss_config=loss_config,
        development_evaluated=evaluation_split == "development",
        early_stopping=asdict(stopper),
        subphase_complete=False,
        dynamic_validation="pending",
    )
    save_json(output / "status.json", status)
    reason = None
    try:
        while epoch < config["max_epochs"] and time.monotonic() < deadline:
            model.train()
            losses = []
            contributions = []
            completed_epoch = True
            for batch in loader:
                if time.monotonic() >= deadline:
                    completed_epoch = False
                    break
                inputs, target, _, _ = move(batch, device)
                # Missing-input examples teach explicit low confidence, not oracle filling.
                available = torch.rand(len(inputs["features"]), device=device) >= 0.125
                support = inputs["geometry"][:, :, 3].flatten(2).mean(-1)
                available &= (support > config.get("min_valid_depth_fraction", 0)).all(-1)
                inputs["features"][~available] = 0
                inputs["geometry"][~available] = 0
                optimizer.zero_grad(set_to_none=True)
                loss, terms = estimator_loss(
                    model(**inputs),
                    target,
                    available,
                    gates=config["gates"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                # Clip geometry alone so confidence cannot scale shared gradients either.
                geometry_loss = sum(v for k, v in terms.items() if k != "confidence")
                geometry_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                # Confidence is fitted separately against frozen geometric predictions.
                confidence_weight = model.output.weight[23:].detach().clone()
                confidence_bias = model.output.bias[23:].detach().clone()
                optimizer.step()
                with torch.no_grad():
                    model.output.weight[23:].copy_(confidence_weight)
                    model.output.bias[23:].copy_(confidence_bias)
                total_steps += 1
                losses.append(float(geometry_loss.detach()))
                contributions.append({k: float(v.detach()) for k, v in terms.items()})
                if total_steps % 25 == 0:
                    with (output / "metrics.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                dict(
                                    event="train",
                                    step=total_steps,
                                    loss=float(np.mean(losses[-25:])),
                                    loss_terms={
                                        k: float(np.mean([r[k] for r in contributions[-25:]]))
                                        for k in terms
                                    },
                                    elapsed_s=time.monotonic() - started,
                                    gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6,
                                )
                            )
                            + "\n"
                        )
            if not losses:
                break
            epoch += int(completed_epoch)
            metrics = evaluate(model, development, config, device)
            score = metrics["geometry_score"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite evaluation selection score")
            stagnant = stopper.update(score, epoch) if completed_epoch else False
            if scheduler is not None and completed_epoch:
                scheduler.step(score)
            improved = best_metrics is None or (not metrics["geometry_passed"], score) < (
                not best_metrics["geometry_passed"],
                best,
            )
            if improved:
                best = score
                best_metrics = metrics
            payload = dict(
                schema="b1.perception.checkpoint.v3",
                confidence_scope=STATE_CONFIDENCE,
                config=config,
                dataset=manifest,
                scope=scope,
                loss_config=loss_config,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=scheduler.state_dict() if scheduler is not None else None,
                epoch=epoch,
                steps=total_steps,
                best_score=best,
                metrics=metrics,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all(),
                sampler_rng=generator.get_state(),
                partial_epoch=not completed_epoch,
                early_stopping=asdict(stopper),
                best_metrics=best_metrics,
                confidence_qualification=dict(qualified=False, reason="geometry_updated"),
            )
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            with (output / "metrics.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(
                            event=evaluation_split,
                            epoch=epoch,
                            step=total_steps,
                            train_loss=float(np.mean(losses)),
                            loss_terms={
                                k: float(np.mean([r[k] for r in contributions]))
                                for k in contributions[0]
                            },
                            elapsed_s=time.monotonic() - started,
                            gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            early_stopping=asdict(stopper),
                            **metrics,
                        )
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    dict(
                        epoch=epoch,
                        step=total_steps,
                        best_score=best,
                        offline_passed=metrics["offline_passed"],
                        geometry_passed=metrics["geometry_passed"],
                        evaluation_split=evaluation_split,
                        per_door=metrics["per_door"],
                    )
                ),
                flush=True,
            )
            status.update(
                epochs=epoch,
                steps=total_steps,
                elapsed_s=time.monotonic() - started,
                best_score=best,
                early_stopping=asdict(stopper),
            )
            save_json(output / "status.json", status)
            if fit_check and metrics["geometry_passed"]:
                reason = "train_state_passed"
                break
            if refine_from is not None and metrics["geometry_passed"]:
                reason = "train_geometry_passed"
                break
            if not completed_epoch or time.monotonic() >= deadline:
                reason = "time_budget"
                break
            if stagnant:
                reason = evaluation_split + "_stagnation"
                break
        status.update(
            state="finished",
            reason=reason or ("epoch_limit" if epoch >= config["max_epochs"] else "time_budget"),
        )
    except BaseException as error:
        status.update(state="failed", reason="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(
            epochs=epoch,
            steps=total_steps,
            elapsed_s=time.monotonic() - started,
            best_score=best if np.isfinite(best) else None,
            early_stopping=asdict(stopper),
        )
        save_json(output / "status.json", status)
        save_json(
            output / "summary.json",
            dict(
                **status,
                **{f"best_{evaluation_split}": best_metrics},
                fit_check_passed=(
                    bool(best_metrics and best_metrics["geometry_passed"]) if fit_check else None
                ),
                best_checkpoint=str(output / "best.pt") if (output / "best.pt").exists() else None,
                last_checkpoint=str(output / "last.pt") if (output / "last.pt").exists() else None,
            ),
        )


def calibrate_confidence(
    checkpoint, train_data, development, config, output, *, device, hours=1 / 6
):
    """Fit confidence after freezing geometry, then qualify on untouched development doors."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Confidence fitting requires CUDA")
    if not 0 < hours <= 1 / 6:
        raise ValueError("Confidence fitting is limited to ten minutes")
    training_scope(train_data, development, False)
    model, payload = load_checkpoint(checkpoint, config, device)
    manifest = json.loads((train_data.root / "index.json").read_text())
    if payload["dataset"] != manifest or development.root != train_data.root:
        raise ValueError("Confidence fitting requires the checkpoint feature corpus")
    for dataset, split in ((train_data, "train"), (development, "development")):
        expected_pairs = {
            (e["asset_id"], e["condition"]) for e in manifest["episodes"] if e["split"] == split
        }
        if expected_pairs != {(e["asset_id"], e["condition"]) for e in dataset.episodes}:
            raise ValueError("Confidence qualification requires complete train/development sets")
    expected = {e["asset_id"] for e in manifest["episodes"] if e["split"] == "train"}
    if payload.get("scope", {}).get("train_doors") != sorted(expected):
        raise ValueError("Confidence qualification requires geometry trained on the full train set")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval().requires_grad_(False)
    model.output.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.output.parameters(), lr=1e-3, weight_decay=0)
    started = time.monotonic()
    deadline = started + hours * 3600
    steps, counts = 0, [0, 0]
    loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    for _ in range(4):
        for batch in loader:
            if time.monotonic() >= deadline:
                break
            inputs, target, _, _ = move(batch, device)
            available = torch.rand(len(inputs["features"]), device=device) >= 0.125
            support = inputs["geometry"][:, :, 3].flatten(2).mean(-1)
            available &= (support > config.get("min_valid_depth_fraction", 0)).all(-1)
            inputs["features"][~available] = 0
            inputs["geometry"][~available] = 0
            prediction = model(**inputs)
            labels = (
                usable_geometry(
                    geometry_errors(prediction, target), geometry_tolerances(config["gates"])
                )
                & available
            )
            counts[1] += int(labels.sum())
            counts[0] += int((~labels).sum())
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy(
                prediction["confidence"], labels.float()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite confidence loss")
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 25 == 0:
                with (output / "metrics.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            dict(
                                step=steps,
                                confidence_loss=float(loss.detach()),
                                elapsed_s=time.monotonic() - started,
                            )
                        )
                        + "\n"
                    )
        if time.monotonic() >= deadline:
            break
    for key, value in model.state_dict().items():
        kept = slice(None, 23) if key in ("output.weight", "output.bias") else slice(None)
        if not torch.equal(value[kept], before[key][kept]):
            raise RuntimeError("Confidence fitting changed geometry")
    metrics = evaluate(model, development, config, device)
    qualification = dict(
        qualified=bool(steps and all(counts) and metrics["offline_passed"]),
        reason="development_passed"
        if steps and all(counts) and metrics["offline_passed"]
        else "development_or_class_support_failed",
        evaluation_split="development",
        geometry_unchanged=True,
        steps=steps,
        elapsed_s=time.monotonic() - started,
        training_labels=dict(inaccurate_or_missing=counts[0], accurate=counts[1]),
        source_checkpoint=str(Path(checkpoint).resolve()),
    )
    payload.update(
        model=model.state_dict(), confidence_qualification=qualification, confidence_metrics=metrics
    )
    # A calibrated artifact is for inference; geometry resume must start from the source run.
    payload.pop("optimizer", None)
    save_checkpoint(output / "calibrated.pt", payload)
    save_json(output / "summary.json", dict(**qualification, development=metrics))
    return qualification
