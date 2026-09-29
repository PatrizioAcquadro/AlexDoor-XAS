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
    DoorEstimator,
    estimator_loss,
    geometry_errors,
    geometry_tolerances,
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
        available = (inputs["geometry"][:, :, 3].flatten(2).sum(-1) > 0).all(-1)
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
            passed=geometry_passed and valid_coverage >= gates["valid_coverage"],
        )
    geometry_score = float(np.mean([r["geometry_score"] for r in result.values()]))
    score = geometry_score + float(np.mean([1 - r["valid_coverage"] for r in result.values()]))
    return dict(
        per_door=result,
        confidence_scope=confidence_scope,
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
        payload["schema"] not in ("b1.perception.checkpoint.v1", "b1.perception.checkpoint.v2")
        or payload["config"] != config
    ):
        raise ValueError("Incompatible perception checkpoint")
    model = DoorEstimator(config["hidden_size"]).to(device)
    model.load_state_dict(payload["model"])
    if payload["schema"] == "b1.perception.checkpoint.v2":
        if payload.get("confidence_scope") != STATE_CONFIDENCE:
            raise ValueError("Checkpoint has an incompatible confidence contract")
    else:
        model.confidence_scope = "contact-only"
    return model, payload


def training_scope(train_data, evaluation_data, fit_check):
    if any(e["split"] != "train" for e in train_data.episodes):
        raise ValueError("Training requires train episodes only")
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
):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Training requires the actual CUDA device; no CPU fallback")
    if not 0 < hours <= 10:
        raise ValueError("Training budget must be in (0, 10] hours")
    if fit_check and hours > 1 / 6:
        raise ValueError("A small fit check is limited to ten minutes")
    if loss_config != {"recipe": "articulated-state-v3"}:
        raise ValueError("Unknown perception loss recipe")
    scope = training_scope(train_data, development, fit_check)
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
    model = DoorEstimator(config["hidden_size"]).to(device)
    mean, std = train_data.normalization()
    model.proprio_mean.copy_(torch.as_tensor(mean, device=device))
    model.proprio_std.copy_(torch.as_tensor(std, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    epoch, best, total_steps = 0, float("inf"), 0
    best_metrics = None
    if resume is not None:
        model, state = load_checkpoint(resume, config, device)
        validate_resume_recipe(state, loss_config, scope)
        if state["dataset"] != manifest:
            raise ValueError("Cannot resume against a different feature dataset")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        optimizer.load_state_dict(state["optimizer"])
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
        development_evaluated=not fit_check,
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
                geometry_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                terms["confidence"].backward()
                optimizer.step()
                total_steps += 1
                losses.append(float(loss.detach()))
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
            score = metrics["selection_score"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite evaluation selection score")
            stagnant = stopper.update(score, epoch) if completed_epoch else False
            improved = best_metrics is None or (not metrics["offline_passed"], score) < (
                not best_metrics["offline_passed"],
                best,
            )
            if improved:
                best = score
                best_metrics = metrics
            payload = dict(
                schema="b1.perception.checkpoint.v2",
                confidence_scope=STATE_CONFIDENCE,
                config=config,
                dataset=manifest,
                scope=scope,
                loss_config=loss_config,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
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
            if fit_check and metrics["offline_passed"]:
                reason = "train_state_passed"
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
                    bool(best_metrics and best_metrics["offline_passed"]) if fit_check else None
                ),
                best_checkpoint=str(output / "best.pt") if (output / "best.pt").exists() else None,
                last_checkpoint=str(output / "last.pt") if (output / "last.pt").exists() else None,
            ),
        )
