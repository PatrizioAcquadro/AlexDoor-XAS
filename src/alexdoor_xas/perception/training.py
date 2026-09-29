"""Bounded CUDA training and development evaluation; importing never launches work."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import DoorEstimator, estimator_loss, rotation_error


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
    rows = {e["asset_id"]: [] for e in dataset.episodes}
    for batch in DataLoader(dataset, batch_size=config["batch_size"], shuffle=False):
        inputs, target, episodes, ends = move(batch, device)
        prediction = model(**inputs)
        position = (prediction["contact_position"] - target["contact_position"]).norm(dim=-1)
        rotation = (
            rotation_error(prediction["contact_rotation"], target["contact_rotation"]) * 180 / np.pi
        )
        finite = torch.stack(
            [
                v.flatten(1).isfinite().all(1) if v.ndim > 1 else v.isfinite()
                for v in prediction.values()
            ]
        ).all(0)
        available = (inputs["geometry"][:, :, 3].flatten(2).sum(-1) > 0).all(-1)
        valid = finite & available & (prediction["confidence"] >= config["confidence_threshold"])
        for i, episode in enumerate(episodes.tolist()):
            entry = dataset.episodes[episode]
            # Phase is scoring metadata, never an estimator input.
            phase = dataset.window_phases[(episode, int(ends[i]))]
            rows[entry["asset_id"]].append(
                (float(position[i]), float(rotation[i]), bool(valid[i]), phase)
            )
    result = {}
    gates = config["gates"]
    for asset, values in rows.items():
        array = np.asarray(values)
        manipulation = array[np.isin(array[:, 3], [1, 2, 3])]
        if not len(manipulation):
            raise ValueError(f"No manipulation samples: {asset}")
        pe, re = np.percentile(manipulation[:, :2], 95, axis=0)
        coverage = float(manipulation[:, 2].mean())
        entry = next(e for e in dataset.episodes if e["asset_id"] == asset)
        result[asset] = dict(
            position_p95_m=float(pe),
            orientation_p95_deg=float(re),
            valid_coverage=coverage,
            handedness=entry["handedness"],
            frames=len(values),
            passed=bool(
                np.isfinite([pe, re]).all()
                and pe <= gates["contact_position_p95_m"]
                and re <= gates["contact_orientation_p95_deg"]
                and coverage >= gates["valid_coverage"]
            ),
        )
    score = float(
        np.mean(
            [
                r["position_p95_m"] / gates["contact_position_p95_m"]
                + r["orientation_p95_deg"] / gates["contact_orientation_p95_deg"]
                + (1 - r["valid_coverage"])
                for r in result.values()
            ]
        )
    )
    return dict(
        per_door=result,
        selection_score=score,
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
        },
    )


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, config, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["schema"] != "b1.perception.checkpoint.v1" or payload["config"] != config:
        raise ValueError("Incompatible perception checkpoint")
    model = DoorEstimator(config["hidden_size"]).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train(train_data, development, config, output, *, hours=1, resume=None, device="cuda:0"):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Training requires the actual CUDA device; no CPU fallback")
    if not 0 < hours <= 10:
        raise ValueError("Training budget must be in (0, 10] hours")
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
    if resume is not None:
        model, state = load_checkpoint(resume, config, device)
        if state["dataset"] != manifest:
            raise ValueError("Cannot resume against a different feature dataset")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        optimizer.load_state_dict(state["optimizer"])
        epoch, best, total_steps = state["epoch"], state["best_score"], state["steps"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        generator.set_state(state["sampler_rng"].cpu())
    sampler = WeightedRandomSampler(
        train_data.weights(), len(train_data), replacement=True, generator=generator
    )
    loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=sampler)
    started = time.monotonic()
    deadline = started + hours * 3600
    status = dict(state="running", device=device, budget_hours=hours, training_started=True)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    try:
        while epoch < config["max_epochs"] and time.monotonic() < deadline:
            model.train()
            losses = []
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
                loss, _ = estimator_loss(model(**inputs), target, available)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1
                losses.append(float(loss.detach()))
                if total_steps % 25 == 0:
                    with (output / "metrics.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                dict(
                                    event="train",
                                    step=total_steps,
                                    loss=float(np.mean(losses[-25:])),
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
            improved = np.isfinite(score) and score < best
            if improved:
                best = score
            payload = dict(
                schema="b1.perception.checkpoint.v1",
                config=config,
                dataset=manifest,
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
            )
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            with (output / "metrics.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(
                            event="development",
                            epoch=epoch,
                            step=total_steps,
                            train_loss=float(np.mean(losses)),
                            elapsed_s=time.monotonic() - started,
                            gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6,
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
                    )
                ),
                flush=True,
            )
            if not completed_epoch:
                break
        status.update(
            state="finished",
            reason="budget_or_epoch_limit",
            steps=total_steps,
            elapsed_s=time.monotonic() - started,
            best_score=best,
            subphase_complete=False,
        )
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
