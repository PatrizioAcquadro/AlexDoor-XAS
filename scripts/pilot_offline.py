#!/usr/bin/env python
"""Bounded pilot preparation and four CUDA training paths; no physical dispatch."""

import argparse
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.action.b1 import apply_pose_delta  # noqa: E402
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA  # noqa: E402
from alexdoor_xas.assets.identity import RobotAssetRef  # noqa: E402
from alexdoor_xas.dataset.b1 import export_dataset, prepare_recording  # noqa: E402
from alexdoor_xas.policies.b1 import (  # noqa: E402
    B1Policy,
    batch_factories,
    load_b1_data,
    make_model,
    save_policy,
)
from alexdoor_xas.policies.common.model import OBS_CLIP  # noqa: E402
from alexdoor_xas.policies.encoding import (  # noqa: E402
    FIXED_UNITS_RECIPE,
    LEARNED_RECIPE,
    RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from alexdoor_xas.policies.observations import B1Observer  # noqa: E402
from alexdoor_xas.policies.purdue import PurdueFK  # noqa: E402
from alexdoor_xas.recording.b1 import PHASES  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(root, protocol, *, export=True, recipe=RECIPE, copy_raw=True, urdf=None):
    episodes, audit = [], []
    splits = {e["asset_id"]: e["split"] for e in protocol["episodes"]}
    binding, robot = None, None
    for entry in protocol["episodes"]:
        directory = root / entry.get("directory", entry["asset_id"])
        if entry.get("use_for_imitation") is False:
            reason = entry.get("imitation_exclusion_reason")
            path = directory / "episode.hdf5"
            if not reason or not path.is_file():
                raise ValueError("Excluded collection requires retained raw evidence and a reason")
            audit.append(
                dict(
                    asset_id=entry["asset_id"],
                    condition=entry.get("condition"),
                    used_for_imitation=False,
                    reason=reason,
                    source_recording=str(path.resolve()),
                )
            )
            print(json.dumps(audit[-1]), flush=True)
            continue
        runtime = json.loads((directory / "runtime.json").read_text())
        identity = RobotAssetRef.from_dict(runtime["robot_asset"])
        if robot is not None and identity != robot:
            raise ValueError("Different robot across physical episodes")
        robot = identity
        urdf_path = Path(runtime["urdf"]) if urdf is None else Path(urdf)
        if hashlib.sha256(urdf_path.read_bytes()).hexdigest() != identity.sha256:
            raise ValueError("Preparation URDF differs from the recorded robot asset")
        fk = PurdueFK(urdf_path)
        path = directory / "episode.hdf5"
        with h5py.File(path) as source:
            calibration = json.loads(source["metadata"].attrs["calibration"])
            metadata = json.loads(source["metadata"].attrs["episode"])
            candidate = pilot_binding(
                metadata["inspection"],
                depth_interval_m=calibration["depth_interval_m"],
                recipe=recipe,
            )
        if binding is not None and candidate != binding:
            raise ValueError("Different observed-input recipes")
        binding = candidate
        observer = B1Observer(None, binding, encoder=RGBDEncoder(binding))
        episode = prepare_recording(path, observer, fk, splits)
        with h5py.File(path) as source:
            obs, commands = source["observations"], source["commands"]
            start = next(
                i for i, phase in enumerate(commands["phase"][:]) if PHASES[int(phase)] != "inspect"
            )
            q = obs["joint_position"][start:]
            exact_a1 = commands["joint_target"][start:] - q[:-1, :7]
            error_a1 = float(np.abs(exact_a1 - episode.actions[A1_JOINT_DELTA]).max())
            p_errors, r_errors = [], []
            for tick, action in enumerate(episode.actions[A2_EE_DELTA]):
                reconstructed = apply_pose_delta(fk(q[tick], calibration), action)
                p_errors.append(
                    np.linalg.norm(reconstructed.origin - commands["tool_position"][start + tick])
                )
                r_errors.append(
                    Rotation.from_matrix(
                        reconstructed.rot @ commands["tool_rotation"][start + tick].T
                    ).magnitude()
                )
            if max(error_a1, max(p_errors), max(r_errors)) > 1e-7:
                raise ValueError("Labels fail observed-state command reconstruction")
            audit.append(
                dict(
                    asset_id=entry["asset_id"],
                    fk_urdf=str(urdf_path.resolve()),
                    split=entry["split"],
                    learned_ticks=len(exact_a1),
                    raw_observations=len(obs["time_s"]),
                    a1_reconstruction_max_rad=error_a1,
                    a2_reconstruction_max_m=max(p_errors),
                    a2_reconstruction_max_rad=max(r_errors),
                    a2_rotation_label_max_rad=float(
                        np.linalg.norm(episode.actions[A2_EE_DELTA][:, 3:], axis=1).max()
                    ),
                    phases={
                        PHASES[int(p)]: int((commands["phase"][:] == p).sum())
                        for p in np.unique(commands["phase"][:])
                    },
                    physics_feedback_samples=len(source["physics_feedback/time_s"]),
                    finite_policy_features=bool(np.isfinite(episode.observations).all()),
                    outcome=json.loads(source["metadata"].attrs["outcome"]),
                )
            )
        episodes.append(episode)
        print(json.dumps(dict(prepared=entry["asset_id"], ticks=len(exact_a1))), flush=True)
    if export and not episodes:
        raise ValueError("No eligible imitation episodes")
    if export:
        export_dataset(episodes, root / "data", binding, robot, splits, copy_raw=copy_raw)
    name = "data-audit.json" if export else "preflight-label-audit.json"
    write_json(root / name, audit)


def inference_batches(batches):
    """Match the maintained policy wrapper's observation clipping during evaluation."""
    for batch in batches:
        yield dict(batch, obs=np.clip(batch["obs"], -OBS_CLIP, OBS_CLIP))


def evaluate(model, family, batches, protocol):
    from alexdoor_xas.policies.act.train import evaluate_l1
    from alexdoor_xas.policies.diffusion.train import evaluate_sampled_l1

    if family == "act":
        return evaluate_l1(model, inference_batches(batches), torch.device("cuda:0"))
    return evaluate_sampled_l1(
        model,
        inference_batches(batches),
        torch.device("cuda:0"),
        protocol["ddim_steps"],
        seed=12345,
    )


def fit(model, family, batches, validation, protocol, *, overfit=False, callback=None):
    from alexdoor_xas.policies.act.config import ActTrainCfg
    from alexdoor_xas.policies.act.train import train_act
    from alexdoor_xas.policies.diffusion.config import DiffusionTrainCfg
    from alexdoor_xas.policies.diffusion.schedulers import make_train_scheduler
    from alexdoor_xas.policies.diffusion.train import EmaModel, train_diffusion

    policy_validation = None if validation is None else lambda: inference_batches(validation())
    options = dict(
        epochs=protocol["overfit_updates"] if overfit else protocol["epochs"],
        batch_size=protocol["batch_size"],
        lr=protocol["lr"],
        seed=protocol["training_seed"],
        device="cuda:0",
        val_every=protocol["overfit_updates"] if overfit else protocol["epochs"],
    )
    if family == "act":
        history = train_act(
            model,
            batches,
            ActTrainCfg(**options),
            make_val_batches=policy_validation,
            on_epoch=callback,
        )
        return model, history
    cfg = DiffusionTrainCfg(
        **options,
        lr_warmup_steps=0 if overfit else protocol["diffusion_warmup_steps"],
        lr_schedule="constant",
        val_inference_steps=protocol["ddim_steps"],
    )
    ema = EmaModel(model, cfg.ema_decay)
    history = train_diffusion(
        model,
        make_train_scheduler(model.cfg),
        batches,
        cfg,
        make_val_batches=policy_validation,
        ema=ema,
        on_epoch=callback,
    )
    return ema.module, history


def roundtrip(path, family, model, data, protocol, *, metadata):
    from alexdoor_xas.policies.act.policy import ActPolicy
    from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy

    save_policy(path, family, model, data, meta=metadata)
    kwargs = dict(device="cuda:0")
    if family == "diffusion":
        kwargs.update(sampler="ddim", num_inference_steps=protocol["ddim_steps"])
    original = B1Policy(
        (ActPolicy if family == "act" else DiffusionPolicy)(model, data.stats, **kwargs),
        family,
        data.dataset.contract,
        robot_asset=data.robot_asset,
    )
    loaded = B1Policy.from_checkpoint(
        path,
        family,
        binding=data.dataset.binding,
        runtime_asset=data.robot_asset,
        device="cuda:0",
        num_inference_steps=protocol["ddim_steps"],
    )
    record = data.dataset.by_id(data.train_ids[0])
    features = np.concatenate([record.obs[key][0] for key in data.stats.obs_keys])
    original.reset(protocol["inference_seed"])
    loaded.reset(protocol["inference_seed"])
    first, second = original.predict(features), loaded.predict(features)
    error = float(np.abs(first - second).max())
    if error > protocol["criteria"]["checkpoint_prediction_atol"]:
        raise ValueError("Checkpoint prediction mismatch")
    return loaded, error


def raw_metrics(policy, data, protocol):
    from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler

    sampler = ChunkSampler(
        data.dataset, protocol["chunk_ticks"], data.stats.obs_keys, list(data.val_ids)
    )
    batch = next(
        iter(
            BatchIterator(
                sampler,
                min(len(sampler), protocol["evaluation_chunks"]),
                seed=protocol["training_seed"],
            )
        )
    )
    predictions, times = [], []
    policy.reset(protocol["inference_seed"])
    for features in batch["obs"]:
        torch.cuda.synchronize()
        begin = perf_counter()
        predictions.append(policy.predict(features))
        torch.cuda.synchronize()
        times.append((perf_counter() - begin) * 1000)
    valid = ~batch["is_pad"]
    pred, target = np.asarray(predictions)[valid], batch["actions"][valid]
    error = pred - target
    metrics = dict(
        chunks=len(predictions),
        valid_actions=len(error),
        mean_absolute_error_per_column=np.abs(error).mean(axis=0).tolist(),
        inference_p50_ms=float(np.median(times)),
        inference_p95_ms=float(np.percentile(times, 95)),
        inference_max_ms=max(times),
        observation_clip_fraction=float(
            (np.abs(data.stats.obs.normalize(batch["obs"])) > policy.policy.obs_clip).mean()
        ),
    )
    if data.stats.action_space == A2_EE_DELTA:
        translation = np.linalg.norm(error[:, :3], axis=1)
        rotation = (
            Rotation.from_rotvec(pred[:, 3:]) * Rotation.from_rotvec(target[:, 3:]).inv()
        ).magnitude()
        metrics.update(
            translation_error_mean_m=float(translation.mean()),
            translation_error_p95_m=float(np.percentile(translation, 95)),
            rotation_error_mean_rad=float(rotation.mean()),
            rotation_error_p95_rad=float(np.percentile(rotation, 95)),
        )
    return metrics


def train(root, protocol):
    from alexdoor_xas.policies.act.config import ActModelCfg
    from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg

    destination = root / "training"
    destination.mkdir(exist_ok=False)
    report = {}
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        data = load_b1_data(root / "data", space)
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            path = destination / name
            path.mkdir()
            factory, development = batch_factories(
                data,
                family,
                protocol["chunk_ticks"],
                protocol["batch_size"],
                protocol["training_seed"],
            )
            val_batches = list(
                islice(development(), protocol["evaluation_chunks"] // protocol["batch_size"])
            )
            cfg = (
                ActModelCfg(chunk_size=protocol["chunk_ticks"])
                if family == "act"
                else DiffusionModelCfg(horizon=protocol["chunk_ticks"])
            )
            if protocol["overfit_updates"] > 0:
                fixed = next(iter(factory(0)))
                model = make_model(data, family, cfg, seed=protocol["training_seed"])
                before = evaluate(model, family, [fixed], protocol)
                learned, history = fit(
                    model,
                    family,
                    lambda epoch, fixed=fixed: iter([fixed]),
                    None,
                    protocol,
                    overfit=True,
                )
                after = evaluate(learned, family, [fixed], protocol)
                _, overfit_error = roundtrip(
                    path / "overfit.pt",
                    family,
                    learned,
                    data,
                    protocol,
                    metadata=dict(scope="fixed_train_minibatch_only"),
                )
                overfit = dict(
                    initial_l1=before,
                    final_l1=after,
                    fractional_reduction=(before - after) / max(before, 1e-12),
                    criterion_passed=bool(
                        after
                        <= before * (1 - protocol["criteria"]["overfit_fractional_l1_reduction"])
                    ),
                    checkpoint_prediction_max_error=overfit_error,
                    history=history.to_dict(),
                )
                del learned, model
            else:
                overfit = dict(skipped=True, optimizer_updates=0)
            write_json(path / "overfit.json", overfit)
            model = make_model(data, family, cfg, seed=protocol["training_seed"])
            initial = evaluate(model, family, val_batches, protocol)
            epochs = []

            def on_epoch(stats, best, epochs=epochs, name=name):
                row = dict(stats.to_dict(), gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20)
                epochs.append(row)
                print(json.dumps(dict(path=name, **row)), flush=True)

            learned, history = fit(
                model,
                family,
                lambda epoch, factory=factory: islice(
                    factory(epoch), protocol["max_batches_per_epoch"]
                ),
                lambda val_batches=val_batches: iter(val_batches),
                protocol,
                callback=on_epoch,
            )
            final = evaluate(learned, family, val_batches, protocol)
            policy, checkpoint_error = roundtrip(
                path / "final.pt",
                family,
                learned,
                data,
                protocol,
                metadata=dict(
                    scope="bounded_pilot_final_epoch_no_selection",
                    protocol=protocol,
                    history=history.to_dict(),
                ),
            )
            result = dict(
                overfit=overfit,
                development_initial_l1=initial,
                development_final_l1=final,
                normalized_units="z-score" if family == "act" else "minmax [-1,1]",
                training=epochs,
                optimizer_updates=sum(row["batch_count"] for row in epochs),
                device=torch.cuda.get_device_name(),
                checkpoint_prediction_max_error=checkpoint_error,
                development=raw_metrics(policy, data, protocol),
                train_episode_ids=data.train_ids,
                development_episode_ids=data.val_ids,
            )
            write_json(path / "report.json", result)
            report[name] = result
            write_json(destination / "report.json", report)
            del policy, learned, model
            torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/b1_policy_pilot.json")
    parser.add_argument(
        "--recipe", choices=(RECIPE, FIXED_UNITS_RECIPE, LEARNED_RECIPE), default=RECIPE
    )
    parser.add_argument(
        "--reference-raw",
        action="store_true",
        help="Retain original raw recordings separately rather than duplicating them in exports",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        help="Relocated byte-identical robot URDF for Isaac-free preparation on Gilbreth",
    )
    parser.add_argument(
        "--asset-id", help="Single declared episode for the read-only preflight audit"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; do not silently fall back to CPU")
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(4)
    if args.mode == "audit":
        protocol["episodes"] = [e for e in protocol["episodes"] if e["asset_id"] == args.asset_id]
        if len(protocol["episodes"]) != 1:
            parser.error("Audit requires one declared identity")
        prepare(args.root, protocol, export=False, recipe=args.recipe, urdf=args.urdf)
    else:
        if args.mode == "prepare":
            prepare(
                args.root,
                protocol,
                recipe=args.recipe,
                copy_raw=not args.reference_raw,
                urdf=args.urdf,
            )
        else:
            train(args.root, protocol)
