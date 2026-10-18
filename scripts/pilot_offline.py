#!/usr/bin/env python
"""Bounded pilot preparation and four CUDA training paths; no physical dispatch."""

import argparse
import gc
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
from alexdoor_xas.resources import host_memory_snapshot, require_host_memory  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def memory_checkpoint(root, stage, *, additional_bytes=0):
    state = host_memory_snapshot()
    with (root / "host-memory.jsonl").open("a") as stream:
        stream.write(
            json.dumps(state | dict(stage=stage, additional_bytes=additional_bytes)) + "\n"
        )
    require_host_memory(state, additional_bytes=additional_bytes)


def dataset_ram_bytes(directory):
    """Use logical observation size, including compact/VDS datasets, not file size."""
    metadata = json.loads((directory / "meta.json").read_text())
    size = 0
    for name in metadata["episodes"]:
        if Path(name).name != name:
            raise ValueError("Episode path must stay inside its dataset")
        with h5py.File(directory / name, "r") as source:
            values = source["observations"]
            size += int(values.size) * values.dtype.itemsize
    return size


def prepare(root, protocol, *, export=True, recipe=RECIPE, copy_raw=True, urdf=None):
    episodes, audit = [], []
    splits = {e["asset_id"]: e["split"] for e in protocol["episodes"]}
    binding, robot = None, None
    for entry in protocol["episodes"]:
        directory = root / entry.get("directory", entry["asset_id"])
        if entry.get("use_for_imitation") is False:
            reason = entry.get("imitation_exclusion_reason")
            path = directory / "episode.hdf5"
            if not reason:
                raise ValueError("Excluded collection requires an explicit reason")
            audit.append(
                dict(
                    asset_id=entry["asset_id"],
                    condition=entry.get("condition"),
                    used_for_imitation=False,
                    reason=reason,
                    source_recording=str(path.resolve()),
                    recording_available_here=path.is_file(),
                    retention_verified_here=False,
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


def fit(
    model,
    family,
    batches,
    validation,
    protocol,
    *,
    overfit=False,
    callback=None,
    resume_state=None,
    on_checkpoint=None,
):
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
            resume_state=resume_state,
            on_checkpoint=on_checkpoint,
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
        resume_state=resume_state,
        on_checkpoint=on_checkpoint,
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
    if not data.val_ids:
        return dict(evaluated=False, reason="development_records_not_on_this_host")
    from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler

    memory_root = data.dataset.dataset_dir.parent
    memory_checkpoint(memory_root, "before_development_batch")
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
    for index, features in enumerate(batch["obs"]):
        if index % 32 == 0:
            memory_checkpoint(memory_root, f"development_chunk_{index}")
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


def save_training_state(path, model, state, protocol, family, space, epochs):
    """Atomically retain optimizer, RNG, EMA and model state at epoch boundaries."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            model_state=model.state_dict(),
            training_state=state,
            protocol=protocol,
            family=family,
            action_space=space,
            epochs=epochs,
        ),
        temporary,
    )
    temporary.replace(path)


def validate_epoch_extension(source, target):
    """Continue the same experiment; only its declared final epoch may increase."""
    if (
        source.get("overfit_updates", 0)
        or target.get("overfit_updates", 0)
        or not isinstance(target.get("epochs"), int)
        or target["epochs"] <= source["epochs"]
        or source | {"epochs": target["epochs"]} != target
    ):
        raise ValueError("Continuation requires only a strict epoch increase")


def continuation_source(directory, protocol):
    directory = directory.resolve()
    source_protocol = json.loads((directory / "protocol.json").read_text())
    validate_epoch_extension(source_protocol, protocol)
    report = json.loads((directory / "report.json").read_text())
    expected = source_protocol["epochs"] * source_protocol["max_batches_per_epoch"]
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            cell = directory / name
            if (
                report.get(name, {}).get("optimizer_updates") != expected
                or not (cell / "final.pt").is_file()
                or not (cell / "training-state.pt").is_file()
            ):
                raise ValueError("Continuation requires four complete source cells")
    return dict(directory=str(directory), protocol=source_protocol)


def restore_training_state(path, model, protocol, family, space):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["protocol"] != protocol
        or payload["family"] != family
        or payload["action_space"] != space
    ):
        raise ValueError("Resume protocol/family/action space differs")
    model.load_state_dict(payload["model_state"])
    return payload["training_state"], payload["epochs"]


def train(root, protocol, *, resume=False, continue_from=None):
    from alexdoor_xas.policies.act.config import ActModelCfg
    from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg

    destination = root / "training"
    if resume and continue_from is not None:
        raise ValueError("Use resume for a partial destination, or continue-from for a new one")
    continuation = (
        continuation_source(continue_from, protocol) if continue_from is not None else None
    )
    destination.mkdir(exist_ok=resume)
    frozen = destination / "protocol.json"
    if resume:
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError("Resume requires the unchanged training protocol")
        if protocol["overfit_updates"]:
            raise ValueError("Fixed-minibatch studies do not support partial resumption")
    else:
        write_json(frozen, protocol)
    continuation_path = destination / "continuation.json"
    if resume and continuation_path.exists():
        continuation = json.loads(continuation_path.read_text())
        if continuation_source(Path(continuation["directory"]), protocol) != continuation:
            raise ValueError("Continuation source changed")
    elif continuation is not None:
        write_json(continuation_path, continuation)
    report_path = destination / "report.json"
    report = json.loads(report_path.read_text()) if resume and report_path.exists() else {}
    names = {
        f"{family}-{space}"
        for space in (A1_JOINT_DELTA, A2_EE_DELTA)
        for family in ("act", "diffusion")
    }
    for name, result in report.items():
        if (
            name not in names
            or not (destination / name / "final.pt").is_file()
            or result.get("optimizer_updates")
            != protocol["epochs"] * protocol["max_batches_per_epoch"]
        ):
            raise ValueError("Incomplete reported final cell")
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        if all(f"{family}-{space}" in report for family in ("act", "diffusion")):
            continue
        memory_checkpoint(
            root, f"{space}_before_load", additional_bytes=dataset_ram_bytes(root / "data")
        )
        data = load_b1_data(root / "data", space)
        memory_checkpoint(root, f"{space}_loaded")
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            path = destination / name
            if name in report:
                continue
            path.mkdir(exist_ok=resume)
            factory, development = batch_factories(
                data,
                family,
                protocol["chunk_ticks"],
                protocol["batch_size"],
                protocol["training_seed"],
            )
            val_batches = (
                list(islice(development(), protocol["evaluation_chunks"] // protocol["batch_size"]))
                if development is not None
                else []
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
            resume_path = path / "training-state.pt"
            source_path, source_protocol = resume_path, protocol
            if continuation is not None:
                original = Path(continuation["directory"]) / name
                final_payload = torch.load(
                    original / "final.pt", map_location="cpu", weights_only=False
                )
                if final_payload["norm_stats"] != data.stats.to_dict():
                    raise ValueError("Continuation training statistics/data differ")
                if not resume_path.exists():
                    source_path = original / "training-state.pt"
                    source_protocol = continuation["protocol"]
            state, epochs = (
                restore_training_state(source_path, model, source_protocol, family, space)
                if (resume or continuation is not None) and source_path.exists()
                else (None, [])
            )
            if (
                continuation is not None
                and source_path != resume_path
                and (
                    state is None
                    or state["next_epoch"] != source_protocol["epochs"]
                    or state["global_step"]
                    != source_protocol["epochs"] * source_protocol["max_batches_per_epoch"]
                    or sum(row["batch_count"] for row in epochs) != state["global_step"]
                    or family == "diffusion"
                    and state.get("ema_state") is None
                )
            ):
                raise ValueError("Source training state is incomplete")
            initial = evaluate(model, family, val_batches, protocol) if val_batches else None
            if resume and state is None and any(path.glob("*.pt")):
                raise ValueError("Existing partial cell lacks resumable training state")

            def on_epoch(stats, best, epochs=epochs, name=name):
                row = dict(stats.to_dict(), gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20)
                epochs.append(row)
                print(json.dumps(dict(path=name, **row)), flush=True)

            def checkpoint(
                state,
                path=resume_path,
                model=model,
                family=family,
                space=space,
                epochs=epochs,
                name=name,
            ):
                save_training_state(path, model, state, protocol, family, space, epochs)
                memory_checkpoint(root, f"{name}_checkpoint_saved")

            learned, history = fit(
                model,
                family,
                lambda epoch, factory=factory: islice(
                    factory(epoch), protocol["max_batches_per_epoch"]
                ),
                (lambda val_batches=val_batches: iter(val_batches)) if val_batches else None,
                protocol,
                callback=on_epoch,
                resume_state=state,
                on_checkpoint=checkpoint,
            )
            final = evaluate(learned, family, val_batches, protocol) if val_batches else None
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
        del data, factory, development
        gc.collect()
        memory_checkpoint(root, f"{space}_released")
    return report


def evaluate_checkpoints(root, protocol, checkpoints):
    """Workstation-only development inference with returned, frozen train statistics."""
    from alexdoor_xas.dataset.b1 import B1Dataset
    from alexdoor_xas.policies.common.data import PolicyData

    destination = root / "evaluation"
    destination.mkdir()
    report = {}
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        memory_checkpoint(
            root, f"{space}_before_load", additional_bytes=dataset_ram_bytes(root / "data")
        )
        dataset = B1Dataset(root / "data", space)
        memory_checkpoint(root, f"{space}_loaded")
        train = tuple(r.episode_id for r in dataset.records if r.split == "train")
        development = tuple(r.episode_id for r in dataset.records if r.split == "development")
        if not development:
            raise ValueError("Local development recordings are required for evaluation")
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            checkpoint = checkpoints / name / "final.pt"
            policy = B1Policy.from_checkpoint(
                checkpoint,
                family,
                binding=dataset.binding,
                runtime_asset=dataset.robot_asset,
                device="cuda:0",
                num_inference_steps=protocol["ddim_steps"],
            )
            if policy.policy.stats.train_episode_ids != train:
                raise ValueError("Returned checkpoint train identities differ")
            data = PolicyData(dataset, train, development, policy.policy.stats, dataset.robot_asset)
            before_stats = policy.policy.stats.to_dict()
            model = policy.policy.model
            before_weights = {key: value.clone() for key, value in model.state_dict().items()}
            metrics = raw_metrics(policy, data, protocol)
            if policy.policy.stats.to_dict() != before_stats or any(
                not torch.equal(before_weights[key], value)
                for key, value in model.state_dict().items()
            ):
                raise ValueError("Offline evaluation changed weights or normalization")
            report[name] = dict(
                checkpoint=str(checkpoint.resolve()),
                offline_development=metrics,
                train_episode_ids=data.train_ids,
                development_episode_ids=data.val_ids,
                frozen_weights_and_statistics=True,
                device=torch.cuda.get_device_name(),
                scope=(
                    "Offline development inference only; no closed-loop outcome or model selection"
                ),
            )
            write_json(destination / f"{name}.json", report[name])
            write_json(destination / "report.json", report)
            print(json.dumps(dict(evaluated=name, **report[name])), flush=True)
            del policy, model, before_weights, data
            torch.cuda.empty_cache()
        del dataset
        gc.collect()
        memory_checkpoint(root, f"{space}_released")
    return report


def evaluate_selection_checkpoints(root, protocol, checkpoints, selection, *, repeats=1):
    """Stream fixed phase-balanced development windows; never fit normalization."""
    from alexdoor_xas.policies.common.b1_contract import PilotBinding
    from alexdoor_xas.policies.pilot_selection import action_errors

    if repeats not in (1, 2) or selection["horizon"] != protocol["chunk_ticks"]:
        raise ValueError("Selection uses the frozen action horizon and one/two repeat passes")
    directory = root / "data"
    metadata = json.loads((directory / "meta.json").read_text())
    binding = PilotBinding.from_dict(metadata["perception"])
    robot = RobotAssetRef.from_dict(metadata["robot_asset"])
    suite, train_ids = [], []
    horizon = selection["horizon"]
    for name in metadata["episodes"]:
        if Path(name).name != name:
            raise ValueError("Episode path escapes dataset")
        with h5py.File(directory / name) as data:
            if data.attrs["split"] == "train":
                train_ids.append(data.attrs["episode_id"])
                continue
            if data.attrs["split"] != "development":
                raise ValueError("Unexpected selection split")
            with h5py.File(data.attrs["source_recording"]) as raw:
                identity = json.loads(raw["metadata"].attrs["episode"])
                if any(
                    identity[key] != data.attrs[key] for key in ("episode_id", "asset_id", "split")
                ):
                    raise ValueError(
                        "Raw phase labels differ from the prepared development episode"
                    )
                starts = data[f"{A1_JOINT_DELTA}/starts"][:]
                np.testing.assert_array_equal(starts, data[f"{A2_EE_DELTA}/starts"][:])
                times = data["time_s"][:][starts]
                raw_times = raw["commands/time_s"][:]
                positions = np.searchsorted(raw_times, times)
                np.testing.assert_allclose(raw_times[positions], times, atol=1e-8, rtol=0)
                phases = raw["commands/phase"][:][positions]
            windows = np.lib.stride_tricks.sliding_window_view(phases, horizon)
            buckets = {}
            for phase in selection["phases"]:
                eligible = np.flatnonzero(np.all(windows == PHASES.index(phase), axis=1))
                count = selection["chunks_per_phase_per_development_episode"]
                if len(eligible) < count:
                    raise ValueError("Insufficient complete windows for the frozen phase sample")
                chosen = eligible[np.linspace(0, len(eligible) - 1, count, dtype=int)]
                if len(np.unique(chosen)) != count:
                    raise ValueError("Duplicate selection starts")
                buckets[phase] = chosen.tolist()
            suite.append(dict(file=name, episode_id=data.attrs["episode_id"], buckets=buckets))
    if len(suite) != 2 or len(train_ids) != 2:
        raise ValueError(
            "This frozen selection suite requires two train and two development records"
        )
    suite_record = dict(selection=selection, episodes=suite, train_episode_ids=train_ids)
    suite_path = root / "selection-suite.json"
    if suite_path.exists():
        if json.loads(suite_path.read_text()) != suite_record:
            raise ValueError("Frozen development sample changed")
    else:
        write_json(suite_path, suite_record)
    destination = root / "selection-evaluation"
    destination.mkdir()
    report = {}
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        for family in ("act", "diffusion"):
            cell = f"{family}-{space}"
            memory_checkpoint(root, f"{cell}_selection_before_load", additional_bytes=512 * 2**20)
            checkpoint = checkpoints / cell / "final.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            policy = B1Policy.from_checkpoint(
                checkpoint,
                family,
                binding=binding,
                runtime_asset=robot,
                device="cuda:0",
                num_inference_steps=protocol["ddim_steps"],
            )
            if policy.policy.stats.train_episode_ids != tuple(train_ids):
                raise ValueError("Selection checkpoint training membership differs")
            before_stats = policy.policy.stats.to_dict()
            before_weights = {
                key: value.clone() for key, value in policy.policy.model.state_dict().items()
            }
            passes, saved_predictions, saved_targets, labels, indices = [], [], [], [], []
            for repeat in range(repeats):
                policy.reset(selection["inference_seed"])
                buckets = {}
                predictions, targets = [], []
                for entry in suite:
                    with h5py.File(directory / entry["file"]) as data:
                        for phase, chosen in entry["buckets"].items():
                            phase_pred, phase_target = [], []
                            for index in chosen:
                                observation_index = int(data[f"{space}/starts"][index])
                                # Model inputs contain only the causal observation.
                                features = data["observations"][observation_index]
                                prediction = policy.predict(features)
                                target = data[f"{space}/actions"][index : index + horizon]
                                phase_pred.append(prediction)
                                phase_target.append(target)
                                if repeat == 0:
                                    labels.append(entry["episode_id"] + "/" + phase)
                                    indices.append(index)
                            pred = np.concatenate(phase_pred)
                            truth = np.concatenate(phase_target)
                            errors, components = action_errors(space, pred, truth)
                            key = entry["episode_id"] + "/" + phase
                            buckets[key] = dict(
                                mean=float(errors.mean()),
                                p95=float(np.percentile(errors, 95)),
                                **{
                                    name: dict(
                                        mean=float(values.mean()),
                                        p95=float(np.percentile(values, 95)),
                                    )
                                    for name, values in components.items()
                                },
                            )
                            predictions.extend(phase_pred)
                            targets.extend(phase_target)
                            memory_checkpoint(root, f"{cell}_selection_{repeat}_{phase}")
                passes.append(
                    dict(
                        primary=float(np.mean([row["mean"] for row in buckets.values()])),
                        bucket_p95={key: row["p95"] for key, row in buckets.items()},
                        buckets=buckets,
                    )
                )
                saved_predictions.append(np.asarray(predictions))
                saved_targets.append(np.asarray(targets))
            repeat_difference = (
                0.0
                if repeats == 1
                else float(np.max(np.abs(saved_predictions[0] - saved_predictions[1])))
            )
            relative_repeat = (
                0.0
                if repeats == 1
                else abs(passes[0]["primary"] - passes[1]["primary"])
                / max(passes[0]["primary"], 1e-12)
            )
            if relative_repeat > selection["maximum_baseline_repeat_relative_difference"]:
                raise ValueError(
                    "Inference repeat variability invalidates the declared effect threshold"
                )
            if (
                before_stats != policy.policy.stats.to_dict()
                or any(
                    not torch.equal(before_weights[key], value)
                    for key, value in policy.policy.model.state_dict().items()
                )
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Selection changed model, checkpoint or normalization")
            np.savez_compressed(
                destination / f"{cell}-actions.npz",
                predicted=np.asarray(saved_predictions),
                target=np.asarray(saved_targets),
                bucket=np.asarray(labels),
                start=np.asarray(indices),
            )
            report[cell] = dict(
                **passes[0],
                checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=digest,
                repeat_max_action_difference=repeat_difference,
                repeat_relative_score_difference=relative_repeat,
                repeats=repeats,
                inferred_chunks=len(labels),
                valid_actions=len(labels) * horizon,
                frozen_weights_and_statistics=True,
                device=torch.cuda.get_device_name(),
                scope=(
                    "Local phase-balanced development action inference; "
                    "one motion/two lighting variants, no closed-loop claim"
                ),
            )
            write_json(destination / "report.json", report)
            print(
                json.dumps(
                    dict(
                        cell=cell,
                        primary=passes[0]["primary"],
                        repeat_relative_difference=relative_repeat,
                    )
                ),
                flush=True,
            )
            del policy, before_weights, before_stats, saved_predictions, saved_targets
            gc.collect()
            torch.cuda.empty_cache()
            memory_checkpoint(root, f"{cell}_selection_released")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prepare", "train", "evaluate"))
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
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        help="Returned training directory containing the four frozen final checkpoints",
    )
    parser.add_argument(
        "--selection-protocol",
        type=Path,
        help="Frozen phase-balanced action-inference selection rules; evaluate only",
    )
    parser.add_argument("--selection-repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue unchanged real cells from saved optimizer/RNG/EMA state",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        help="Completed four-cell training directory; only epochs may increase, in a new root",
    )
    args = parser.parse_args()
    if args.selection_protocol is not None and args.mode != "evaluate":
        parser.error("--selection-protocol is inference-only evaluation")
    if args.mode == "evaluate" and args.checkpoints_root is None:
        parser.error("evaluate requires --checkpoints-root")
    if args.continue_from is not None and (args.mode != "train" or args.resume):
        parser.error("--continue-from requires train without --resume")
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid for train")
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
        elif args.mode == "evaluate":
            if args.selection_protocol is None:
                evaluate_checkpoints(args.root, protocol, args.checkpoints_root)
            else:
                evaluate_selection_checkpoints(
                    args.root,
                    protocol,
                    args.checkpoints_root,
                    json.loads(args.selection_protocol.read_text()),
                    repeats=args.selection_repeats,
                )
        else:
            train(args.root, protocol, resume=args.resume, continue_from=args.continue_from)
