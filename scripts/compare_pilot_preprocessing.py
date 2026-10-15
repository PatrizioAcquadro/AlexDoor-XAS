#!/usr/bin/env python
"""One paired ACT A2 preprocessing comparison on unchanged historical episodes."""

import argparse
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pilot_diagnosis import clipping, metrics  # noqa: E402

from alexdoor_xas.action.spaces import A2_EE_DELTA as SPACE  # noqa: E402
from alexdoor_xas.dataset.observations import obs_matrix  # noqa: E402
from alexdoor_xas.policies.act.config import ActModelCfg, ActTrainCfg  # noqa: E402
from alexdoor_xas.policies.act.train import train_act  # noqa: E402
from alexdoor_xas.policies.b1 import (  # noqa: E402
    B1Policy,
    batch_factories,
    load_b1_data,
    make_model,
    save_policy,
)
from alexdoor_xas.policies.encoding import (  # noqa: E402
    DEPTH_SCALE_M,
    FIXED_UNITS_RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from alexdoor_xas.recording.b1 import PHASES  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def depth_columns():
    return np.concatenate([np.arange(i * 320 + 192, i * 320 + 256) for i in range(4)])


def prepare(root, output):
    """Linear unit conversion of saved sufficient statistics; never copy/change raw data."""
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((REPO / "configs/b1_policy_pilot.json").read_text())
    data = load_b1_data(root / "data", SPACE)
    cfg = data.dataset.binding.config
    binding = pilot_binding(
        cfg["inspection"],
        depth_interval_m=cfg["encoding"]["depth_interval_m"],
        warmup_s=cfg["warmup_s"],
        recipe=FIXED_UNITS_RECIPE,
    )
    destination = output / "fixed-data"
    destination.mkdir(exist_ok=False)
    metadata = json.loads((root / "data/meta.json").read_text())
    metadata["perception"] = binding.to_dict()
    for filename in metadata["episodes"]:
        with (
            h5py.File(root / "data" / filename) as old,
            h5py.File(destination / filename, "x") as new,
        ):
            new.attrs.update(old.attrs)
            new.attrs["source_dataset_episode"] = str((root / "data" / filename).resolve())
            x = old["observations"][:]
            x[:, depth_columns()] *= cfg["encoding"]["depth_interval_m"][1] / DEPTH_SCALE_M
            new.create_dataset("observations", data=x)
            for key in ("time_s", "A1_joint_delta", SPACE):
                old.copy(key, new)
    write(destination / "meta.json", metadata)
    corrected = load_b1_data(destination, SPACE)
    assert data.train_ids == corrected.train_ids and data.val_ids == corrected.val_ids
    assert data.stats.action.to_dict() == corrected.stats.action.to_dict()
    write(output / "original-stats.json", data.stats.to_dict())
    write(output / "fixed-stats.json", corrected.stats.to_dict())
    original_checkpoint = root / "training/act-A2_ee_delta/final.pt"
    with original_checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write(
        output / "protocol.json",
        dict(
            baseline_commit="0b986e6e314ffa4357ae701a83d24839c93aa4b3",
            original_root=str(root),
            original_checkpoint_sha256=digest,
            original_checkpoint=str(original_checkpoint),
            intervention="one fixed-unit observation recipe, no empirical observation z-score",
            fixed_depth_scale_m=DEPTH_SCALE_M,
            sensor_interval_m=cfg["encoding"]["depth_interval_m"],
            action_normalization="unchanged train-only z-score",
            clip=10,
            input_fit="none; min/max/count recorded from train only, center=0 scale=1",
            training=dict(
                seed=17,
                epochs=3,
                max_batches_per_epoch=32,
                batch_size=32,
                optimizer_updates=96,
                lr=0.0001,
                model=vars(ActModelCfg(chunk_size=12)),
                order=["baseline", "fixed"],
                selection="final, no development selection",
            ),
            evaluation="every sixth tick; target phase and chunk position; exclude padding",
            executed_prefix="positions 0-5, nominal historical schedule, no physical execution",
            baselines="zero and global componentwise train median, no development fit",
            original_protocol=protocol,
            source_data="same historical controller; no new demonstrations",
            qualification=False,
            sealed_test=False,
            rollout=False,
        ),
    )
    input_report = {}
    for old, new in zip(data.dataset.records, corrected.dataset.records, strict=True):
        np.testing.assert_array_equal(old.actions, new.actions)
        np.testing.assert_array_equal(old.times, new.times)
        x, y = obs_matrix(old, data.stats.obs_keys), obs_matrix(new, corrected.stats.obs_keys)
        unchanged = np.setdiff1d(np.arange(data.obs_dim), depth_columns())
        np.testing.assert_array_equal(x[:, unchanged], y[:, unchanged])
        stages = phases(root, old)
        input_report[old.asset_id] = dict(
            split=old.split,
            rows=len(x),
            original=clipping(x, stages, data.stats),
            fixed=clipping(y, stages, corrected.stats),
            fixed_max_abs_input=float(np.abs(y).max()),
        )
    write(output / "input-report.json", input_report)


def phases(root, record):
    with h5py.File(root / record.asset_id / "episode.hdf5") as source:
        values = np.array([PHASES[int(p)] for p in source["commands/phase"][:]])
        first = np.flatnonzero(values != "inspect")[0]
        np.testing.assert_allclose(
            source["observations/time_s"][first:-1], record.times, atol=1e-9, rtol=0
        )
    return values[first:]


def verify_raw(root, output):
    """All encoded raw frames from reset, plus exact 60Hz proprioception/time joins."""
    data = load_b1_data(output / "fixed-data", SPACE)
    report_path = output / "raw-consistency.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for record in data.dataset.records:
        if record.asset_id in report:
            continue
        encoder = RGBDEncoder(data.dataset.binding)
        expected = obs_matrix(record, data.stats.obs_keys)
        maximum, encoded_count, compared = 0.0, 0, 0
        with h5py.File(root / record.asset_id / "episode.hdf5") as source:
            raw = source["observations"]
            times = raw["time_s"][:]
            start = np.flatnonzero(
                np.array(source["commands/phase"][:]) != PHASES.index("inspect")
            )[0]
            q, dq = raw["joint_position"][start:-1], raw["joint_velocity"][start:-1]
            np.testing.assert_array_equal(np.c_[q, dq], expected[:, -18:])
            # Encode only acquisitions used by the causal 10Hz encoder; skipped
            # 60Hz frames cannot change its state. Keep inspection and terminal.
            for i, t in enumerate(times):
                if encoder.last is not None and t - encoder.last.recent.acquired_s < 0.1 - 1e-9:
                    continue
                sample = {key: raw[key][i] for key in ("rgb", "depth_m", "valid_depth")}
                sample.update(time_s=t, frame=int(raw["frame"][i]))
                value = encoder.update(sample, generation=1)
                encoded_count += 1
                if start <= i < start + len(expected):
                    vector = np.r_[value.static.values, value.recent.values]
                    delta = float(np.max(np.abs(vector - expected[i - start, :1280])))
                    maximum = max(maximum, delta)
                    np.testing.assert_allclose(
                        vector, expected[i - start, :1280], atol=3e-6, rtol=2e-6
                    )
                    compared += 1
        report[record.asset_id] = dict(
            encoded_frames=encoded_count,
            learned_frames=compared,
            max_feature_difference=maximum,
            proprioception_exact=True,
            note="float32 pooling/unit conversion roundoff; same causal history from reset",
        )
        write(report_path, report)
        print(json.dumps(dict(raw_verified=record.asset_id, **report[record.asset_id])), flush=True)


def score(policy, data, root, destination):
    median = np.median(
        np.concatenate([r.actions for r in data.dataset.records if r.split == "train"]), axis=0
    )
    report = dict(train_median_native=median.tolist(), episodes={})
    for r in data.dataset.records:
        features = obs_matrix(r, data.stats.obs_keys)
        starts = np.arange(0, len(features), 6)
        pred = np.stack([policy.predict(features[t]) for t in starts])
        index = starts[:, None] + np.arange(12)
        valid = index < len(features)
        safe = np.minimum(index, len(features) - 1)
        target, stage = r.actions[safe], phases(root, r)[safe]
        position = np.broadcast_to(np.arange(12), valid.shape)
        values = dict(
            final=pred, zero=np.zeros_like(pred), train_median=np.broadcast_to(median, pred.shape)
        )
        comparisons = {}
        for name, value in values.items():
            row = {}
            for part, use in dict(
                full=valid,
                executed_prefix=valid & (position < 6),
                discarded_suffix=valid & (position >= 6),
            ).items():
                row[part] = dict(
                    overall=metrics(value[use], target[use], SPACE),
                    by_phase={
                        p: metrics(value[use & (stage == p)], target[use & (stage == p)], SPACE)
                        for p in np.unique(stage[use])
                    },
                )
            row["by_position"] = {
                str(j): metrics(
                    value[valid & (position == j)], target[valid & (position == j)], SPACE
                )
                for j in range(12)
            }
            row["by_phase_and_position"] = {
                p: {
                    str(j): metrics(value[mask], target[mask], SPACE)
                    for j in range(12)
                    if (mask := valid & (stage == p) & (position == j)).any()
                }
                for p in np.unique(stage)
            }
            comparisons[name] = row
        report["episodes"][r.asset_id] = dict(
            split=r.split, starts=len(starts), comparators=comparisons
        )
        np.savez_compressed(
            destination / (r.asset_id + ".npz"),
            starts=starts,
            prediction=pred,
            target=target,
            valid=valid,
            phase=stage,
        )
    write(destination / "physical-errors.json", report)


def train_variant(root, output, variant):
    destination = output / variant
    if (destination / "complete.json").exists():
        return
    destination.mkdir(exist_ok=True)
    data = load_b1_data(root / "data" if variant == "baseline" else output / "fixed-data", SPACE)
    train, _ = batch_factories(data, "act", 12, 32, 17)
    model = make_model(data, "act", ActModelCfg(chunk_size=12), seed=17)
    write(destination / "stats.json", data.stats.to_dict())
    # Trainer's existing epoch checkpoints allow only unfinished epochs to resume.
    resume = None
    resume_path = destination / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, weights_only=False, map_location="cuda:0")
        model.load_state_dict(saved["model"])
        resume = saved["training"]
    initial_path = destination / "initial.pt"
    if not initial_path.exists():
        save_policy(
            initial_path, "act", model, data, meta=dict(scope="paired initialization seed 17")
        )

    def checkpoint(state):
        temporary = destination / "resume.tmp.pt"
        torch.save(dict(model=model.state_dict(), training=state), temporary)
        temporary.replace(resume_path)

    cfg = ActTrainCfg(epochs=3, batch_size=32, lr=0.0001, seed=17, device="cuda:0", val_every=3)
    history = train_act(
        model,
        lambda epoch: islice(train(epoch), 32),
        cfg,
        resume_state=resume,
        on_checkpoint=checkpoint,
        on_epoch=lambda row, _: print(
            json.dumps(dict(variant=variant, **row.to_dict())), flush=True
        ),
    )
    assert sum(e.n_batches for e in history.epochs) == 96
    final = save_policy(
        destination / "final.pt",
        "act",
        model,
        data,
        meta=dict(scope="paired final, no development selection", history=history.to_dict()),
    )
    policy = B1Policy.from_checkpoint(
        final, "act", binding=data.dataset.binding, runtime_asset=data.robot_asset
    )
    # Compare actual training tensors with loaded inference normalization, including clip.
    batch = next(iter(train(0)))
    from alexdoor_xas.policies.act.policy import ActPolicy

    direct = ActPolicy(model, data.stats, device="cuda:0")
    features = obs_matrix(data.dataset.records[0], data.stats.obs_keys)[0]
    np.testing.assert_array_equal(policy.predict(features), direct.predict(features))
    if variant == "fixed":
        # A forward pre-hook observes the exact tensor supplied by the wrapper.
        captured = []
        hook = model.obs_proj.register_forward_pre_hook(
            lambda _, args: captured.append(args[0].detach().cpu().numpy())
        )
        direct.predict(features)
        hook.remove()
        np.testing.assert_array_equal(
            captured[-1][0], np.clip(features, -10, 10).astype(np.float32)
        )
    score(policy, data, root, destination)
    write(
        destination / "complete.json",
        dict(
            history=history.to_dict(),
            updates=96,
            checkpoint_roundtrip_exact=True,
            device=torch.cuda.get_device_name(),
            batch_shape=list(batch["obs"].shape),
            qualification=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify-raw", "baseline", "fixed"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "prepare":
        prepare(args.root, args.output)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        if args.mode == "verify-raw":
            verify_raw(args.root, args.output)
        else:
            train_variant(args.root, args.output, args.mode)


if __name__ == "__main__":
    main()
