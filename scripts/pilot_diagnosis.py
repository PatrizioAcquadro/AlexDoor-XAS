#!/usr/bin/env python
"""Read-only physical-unit diagnosis of the four existing pilot checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA  # noqa: E402
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS  # noqa: E402
from alexdoor_xas.dataset.observations import obs_matrix  # noqa: E402
from alexdoor_xas.policies.act.config import ActModelCfg  # noqa: E402
from alexdoor_xas.policies.act.policy import ActPolicy  # noqa: E402
from alexdoor_xas.policies.b1 import B1Policy, load_b1_data, make_model  # noqa: E402
from alexdoor_xas.policies.common.model import OBS_CLIP  # noqa: E402
from alexdoor_xas.recording.b1 import PHASES  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary(values):
    values = np.asarray(values)
    return dict(
        count=len(values),
        mean=float(values.mean()),
        p50=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        max=float(values.max()),
    )


def errors(prediction, target, space):
    difference = prediction - target
    if space == A1_JOINT_DELTA:
        return {
            name + "_deg": np.rad2deg(np.abs(difference[:, i])) for i, name in enumerate(ARM_JOINTS)
        }
    angle = (
        Rotation.from_rotvec(prediction[:, 3:]) * Rotation.from_rotvec(target[:, 3:]).inv()
    ).magnitude()
    return dict(
        translation_mm=np.linalg.norm(difference[:, :3], axis=1) * 1000,
        rotation_deg=np.rad2deg(angle),
    )


def metrics(prediction, target, space):
    result = {key: summary(value) for key, value in errors(prediction, target, space).items()}
    result["column_mae_native"] = np.abs(prediction - target).mean(axis=0).tolist()
    return result


def feature_names():
    names = []
    for block in ("static", "history_oldest", "history_middle", "history_newest"):
        for channel in ("red", "green", "blue", "depth_over_far", "coverage"):
            names.extend(f"{block}/{channel}/{row}/{col}" for row in range(8) for col in range(8))
    names.extend("q/" + name for name in ARM_JOINTS + NECK_JOINTS)
    names.extend("dq/" + name for name in ARM_JOINTS + NECK_JOINTS)
    return names


def clipping(features, phases, stats):
    normalized = stats.obs.normalize(features)
    clipped = np.abs(normalized) > OBS_CLIP
    names = feature_names()
    assert len(names) == features.shape[1]
    columns = []
    for i in np.flatnonzero(clipped.any(axis=0)):
        columns.append(
            dict(
                index=int(i),
                name=names[i],
                clipped_rows=int(clipped[:, i].sum()),
                fraction=float(clipped[:, i].mean()),
                train_std=float(stats.obs.std[i]),
                train_min=float(stats.obs.min[i]),
                train_max=float(stats.obs.max[i]),
                raw_min=float(features[:, i].min()),
                raw_max=float(features[:, i].max()),
                max_abs_z=float(np.abs(normalized[:, i]).max()),
            )
        )
    groups = {}
    for prefix in sorted({"/".join(name.split("/")[:2]) for name in names}):
        index = [
            i for i, name in enumerate(names) if name == prefix or name.startswith(prefix + "/")
        ]
        groups[prefix] = dict(
            entries=int(clipped[:, index].sum()),
            fraction=float(clipped[:, index].mean()),
            dimensions=len(index),
        )
    return dict(
        fraction=float(clipped.mean()),
        rows_with_clipping=int(clipped.any(axis=1).sum()),
        by_phase={str(p): float(clipped[phases == p].mean()) for p in np.unique(phases)},
        groups=groups,
        columns=sorted(columns, key=lambda row: -row["clipped_rows"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((REPO / "configs/b1_policy_pilot.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "protocol.json",
        dict(
            checkpoints="four final.pt; ACT seed-17 initialization is an untrained reference",
            starts="every sixth tick plus historical raw-metric 256 development windows",
            attribution="target phase and 0-based chunk position; exclude end padding",
            baselines="zero and global componentwise train median; no development fit",
            clipping="unchanged checkpoint z-score / +/-10; full episode feature audit",
            inference_seed=23,
            ddim_steps=10,
            training=False,
            sealed_test=False,
        ),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(4)
    report = dict(device=torch.cuda.get_device_name(), qualification=False, paths={}, clipping={})
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        data = load_b1_data(args.root / "data", space)
        train_actions = np.concatenate(
            [r.actions for r in data.dataset.records if r.split == "train"]
        )
        median = np.median(train_actions, axis=0)
        records = []
        for record in data.dataset.records:
            with h5py.File(args.root / record.asset_id / "episode.hdf5", "r") as raw:
                stages = np.array([PHASES[int(p)] for p in raw["commands/phase"][:]])
                start = np.flatnonzero(stages != "inspect")[0]
                stages = stages[start:]
                np.testing.assert_allclose(
                    raw["observations/time_s"][start:-1], record.times, atol=1e-9, rtol=0
                )
            assert len(stages) == len(record.actions)
            features = obs_matrix(record, data.stats.obs_keys)
            if space == A1_JOINT_DELTA:
                report["clipping"][record.asset_id] = clipping(features, stages, data.stats)
            starts = np.arange(0, len(features), protocol["replan_ticks"])
            historical = np.array([], dtype=int)
            if record.split == "development":
                historical = np.random.default_rng(protocol["training_seed"]).permutation(
                    len(features)
                )[: protocol["evaluation_chunks"]]
            records.append((record, features, stages, starts, historical))
        for family in ("act", "diffusion"):
            name = family + "-" + space
            policy = B1Policy.from_checkpoint(
                args.root / "training" / name / "final.pt",
                family,
                binding=data.dataset.binding,
                runtime_asset=data.robot_asset,
                device="cuda:0",
                num_inference_steps=protocol["ddim_steps"],
            )
            assert policy.policy.stats.to_dict() == data.stats.to_dict()
            initial = None
            if family == "act":
                model = make_model(
                    data,
                    family,
                    ActModelCfg(**vars(policy.policy.model.cfg)),
                    seed=protocol["training_seed"],
                    device="cuda:0",
                )
                initial = ActPolicy(model, data.stats, device="cuda:0")
            path_report = dict(train_median_native=median.tolist(), episodes={})
            for record, features, stages, starts, historical in records:
                query = np.unique(np.r_[starts, historical])
                policy.reset(protocol["inference_seed"])
                prediction = np.stack([policy.predict(features[t]) for t in query])
                predictions = dict(
                    final=prediction,
                    zero=np.zeros_like(prediction),
                    train_median=np.broadcast_to(median, prediction.shape),
                )
                if initial is not None:
                    predictions["initial_act"] = np.stack(
                        [initial.predict(features[t]) for t in query]
                    )
                target_index = query[:, None] + np.arange(protocol["chunk_ticks"])
                valid = target_index < len(features)
                safe_index = np.minimum(target_index, len(features) - 1)
                target = record.actions[safe_index]
                target_phases = stages[safe_index]
                position = np.broadcast_to(np.arange(protocol["chunk_ticks"]), valid.shape)
                use = valid & np.isin(query, starts)[:, None]
                historical_use = valid & np.isin(query, historical)[:, None]
                episode_report = dict(
                    split=record.split,
                    starts=len(starts),
                    queries=len(query),
                    valid_actions=int(use.sum()),
                    comparators={},
                )
                for comparator, value in predictions.items():
                    comparisons = dict(
                        overall=metrics(value[use], target[use], space),
                        by_phase={},
                        by_chunk_position={},
                        by_phase_and_position={},
                    )
                    if historical_use.any():
                        comparisons["historical_256"] = metrics(
                            value[historical_use], target[historical_use], space
                        )
                    for p in np.unique(stages):
                        mask = use & (target_phases == p)
                        comparisons["by_phase"][p] = metrics(value[mask], target[mask], space)
                        comparisons["by_phase_and_position"][p] = {}
                        for j in range(protocol["chunk_ticks"]):
                            mask = use & (target_phases == p) & (position == j)
                            if mask.any():
                                comparisons["by_phase_and_position"][p][str(j)] = metrics(
                                    value[mask], target[mask], space
                                )
                    for j in range(protocol["chunk_ticks"]):
                        mask = use & (position == j)
                        comparisons["by_chunk_position"][str(j)] = metrics(
                            value[mask], target[mask], space
                        )
                    episode_report["comparators"][comparator] = comparisons
                path_report["episodes"][record.asset_id] = episode_report
                np.savez_compressed(
                    args.output / f"{name}-{record.asset_id}.npz",
                    query=query,
                    starts=starts,
                    historical=historical,
                    target=target,
                    valid=valid,
                    phases=target_phases,
                    **predictions,
                )
                print(
                    json.dumps(dict(path=name, episode=record.asset_id, queries=len(query))),
                    flush=True,
                )
            report["paths"][name] = path_report
            write(args.output / "report.json", report)
    print(json.dumps(dict(completed=True, report=str(args.output / "report.json"))), flush=True)


if __name__ == "__main__":
    main()
