"""Train/development-only feature preparation and causal windows for B1 perception."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from alexdoor_xas.perception.model import preprocess
from alexdoor_xas.recording.b1 import validate_episode

TARGETS = (
    "hinge_origin",
    "hinge_rotation",
    "signed_angle",
    "dimensions",
    "contact_local",
    "contact_rotation_local",
    "contact_position",
    "contact_rotation",
)


def episode_paths(root, corpus, *, complete_campaign=True):
    entries = {e["asset_id"]: e for e in corpus["doors"] if e["split"] != "test"}
    paths = sorted(Path(root).glob("*/*/episode.hdf5"))
    found = set()
    for path in paths:
        validate_episode(path, images=False)
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5.attrs["metadata"])
        entry = entries.get(meta["asset_id"])
        if entry is None or any(
            meta[k] != entry[k] for k in ("split", "handedness", "records_sha256")
        ):
            raise ValueError(f"Episode does not belong to the frozen corpus: {path}")
        key = (meta["asset_id"], meta["condition"])
        if key in found or meta["condition"] not in ("nominal", "light"):
            raise ValueError("Duplicate or unknown perception condition")
        found.add(key)
    if not paths:
        raise ValueError("No B1 recordings")
    expected = {(asset, condition) for asset in entries for condition in ("nominal", "light")}
    if complete_campaign and found != expected:
        raise ValueError(f"Incomplete perception campaign: {len(found)}/{len(expected)} episodes")
    return paths


@torch.no_grad()
def prepare_features(paths, destination, backbone, config, device):
    """Cache deterministic frozen features, not fitted statistics. Refuse overwrite."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    index = []
    with (Path(backbone.encoder.config._name_or_path) / "model.safetensors").open("rb") as stream:
        backbone_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    for path in paths:
        with h5py.File(path, "r") as source:
            meta = json.loads(source.attrs["metadata"])
            filename = f"{meta['asset_id']}--{meta['condition']}.hdf5"
            out = destination / filename
            obs = source["observations"]
            dt = meta["control_dt"]
            stride = round(1 / config["sample_hz"] / dt)
            if not np.isclose(stride * dt, 1 / config["sample_hz"]):
                raise ValueError("Perception sampling must divide control cadence")
            ids = np.arange(0, len(obs["time_s"]), stride)
            with h5py.File(out, "x") as cache:
                cache.attrs.update(
                    metadata=json.dumps(meta),
                    source=str(path.resolve()),
                    config=json.dumps(config),
                    complete=False,
                )
                cache.create_dataset("time_s", data=obs["time_s"][ids])
                cache.create_dataset("frame", data=obs["frame"][ids])
                cache.create_dataset(
                    "proprio",
                    data=np.concatenate(
                        (obs["joint_position"][ids], obs["joint_velocity"][ids]), -1
                    ),
                )
                cache.create_dataset("camera", data=obs["camera_world"][ids].astype(np.float32))
                for key in (*TARGETS, "phase"):
                    cache.create_dataset("labels/" + key, data=source["annotations/" + key][ids])
                for start in range(0, len(ids), 16):
                    batch = ids[start : start + 16]
                    values = {
                        k: torch.as_tensor(np.stack([obs[k][int(i)] for i in batch]), device=device)
                        for k in ("rgb", "depth_m", "valid_depth", "intrinsics")
                    }
                    rgb, geometry, _ = preprocess(
                        values["rgb"],
                        values["depth_m"],
                        values["valid_depth"],
                        values["intrinsics"],
                        config["image_size"],
                    )
                    features = backbone(rgb)
                    # Preserve masked XYZ sums; the head divides by valid fraction.
                    geometry = F.adaptive_avg_pool2d(geometry, features.shape[-2:])
                    for key, value in (("features", features), ("geometry", geometry)):
                        dtype = np.float16 if key == "features" else np.float32
                        array = value.cpu().numpy().astype(dtype)
                        if key not in cache:
                            cache.create_dataset(
                                key,
                                shape=(len(ids), *array.shape[1:]),
                                dtype=dtype,
                                chunks=(1, *array.shape[1:]),
                                compression="lzf",
                            )
                        cache[key][start : start + len(batch)] = array
                cache.attrs["complete"] = True
            index.append(
                dict(
                    path=filename,
                    asset_id=meta["asset_id"],
                    split=meta["split"],
                    handedness=meta["handedness"],
                    condition=meta["condition"],
                    frames=len(ids),
                )
            )
            print(json.dumps(dict(prepared=filename, frames=len(ids))), flush=True)
    (destination / "index.json").write_text(
        json.dumps(dict(config=config, backbone_sha256=backbone_sha256, episodes=index), indent=2)
        + "\n"
    )


class PerceptionWindows(Dataset):
    def __init__(self, root, split, config):
        if split not in ("train", "development"):
            raise ValueError("Only train/development can be loaded")
        self.root, self.config = Path(root), config
        manifest = json.loads((self.root / "index.json").read_text())
        if manifest["config"] != config:
            raise ValueError("Feature preprocessing/config mismatch")
        self.episodes = [e for e in manifest["episodes"] if e["split"] == split]
        self.windows, self.groups = [], []
        self.window_phases = {}
        for number, entry in enumerate(self.episodes):
            with h5py.File(self.root / entry["path"], "r") as h5:
                if not h5.attrs["complete"]:
                    raise ValueError("Incomplete feature cache")
                times, phases = h5["time_s"][:], h5["labels/phase"][:]
                for end in range(config["history"] - 1, len(times)):
                    timespan = times[end - config["history"] + 1 : end + 1]
                    if not np.allclose(np.diff(timespan), 1 / config["sample_hz"], atol=1e-6):
                        raise ValueError("History contains missing or noncausal samples")
                    self.windows.append((number, end))
                    self.groups.append((entry["asset_id"], int(phases[end])))
                    self.window_phases[(number, end)] = int(phases[end])
        if not self.windows:
            raise ValueError(f"No causal {split} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        number, end = self.windows[index]
        entry = self.episodes[number]
        start = end - self.config["history"] + 1
        with h5py.File(self.root / entry["path"], "r") as h5:
            inputs = {
                key: torch.from_numpy(h5[key][start : end + 1].astype(np.float32))
                for key in ("features", "geometry", "proprio", "camera")
            }
            target = {
                key: torch.as_tensor(h5["labels/" + key][end], dtype=torch.float32)
                for key in TARGETS
            }
        return inputs, target, number, end

    def weights(self):
        from collections import Counter

        counts = Counter(self.groups)
        phases = Counter(asset for asset, _ in counts)
        return torch.tensor(
            [1 / (counts[group] * phases[group[0]]) for group in self.groups], dtype=torch.double
        )

    def normalization(self):
        if any(e["split"] != "train" for e in self.episodes):
            raise ValueError("Normalization may only be fitted to train doors")
        values = []
        for entry in self.episodes:
            with h5py.File(self.root / entry["path"], "r") as h5:
                values.append(h5["proprio"][:].astype(np.float64))
        # Equal episode weighting prevents long opening traces dominating scale.
        mean = np.mean([v.mean(0) for v in values], 0)
        second = np.mean([(v * v).mean(0) for v in values], 0)
        return mean.astype(np.float32), np.sqrt(np.maximum(second - mean * mean, 1e-6)).astype(
            np.float32
        )


def validate_feature_corpus(root, paths, config, backbone_path):
    """Bind cached features to the active corpus episodes and frozen backbone."""
    manifest = json.loads((Path(root) / "index.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Feature configuration differs")
    with (Path(backbone_path) / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if manifest["backbone_sha256"] != digest:
        raise ValueError("Backbone weights differ from cached features")
    sources = {}
    for path in paths:
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5.attrs["metadata"])
        sources[(meta["asset_id"], meta["condition"])] = (path.resolve(), meta)
    seen = set()
    for entry in manifest["episodes"]:
        key = (entry["asset_id"], entry["condition"])
        if key not in sources or key in seen:
            raise ValueError("Feature corpus has an unknown or duplicate episode")
        seen.add(key)
        source, meta = sources[key]
        path = (Path(root) / entry["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Feature path escapes dataset")
        with h5py.File(path, "r") as cache:
            if not cache.attrs["complete"] or json.loads(cache.attrs["metadata"]) != meta:
                raise ValueError("Stale or incomplete feature metadata")
            if Path(cache.attrs["source"]).resolve() != source:
                raise ValueError("Feature source differs")
            if entry["split"] != meta["split"] or entry["handedness"] != meta["handedness"]:
                raise ValueError("Feature split/handedness differs")
    if seen != set(sources):
        raise ValueError("Feature corpus is incomplete")
    return manifest
