"""Prevent overlapping full datasets and preserve progress before RAM exhaustion."""

import importlib.util
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from alexdoor_xas.dataset import b1
from alexdoor_xas.resources import GIB, require_host_memory


@pytest.fixture
def offline():
    path = Path(__file__).resolve().parents[2] / "scripts/pilot_offline.py"
    spec = importlib.util.spec_from_file_location("pilot_offline_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_releases_A1_before_constructing_A2(tmp_path, monkeypatch, offline):
    datasets = []

    class Dataset:
        def __init__(self, directory, space):
            assert all(ref() is None for ref in datasets), "previous dataset still owned"
            datasets.append(weakref.ref(self))
            self.records = [
                SimpleNamespace(episode_id="train", split="train"),
                SimpleNamespace(episode_id="dev", split="development"),
            ]
            self.binding = self.robot_asset = object()
            self.space = space

        def close(self):
            pass

    class Policy:
        @staticmethod
        def from_checkpoint(path, family, **kwargs):
            stats = SimpleNamespace(train_episode_ids=("train",), to_dict=lambda: {})
            return SimpleNamespace(
                policy=SimpleNamespace(stats=stats, model=SimpleNamespace(state_dict=lambda: {}))
            )

    monkeypatch.setattr(b1, "B1Dataset", Dataset)
    monkeypatch.setattr(offline, "B1Policy", Policy)
    monkeypatch.setattr(offline, "raw_metrics", lambda *args: {"evaluated": True})
    monkeypatch.setattr(offline, "memory_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(offline, "dataset_loading_bytes", lambda *args: 0)
    monkeypatch.setattr(offline.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(offline.torch.cuda, "get_device_name", lambda: "unit-test-no-model")
    report = offline.evaluate_checkpoints(tmp_path, {"ddim_steps": 10}, tmp_path / "models")
    assert len(report) == 4 and len(datasets) == 2
    assert all(ref() is None for ref in datasets)


def test_loading_budget_counts_eager_arrays_and_leaves_images_on_disk(tmp_path, offline):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as file:
        # A chunked image dataset occupies no disk space until written.
        file.create_dataset("observations", (16, 128), maxshape=(None, 128), dtype="f4")
        file.create_dataset("time_s", (16,), maxshape=(None,), dtype="f8")
        for space, width in ((offline.A1_JOINT_DELTA, 7), (offline.A2_EE_DELTA, 6)):
            file.create_dataset(f"{space}/actions", (15, width), dtype="f4")
    meta = {"episodes": [path.name], "perception": {"config": {"visual_dims": [128]}}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    budget = offline.dataset_loading_bytes(tmp_path)
    with h5py.File(path, "r+") as file:
        file["observations"].resize((2**24, 128))
        file["time_s"].resize((32,))
    assert offline.dataset_loading_bytes(tmp_path) == budget + 16 * 8
    assert budget < GIB


def test_low_RAM_is_recorded_and_refused_before_allocation(tmp_path, monkeypatch, offline):
    state = dict(host_available_bytes=20 * GIB, process_rss_bytes=GIB)
    monkeypatch.setattr(offline, "host_memory_snapshot", lambda: state)
    with pytest.raises(MemoryError, match="host_ram_reserve"):
        offline.memory_checkpoint(tmp_path, "before_A2_load", additional_bytes=24 * GIB)
    record = json.loads((tmp_path / "host-memory.jsonl").read_text())
    assert record["stage"] == "before_A2_load" and record["additional_bytes"] == 24 * GIB
    require_host_memory(state, additional_bytes=8 * GIB)
    with pytest.raises(ValueError, match="nonnegative"):
        require_host_memory(state, additional_bytes=-1)
