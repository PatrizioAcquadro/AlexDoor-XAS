"""Prevent overlapping full datasets and preserve progress before RAM exhaustion."""

import importlib.util
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
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
    monkeypatch.setattr(offline, "dataset_ram_bytes", lambda *args: 0)
    monkeypatch.setattr(offline.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(offline.torch.cuda, "get_device_name", lambda: "unit-test-no-model")
    report = offline.evaluate_checkpoints(tmp_path, {"ddim_steps": 10}, tmp_path / "models")
    assert len(report) == 4 and len(datasets) == 2
    assert all(ref() is None for ref in datasets)


def test_logical_VDS_bytes_are_not_mistaken_for_compact_file_size(tmp_path, offline):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w", libver="latest") as file:
        file.create_dataset("unique", data=np.ones((1, 128)), compression="gzip")
        layout = h5py.VirtualLayout((1024, 128), dtype="f8")
        source = h5py.VirtualSource(".", "unique", shape=(1, 128))
        for index in range(1024):
            layout[index : index + 1] = source
        file.create_virtual_dataset("observations", layout)
    (tmp_path / "meta.json").write_text(json.dumps({"episodes": [path.name]}))
    assert offline.dataset_ram_bytes(tmp_path) == 1024 * 128 * 8
    assert path.stat().st_size < offline.dataset_ram_bytes(tmp_path)


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
