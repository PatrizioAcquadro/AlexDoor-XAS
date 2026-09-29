"""Autonomous lifecycle checks; no training or model weight updates."""

import json
import subprocess
from dataclasses import asdict

import pytest

from alexdoor_xas.perception.run import EarlyStopping, launch_detached, save_json


def test_small_cumulative_gains_reset_patience_and_resume(tmp_path):
    settings = dict(patience_evaluations=3, min_relative_improvement=0.005, min_epochs=5)
    stopper = EarlyStopping(settings)
    assert not stopper.update(100, 1)
    assert not stopper.update(99.8, 2)
    assert not stopper.update(99.6, 3)
    assert not stopper.update(99.4, 4)
    assert stopper.bad_evaluations == 0
    assert not stopper.update(99.3, 5)
    path = tmp_path / "state.json"
    save_json(path, asdict(stopper))
    resumed = EarlyStopping(**json.loads(path.read_text()))
    assert not resumed.update(99.2, 6)
    assert resumed.update(99.1, 7)
    assert resumed.reference_score == 99.4


def test_warmup_is_not_shortened_by_a_flat_curve_and_nan_fails():
    stopper = EarlyStopping(
        dict(patience_evaluations=2, min_relative_improvement=0.005, min_epochs=5)
    )
    for epoch in range(1, 5):
        assert not stopper.update(10, epoch)
    assert stopper.update(10, 5)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        stopper.update(float("nan"), 6)
    with pytest.raises(ValueError, match="positive integer"):
        EarlyStopping(dict(patience_evaluations=0, min_relative_improvement=0.005, min_epochs=5))


def test_launch_uses_service_and_persistent_files_without_waiting(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    output = tmp_path / "run-01"
    command = ["/some path/isaaclab.sh", "-p", "script.py", "train"]
    record = launch_detached(command, output, tmp_path)
    submitted, options = calls[0]
    assert submitted[-len(command) :] == command
    assert options == {"check": True}
    assert "--wait" not in submitted
    assert "--property=Restart=no" in submitted
    assert f"--property=StandardOutput=append:{tmp_path / 'run-01.console.log'}" in submitted
    assert json.loads((tmp_path / "run-01.launch.jsonl").read_text()) == record
    assert not output.exists()  # The actual trainer owns creation of its run directory.
    output.mkdir()
    with pytest.raises(ValueError, match="new run"):
        launch_detached(command, output, tmp_path)
    launch_detached(command, output, tmp_path, resume=True)
    assert len((tmp_path / "run-01.launch.jsonl").read_text().splitlines()) == 2
