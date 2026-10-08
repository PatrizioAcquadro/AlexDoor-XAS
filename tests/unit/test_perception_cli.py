"""CLI process failures are distinct from diagnostic quality scores."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def cli_module():
    path = Path(__file__).resolve().parents[2] / "scripts/perception.py"
    spec = importlib.util.spec_from_file_location("perception_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_does_not_require_recordings(tmp_path, monkeypatch):
    from alexdoor_xas.perception.diagnostics import live
    from alexdoor_xas.recording import b1

    calls = []
    monkeypatch.setattr(live, "run_live_smoke", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(b1, "episode_paths", lambda *a: pytest.fail("live reads recordings"))
    assert (
        cli_module().main(
            [
                "point2pose-live",
                "--case",
                "camera",
                "--output",
                str(tmp_path / "run"),
                "--data",
                str(tmp_path / "missing"),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0][1]["case"] == "camera"


@pytest.mark.parametrize("outcome", ["complete", "exit_error", "missing", "failure"])
def test_live_matrix_propagates_child_failures(tmp_path, monkeypatch, outcome):
    cli = cli_module()

    def child(command, **kwargs):
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir(parents=True)
        if outcome != "missing":
            (folder / "report.json").write_text(json.dumps({"diagnostic_passed": False}))
        if outcome == "failure":
            (folder / "failure.json").write_text("{}")
        return SimpleNamespace(returncode=1 if outcome == "exit_error" else 0)

    monkeypatch.setattr(cli.subprocess, "run", child)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: "test-revision")
    args = SimpleNamespace(
        output=tmp_path / "matrix", asset="pilot", models="models", config="config"
    )
    recipe = SimpleNamespace(to_dict=lambda: {})
    assert cli.run_live_matrix(args, recipe) == (0 if outcome == "complete" else 1)
    assert len(json.loads((args.output / "runs.json").read_text())) == 4
