"""Explicit single diagnostics; process completion is separate from scientific quality."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

PILOT = "animated-door-1-88abf40"


def cli_module():
    path = Path(__file__).resolve().parents[2] / "scripts/perception.py"
    spec = importlib.util.spec_from_file_location("perception_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("complete", [True, False])
def test_live_does_not_require_recordings(tmp_path, monkeypatch, complete):
    from alexdoor_xas.perception.diagnostics import live
    from alexdoor_xas.recording import b1

    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"complete": complete, "diagnostic_passed": False}

    monkeypatch.setattr(live, "run_live", run)
    monkeypatch.setattr(b1, "episode_paths", lambda *a: pytest.fail("live reads recordings"))
    assert cli_module().main(
        [
            "point2pose-live",
            "--case",
            "camera",
            "--asset",
            PILOT,
            "--output",
            str(tmp_path / "run"),
        ]
    ) == int(not complete)
    assert len(calls) == 1 and calls[0][1]["case"] == "camera"


@pytest.fixture
def recorded_diagnostic(tmp_path, monkeypatch):
    from alexdoor_xas.recording import b1

    episode = tmp_path / PILOT / "nominal/episode.hdf5"
    episode.parent.mkdir(parents=True)
    episode.touch()
    output = tmp_path / "run"

    def validate(path, corpus, *, complete_campaign):
        assert path == episode and not complete_campaign
        return [path]

    monkeypatch.setattr(b1, "episode_paths", validate)
    return episode, output


@pytest.mark.parametrize(
    "command,handler",
    [
        ("smoke", "smoke.run_sam3_smoke"),
        ("diagnose-scan", "scan.run_scan_diagnostic"),
        ("point2pose-smoke", "smoke.run_point2pose_smoke"),
        ("point2pose-offline", "offline.offline_episode"),
        ("point2pose-replay", "replay.run_replay"),
        ("point2pose-operational", "operational.recorded_operational"),
    ],
)
def test_recorded_command_dispatches_once(recorded_diagnostic, monkeypatch, command, handler):
    episode, output = recorded_diagnostic
    worker = Mock(return_value={"complete": True, "diagnostic_passed": False})
    monkeypatch.setattr(f"alexdoor_xas.perception.diagnostics.{handler}", worker)
    assert cli_module().main([command, "--episode", str(episode), "--output", str(output)]) == 0
    worker.assert_called_once()
    assert worker.call_args.args[0] == episode


@pytest.mark.parametrize(
    "report",
    [{"complete": False}, {}, {"complete": True, "failure": "lost"}, {"complete": True}, None],
    ids=["incomplete", "missing-status", "reported-failure", "failure-file", "exception"],
)
def test_recorded_failure_propagates_without_retry(recorded_diagnostic, monkeypatch, report):
    episode, output = recorded_diagnostic

    def run(*args, **kwargs):
        if report is None:
            raise RuntimeError("worker failed")
        if report == {"complete": True}:
            output.mkdir()
            (output / "failure.json").write_text("{}")
        return report

    worker = Mock(side_effect=run)
    monkeypatch.setattr("alexdoor_xas.perception.diagnostics.offline.offline_episode", worker)
    argv = ["point2pose-offline", "--episode", str(episode), "--output", str(output)]
    if report is None:
        with pytest.raises(RuntimeError, match="worker failed"):
            cli_module().main(argv)
    else:
        assert cli_module().main(argv) == 1
    worker.assert_called_once()


@pytest.mark.parametrize(
    "argv",
    [
        ["smoke"],
        ["diagnose-scan"],
        ["point2pose-smoke"],
        ["point2pose-offline"],
        ["point2pose-replay"],
        ["point2pose-operational"],
        ["point2pose-live"],
        ["point2pose-live", "--asset", PILOT],
        ["point2pose-live", "--case", "camera"],
        ["point2pose-live-smoke"],
    ],
)
def test_diagnostics_require_explicit_inputs(tmp_path, argv):
    with pytest.raises(SystemExit) as stopped:
        cli_module().main([*argv, "--output", str(tmp_path / "run")])
    assert stopped.value.code == 2
    assert not (tmp_path / "run").exists()


def test_single_episode_cannot_bypass_recording_admission(tmp_path, monkeypatch):
    from alexdoor_xas.perception.diagnostics import offline
    from alexdoor_xas.recording import b1

    def reject(*args, **kwargs):
        raise ValueError("outside frozen corpus")

    monkeypatch.setattr(b1, "episode_paths", reject)
    monkeypatch.setattr(offline, "offline_episode", lambda *a, **kw: pytest.fail("worker started"))
    episode = tmp_path / PILOT / "nominal/episode.hdf5"
    episode.parent.mkdir(parents=True)
    episode.touch()
    with pytest.raises(ValueError, match="frozen corpus"):
        cli_module().main(
            [
                "point2pose-offline",
                "--episode",
                str(tmp_path / PILOT / "nominal/episode.hdf5"),
                "--output",
                str(tmp_path / "result"),
            ]
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "command,extra",
    [
        ("smoke", ["--data", "unused"]),
        ("smoke", ["--seed", "unused"]),
        ("diagnose-scan", ["--selection", "unused"]),
        ("point2pose-replay", ["--mode", "replay"]),
        ("point2pose-live", ["--episode", "unused"]),
    ],
)
def test_options_are_scoped_to_their_subcommand(tmp_path, command, extra):
    episode = tmp_path / PILOT / "nominal/episode.hdf5"
    episode.parent.mkdir(parents=True)
    episode.touch()
    required = (
        ["--asset", PILOT, "--case", "camera"]
        if command == "point2pose-live"
        else ["--episode", str(episode)]
    )
    with pytest.raises(SystemExit) as stopped:
        cli_module().main([command, *required, *extra, "--output", str(tmp_path / "run")])
    assert stopped.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["missing", "directory", "nonpilot", "existing_output"])
def test_invalid_paths_fail_before_any_worker(tmp_path, monkeypatch, kind):
    from alexdoor_xas.perception.diagnostics import smoke

    episode = tmp_path / (PILOT if kind != "nonpilot" else "other") / "nominal/episode.hdf5"
    episode.parent.mkdir(parents=True)
    if kind == "directory":
        episode.mkdir()
    elif kind != "missing":
        episode.touch()
    output = tmp_path / "output"
    if kind == "existing_output":
        output.mkdir()
    monkeypatch.setattr(smoke, "run_sam3_smoke", lambda *a: pytest.fail("worker started"))
    with pytest.raises(SystemExit) as stopped:
        cli_module().main(["smoke", "--episode", str(episode), "--output", str(output)])
    assert stopped.value.code == 2
