"""Explicit single diagnostics; process completion is separate from scientific quality."""

import importlib.util
from pathlib import Path

import pytest

PILOT = "animated-door-1-88abf40"


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

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"complete": True, "diagnostic_passed": False}

    monkeypatch.setattr(live, "run_live", run)
    monkeypatch.setattr(b1, "episode_paths", lambda *a: pytest.fail("live reads recordings"))
    assert (
        cli_module().main(
            [
                "point2pose-live",
                "--case",
                "camera",
                "--asset",
                PILOT,
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0][1]["case"] == "camera"


@pytest.mark.parametrize(
    "command",
    [
        "smoke",
        "diagnose-scan",
        "point2pose-smoke",
        "point2pose-offline",
        "point2pose-replay",
        "point2pose-operational",
    ],
)
@pytest.mark.parametrize("outcome", ["complete", "incomplete", "failure", "missing", "exception"])
def test_single_episode_propagates_failures_without_retry(tmp_path, monkeypatch, command, outcome):
    from alexdoor_xas.perception.diagnostics import offline, operational, replay, scan, smoke
    from alexdoor_xas.recording import b1

    output = tmp_path / "run"
    episode = tmp_path / PILOT / "nominal/episode.hdf5"
    episode.parent.mkdir(parents=True)
    episode.touch()
    calls = []

    def run(path, *args, **kwargs):
        calls.append(path)
        if outcome == "exception":
            raise RuntimeError("worker failed")
        if outcome == "failure":
            output.mkdir()
            (output / "failure.json").write_text("{}")
        return (
            {}
            if outcome == "missing"
            else {
                "complete": outcome != "incomplete",
                "diagnostic_passed": False,
            }
        )

    monkeypatch.setattr(smoke, "run_sam3_smoke", run)
    monkeypatch.setattr(smoke, "run_point2pose_smoke", run)
    monkeypatch.setattr(scan, "run_scan_diagnostic", run)
    monkeypatch.setattr(offline, "offline_episode", run)
    monkeypatch.setattr(replay, "run_replay", run)
    monkeypatch.setattr(operational, "recorded_operational", run)

    def validate(path, corpus, *, complete_campaign):
        assert path == episode and not complete_campaign
        return [path]

    monkeypatch.setattr(b1, "episode_paths", validate)
    argv = [command, "--episode", str(episode), "--output", str(output)]
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="worker failed"):
            cli_module().main(argv)
    else:
        assert cli_module().main(argv) == (0 if outcome == "complete" else 1)
    assert calls == [episode]


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
    "argv",
    [
        ["smoke", "--data", "unused"],
        ["smoke", "--seed", "unused"],
        ["diagnose-scan", "--selection", "unused"],
        ["point2pose-replay", "--mode", "replay"],
        ["point2pose-live", "--episode", "unused"],
    ],
)
def test_options_are_scoped_to_their_subcommand(tmp_path, argv):
    with pytest.raises(SystemExit) as stopped:
        cli_module().main([*argv, "--output", str(tmp_path / "run")])
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
