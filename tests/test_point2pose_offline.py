"""Offline scheduling/scoring and native versus integration failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alexdoor_xas.action.frames import rot_z
from alexdoor_xas.perception import point2pose_offline as offline
from alexdoor_xas.perception.point2pose_seed import initialization_checks
from alexdoor_xas.perception.point2pose_worker import pack_array
from alexdoor_xas.perception.provider import PrototypeRecipe
from alexdoor_xas.recording.b1 import OBS_KEYS
from test_point2pose_temporal import make_surface, sensor


def recording(path, rows=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata").attrs["calibration"] = json.dumps(
            dict(depth_interval_m=[0.1, 3])
        )
        observations = h5.create_group("observations")
        samples = [sensor(4 + (row - 2) / 60, row + 8) for row in range(rows)]
        for key in OBS_KEYS:
            observations[key] = np.stack([sample[key] for sample in samples])
        labels = h5.create_group("annotations")
        labels["hinge_origin"] = np.zeros((rows, 3))
        labels["hinge_rotation"] = np.tile(np.eye(3), (rows, 1, 1))
        labels["signed_angle"] = (np.arange(rows) - 2) * 0.01
        h5.create_group("commands")["privileged"] = np.ones(rows)
    return path


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        workers=[],
        latency=0.02,
        rejected=False,
        loss=None,
        sparse=None,
        fail=None,
        invalid=None,
        interrupt=None,
    )

    class Visual:
        def __init__(self, *_):
            self.closed = False
            self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)

        def infer(self, rgb):
            return dict(latency_s=state.latency)

        def close(self):
            assert not self.closed
            self.closed = True

    class Native:
        def __init__(self, *_, diagnostic_only=False):
            assert diagnostic_only
            self.boot_latency_s = 1.0
            self.calls = []
            self.closed = False
            self.seed_frame = None
            self.camera = None
            state.workers.append(self)

        def infer(self, observed, masks=None):
            assert set(observed) == set(OBS_KEYS)
            self.calls.append({key: np.array(value, copy=True) for key, value in observed.items()})
            frame = int(observed["frame"])
            if masks is not None:
                assert self.seed_frame is None
                self.seed_frame = frame
                self.camera = observed["camera_world"].copy()
            relative_frame = frame - self.seed_frame
            if relative_frame == state.interrupt:
                raise KeyboardInterrupt
            if relative_frame == state.fail:
                raise RuntimeError("native_test_failure")
            transform = np.eye(4)
            pose_frame = relative_frame - 1 if relative_frame == state.loss else relative_frame
            transform[:3, :3] = rot_z(pose_frame * 0.01)
            transform = np.linalg.inv(observed["camera_world"]) @ transform @ self.camera
            if relative_frame == state.invalid:
                transform[0, 0] = np.nan
            source = np.array(
                [
                    [-0.25, -0.25, 1],
                    [0.25, -0.25, 1],
                    [0.25, 0.25, 1],
                    [-0.25, 0.25, 1],
                    [0, 0.2, 1],
                    [0.2, 0, 1],
                ]
            )
            measured = np.full(6, relative_frame != state.sparse)
            obj = dict(
                object_id=0,
                camera_from_map=pack_array(transform),
                lost=relative_frame == state.loss,
                source_points=pack_array(source),
                current_points=pack_array(source @ transform[:3, :3].T + transform[:3, 3]),
                measured_correspondences=pack_array(measured),
                inliers=pack_array(np.ones(6, bool)),
                tsdf_gpu=True,
                tsdf_voxels=100,
            )
            return dict(
                frame=frame,
                capture_s=float(observed["time_s"]),
                objects=[obj],
                latency_s=state.latency,
                model_latency_s=state.latency,
                initialization_checks=[
                    dict(
                        object_id=0,
                        accepted=not state.rejected,
                        reason="mask_rejected" if state.rejected else None,
                    )
                ],
            )

        def close(self):
            assert not self.closed
            self.closed = True

    monkeypatch.setattr(offline, "ModelWorker", Visual)
    monkeypatch.setattr(offline, "Point2PoseWorker", Native)
    monkeypatch.setattr(offline, "surfaces", lambda cue, observed, config: [make_surface(observed)])
    state.recipe = PrototypeRecipe(
        json.dumps(
            dict(
                scan_fusion="object-v1",
                plane_tolerance_m=0.004,
                voxel_m=0.004,
                association_angle_deg=5,
                association_distance_m=0.025,
                min_points=120,
                semantic_period_s=0.2,
                inspection=dict(sample_times_s=[4.0, 25.0]),
            )
        )
    )
    return state


def run(path, output, runtime, **kwargs):
    report = offline.offline_episode(path, runtime.recipe, output, Path("unused"), **kwargs)
    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    return report, frames


def test_all_frames_last_capture_and_latency_independence(tmp_path, runtime):
    path = recording(tmp_path / "episode.hdf5")
    fast, fast_frames = run(path, tmp_path / "fast", runtime)
    runtime.latency = 2.0
    slow, slow_frames = run(path, tmp_path / "slow", runtime)
    assert fast["primary"] == slow["primary"]
    assert fast["native_result_rows"] == slow["native_result_rows"] == 6
    assert [r["row"] for r in slow_frames] == list(range(8))
    assert [int(r["frame"]) for r in runtime.workers[1].calls] == list(range(10, 16))
    assert fast_frames[-1]["frame"] == slow_frames[-1]["frame"] == 15
    assert slow["latency"]["native_ipc_s"]["p95"] == 2
    assert (
        slow["primary"]["all_frames"]["errors_non_seed"]["native_tracking"]["position_m"]["maximum"]
        < 1e-10
    )
    assert all(worker.closed for worker in runtime.workers)


@pytest.mark.parametrize("interrupt", [None, 1])
def test_bounded_window_retains_original_rows_and_complete_denominator(
    tmp_path, runtime, interrupt
):
    runtime.interrupt = interrupt
    report, frames = run(
        recording(tmp_path / "episode.hdf5"),
        tmp_path / "bounded",
        runtime,
        capture_window_s=(4.0, 4.05),
    )
    assert [r["row"] for r in frames] == [2, 3, 4, 5]
    assert [r["frame"] for r in frames] == [10, 11, 12, 13]
    assert report["scheduled_rows"] == 4 and report["recorded_rows"] == 8
    assert report["capture_window_s"] == (4.0, 4.05)
    assert report["not_processed_rows"] == (3 if interrupt else 0)
    assert report["complete"] == (interrupt is None)
    assert len(runtime.workers) == 1 and runtime.workers[0].closed


def test_bounded_window_initializes_after_inspection_schedule(tmp_path, runtime):
    config = runtime.recipe.config
    config["inspection"]["sample_times_s"] = [0.0, 1.0]
    runtime.recipe = PrototypeRecipe(json.dumps(config))
    report, frames = run(
        recording(tmp_path / "episode.hdf5"),
        tmp_path / "late-window",
        runtime,
        capture_window_s=(4.0, 4.05),
    )
    assert report["complete"] and report["native_result_rows"] == 4
    assert report["initialization"]["seed_time_s"] == 4.0
    assert frames[0]["is_seed"]
    assert [int(r["frame"]) for r in runtime.workers[0].calls] == [10, 11, 12, 13]


def test_truth_only_changes_scoring_and_source_time_is_used(tmp_path, runtime):
    path = recording(tmp_path / "episode.hdf5")
    before, _ = run(path, tmp_path / "before", runtime)
    with h5py.File(path, "r+") as h5:
        h5["annotations/signed_angle"][:] *= 2
        h5["commands/privileged"][:] = 999
    runtime.latency = 10.0
    after, _ = run(path, tmp_path / "after", runtime)
    for a, b in zip(runtime.workers[0].calls, runtime.workers[1].calls, strict=True):
        for key in OBS_KEYS:
            np.testing.assert_array_equal(a[key], b[key])
    error = "errors_non_seed"
    assert (
        before["primary"]["all_frames"][error]["native_tracking"]["rotation_deg"]["maximum"] < 1e-10
    )
    assert after["primary"]["all_frames"][error]["native_tracking"]["rotation_deg"]["maximum"] > 2


def test_loss_and_integration_rejection_have_separate_recovery(tmp_path, runtime):
    runtime.sparse, runtime.loss = 1, 2
    report, frames = run(recording(tmp_path / "episode.hdf5"), tmp_path / "result", runtime)
    assert frames[3]["objects"][0]["native_tracking"]
    assert not frames[3]["objects"][0]["integration_accepted"]
    assert frames[4]["objects"][0]["native_pose_valid"]
    assert not frames[4]["objects"][0]["native_tracking"]
    primary = report["primary"]
    assert primary["native_continuity"]["losses"] == 1
    assert primary["native_continuity"]["recoveries"] == 1
    assert primary["integration_continuity"]["losses"] == 1
    assert primary["integration_continuity"]["interruptions"][0]["duration_s"] == pytest.approx(
        2 / 60
    )
    assert primary["all_frames"]["errors_non_seed"]["native_lost_finite"]["position_m"]["n"] == 1
    assert (
        primary["all_frames"]["errors_non_seed"]["native_lost_finite"]["rotation_deg"]["maximum"]
        > 0
    )


def test_rejected_initialization_continues_native_and_never_becomes_accepted(tmp_path, runtime):
    runtime.rejected = True
    report, frames = run(recording(tmp_path / "episode.hdf5"), tmp_path / "result", runtime)
    assert report["initialization"]["native_completed"]
    assert len(runtime.workers[0].calls) == 6
    assert report["primary"]["after_seed"]["native_coverage"] == 1
    assert report["primary"]["all_frames"]["integration_coverage"] == 0
    assert all(r["objects"][0]["integration_rejections"] == ["mask_rejected"] for r in frames[2:])


def test_failure_retains_full_denominator_and_stops_native_calls(tmp_path, runtime):
    runtime.fail = 2
    report, frames = run(recording(tmp_path / "episode.hdf5"), tmp_path / "result", runtime)
    assert not report["complete"]
    assert report["scheduled_rows"] == report["primary"]["all_frames"]["rows"] == 8
    assert report["statuses"] == dict(
        not_initialized=2, native_result=2, process_error=1, not_processed=3
    )
    assert len(runtime.workers[0].calls) == 3
    assert frames[-1]["status"] == "not_processed"
    assert (tmp_path / "result/failure.json").exists()


def test_invalid_native_pose_is_explicit_and_does_not_refresh_tracking(tmp_path, runtime):
    runtime.invalid = 1
    report, frames = run(recording(tmp_path / "episode.hdf5"), tmp_path / "result", runtime)
    obj = frames[3]["objects"][0]
    assert obj["world_pose"] is None and obj["capture_error"] is None
    assert not obj["native_tracking"] and not obj["integration_accepted"]
    assert "invalid_native_pose" in obj["integration_rejections"]
    assert report["primary"]["all_frames"]["invalid_native_pose_rows"] == 1


def test_three_fresh_trials_per_pilot_and_one_second_extra_window(tmp_path, runtime):
    paths = [
        recording(tmp_path / pilot / condition / "episode.hdf5", rows=70)
        for pilot in offline.PILOTS
        for condition in ("nominal", "light")
    ]
    reports = offline.run_offline(paths, runtime.recipe, tmp_path / "campaign", Path("unused"))
    assert len(reports) == len(runtime.workers) == 12
    assert [r["scheduled_rows"] for r in reports] == [70, 63, 63] * 4
    assert len({id(worker) for worker in runtime.workers}) == 12
    assert all(worker.closed for worker in runtime.workers)
    campaign = json.loads((tmp_path / "campaign/report.json").read_text())
    assert campaign["full_recording_frames"] == 280
    assert campaign["native_initializations"] == 12
    assert not campaign["qualified"] and not campaign["offline_passed"]


def test_user_interrupt_preserves_prefix_denominator_and_stops_campaign(tmp_path, runtime):
    runtime.interrupt = 3
    paths = [
        recording(tmp_path / pilot / condition / "episode.hdf5")
        for pilot in offline.PILOTS
        for condition in ("nominal", "light")
    ]
    output = tmp_path / "campaign"
    reports = offline.run_offline(paths, runtime.recipe, output, Path("unused"))
    assert len(reports) == len(runtime.workers) == 1
    assert runtime.workers[0].closed
    report = reports[0]
    assert report["termination"] == "user_requested_stop" and not report["complete"]
    assert report["scheduled_rows"] == 8 and report["not_processed_rows"] == 3
    assert report["native_result_rows"] == 3
    assert report["primary"]["all_frames"]["native_coverage"] == 3 / 8
    frames = [
        json.loads(line) for line in next(output.rglob("frames.jsonl")).read_text().splitlines()
    ]
    assert [r["row"] for r in frames] == list(range(8))
    assert all(r["objects"] == [] and r["latency"] is None for r in frames[5:])
    campaign = json.loads((output / "report.json").read_text())
    assert campaign["completed_attempts"] == 0 and campaign["not_started_attempts"] == 11
    assert campaign["expected_full_recording_frames"] == 32
    assert campaign["not_processed_full_recording_frames"] == 27


@pytest.mark.parametrize("kind", ["native", "visual"])
def test_interrupted_worker_startup_closes_unreturned_process(tmp_path, monkeypatch, kind):
    from io import BytesIO

    from alexdoor_xas.perception import point2pose_runtime, provider

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdin, self.stdout = BytesIO(), BytesIO()
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, **kwargs):
            self.stopped = True

    process = Process()
    module = point2pose_runtime if kind == "native" else provider
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(module, "send", lambda *args: None)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "receive", interrupt)
    monkeypatch.setattr(
        point2pose_runtime.threading,
        "Thread",
        lambda **kw: SimpleNamespace(start=lambda: None, join=lambda **kw: None),
    )
    with pytest.raises(KeyboardInterrupt):
        if kind == "native":
            point2pose_runtime.Point2PoseWorker(tmp_path, [0.1, 3], 0.004, 1, tmp_path / "logs")
        else:
            provider.ModelWorker(tmp_path, {})
    assert process.stopped and process.stdin.closed and process.stdout.closed


def test_mask_guard_is_strict_operationally_and_diagnostic_per_candidate():
    expected = np.ones((8, 8), bool)
    actual = expected.copy()
    actual[3, 3] = False
    prompts = np.array([[3, 3]])
    with pytest.raises(ValueError, match="sam2_initialization_candidate_mismatch"):
        initialization_checks([expected], [actual], [prompts])
    checks = initialization_checks(
        [expected, expected], [actual, expected], [prompts, prompts], diagnostic_only=True
    )
    assert not checks[0]["accepted"] and checks[1]["accepted"]
