"""Complete/failure/interruption and requested versus applied command semantics."""

import json

import h5py
import numpy as np
import pytest

from helpers import temporal_command as record_command
from helpers import temporal_recorder as recorder


@pytest.mark.parametrize("image_compression", [None, "lzf", "gzip"])
def test_failed_episode_native_clocks_requests_and_lossless_images(tmp_path, image_compression):
    path = tmp_path / "failure.hdf5"
    log = recorder(path, image_compression=image_compression)
    log.reset(0)
    depth = np.ones((4, 6, 1), "f4")
    depth.view("u4")[0, 0, 0] = 0x7FC00017
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    log.capture(
        np.zeros((4, 6, 3), "u1"),
        depth,
        np.isfinite(depth),
        np.eye(3),
        np.eye(4),
        acquired=0,
        available=0,
        generation=0,
    )
    record_command(log, 0)
    with pytest.raises(ValueError, match="pre-state"):
        record_command(log, 0)
    log.feedback(np.zeros(7), acquired=1 / 120, available=1 / 120, healthy=True, generation=0)
    log.state(np.zeros(9), np.zeros(9), acquired=1 / 60, available=1 / 60, generation=0)
    record_command(log, 1 / 60, accepted=False)
    log.state(np.zeros(9), np.zeros(9), acquired=2 / 60, available=2 / 60, generation=0)
    log.finish(dict(passed=False, termination="safe_refusal"))
    log.close()
    with h5py.File(path) as file:
        assert file.attrs["complete"]
        assert file["captures/depth_m"].compression == image_compression
        assert not json.loads(file["metadata"].attrs["outcome"])["passed"]
        np.testing.assert_array_equal(file["captures/depth_m"][0].view("u4"), depth.view("u4"))
        np.testing.assert_array_equal(file["commands/requested_target"][0], 1)
        np.testing.assert_array_equal(file["commands/applied_target"][0], 0.5)
        assert not file["commands/accepted"][1]
        assert np.isnan(file["commands/dispatch_s"][1])
        assert np.isnan(file["commands/applied_s"][1])
        assert np.isnan(file["commands/applied_target"][1]).all()
        assert len(file["control/acquired"]) == 3 and len(file["captures/acquired"]) == 1
        from alexdoor_xas.dataset.temporal import raw_streams
        from alexdoor_xas.policies.temporal import CausalObservationBuilder, TemporalConfig

        cfg = TemporalConfig(image_size=(2, 3), static_time_s=0, warmup_s=0)
        streams, *_ = raw_streams(file, cfg)
        np.testing.assert_array_equal(streams["target"]["acquired"], [0])
        snapshot, _ = CausalObservationBuilder(cfg, streams, episode_id="failed").build_support(
            2 / 60, 0
        )
        np.testing.assert_array_equal(snapshot["obs"]["robot"][-1, 25:32], 0.5)
        assert snapshot["obs"]["robot_valid"][-1, 3]


def test_interruption_flushes_without_certifying_completion(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    log = recorder(path)
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    record_command(log, 0)
    with pytest.raises(ValueError, match="terminal"):
        log.finish(dict(passed=False, termination="interrupted"))
    log.close()
    with h5py.File(path) as file:
        assert not file.attrs["complete"]
        assert len(file["commands/dispatch_s"]) == 1
    with pytest.raises(FileExistsError):
        recorder(path)


def test_missing_state_keeps_raw_nan_and_field_mask(tmp_path):
    path = tmp_path / "missing.hdf5"
    log = recorder(path)
    log.reset(0)
    q = np.full(9, np.nan)
    log.state(q, np.zeros(9), acquired=0, available=0, generation=0)
    log.close()
    with h5py.File(path) as file:
        assert np.isnan(file["control/joint_position"][0]).all()
        assert not file["control/q_valid"][0]
        assert file["control/dq_valid"][0]
        assert not file.attrs["complete"]


def test_refusal_cannot_claim_delivery_or_application(tmp_path):
    log = recorder(tmp_path / "refusal.hdf5")
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    with pytest.raises(ValueError, match="Refusal"):
        log.command(
            request_s=0,
            dispatch_s=0,
            applied_s=0,
            requested_target=np.zeros(7),
            applied_target=np.zeros(7),
            q_before=np.zeros(7),
            accepted=False,
            generation=0,
            request_kind=1,
            request=np.zeros(7),
            neck_target=np.zeros(2),
            gripper_target=np.zeros(1),
            effort_target_nm=np.zeros(9),
        )
    record_command(log, 0, accepted=False)
    log.interrupt(dict(passed=False, termination="interrupted"))
    log.close()
    with h5py.File(tmp_path / "refusal.hdf5") as file:
        assert not file.attrs["complete"]
        assert not file["metadata"].attrs["terminal_support"]
        assert json.loads(file["metadata"].attrs["outcome"])["termination"] == "interrupted"


def test_phase_quality_keeps_released_tail_and_masks_unqualified_load(tmp_path):
    from alexdoor_xas.recording.quality import teacher_eligibility

    path = tmp_path / "phases.hdf5"
    log = recorder(path)
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    for i, phase_valid in enumerate((True, False, True, True)):
        record_command(log, i / 60)
        log.annotate(
            dict(
                phase_valid=np.bool_(phase_valid),
                physical_safe=np.bool_(True),
                contact_valid=np.bool_(i == 0),
            )
        )
        log.state(
            np.zeros(9), np.zeros(9), acquired=(i + 1) / 60, available=(i + 1) / 60, generation=0
        )
    log.finish(dict(passed=True, termination="qualified_fixture", released=True, stabilized=True))
    log.close()
    with h5py.File(path, "r+") as file:
        np.testing.assert_array_equal(teacher_eligibility(file), [True, False, True, True])
        file.attrs["complete"] = False
        assert not teacher_eligibility(file).any()


def test_native_unverified_order_masks_tau_offline(tmp_path):
    from alexdoor_xas.dataset.temporal import raw_streams
    from alexdoor_xas.policies.temporal import TemporalConfig

    path = tmp_path / "order.hdf5"
    log = recorder(path)
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    depth = np.ones((4, 6, 1), "f4")
    log.capture(
        np.zeros((4, 6, 3), "u1"),
        depth,
        np.ones_like(depth, bool),
        np.eye(3),
        np.eye(4),
        acquired=0,
        available=0,
        generation=0,
    )
    log.feedback(np.ones(7), acquired=0, available=0, healthy=True, generation=0)
    record_command(log, 0)
    log.state(np.zeros(9), np.zeros(9), acquired=1 / 60, available=1 / 60, generation=0)
    log.finish(dict(passed=False, termination="fixture"))
    log.close()
    with h5py.File(path, "r+") as file:
        cfg = TemporalConfig(image_size=(2, 3), static_time_s=0)
        streams, *_ = raw_streams(file, cfg)
        assert len(streams["tau"]["acquired"]) == 1
        contract = json.loads(file["metadata"].attrs["feedback_contract"])
        contract["joint_names"] = contract["joint_names"][::-1]
        file["metadata"].attrs["feedback_contract"] = json.dumps(contract)
        streams, *_ = raw_streams(file, cfg)
        assert not len(streams["tau"]["acquired"])


def test_contact_acquisition_does_not_promote_later_ambiguous_load(tmp_path):
    from alexdoor_xas.recording.quality import teacher_eligibility

    path = tmp_path / "contact.hdf5"
    log = recorder(path)
    episode = json.loads(log.file["metadata"].attrs["episode"])
    episode["phase_names"] = ["contact"]
    log.file["metadata"].attrs["episode"] = json.dumps(episode)
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    for i, valid in enumerate((False, True, False, True)):
        record_command(log, i / 60)
        log.annotate(
            dict(
                phase=np.int8(0),
                phase_valid=np.bool_(True),
                physical_safe=np.bool_(True),
                contact_valid=np.bool_(valid),
            )
        )
        log.state(
            np.zeros(9), np.zeros(9), acquired=(i + 1) / 60, available=(i + 1) / 60, generation=0
        )
    log.finish(dict(passed=True, termination="fixture", released=True, stabilized=True))
    log.close()
    with h5py.File(path) as file:
        np.testing.assert_array_equal(teacher_eligibility(file), [True, True, False, True])


def test_full_robot_diagnostics_declare_rotational_and_jaw_effort_units(tmp_path):
    from alexdoor_xas.recording.b1_runtime import joint_si_units

    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot><joint name="arm" type="revolute"/><joint name="jaw" type="prismatic"/></robot>'
    )
    units = joint_si_units(urdf, ["arm", "jaw"])
    assert units == dict(position=["rad", "m"], velocity=["rad/s", "m/s"], effort=["N m", "N"])


def test_native_control_diagnostics_retain_actual_oscillation_separately_from_feedback(tmp_path):
    path = tmp_path / "native-diagnostics.hdf5"
    log = recorder(path)
    log.reset(0)
    for i in range(5):
        log.feedback(
            np.full(7, 3.0),
            acquired=i / 240,
            available=i / 240,
            healthy=True,
            generation=0,
            diagnostic_state=dict(
                diagnostic_joint_position=np.full(7, i * 0.001),
                diagnostic_joint_velocity=np.full(7, (-1.0) ** i * (i + 1)),
                diagnostic_position_target=np.ones(7),
                diagnostic_computed_effort_nm=np.full(7, 100.0),
                diagnostic_applied_effort_nm=np.full(7, 80.0),
            ),
        )
    log.close()
    with h5py.File(path) as file:
        native = file["physics_feedback"]
        np.testing.assert_array_equal(native["diagnostic_joint_velocity"][:, 0], [1, -2, 3, -4, 5])
        np.testing.assert_array_equal(native["torque_nm"][:], np.full((5, 7), 3.0))
        np.testing.assert_array_equal(
            native["diagnostic_applied_effort_nm"][:], np.full((5, 7), 80.0)
        )
        # Diagnostic effort and velocity never replace the allowlisted measured
        # torque stream consumed by the observation reader.
        assert len(native["acquired"]) == len(native["diagnostic_joint_position"]) == 5


def test_teacher_native_diagnostics_never_substitute_commanded_effort_for_feedback():
    from types import SimpleNamespace

    from alexdoor_xas.assets.purdue import ARM_JOINTS
    from alexdoor_xas.perception.contracts import RobotFeedback
    from alexdoor_xas.recording.teacher import TemporalExpertRecorder

    ids = [1, 3, 5, 7, 9, 11, 13]
    data = SimpleNamespace(
        joint_vel_target=np.zeros((1, 14)),
        computed_torque=np.arange(14)[None] + 100,
        applied_torque=np.arange(14)[None] + 80,
        gravity_compensation_forces=np.arange(14)[None] + 10,
    )
    env = SimpleNamespace(
        arm_ids=ids,
        generation=2,
        targets=np.arange(14)[None],
        robot=SimpleNamespace(data=data),
    )
    stored, observed = [], []
    log = TemporalExpertRecorder.__new__(TemporalExpertRecorder)
    log.env = env
    log.writer = SimpleNamespace(feedback=lambda tau, **kw: stored.append((tau, kw)))
    log._append = lambda *args, **kw: observed.append((args, kw))
    feedback = RobotFeedback(
        0.25, 0.25, ARM_JOINTS, np.full(7, 3.0), "simulated_actuator_feedback", "joint_output", True
    )
    sample = SimpleNamespace(feedback=feedback, q=np.zeros(7), dq=np.full(7, 4.47))
    log._feedback(sample)
    np.testing.assert_array_equal(stored[0][0], 3)
    diagnostics = stored[0][1]["diagnostic_state"]
    np.testing.assert_array_equal(diagnostics["diagnostic_joint_velocity"], 4.47)
    np.testing.assert_array_equal(diagnostics["diagnostic_position_target"], ids)
    np.testing.assert_array_equal(diagnostics["diagnostic_computed_effort_nm"], np.array(ids) + 100)
    assert len(observed) == 1 and observed[0][0][0] == "tau"
    np.testing.assert_array_equal(observed[0][0][1], 3)
