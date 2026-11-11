"""Actual guarded-recorder/IO dispatch on a numerical plant, not Isaac evidence."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace as S

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_observed_contact import begin_elastic_unloading, establish, inputs, monitor

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.kinematics.pose_control import bounded_joint_step
from alexdoor_xas.perception.contracts import OPERATIONAL_V1
from alexdoor_xas.perception.control import ObservedControlChecks
from alexdoor_xas.perception.local_admission import admit_local_action
from alexdoor_xas.policies.purdue import PurdueIO
from alexdoor_xas.recording.guarded_teacher import GuardedTeacherRecorder


@pytest.mark.parametrize(
    "defect", [None, "clipped", "admission", "stale", "preview", "step", "delivery"]
)
def test_guarded_preview_admission_dispatch_and_reference_memory(monkeypatch, defect):
    import alexdoor_xas.policies.observations as observations

    monkeypatch.setattr(observations, "require_features", lambda *_: None)
    m = monitor()
    establish(m)
    now, estimate, tool, load, base = inputs(1.2)
    events, delivered = [], []
    env = S(
        device="cpu",
        motion_stopped=False,
        arm_ids=list(range(7)),
        capture=S(sample=S(time_s=now, joint_position=torch.zeros(1, 9))),
        targets=torch.tensor([[0.0015, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        last_goal=None,
        pending=None,
    )

    def fk(q, _):
        return ObjectFrame(tool.origin + q[:3], Rotation.from_rotvec(q[3:6]).as_matrix())

    def preview_targets(*, world_goal=None, joint_delta=None, **_):
        events.append("preview")
        assert env.last_goal is None and not m.elastic_unloading
        if joint_delta is None:
            delta = torch.zeros(1, 7)
            delta[:, :3] = (
                env.targets[:, :3]
                + (world_goal[0] - torch.tensor(tool.origin, dtype=torch.float32)) * 0.5
            )
        else:
            delta = joint_delta
        q = env.capture.sample.joint_position[:, :7]
        target_limits = torch.tensor([[[-1.0, 1.0]] * 7])
        if defect == "clipped":
            target_limits[0, 0, 0] = 0.00148
        target, clipped = bounded_joint_step(
            delta,
            q,
            target_limits,
            torch.ones(1, 7) * 0.5,
            1 / 60,
            env.targets,
        )
        return target, clipped, torch.zeros(1, 6)

    env.preview_targets = preview_targets

    def command_pose(position, quaternion, **options):
        events.append("setter")
        env.pending = (
            torch.tensor(position, dtype=torch.float32)[None],
            torch.tensor(quaternion, dtype=torch.float32)[None],
        )

    def step(_):
        events.append("step")
        if defect == "step":
            raise RuntimeError("fixture_step_failure")
        target, _, _ = env.preview_targets(world_goal=env.pending)
        env.targets = target
        if defect == "delivery":
            env.targets = target + 0.0001
        env.last_goal = env.pending
        env.capture.sample.time_s += 1 / 60

    env.command_pose, env.step = command_pose, step
    io = PurdueIO.__new__(PurdueIO)
    io.env, io.dt, io.profile = env, 1 / 60, OPERATIONAL_V1
    io.response_violation, io.calibration = None, {}
    io.tool_fk = fk
    io.tool_pose = lambda: tool
    io.safety = S(
        provider=S(last_estimate=estimate),
        checks=S(contact_monitor=m, on_contact_evidence=delivered.append),
        before_command=lambda _: "",
        check=lambda _: "",
    )
    io.safety_reason = lambda _: ""
    io.step_recorder = lambda *_: events.append("recorded")
    recorder = GuardedTeacherRecorder.__new__(GuardedTeacherRecorder)
    recorder.env, recorder.io = env, io
    recorder.observer, recorder.latest_observation = S(binding=None), S()
    recorder.brake_anchor = None
    recorder.admitted_commands, recorder.guard_refusal = 0, None

    def motion(_, measured, actual_delta, ticks):
        events.append("admit")
        assert ticks == 1 and not m.elastic_unloading and env.last_goal is None
        endpoint = fk(np.r_[actual_delta, 0, 0], {})
        proposal = replace(base.proposal, action=actual_delta[:6], world_poses=(measured, endpoint))
        support = replace(base.local_evidence, world_poses=proposal.world_poses)
        limits = replace(base.limits, robot_position_bound_m=0.005)
        receipt = admit_local_action(estimate, proposal, support, limits, now, generation=0)
        assert receipt.admitted
        if defect == "admission":
            return replace(receipt, admitted=False, reason="fixture_admission_refusal")
        if defect == "stale":
            io.safety.provider.last_estimate = replace(
                estimate, local=replace(estimate.local, generation=1)
            )
        if defect == "preview":
            io.preview = lambda _: apply_pose_delta(endpoint, [0.01, 0, 0, 0, 0, 0])
        # Feed the same current admitted observation/load used by the real guard.
        assert m.update(now, estimate, measured, load, receipt).status == "contact"
        return receipt

    recorder.motion = motion
    requested = apply_pose_delta(tool, [-0.0001, 0, 0, 0, 0, 0])
    if defect and defect != "clipped":
        with pytest.raises((ValueError, RuntimeError)):
            recorder.execute(requested.origin, requested.rot, "release")
        assert recorder.admitted_commands == 0 and not m.elastic_unloading
        assert env.last_goal is None if defect != "delivery" else env.last_goal is not None
        if defect == "delivery":
            assert "recorded" in events  # preserve actions even on delivery mismatch
    else:
        recorder.execute(requested.origin, requested.rot, "release")
        assert recorder.admitted_commands == 1 and m.elastic_unloading
        assert events.index("admit") < events.index("setter") < events.index("recorded")
        assert env.last_goal is not None and len(delivered) == 1
        torch.testing.assert_close(
            env.targets[0, 0], torch.tensor(0.00148 if defect == "clipped" else 0.00145)
        )
        assert delivered[0].decision["delivered_unloading"]["unloading_before"] is False
        assert delivered[0].decision["delivered_unloading"]["unloading_after"] is True
        assert not io.response_violation
        assert len(delivered[0].decision["delivered_unloading"]["admitted_world_poses"]) == 2
        json.dumps(asdict(delivered[0]), allow_nan=False)


@pytest.mark.parametrize("stage", ["push", "hold"])
def test_early_intentional_unloading_cannot_supply_required_task_support(stage):
    m = monitor()
    begin_elastic_unloading(m)
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        m.update(*inputs(float(t), force=0))
    now, estimate, tool, load, receipt = inputs(1.35, force=0)
    checks = ObservedControlChecks(
        S(max_joint_speed=0.5, orientation_tolerance=0.1, force_limit_n=10), contact_monitor=m
    )
    sensor = dict(
        time_s=now,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        rgb=np.ones((1, 1, 3)),
        valid_depth=np.ones((1, 1), bool),
    )
    assert (
        checks.check(
            sensor,
            estimate,
            tool,
            np.tile([-2, 2], (7, 1)),
            stage,
            np.zeros(2),
            profile=OPERATIONAL_V1,
            feedback=load.feedback,
            admission=receipt,
            load=load,
        )
        == "observed_contact_not_established"
    )
