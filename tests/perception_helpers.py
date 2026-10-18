"""Shared observed-only static scan fixtures; no model execution."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from alexdoor_xas.action.b1 import apply_pose_delta, relative_pose
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.admission import (
    ActionLimits,
    ActionProposal,
    HypothesisActionSupport,
    RotationLevers,
)
from alexdoor_xas.perception.contracts import (
    ContactSelection,
    DoorEstimate,
    FieldSupport,
    HingeHypothesis,
    LocalContactSelection,
    LocalMaterialState,
    LocalPatchState,
    OperationalState,
    RobotFeedback,
)
from alexdoor_xas.perception.geometry import Surface, SurfaceObservation, deproject, surfaces
from alexdoor_xas.perception.load import LoadCalibration, ResidualLoadModel
from alexdoor_xas.perception.local_admission import LocalActionSupport
from alexdoor_xas.perception.provider import load_recipe


def sensor(t=4.0, frame=248):
    return dict(
        time_s=t,
        frame=frame,
        rgb=np.ones((16, 16, 3), np.uint8),
        depth_m=np.ones((16, 16, 1)),
        valid_depth=np.ones((16, 16, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.array([[20, 0, 8], [0, 20, 8], [0, 0, 1]]),
    )


class EmptyWorker:
    def infer(self, rgb):
        return dict(
            latency_s=0.05,
            shape=rgb.shape[:2],
            masks=[],
        )


def recipe():
    root = Path(__file__).resolve().parents[1]
    return load_recipe(root / "configs/perception_geometry.json", root)


def tracking_sensor(t=0.0, frame=0):
    camera = np.eye(4)
    camera[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    return dict(
        time_s=np.asarray(t),
        frame=np.asarray(frame),
        rgb=np.ones((80, 80, 3), np.uint8),
        depth_m=np.ones((80, 80, 1)),
        valid_depth=np.ones((80, 80, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=camera,
        intrinsics=np.array([[60, 0, 40], [0, 60, 40], [0, 0, 1.0]]),
    )


def tracking_surface(captured):
    mask = np.zeros((80, 80), bool)
    mask[15:65, 15:65] = True
    v, u = np.nonzero(mask)
    points = deproject(
        np.ones(len(u)), np.c_[u, v], captured["intrinsics"], captured["camera_world"]
    )
    observation = SurfaceObservation(
        int(captured["frame"]),
        float(captured["time_s"]),
        float(captured["time_s"]),
        0,
        mask.shape,
        np.packbits(mask).tobytes(),
        captured["intrinsics"].copy(),
        captured["camera_world"].copy(),
        points[::20],
        {},
    )
    return Surface(
        points,
        np.array([1.0, 0, 0]),
        1.0,
        0.0,
        1.0,
        observations=(observation,),
    )


def camera_sensor(time=0, frame=8):
    h, w = 240, 240
    return dict(
        rgb=np.ones((h, w, 3), np.uint8),
        depth_m=np.full((h, w, 1), 2.0),
        valid_depth=np.ones((h, w, 1), bool),
        intrinsics=np.array([[120, 0, 120], [0, 120, 120], [0, 0, 1]], float),
        camera_world=np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]], float),
        time_s=time,
        frame=frame,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
    )


def measured_surface(sensor=None, mask=None):
    sample = camera_sensor() if sensor is None else sensor
    mask = np.zeros((240, 240), bool) if mask is None else mask.copy()
    if not mask.any():
        mask[30:211, 60:181] = True
    sample["depth_m"][mask] = 1
    cue = EmptyWorker().infer(sample["rgb"])
    cue["masks"] = [np.packbits(mask).tobytes()]
    return surfaces(cue, sample, recipe().config)[0], sample, cue


def operational_estimate(time=0.0, generation=0, count=1, bound=0.001):
    """Declared numerical support; these values are not a measured robot recipe."""
    support = FieldSupport(time, time, time, generation, bound, bound)
    hinge = ObjectFrame(np.zeros(3), np.eye(3))
    local = ObjectFrame(np.array([0.0, 0.3, 1.0]), np.eye(3))
    hypotheses = tuple(
        HingeHypothesis(
            f"h{i}", ObjectFrame(np.array([0.0, i * 0.3, 0.0]), np.eye(3)), 0.0, support
        )
        for i in range(count)
    )
    contact = ContactSelection("selection-1", "patch-1", time, local, local, support, support)
    state = OperationalState(
        "observed-leaf-1",
        generation,
        "tracking" if count == 1 else "ambiguous",
        support,
        hypotheses,
        "h0",
        support,
        support,
        contact,
    )
    return DoorEstimate(
        time,
        False,
        "legacy_incomplete",
        frame=hinge,
        panel_rotation=np.eye(3),
        signed_angle=0.0,
        contact_position=local.origin,
        contact_rotation=local.rot,
        operational=state,
    )


def action_inputs(estimate):
    contact = estimate.operational.contact
    poses = (contact.world_pose, ObjectFrame(contact.world_pose.origin + [0.001, 0, 0], np.eye(3)))
    now = estimate.timestamp_s
    proposal = ActionProposal(
        "diagnostic",
        A2_EE_DELTA,
        np.array([0.001, 0, 0, 0, 0, 0]),
        np.array([now, now + 0.1]),
        poses,
        contact.selection_id,
        True,
    )
    support = estimate.operational.identity_support
    evidence = tuple(
        HypothesisActionSupport(
            hinge.hypothesis_id,
            relative_pose(
                contact.world_pose, ObjectFrame(hinge.frame.origin, estimate.panel_rotation)
            ),
            poses,
            support,
            (0.1, 0.1),
            0.1,
            False,
            0.001,
            0.001,
            RotationLevers(1.1, 0.31, 1.1, 0.04, 0.1, 0.04, 1.1),
        )
        for hinge in estimate.operational.hypotheses
    )
    limits = ActionLimits(0.2, 0.01, 0.1, 0.1, 0.001, 0.001, 0.01, 0.001)
    return proposal, evidence, limits


def supported_patch(patch_id, time=1.0):
    support = FieldSupport(0, time, time, 0, 0.004, 0.02)
    return LocalPatchState(
        patch_id,
        ObjectFrame(np.array([1, 0, 1.1]), np.eye(3)),
        support,
        support,
        support,
        (0.04, 0.04),
        "tracked_material",
    )


def local_inputs(space=A2_EE_DELTA):
    patch = replace(
        supported_patch("contact"), tracking_state="tracking", ownership="observed_rigid_material"
    )
    state = LocalMaterialState(
        0, "contact", (patch,), LocalContactSelection("s", "contact", 1, "policy")
    )
    estimate = DoorEstimate(1, False, "local_only", local=state)
    row = np.array([0.001, 0, 0, 0, 0, 0]) if space == A2_EE_DELTA else np.zeros(7)
    goal = apply_pose_delta(patch.world_pose, row) if space == A2_EE_DELTA else patch.world_pose
    proposal = ActionProposal(
        "policy", space, row, np.array([1, 1 + 1 / 60]), (patch.world_pose, goal), "s", True
    )
    evidence = LocalActionSupport(
        patch.pose_support, proposal.world_poses, (0.04, 0.04), 0.04, False, 0.001, 0.001, 0.05
    )
    limits = ActionLimits(0.2, 0.01, 0.1, 0.1, 0.001, 0.001, 0.01, 0.001)
    return estimate, proposal, evidence, limits


def load_inputs(*, verified=True, stop=True):
    calibration = LoadCalibration(
        np.full(7, 0.05), -np.ones(7), np.ones(7), np.ones(7), np.ones(7), 1, verified, stop
    )
    feedback = RobotFeedback(
        1,
        1,
        ARM_JOINTS,
        np.array([-4.0, 0, 0, 0, 0, 0, 0]),
        "simulated_actuator_feedback",
        "joint_output",
        True,
    )
    jac = np.c_[np.eye(6), np.zeros(6)]
    model = ResidualLoadModel(calibration)
    estimate = model.estimate(
        feedback, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0]
    )
    return model, jac, estimate
