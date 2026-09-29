"""B1 action semantics and explicit, full-pose object-centric segments."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .frames import ObjectFrame, rot_z, validate_object_frame, world_delta_to_frame
from .spaces import A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK

ACTION_DIMS = {A1_JOINT_DELTA: 7, A2_EE_DELTA: 6, A3_OBJ_REL_EE_DELTA: 6, A4_OBJ_CENTRIC_CHUNK: 17}
STAGES = ("approach", "contact", "push", "hold", "release")
DISCRETE_COLUMNS = (0, 1, 2, 3, 4, 16)
ACTION_SCHEMA = "b1.actions.v1"
ROTATION_TOLERANCE = np.deg2rad(0.5)


def checked_pose(pose):
    if reason := validate_object_frame(pose):
        raise ValueError(reason)
    return pose


def pose_delta(current, goal):
    """World translation and left-multiplied axis-angle rotation, in meters/radians."""
    checked_pose(current)
    checked_pose(goal)
    return np.r_[
        goal.origin - current.origin, Rotation.from_matrix(goal.rot @ current.rot.T).as_rotvec()
    ]


def apply_pose_delta(current, delta):
    checked_pose(current)
    value = finite_vector(delta, 6)
    return ObjectFrame(
        current.origin + value[:3], Rotation.from_rotvec(value[3:]).as_matrix() @ current.rot
    )


def finite_vector(value, dim):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (dim,) or not np.isfinite(value).all():
        raise ValueError(f"Expected {dim} finite action values")
    return value


def primitive_actions(joints, joint_target, current, goal, hinge_frame):
    """Labels reproduce the applied target, not differences of successive targets."""
    a2 = pose_delta(current, goal)
    checked_pose(hinge_frame)
    return {
        A1_JOINT_DELTA: finite_vector(joint_target, 7) - finite_vector(joints, 7),
        A2_EE_DELTA: a2,
        A3_OBJ_REL_EE_DELTA: world_delta_to_frame(a2, hinge_frame),
    }


def relative_pose(world, frame):
    checked_pose(world)
    checked_pose(frame)
    return ObjectFrame(frame.point_from_world(world.origin), frame.rot.T @ world.rot)


def panel_pose(hinge, angle):
    checked_pose(hinge)
    if not np.isfinite(angle):
        raise ValueError("Nonfinite signed hinge angle")
    return ObjectFrame(hinge.origin, hinge.rot @ rot_z(angle))


@dataclass(frozen=True)
class Segment:
    stage: str
    target_panel: ObjectFrame
    hinge_delta: float
    ticks: int
    end_episode: bool = False

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError("Unknown B1 A4 stage")
        checked_pose(self.target_panel)
        if not np.isfinite(self.hinge_delta) or not isinstance(self.ticks, (int, np.integer)):
            raise ValueError("Invalid A4 motion/duration")
        if self.ticks < 1 or not isinstance(self.end_episode, (bool, np.bool_)):
            raise ValueError("A4 duration must be positive and termination boolean")
        if self.end_episode and self.stage != "release":
            raise ValueError("A4 termination requires release")

    def encode(self):
        row = np.zeros(17)
        row[STAGES.index(self.stage)] = 1
        row[5:8] = self.target_panel.origin
        row[8:14] = self.target_panel.rot[:, :2].T.reshape(6)
        row[14:] = self.hinge_delta, self.ticks, self.end_episode
        return row

    @classmethod
    def decode(cls, row, *, remaining_ticks):
        row = finite_vector(row, 17)
        if row[15] <= 0 or row[15] > remaining_ticks + 0.5:
            raise ValueError("A4 duration exceeds remaining budget or is nonpositive")
        ticks = int(np.floor(row[15] + 0.5))
        if not 1 <= ticks <= remaining_ticks:
            raise ValueError("A4 rounded duration outside budget")
        # Decode continuous 6D rotation, rejecting degenerate columns instead of repairing them.
        a, b = row[8:11], row[11:14]
        if np.linalg.norm(a) < 1e-6:
            raise ValueError("Degenerate A4 rotation")
        a = a / np.linalg.norm(a)
        b = b - a * np.dot(a, b)
        if np.linalg.norm(b) < 1e-6:
            raise ValueError("Degenerate A4 rotation")
        b = b / np.linalg.norm(b)
        if np.count_nonzero(row[:5] == row[:5].max()) != 1:
            raise ValueError("Ambiguous A4 stage")
        return cls(
            STAGES[int(row[:5].argmax())],
            ObjectFrame(row[5:8].copy(), np.column_stack((a, b, np.cross(a, b)))),
            float(row[14]),
            ticks,
            bool(row[16] >= 0.5),
        )


class StageSequence:
    """Validate predicted boundaries; never supply a missing stage or release."""

    def __init__(self):
        self.stage = -1
        self.ended = False

    def accept(self, segment):
        index = STAGES.index(segment.stage)
        if self.ended or index not in (self.stage, self.stage + 1):
            raise ValueError("A4 stages must start at approach and cannot skip or regress")
        self.stage, self.ended = index, segment.end_episode


@dataclass(frozen=True)
class SegmentMotion:
    segment: Segment
    start_panel: ObjectFrame
    start_angle: float

    @classmethod
    def start(cls, segment, tool, hinge, angle):
        return cls(segment, relative_pose(tool, panel_pose(hinge, angle)), float(angle))

    def goal(self, tick, hinge):
        """Interpolate predicted motion, anchored to the latest observed hinge frame.

        Angle progress is relative to segment admission, never repeatedly added to
        a moving measured angle. There is no teacher tracking compensation.
        """
        if not 1 <= tick <= self.segment.ticks:
            raise ValueError("Tick outside A4 segment")
        alpha = tick / self.segment.ticks
        endpoint = self.segment.target_panel
        position = self.start_panel.origin * (1 - alpha) + endpoint.origin * alpha
        rotation = Rotation.from_matrix(endpoint.rot @ self.start_panel.rot.T).as_rotvec()
        rotation = Rotation.from_rotvec(alpha * rotation).as_matrix() @ self.start_panel.rot
        panel = panel_pose(hinge, self.start_angle + alpha * self.segment.hinge_delta)
        return ObjectFrame(panel.point_to_world(position), panel.rot @ rotation)


def fit_segments(
    tools, goals, frames, angles, stages, *, position_tol=0.001, rotation_tol=ROTATION_TOLERANCE
):
    """Fit command goals with stage-preserving subdivisions; return starts and rows.

    tools/frames/angles contain N+1 causal samples, goals/stages N commands.
    Subdivision reaches single-tick exact endpoints when a longer segment fails.
    """
    n = len(goals)
    if (
        not n
        or len(stages) != n
        or len(tools) != n + 1
        or len(frames) != n + 1
        or len(angles) != n + 1
    ):
        raise ValueError("A4 fitting requires N commands and N+1 observations")
    if not 0 < position_tol <= 0.001 or not 0 < rotation_tol <= np.deg2rad(0.5):
        raise ValueError("A4 reconstruction tolerances cannot exceed 1 mm / 0.5 deg")
    starts, rows = [], []

    def fit(start, stop):
        angle_delta = float(angles[stop] - angles[start])
        # At the final executed tick, the adapter has the observation at stop-1.
        endpoint_frame = panel_pose(frames[stop - 1], angles[start] + angle_delta)
        segment = Segment(
            stages[start],
            relative_pose(goals[stop - 1], endpoint_frame),
            angle_delta,
            stop - start,
            stop == n,
        )
        motion = SegmentMotion.start(segment, tools[start], frames[start], angles[start])
        for index in range(start, stop):
            predicted = motion.goal(index - start + 1, frames[index])
            error = pose_delta(predicted, goals[index])
            if np.linalg.norm(error[:3]) > position_tol or np.linalg.norm(error[3:]) > rotation_tol:
                if stop - start == 1:
                    raise ValueError("A4 single-tick reconstruction failed")
                middle = (start + stop) // 2
                fit(start, middle)
                fit(middle, stop)
                return
        starts.append(start)
        rows.append(segment.encode())

    boundaries = [0] + [i for i in range(1, n) if stages[i] != stages[i - 1]] + [n]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        fit(start, stop)
    sequence = StageSequence()
    for row in rows:
        sequence.accept(Segment.decode(row, remaining_ticks=n))
    if not sequence.ended:
        raise ValueError("Incomplete A4 sequence")
    return np.asarray(starts, dtype=np.int64), np.stack(rows)
