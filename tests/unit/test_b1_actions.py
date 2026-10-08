"""Numerical B1 semantics, full rotations and staged command reconstruction."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.b1 import (
    Segment,
    SegmentMotion,
    StageSequence,
    apply_pose_delta,
    fit_segments,
    panel_pose,
    pose_delta,
    primitive_actions,
)
from alexdoor_xas.action.frames import ObjectFrame, frame_delta_to_world, rot_z
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA


def pose(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    return ObjectFrame(np.array(position), Rotation.from_rotvec(rotation).as_matrix())


@pytest.mark.parametrize("sign", [-1, 1])
def test_all_primitive_targets_reconstruct_with_tool_offset_and_rotations(sign):
    wrist = pose([0.1, 0.2, 1.1], [0.3, -0.1, 0.2])
    tool = ObjectFrame(wrist.point_to_world([0.12, 0.02, -0.01]), wrist.rot)
    goal = pose([0.2, 0.3, 1.0], [sign * 0.4, -0.2, 0.1])
    frame = ObjectFrame(np.array([2.0, 0.0, 0.0]), rot_z(sign * 0.5))
    q, target = np.arange(7) * 0.1, np.arange(7) * 0.03
    rows = primitive_actions(q, target, tool, goal, frame)
    np.testing.assert_allclose(q + rows[A1_JOINT_DELTA], target, atol=1e-15)
    for delta in (rows[A2_EE_DELTA], frame_delta_to_world(rows[A3_OBJ_REL_EE_DELTA], frame)):
        np.testing.assert_allclose(pose_delta(apply_pose_delta(tool, delta), goal), 0, atol=1e-15)


def test_segment_codec_and_complete_sequence():
    sequence = StageSequence()
    for stage in ("approach", "approach", "contact", "push", "hold", "release"):
        original = Segment(
            stage, pose([0.1, 0.2, 0.3], [0.3, -0.2, 0.5]), -0.2, 3, stage == "release"
        )
        decoded = Segment.decode(original.encode(), remaining_ticks=3)
        np.testing.assert_allclose(decoded.encode(), original.encode(), atol=1e-15)
        sequence.accept(decoded)
    assert sequence.ended
    with pytest.raises(ValueError, match="skip or regress"):
        sequence.accept(decoded)


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "zero_rotation",
        "parallel_rotation",
        "zero_ticks",
        "budget",
        "early_end",
        "ambiguous_stage",
    ],
)
def test_invalid_segments_fail_without_repair(kind):
    row = Segment("approach", pose(), 0.1, 2).encode()
    if kind == "nan":
        row[5] = np.nan
    elif kind == "zero_rotation":
        row[8:11] = 0
    elif kind == "parallel_rotation":
        row[11:14] = row[8:11]
    elif kind == "zero_ticks":
        row[15] = 0
    elif kind == "budget":
        row[15] = 4
    elif kind == "early_end":
        row[-1] = 1
    else:
        row[:5] = 0
    with pytest.raises(ValueError):
        Segment.decode(row, remaining_ticks=3)


def test_duration_rounding_and_missing_stages():
    row = Segment("approach", pose(), 0, 3).encode()
    row[15] = 2.6
    assert Segment.decode(row, remaining_ticks=3).ticks == 3
    sequence = StageSequence()
    sequence.accept(Segment("approach", pose(), 0, 3))
    with pytest.raises(ValueError, match="skip or regress"):
        sequence.accept(Segment("push", pose(), 0, 3))


@pytest.mark.parametrize("sign", [-1, 1])
def test_segment_fitting_subdivides_curvature_and_replays_commands(sign):
    n = 15
    stages = [s for s in ("approach", "contact", "push", "hold", "release") for _ in range(3)]
    frames = [pose([0.001 * i, 0, 0], [0, 0, 0.2]) for i in range(n + 1)]
    angles = sign * np.arange(n + 1) * 0.02
    tools = []
    for i in range(n + 1):
        panel = panel_pose(frames[i], angles[i])
        local = np.array([0.1 + 0.03 * (i % 2), 0.3, 0.5])
        tools.append(ObjectFrame(panel.point_to_world(local), panel.rot @ rot_z(0.1 * i)))
    goals = tools[1:]
    starts, rows = fit_segments(tools, goals, frames, angles, stages)
    assert len(rows) > 5
    for start, row in zip(starts, rows, strict=True):
        segment = Segment.decode(row, remaining_ticks=n - start)
        motion = SegmentMotion.start(segment, tools[start], frames[start], angles[start])
        for k in range(segment.ticks):
            error = pose_delta(motion.goal(k + 1, frames[start + k]), goals[start + k])
            assert np.linalg.norm(error[:3]) <= 0.001 + 1e-12
            assert np.linalg.norm(error[3:]) <= np.deg2rad(0.5) + 1e-12
