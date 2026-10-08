"""Canonical B1 action-space tags and dimensions."""

from __future__ import annotations

A1_JOINT_DELTA = "A1_joint_delta"
A2_EE_DELTA = "A2_ee_delta"
A3_OBJ_REL_EE_DELTA = "A3_obj_rel_ee_delta"
A4_OBJ_CENTRIC_CHUNK = "A4_obj_centric_chunk"

ALL_ACTION_SPACES: tuple[str, ...] = (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)

# A2/A3 per-step action layout: (dx, dy, dz, drx, dry, drz) for one end-effector.
EE_DELTA_DIM = 6
