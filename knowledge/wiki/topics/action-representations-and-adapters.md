# Action Representations and Adapters

AlexDoor-XAS changes the action representation while holding the robot, task, physical episode, and evaluation protocol fixed.

Purdue now provides the low-level A1/A2/A3 executor. Phase 6 still owns learned
A1–A4 integration, estimated object frames, and observations. Reusable
action/export structures remain available; they are not the Purdue policy path.

## Canonical Representations

| Tag | Meaning | Frame and form | Current use |
|---|---|---|---|
| `A1_joint_delta` | Joint-target delta | Robot joint coordinates | Seven-joint Purdue execution; numerical exports |
| `A2_ee_delta` | End-effector delta | World-frame 6D delta | Full-pose Purdue execution; numerical models |
| `A3_obj_rel_ee_delta` | Object-relative end-effector delta | Static hinge-anchored door-frame 6D delta | Supplied-frame transform to full-pose A2 |
| `A4_obj_centric_chunk` | Object-centric contact-intent chunk | Contact targets in the moving panel frame | Numerical export structure; B1 execution deferred |

Frames are Z-up, distances are meters, angles are radians, and quaternions use `(x, y, z, w)`. The A3 frame is fixed at the hinge with +Z along the hinge axis. A4 contact targets move with the panel.

## Purdue Execution Boundary

A1 uses seven deltas in the right-arm order defined by the
[[topics/purdue-b1-robot-and-contact|robot contract]]. A2 uses six world-frame
translation/axis-angle deltas; both position and orientation are actuated through
the offset tool Jacobian. Damped IK includes deterministic nullspace joint
centering. Physical position limits and command-to-command velocity limits are
applied, with clamp and pose-error telemetry.

A3's `step_a3(delta, frame)` rotates both vectors and executes A2. The frame must
be explicitly supplied and a proper finite rotation. It does not read the door,
cache simulator geometry, or impose a panel-derived orientation. A qualification
expert may later supply truth; a learned policy must obtain its frame through
the Phase 6 observation/perception boundary.

The B0 adapters, rollout driver and scripted controller are retired. Frame
validation lives with action math and is shared by the Purdue A3 executor.
There is no connected B1 learned adapter or A4 execution path yet.

## Contact and Force Semantics

Purdue reports each physics substep's contact diagnostics separately from RGB-D
observations. Only the authorized distal surfaces against an exact panel body
are valid push contacts; lateral/mount, frame, handle, other robot and pedestal
contacts are forbidden. Structural support pairs are explicit. Forces are
normal-only. See the robot contract for geometry and approximation limits.

## Matched Comparison

A2 and A3 products derived from one physical episode share episode identity, outcome, pose, split, and evaluation seeds while retaining different action arrays and normalization. A4 preserves the same physical identity as a staged object-centric representation.

This controls major task-distribution confounds but does not prove that representation is the only cause of every learning difference.

## Primary References

- `src/alexdoor_xas/action/spaces.py`
- `src/alexdoor_xas/action/frames.py`
- `tests/test_action_spaces.py`
