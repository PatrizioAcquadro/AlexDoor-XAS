# Action Representations and Adapters

AlexDoor-XAS changes the action representation while holding the robot, task, physical episode, and evaluation protocol fixed.

Purdue provides the low-level executor. The separate B1 software path now connects
observed inputs, ACT/Diffusion outputs and A1-A4 adapters. Final integration and
physical rollout validation await qualified/frozen 6.0 perception. Historical
numerical action/export structures are retired; B1 is the maintained contract.

## Canonical Representations

| Tag | Meaning | Frame and form | Current use |
|---|---|---|---|
| `A1_joint_delta` | Joint-target delta | Robot joint coordinates | Seven-joint Purdue execution and B1 labels |
| `A2_ee_delta` | End-effector delta | World-frame 6D delta | Full-pose Purdue execution; numerical models |
| `A3_obj_rel_ee_delta` | Object-relative end-effector delta | Static hinge-anchored door-frame 6D delta | Supplied-frame transform to full-pose A2 |
| `A4_obj_centric_chunk` | Object-centric full-pose segment | Tool targets in the moving panel frame | B1 encoding/adapter implemented; physical validation pending |

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
The B1 adapters are in `policies/rollout.py`; their simulator bridge is in
`policies/purdue.py`. The old numerical chunk type and its reader/export path are retired.

## Contact and Force Semantics

Purdue reports each physics substep's contact diagnostics separately from RGB-D
observations. Only the authorized distal surfaces against an exact panel body
are valid push contacts; lateral/mount, frame, handle, other robot and pedestal
contacts are forbidden. Structural support pairs are explicit. Forces are
normal-only. See the robot contract for geometry and approximation limits.

## Matched Comparison

A2 and A3 products derived from one physical episode share episode identity, outcome, pose, split, and evaluation seeds while retaining different action arrays and normalization. A4 preserves the same physical identity as a staged object-centric representation.

This controls major task-distribution confounds but does not prove that representation is the only cause of every learning difference.

## B1 Learning Contracts (6.1 validation pending)

`action/b1.py` defines separate B1 numerical contracts. A1 labels are the applied
seven-joint target minus the **current observed joint position**, matching
`step_a1`; differences between consecutive targets are not equivalent. A2 labels
include world translation and left-multiplied axis-angle rotation from the
observed tool pose to the commanded goal. A3 rotates both components into the
causally estimated hinge frame. Tool poses include the calibrated distal offset.

A4 uses explicit segments and the ordered stages `approach`, `contact`, `push`,
`hold`, `release`. Its 17 columns are five stage scores, three panel-local tool
coordinates, six rotation values (first two matrix columns), signed hinge motion,
duration in control ticks, and episode termination. Training labels use one-hot
stages and boolean termination; decoding uses argmax and a 0.5 termination
threshold. Durations round to the nearest positive tick and must fit the remaining
budget. Degenerate rotations, ambiguous stages and premature termination fail.
Repeated stages are allowed; missing stages are never synthesized.

Segment motion interpolates panel-local tool position and orientation. Predicted
hinge motion is relative to the angle observed at segment admission; it is not
repeatedly added to the moving angle. Every control tick uses the latest fresh
estimated hinge frame. Fitting subdivides recorded commands until reconstruction
is within 1 mm and 0.5 degrees, preserving stage boundaries and complete release.
These are coding tolerances, not physical qualification gates. Numerical tests do
not establish matched physical replay or observed-geometry rollout validity.

The B1 bridge reconstructs A2/A3 absolute goals from observed robot FK and calls
the same `command_pose` route used by the recorded teacher and A4. This preserves
the recorded target instead of applying the raw Gym delta interface's additional
component clips. All pose paths retain seven-joint IK, physical position limits,
velocity bounds and shared contact/force monitoring. A1 dispatches directly to
`step_a1`, without learned-action IK.

`B1Runner` replans A4 at segment boundaries, executes the first predicted segment
and discards the remaining predicted horizon. Loss/stale input, invalid outputs
and safety stops clear pending chunks and segment state; no release is appended.
The common neck inspection and parked-tool hold precede learned arm execution.

## Operational interfaces — 6.0A implemented

The [[../implementation_phases/phase-6-0-operational-perception-and-contact|6.0 successor protocol]]
preserves these A4 semantics. Its transform needs the hinge frame, signed angle
and full local target, not total leaf dimensions. A3's free-vector transform uses
frame rotation only. Explicit profile dispatch preserves legacy complete-state
validation and permits operational states without irrelevant dimensions. Finite
features alone do not authorize actions. Operational adapters require an explicit
admission matching the action source, raw prediction, contact selection, episode
generation, schedule and world trajectory. Provisional admission is diagnostic-only.
An operational A4 segment keeps its admitted reference; incompatible updates latch
the adapter and require explicit reset. No target projection or task correction occurs.
Numerical integration does not qualify a provider, physical replay or loaded control.

The policy owns target, stage, opening increment and duration. A diagnostic action
source may test a small admitted movement, but cannot become a hidden policy
fallback. Updating a hinge hypothesis cannot silently redirect an executing segment.
Replan at explicit boundaries while admissible; loss/safety stops latch and clear
actions, without automatic resumption, release or opposite-side trials. Common
low-level compliance tracks the requested action and supports safe stopping; it
must not supply door-following task corrections. All eight cells share its settings.

The [[../experiments/point2pose-consumer-impact-and-failure-windows|saved Point2Pose consumer diagnosis]]
scores observed-zone drift and conditional target transport at teacher commands.
It does not supply a static A3 frame from a moving zone or insert evaluator hinge
geometry into an adapter. Closed yaw/full supported hinge geometry and actual
A4 segments/admissions are absent from those saved runs, so actual A3/A4 action
errors remain unmeasured. The operational A4 reference remains fixed per admission;
the measured publication spike is a consumer risk, not an executed command.

The subsequent [[../experiments/p2p-a3-a4-integration-diagnosis|bounded P2P to A3/A4 diagnosis]]
verifies the two primary saved seeds on the closed leaf and defines a frozen
orientation from its observed normal and calibrated up. Only that rotation is
needed by primitive A3 free-vector conversion; operational control still requires
the complete admitted reference. A4 needs a supported hinge/angle before its
first approach segment. Current motion-axis fits and their sample times are
diagnostic, and the provider remains invalid/local; no bootstrap probe or adapter
change is implemented.

## Primary References

- `src/alexdoor_xas/action/spaces.py`
- `src/alexdoor_xas/action/frames.py`
- `tests/test_action_spaces.py`
