# Phase 4 — Robot and Task Configuration

> Planned. Two subphases prepare the robot, then select its common setup.
> Documentation approval is not implementation or runtime validation.

## Objective

Make the Purdue Alex003 assembly operational and establish one fixed-base setup
on synthetic doors before inspecting collected assets. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]] and
[[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]].

Before implementation, separately audit the superseded local qualification code.
That cleanup is not permission to execute this phase.

## Subphase 4.0 — Operational Alex003, Control, and RGB-D

#### Implementation

Reuse the external Alex `full_convex` Purdue/WSG32/UMI v1 profile, measured
Alex003 pedestal, and ZED X Mini Wide integration. Resolve seven right-arm joints,
two neck joints, fixed closed-finger targets, and a collision-free parked left
arm. Implement the task-owned push frame from the existing distal-finger geometry.
Verify the assembly, fixed mounting height, joint limits, mimic behavior,
collision geometry, and actual contact locations. Adapt force/contact selection
to the right finger surfaces and exact door bodies, including forbidden
robot/frame/handle/pedestal contact. Do not retain the old single-body force
filter without checking its ownership assumptions.

Implement full tool position-and-orientation control through all seven arm
joints. Migrate the A2 executor and A3 transform, including the offset tool
Jacobian and a deterministic redundancy/joint-margin rule. Verify actual rotation
execution and A1 direct addressing of the same joint order. Phase 6 owns learned
model/data/adapter integration, not this low-level controller.

Connect the existing ZED left RGB/depth outputs and derive the valid-depth mask.
Verify optical extrinsics, metric units, image/depth alignment, neck proprioception,
synchronization, and reset. Build this capture path once and reuse it later for
multi-door recording. Keep simulator annotations out of observed inputs.

#### Key Decisions

- The robot/contact topic owns model, fingers, surface, tool frame, collision
  policy, limits, and measured support dimensions. Do not duplicate Alex assets,
  camera factories, or pedestal builders in this repository.
- This is the Purdue Alex003 composition with WSG selected explicitly; the
  generic reference scene defaults to SAKE.
- Use head RGB-D/proprioception without a mandatory wrist camera or GMSL emulation.
  Ideal rendered depth remains an explicit approximation.

#### Problems / Limitations

Complete after GPU checks show stable reset, valid seven-joint pose control,
correct contact/force observability, and synchronized RGB-D. Source inspection
alone is insufficient. No physical-safety or sim-to-real claim follows.

## Subphase 4.1 — Common Pose, Synthetic Reachability, and Visibility

#### Implementation

Use four synthetic single-leaf doors: widths 1.20 m and 0.65 m, each left- and
right-hinged. Record one nominal height/thickness and physics template before
searching. Both widths participate in the complete approach/contact/push/hold/
release check and minimax objective; neither is assumed to be the worst case.

Place Alex003 relative to the center of the closed door opening at floor level:
+Z up, +X into the opening from the robot side, +Y completing the right-handed
frame. Keep this placement reference distinct from A3's hinge frame. Search
robot-plus-pedestal floor X/Y and yaw with measured height, roll, and pitch fixed.
Do not mirror or reposition Alex separately for each handedness.

Use one panel contact fraction from the hinge, one absolute floor height near
the nominal handle region but on the panel, and the frozen finger footprint.
Start at fraction 0.90; evaluate any common adjustment only on synthetics. Keep
the footprint on the panel and clear of frame/handle geometry. The expert follows
that material location along the panel arc, with the tool pointing into the
panel and its vertical axis upward, using finite tracking tolerances.

Measure the maximum opening sustained for 0.5 seconds of valid controlled
contact, followed by safe release. Never end the push at 45 or 50 degrees.
Choose the common pose maximizing the minimum sustained expert angle over the
four cases, subject to valid approach/contact/hold/release and no forbidden
contact. Within a frozen numerical tie tolerance, prefer greater normalized
joint-limit margin, then lower force and greater forbidden-collision clearance.

Check useful panel/contact/frame visibility during these same motions, including
arm occlusion and near-range limits. First seek one fixed neck pose. Visibility
may reopen the synthetic pose search before freezing; do not conceal a visual
failure by lowering the measured physical opening. The whole door need not stay
in the image if the information needed for control remains observable.

Only if the fixed view is insufficient, record the demonstrated deficit and
requirements for bounded deterministic gaze driven by RGB-D/proprioception.
Its perception-dependent implementation belongs to Subphase 6.0. Do not build
active gaze when the fixed view passes, or claim it is validated by this phase.

#### Key Decisions

- Measure a practical controller-qualified envelope, not a global kinematic optimum.
- Respect mechanical stops, force/robot limits, and a common finite horizon.
  Select the horizon on synthetics so ordinary slow progress is not misclassified.
- Distinguish evidenced kinematic limit, mechanical stop, safety stop, solver/
  tracking stall, lost contact, timeout, and invalid physics. Timeout alone
  does not establish a limit. A valid limit is a result, not an asset failure.
- Freeze base/contact/neck setup, ready/parked poses, probe, redundancy rule,
  force/speed limits, tolerances, sustain duration, horizon, and stop classification
  before collected assets. The synthetic minimum is evidence, not a motion target.
- Simulator truth may drive the qualification expert and measure visibility.
  It must not drive learned policies through gaze, adapters, cached door state,
  or completion logic. Robot forward kinematics from proprioception is allowed.

#### Problems / Limitations

Complete with interpretable angle/force/joint/contact evidence and a lightweight
reset repeat for all four cases, plus a passing fixed view or an explicit
Subphase 6.0 gaze dependency. Resolve inadequate reach for the intended 45-degree
admission domain on synthetics, not by per-asset tuning. Unresolved stalls or
horizon stops cannot establish a maximum. No collected asset, split, bootstrap,
or qualification-count selection belongs here. Synthetics are setup evidence,
not B1 training/test data.

## Artifacts

Future outputs: operational Alex003/capture configuration, common setup/probe,
four-case reachability and contact/force evidence, and visibility assessment.
None was produced by this documentation revision.

## Files

Expected consumer surfaces: `src/alexdoor_xas/assets/`, `src/alexdoor_xas/envs/`,
`src/alexdoor_xas/adapters/`, `src/alexdoor_xas/policies/scripted/`,
`src/alexdoor_xas/recording/`, `configs/`, and focused `scripts/` entry points.
External component ownership is documented in the robot topic.
