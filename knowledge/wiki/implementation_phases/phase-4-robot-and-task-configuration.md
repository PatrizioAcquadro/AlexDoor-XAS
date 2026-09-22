# Phase 4 — Robot and Task Configuration

> Planned. The 2026-09-22 revision replaces the former all-in-one Phase 4.
> Documentation approval is not implementation or runtime validation.

## Objective

Establish one practical fixed-base Purdue Alex configuration and a measured
expert reachability reference on synthetic doors before inspecting collected
door assets. Follow the scientific contract in
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]] and
the source-backed [[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]].

Before implementation, audit the superseded local qualification code separately.
That cleanup must preserve useful capabilities and current consumers; it is not
permission to execute this phase.

## Subphase 4.0 — Adopt Purdue, WSG32, Pedestal, and Sensor Mounts

#### Implementation

Adopt the existing Alex `full_convex` Purdue/WSG32/UMI v1 profile, measured
pedestal, and ZED X Mini Wide mount through the external package. Implement the
task-owned closed-finger push frame from the canonical contact geometry. Resolve
the seven right-arm joints, two neck joints, fixed finger targets, and one parked
left-arm pose explicitly. Retire the old asset/calibration assumptions only as
their consumers migrate.

Verify the imported assembly, joint identities/limits, mimic behavior, fixed
mounting height, collision geometry, and actual distal-finger contact locations.
Adapt contact/force selection to the right finger surfaces and exact contacted
door bodies; include forbidden robot/frame/handle/pedestal contacts in validity.
Do not reuse the old single-gripper-body force filter without checking ownership.

#### Key Decisions

- The robot/contact topic owns the selected model, finger opening, surface,
  operational frame, collision policy, limits, and measured support dimensions.
- Reuse generic components from Alex; do not duplicate URDFs, meshes, camera
  factories, or pedestal builders here.
- The phase validates simulation integration, not physical safety or sim-to-real.

#### Problems / Limitations

Complete only after GPU checks establish stable reset, valid joint commands,
collision/force observability, and the intended contact footprint. Current local
source inspection supplies a prescription, not that evidence. Resolve any
contact-model defect before searching for the common pose.

## Subphase 4.1 — Seven-Joint Pose Control and Synthetic Reachability

#### Implementation

First implement full position-and-orientation control of `right_push_tip` using
all seven arm joints. Migrate the A2 executor and A3 frame transform, including
the offset tool Jacobian and a deterministic redundancy/joint-margin rule.
Verify that rotation commands are actually executed and that A1 can address the
same joint order directly; learned model support belongs to Phase 6.

Build four canonical synthetic single-leaf doors: widths 1.20 m and 0.65 m,
each left- and right-hinged. Use a common nominal height/thickness and physics
template recorded before the search. Both widths participate in the minimax
objective and the complete approach/contact/push/hold/release check. Neither
handedness may mirror or reposition Alex independently.

Define the common setup relative to the center of the closed door opening at
floor level: +Z up, +X into the opening from the robot's side, +Y completing the
right-handed frame. Keep this task reference separate from A3's hinge-anchored
frame. Search robot-plus-pedestal floor X/Y and yaw with measured height fixed.

Use one panel-relative contact rule: a fraction of width measured from the hinge,
one height above the floor near the nominal handle region but on the panel,
and the distal-finger footprint/orientation. Start
at fraction 0.90. Evaluate that choice and any justified common adjustment only
on the synthetics. Record the selected fraction and absolute height before
Phase 5; do not substitute each asset's actual handle location. Check that the
whole footprint stays on the panel and clear of handle/frame geometry.

The privileged expert follows the same material location along the panel's arc,
with the tool pointing into the panel and its vertical axis upward. Preserve
this controlled push through finite tracking tolerances, followed by a hold and
safe release. Measure the maximum opening sustained for 0.5 seconds while
maintaining valid controlled contact. No target angle or success crossing may
end the opening attempt at 45 or 50 degrees.

Choose the common pose that maximizes the minimum sustained expert angle across
the four synthetic cases. Feasibility requires valid approach, contact, hold,
release, and no forbidden contact. Treat angles within the frozen numerical
tolerance as ties; prefer larger normalized joint-limit margins, then lower
contact force and greater forbidden-collision clearance. Required panel contact
is not a penalty to eliminate.

#### Key Decisions

- Measure a controller-qualified reachability envelope, not a global kinematic
  optimum. A broad door is not assumed to be the worst case.
- Respect physical hinge stops, robot limits, force limits, and a bounded common
  episode horizon. Select the horizon on synthetics so ordinary slow progress
  is not mistaken for a reachability limit; freeze it before collected assets.
- Report `kinematic_limit` only with joint/workspace/pose-feasibility evidence.
  Separate mechanical stop, safety stop, tracking/solver stall, lost contact,
  time-budget exhaustion, and invalid physics. Timeout alone does not prove a limit.
- A controlled limit reached with a valid hold and release is a measured result,
  not an asset failure. An unresolved stall/horizon stop yields an incomplete
  reachability measurement and requires diagnosis.
- Freeze the controller, redundancy rule, ready/parked poses, force/speed limits,
  tracking/contact tolerances, sustain duration, horizon, and stop classification
  together with base pose and contact rule. The synthetic minimum is evidence,
  never a controller/policy stopping target.

#### Problems / Limitations

Complete after all four cases have interpretable angle/force/joint/contact traces
and a lightweight reset repeat. No collected assets, split assignment, statistical
qualification-count selection, or rollout-percentile threshold belongs here.
If the setup is inadequate for the intended 45-degree admission domain, report
that limitation and revise it on synthetics before freezing; do not hide
the problem through asset-specific tuning. Synthetic geometry is configuration
evidence, not B1 training or test data.

## Subphase 4.2 — Head RGB-D and Visibility Contract

#### Implementation

Connect the existing ZED left RGB/depth outputs to synchronized observations and
derive a sensor-validity mask. Include neck proprioception and correct optical
extrinsics. Verify units, alignment, timestamps, reset behavior, and separation
of observed inputs from simulator annotations.

With the selected base pose, inspect visibility through the complete synthetic
motions, including arm occlusion and near-range limits. First select one fixed
neck pose that retains useful panel/contact/frame information; the entire door
need not occupy the image at all times. A poor result may reopen the synthetic
setup before Phase 4 closes, but must not lower the recorded physical angle to
hide a sensing problem.

If static viewing is insufficient, specify a common bounded deterministic gaze
controller driven only by RGB-D and robot proprioception. Implement and freeze
its perception-dependent execution in Phase 6.1. Phase 4 must record the actual
visibility deficit and gaze requirement, not claim that active gaze is already
validated. If static viewing passes, no gaze controller is required.

#### Key Decisions

- Use head RGB-D plus proprioception for every representation; no mandatory
  wrist camera and no simulated GMSL transport.
- Simulator truth may measure visibility and drive the qualification expert.
  It must not drive learned evaluation through gaze, adapters, hidden reset
  state, or completion logic. Robot forward kinematics from proprioception is allowed.
- Phase 4 freezes sensor geometry and the observation boundary. Later learned
  perception cannot tune the common base/contact setup using collected test doors.

#### Problems / Limitations

Complete after RGB-D capture and synthetic visibility are verified, with either
a passing fixed view or an explicit Phase 6 gaze dependency. Ideal rendered
depth is not a validated stereo-error model. No policy training or collected
door qualification is performed in this phase.

## Artifacts

Future outputs: common setup configuration, four-case reachability measurements
with stop reasons, contact/force evidence, and RGB-D/visibility evidence.
None was produced by the documentation revision.

## Files

Expected consumer changes: `src/alexdoor_xas/assets/`, `src/alexdoor_xas/envs/`,
`src/alexdoor_xas/adapters/`, `src/alexdoor_xas/policies/scripted/`,
`src/alexdoor_xas/recording/`, `configs/`, and focused verification entry points
under `scripts/`. External source ownership is documented in the robot topic.
