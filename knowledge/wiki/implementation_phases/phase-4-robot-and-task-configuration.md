# Phase 4 — Robot and Task Configuration

> Subphase 4.0 completed and GPU-verified on 2026-09-22.
> Subphase 4.1 is in progress; synthetic physics is GPU-verified, common setup is not frozen.

## Objective

Make the Purdue Alex003 assembly operational and establish one fixed-base setup
on synthetic doors before inspecting collected assets. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]] and
[[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]].

## Subphase 4.0 — Operational Alex003, Control, and RGB-D

#### Implementation

The Purdue consumer uses the external `full_convex` WSG32/UMI v1 assembly,
measured Alex003 pedestal, and ZED X Mini Wide. Seven right-arm joints are ordered
explicitly; the two neck joints are separate and both grippers stay closed with
the package's leader targets and mimic followers.

`assets/purdue.py` derives `right_push_tip` from the canonical closed-finger
collision meshes, including the external wrist mount. Fixed-link merging is
resolved through the imported rigid owners rather than assuming URDF link names
survive as rigid bodies. Contact regions use the imported convex mesh, forward
extremum (3 mm geometric tolerance), and surface normal. Raw PhysX points are
classified against exact partner actors; internal pairs are counted once.
Reported force is **normal-only**, not total force including friction.

`kinematics/pose_control.py` provides full-pose damped IK and an exact-SVD
nullspace joint-centering term. Position limits come from Alex; target velocity
is limited against the previous command, not the measured state. This avoids
artificially limiting the available PD tracking error. A1 addresses the same
seven joints; A3 rotates both delta vectors from an explicitly supplied frame.

`recording/rgbd.py` defines copied RGB/depth/mask and nine-joint position/velocity
samples, with simulation time, episode and frame identifiers. Depth validity
uses only finite, positive, in-range depth. Duplicate frames are rejected and
reset invalidates the prior sample. This capture contract does not change the
historical episode/dataset schema or implement a learned observation encoder.

#### Key Decisions

- Replace B0 execution; preserve historical data/checkpoint readers and offline
  training. Full generation and learned evaluation remain unavailable until
  their later migration, with explicit errors before simulator startup.
- Reuse external robot, material, mimic, collision-filter, pedestal and camera
  factories. Keep gravity and model limits active. Use model gravity compensation
  for arm/neck holding; retain external PD gains rather than B0 gains.
- One GPU environment, 120 Hz physics, 60 Hz commands and RGB-D. These are
  commissioning defaults, not the frozen data protocol.
- Commission on small collidable panel/frame/handle fixtures. The common base,
  contact height, neck pose and full synthetic-door expert belong to 4.1.

#### Problems / Limitations

The supported runtime merges fixed finger links into jaw rigid bodies and can
reuse collider leaf names; exact ownership paths and imported geometry resolve
that ambiguity. Speculative zero-load contacts alone do not pass the contact gate.
The existing contact API reports normal forces only, including zero reaction on
some fixed-support pairs; separation/ownership still detect those forbidden pairs.

Target velocity limits initially applied against measured joints restricted PD
tracking error and made pose control sluggish. Bounding against previous targets
fixes this without changing external gains or widening physical limits.

The supported Lab/Fabric runtime leaves the attached optical descendant pose
stale. The consumer now applies the canonical optical transform to the current
physical ZED body pose before image acquisition. Independent head/mount checks,
metric targets, neck motion and reset verify the resulting extrinsics. Reset
also settles renderer history with eight fresh renders at the unchanged physics
state before publishing a sample, avoiding previous-episode ghosting.

Ideal rendered depth, rigid model-reference fingers and model gravity compensation
are simulation approximations; no physical-safety or sim-to-real claim follows.

## Subphase 4.1 — Common Pose, Synthetic Reachability, and Visibility

#### Implementation

Implemented foundation: `assets/synthetic_door.py` authors the four 2.10 m high,
0.04 m thick doors with a 25 kg panel, geometry-derived inertia, 4 Nm s/rad hinge
damping and 0.5 friction. The collidable handle remains a separate rigid actor.
The hinge is offset 85 mm from the frame plane; the first panel/jamb intersection
sets a symmetric 183.2-degree mechanical stop, with 0.2-degree geometric clearance.
The Purdue environment optionally loads these doors and moves robot/pedestal
jointly in floor X/Y/yaw; the existing commissioning fixtures remain available.

GPU foundation evidence in `~/.cache/alexdoor-xas/verification/synthetic-physics-all/`
passes all four cases: two seconds passive drift below 0.00022 degrees, zero
frame drift, ten seconds at 15 Nm reaching the computed stop within 0.00002 degrees,
and closed reset. These are door-physics checks with the robot parked clear;
they do not establish the common expert setup or visibility. The remaining work
below retains the approved acceptance criteria.

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

The full RTX 4090 run is recorded in
`~/.cache/alexdoor-xas/verification/purdue-final/report.json`, with numeric
`control_traces.npz`, raw contact records, `rgbd_sample.npz`, and representative
RGB images before/after neck motion and reset. Final sensor-only evidence with
reset renderer settling is in `purdue-final-rgbd/`; contact force-direction checks are in
`purdue-final-contacts/`, under the same verification cache.

- Three stable consecutive resets; shoulder height 1.266000 m, closed leaders
  and mimic followers within 1 mm; no forbidden contact during free-space motion.
- Seven local translation/rotation/combined targets, held for 0.5 s: maximum
  position error 0.0204 mm and orientation error 0.00209 degrees. Return moves
  also passed the operational 10 mm / 5 degree criterion.
- Seven individual A1 commands: maximum joint error 0.000075 rad.
  Full tool-point Jacobian agrees with independent URDF finite differences
  within 0.00000286 (linear/angular components).
- Loaded contacts on both distal fingers; frame, handle, finger side and pedestal
  contacts are rejected. Raw points, separations and normal vectors are retained;
  robot-internal pairs are recorded only once per physics sample.
- RGB-D: 960 × 600 metric optical-axis depth, known-target error below 0.01 m,
  RGB/depth fixture overlap above 0.90 IoU, correct head/mount extrinsics and
  fresh acquisition after neck motion/reset. Representative images inspected.
- The preserved D0 door-only GPU smoke passes reset, 2 s passive drift (zero),
  and 3 s response to 15 Nm, reaching the historical 90-degree stop. This checks
  the retained preparation consumer, not B1 admission or reachability.

Software validation: 332 tests pass, including historical readers/model contracts,
with Ruff, whitespace and wiki-link/index checks. No corpus, training run or
four-case reachability result was produced.

## Files

- `src/alexdoor_xas/envs/door_task/door_push_purdue_env.py` and configuration.
- `src/alexdoor_xas/envs/door_task/purdue_contacts.py`.
- `src/alexdoor_xas/assets/purdue.py`.
- `src/alexdoor_xas/kinematics/pose_control.py` and `src/alexdoor_xas/recording/rgbd.py`.
- `scripts/verify_purdue_runtime.py`, `scripts/check_env.py` and retired B0 entry points.
- `src/alexdoor_xas/envs/door_task/door_inspection.py` and retained preparation checks.

External component ownership is documented in the robot topic.
