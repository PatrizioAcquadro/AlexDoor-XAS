# Phase 4 — Robot and Task Configuration

> Subphase 4.0 completed and GPU-verified on 2026-09-22.
> Subphase 4.1 completed and GPU-verified on 2026-09-22; common synthetic setup is frozen.

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

- Replace B0 execution. Subsequent cleanup retired B0 compatibility and CLI
  orchestration; numerical learning components remain for later B1 integration.
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

The four synthetic doors have widths 0.65/1.20 m and both handednesses, with a
common 2.10 m height, 0.04 m panel thickness, 25 kg panel, geometry-derived inertia,
4 Nm s/rad hinge damping and 0.5 contact friction. Panel, frame and fixed handle
are collidable; the handle is a separate rigid actor for contact classification.
The hinge offset is 85 mm from the frame plane. The first panel/jamb intersection,
resolved at 0.1 degrees with a 0.2-degree clearance, defines the 183.2-degree joint
stop. The legacy 90-degree limit is not inherited.

The opening-center floor frame has +X into the opening, +Z up and +Y left.
Robot and measured pedestal share one X/Y/yaw transform; root height, roll/pitch,
model collision filters and joint limits are retained. Handedness changes the
door geometry and material-point trajectory, not the robot placement. The old
4.0 fixtures remain available through the same environment configuration.

`configs/purdue_synthetic_probe.json` freezes the reusable common setup. Its
right-arm ready vector, left parked pose and closed WSG targets apply to every
case. The selected placement is **X = -0.400 m, Y = 0.225 m, yaw = 45 degrees**.
Contact is at **0.40 of actual panel width from the hinge, 1.00 m above the floor**.
The fixed neck pose is **NECK_Z = -0.70 rad, NECK_Y = 0.25 rad**.

The expert uses existing full-pose control for approach → contact → push → hold
→ release. It follows the panel material point with tool +X into the panel and
+Z upward. The target joint-speed bound is 0.5 rad/s and nominal opening reference
speed is 1 degree/s; physics/commands remain 120/60 Hz. Damped IK uses 0.01 damping
and 2.0 nullspace centering gain. Approach/contact/release budgets are 6/3/3 s,
with a common 150 s push horizon. These are simulation qualification settings.

A valid sustain interval requires a causal 0.1-second mean normal force of at
least 0.02 N and an authorized contact point within 0.1 mm in both 120 Hz substeps,
an in-panel distal footprint, closed grippers, no forbidden contact,
and material-pose errors within 10 mm / 5 degrees. The reported angle is the
**minimum angle over a contiguous 0.5 s interval**; sustain does not require an
exactly motionless panel. Hold has a 3 s budget. A signed reference offset decays
smoothly over 0.5 s on entry, including when panel inertia puts it ahead of the
command. Release retraces an achieved pose behind the panel and verifies unloaded
separation of at least 1 cm.

The normal-force thresholds are 0.02 N for loaded contact, 0.10 N for declining
mean load over 0.1 s, 50 N for a soft safety stop and 80 N for a hard failure.
After the five-second push transient, a 2.5 mm material-drift reserve or half the
orientation budget also triggers a safety stop. The reserve leaves room to finish
hold and release. There is no 45- or 50-degree termination condition. Mechanical
stop, locally evidenced joint constraint, safety stop, lost contact, solver/tracking
stall, timeout and invalid physics remain distinct; unresolved stops cannot qualify.

Screening derives GPU batched FK/Jacobians from the external URDF, checked against
the imported runtime. Eight deterministic starts (seed 4101) and 5-degree angle
continuation propose candidates; failed local solves are not global reachability
proofs. The coarse domain was X [-0.50, 0.50] m, Y [-0.60, 0.80] m and yaw
[-180, 180] degrees, at 0.10 m / 15 degrees. The final refinement was X
[-0.525, -0.325] m, Y [0.10, 0.40] m, yaw [-15, 45] degrees, at 0.025 m / 5 degrees.
Panel/pedestal swept overlap and fixed jamb/pedestal overlap are screened out.

The initial 0.90 fraction at 1.20 m produced no common pre-contact candidate in
the final coarse screen. Calibration explored fractions 0.30, 0.40, 0.45, 0.50,
0.60 and 0.90 at selected heights between 1.00 and 1.50 m, not a full Cartesian
sweep. The final 0.40/1.00 m setting yielded a 50-degree screened common envelope.
This is filter evidence; physical probes establish the reported envelope.

Three refined floor candidates were physically investigated. The candidate at
(-0.375, 0.225 m, 40 degrees) was rejected below the required domain on the
left-wide door (43.13 degrees with the final contact criterion).
The (-0.375, 0.175 m, 40 degrees) candidate was rejected for wrist/torso contact
during left-wide approach. Neither establishes a competitive qualified setup.
The angular tie band was fixed at 0.5 degrees before comparison; normalized joint
margin, lower peak normal force and larger collision-clearance bound are the
subsequent criteria. Tiny negative numerical margins rank as zero; traces retain
the measured values. Clearance is a conservative robot/door AABB lower bound,
not an exact self-collision distance, and did not decide this selection.

The following results use the lower of two complete cycles per case:

| Door | Sustained angle | Repeat difference | Limiting safety guard |
|---|---:|---:|---|
| left-0.65 | 66.15° | 0.00° | Material tracking margin |
| right-0.65 | 77.91° | 0.00° | Material tracking margin |
| left-1.20 | 46.35° | 0.00° | Declining normal load |
| right-1.20 | 48.07° | 0.00° | Material tracking margin |

Every repeat has the same limiting cause as its pair and agrees within 2 degrees.
All eight cycles satisfy the sustain/contact/release criteria and the fixed-view
check. Visibility compares panel, contact-surround and frame points with current
valid optical-axis depth, including arm occlusion: at least 6/25, 2/8 and 2/14
samples, respectively, in every control frame. Representative RGB-D was inspected.
An exploratory 20-pose neck screen guided selection; sparse pose replay is not
qualification evidence. The continuous trajectories are authoritative. No active
gaze or Subphase 6.0 gaze dependency is required for these four synthetics.

Supported workstation entry points, with explicit configuration and cache output:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/screen_synthetic_setup.py \
  --fraction 0.4 --height 1.0 --center -0.425 0.25 15 --output /tmp/screen.json
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_synthetic_setup.py physics \
  --viz none --device cuda:0 --output /tmp/synthetic-physics
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_synthetic_setup.py search \
  --config configs/purdue_synthetic_probe.json --candidates /tmp/screen.json \
  --cameras --viz none --device cuda:0 --output /tmp/synthetic-search
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_synthetic_setup.py probe \
  --config configs/purdue_synthetic_probe.json --repeats 2 \
  --cameras --viz none --device cuda:0 --output /tmp/synthetic-verification
```

Search records the screening domain and rejects a candidate at its first failed
case. It ranks only four-case controlled completions and leaves `frozen: false`:
new searches still require repeatability, fixed-view review and documentation
closeout. Individual `--case` probes write separate case reports and can share
an output directory without overwriting one another's reports. With `--cameras`,
a probe also exits unsuccessfully if the fixed-view check fails.

#### Key Decisions

- Freeze one configuration and expert before collected assets. Synthetic minimum
  opening is evidence, not a target angle for later qualification.
- Report a controller-qualified envelope within the explored search domain,
  not a global kinematic optimum. All selected limits are measured safety stops.
- Keep simulator truth in the expert and diagnostics. Policy, observation and
  dataset contracts are unchanged; no hidden door-state correction enters adapters.
- Keep both 4.0 fixtures and articulated synthetics. Preserve the external Alex
  assembly, measured mounting height, collision model and physical joint limits.

#### Problems / Limitations

The initial mesh inset reduced leaf widths by 30 mm. The template now authors
exact 0.65/1.20 m panels and geometry-derived handle inertia; explicit USD dimension
and inertia checks and all physics/full-cycle gates were repeated. Earlier
nominal-width runs remain calibration evidence only.

An abrupt push-to-hold target change caused force spikes and failed sustain when
the panel led the reference. Signed continuous blending fixed that failure.
Waiting for 5 mm material drift left insufficient hold margin on the right-narrow
case; the common 2.5 mm reserve fixed it without an angle cutoff or relaxed
acceptance criterion. An initial neck pose approached the right shoulder; the
selected fixed neck clears it and preserves the required view throughout the
qualified motions.

Instantaneous contact force produced false losses during impulse chatter. The
final criterion combines a causal mean with actual substep separation; all final
cycles were rerun after calibration. Zero-load speculative contact alone fails.

Rendered depth is ideal geometry; finger compliance, contact friction and hinge
properties are simulation references. Normal force excludes tangential friction.
Visibility is a geometric observability proxy, not a learned-perception result.
Local joint-limit diagnostics do not prove global unreachability. These four
synthetics establish the common setup only: collected-asset admission, corpus,
splits, demonstrations, training, hardware safety and sim-to-real validation are
not part of this subphase.

## Artifacts

The original Subphase 4.0 RTX 4090 run is recorded in
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
- The historical D0 door-only smoke established the isolated preparation consumer;
  D0 payloads were subsequently retired. `DoorInspectionEnv` remains in B1 preparation.

Subphase 4.0 did not produce a corpus, training run or four-case reachability result.

Subphase 4.1 evidence is in
`~/.cache/alexdoor-xas/verification/synthetic-calibrated-final/`: common setup,
four paired reports, exported scenes, full angle/force/joint/contact traces,
RGB/depth samples, `validation-traces.png` and `representative-rgbd.png`.
An independent reconstruction from raw substep separations and normal impulses
confirms all eight sustain windows. Representative RGB-D was inspected.

The exact-width template gate in sibling `synthetic-physics-exact-width/` passes
all four resets, passive drift below 0.00022 degrees, fixed frame, geometric stop
within 0.00002 degrees, and exported panel dimensions/inertia. Separate exported
USD checks verify handle cuboid inertia. `synthetic-screen/` retains coarse and
refined domains; `synthetic-calibrated-comparison/` retains the rejected 43.13-degree
alternative. `synthetic-comparison-rejected-exact-width/` retains the wrist/torso
approach rejection. Earlier instantaneous-load and inset-width runs remain
calibration evidence, not final qualification.

The full 4.0 regression after the common-setup work passed in sibling
`purdue-after-41/`. Essential numerical tests cover handedness/contact trajectory,
sustain, force chatter/separation, stop classification, minimax and repeatability.
No corpus or policy training was performed.

The complete Purdue gate was rerun after the 2026-09-24 maintenance revision on
the RTX 4090. `~/.cache/alexdoor-xas/verification/quality-20260924/report.json`
passes assembly/reset, tool Jacobian, full-pose control, seven-joint A1, contact
classification and RGB-D checks. RGB/depth overlap is 0.988 IoU; acquisition
refreshes after neck motion and reset. This is synthetic runtime regression
evidence, not expert qualification of the 32 prepared doors.

## Files

- `src/alexdoor_xas/envs/door_task/door_push_purdue_env.py` and configuration.
- `src/alexdoor_xas/envs/door_task/purdue_contacts.py`.
- `src/alexdoor_xas/assets/purdue.py`.
- `src/alexdoor_xas/kinematics/pose_control.py` and `src/alexdoor_xas/recording/rgbd.py`.
- `scripts/verify_purdue_runtime.py`, `scripts/check_env.py`.
- `src/alexdoor_xas/envs/door_task/door_inspection.py` and retained preparation checks.

External component ownership is documented in the robot topic.
