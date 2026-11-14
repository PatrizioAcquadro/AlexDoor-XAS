# Purdue B1 Robot and Contact Contract

This is the implemented Purdue operational contract for Subphases 4.0–4.1. The B0
Alex V2 runtime has been retired. The common synthetic-door setup is GPU-qualified;
B1 learned-action software is available, with final integration and physical
validation still pending in Phase 6.1. Current runtime evidence is maintained in
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]].

## Reuse and Ownership

Reuse the editable `~/Desktop/Alex` package. It owns robot/gripper assets,
actuators, collision geometry and filters, sensor mounts, and measured pedestal
geometry. AlexDoor-XAS owns the task contact frame, control, sensing
contracts, scene placement, and benchmark qualification. Do not copy Alex assets
or generic integration code into this repository or Isaac Lab.

On the Kit stack, use `ihmc_alex_isaaclab.robots.alex_purdue.make_alex_purdue_cfg` with
`fix_base=True`, `variant="full_convex"`, and `end_effector="wsg32_umi_v1"`.
The source is
`~/Desktop/Alex/assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf`.
The factory defaults to SAKE when the end effector is omitted; select WSG explicitly.

## Physics profile and reset

Start the Purdue simulation through `scripts/isaac-run.sh`. The B1
`collect_perception.py` and `qualify_door.py` commands select Isaac Lab's
`SteelSolverCfg` with CUDA Graph and the Newton Warp renderer. The remaining
Kit commands use the local Steel wheel site under
`outputs/generated/isaac-steel-runtime/site`. Select `newton_steel` explicitly
through `ISAAC_PHYSICS_PROFILE` to use that Kit profile for the B1 commands.
Set `NEWTON_SITE` for a different wheel site or select `isaacsim_physx` to
reproduce a GPU PhysX run.

The first native Steel step captures the graph. Each standard environment reset
checks the latched solver status before restoring the scene. It clears solver
warm-start state and contact counts in place and invalidates the host contact
snapshot while keeping the buffers used by the graph. Reset restores the robot
and hinge state and acquires a fresh ZED sample. Capacity checks continue at
each native contact readout, including the first step after reset.

On `main`, the launcher selected Steel without overrides on the RTX 4090.
One session passed three short D4-N push, release, and stabilization cycles
with Alex003, WSG32, and ZED RGB-D. One graph served 8,076 launches and
capacity checks, including warm-up. All 1,983 cycle frames passed the ZED
checks. The first control tick took 0.785 s at capture and 0.037–0.040 s
after reset. An injected saturation flag caused a subsequent reset to raise
before discarding solver or episode state. Local evidence and the repeat
command are under `outputs/generated/alexdoor-d4n-standard-reset-20261113/`.

### Install the native B1 runtime

Check out NVIDIA Isaac Lab at `67fe12806afbdb8bf8e125866661dd201088a54d`.
From that checkout, use its lockfile with
`UV_PROJECT_ENVIRONMENT=/absolute/path/to/.venv uv sync --locked --python 3.12`.
That lockfile selects Newton `7f0c13af6e4dd75253709c31dc65c0edcaf22b44`.
Use persistent storage for the checkout, virtual environment, and package cache.
The launcher uses `outputs/generated/isaac-native-runtime/.venv` by default.
The workstation links `outputs/generated/isaac-native-runtime` to
`/media/pacquadr/DATA/AlexDoor-XAS/isaac-native-runtime`. Mount the local DATA
volume before launching after a restart. The launcher defaults to `cache/` and
`tmp/` under that installation for Warp, CUDA, and temporary files.
Use `ALEXDOOR_NATIVE_PYTHON` to select another interpreter. The editable Alex
source defaults to the sibling `Alex` checkout. Set `ALEXDOOR_ALEX_ROOT` if it
lives elsewhere.

Set `ALEXDOOR_USD_CACHE` to the existing converted robot and pedestal USD
directory or place it under `outputs/generated/isaac-native-runtime/usd-cache`.
The scene references the fixed-base Alex003 WSG32/UMI asset under
`robot/alex_purdue_wsg32_umi_v1_full_convex/` and its pedestal under
`pedestal/purdue_alex003_pedestal/`. Keep the companion layers alongside
each root USDA. The native path uses the Alex URDF joint limits and Purdue
actuator gains with Isaac Lab's current actuator fields. It applies Alex's
collision filters, finger material, and ZED mount at scene composition.

### Repeat the bounded B1 diagnostic

Run from the repository root on the reserved RTX GPU. Use a fresh output
directory with room for the RGB-D recordings.

```bash
asset=psx-interior-wood-002-02e442c
fixture=configs/purdue_b1_bounded_diagnostic.json
output=/absolute/path/to/fresh-results
for condition in nominal light; do
  scripts/isaac-run.sh scripts/collect_perception.py --asset-id "$asset" \
    --bounded-diagnostic "$fixture" --condition "$condition" --output "$output/collection"
  scripts/isaac-run.sh scripts/qualify_door.py --asset-id "$asset" --diagnostic \
    --bounded-diagnostic "$fixture" --condition "$condition" --output-root "$output/evaluation"
done
```

The fixture limits the hinge to 4 degrees on the live stage and uses the
near-panel Alex003 pose at 240 Hz physics and 60 Hz control and RGB-D. The
nominal and varied runs use the same seed. Newton Warp's varied condition
rotates the directional light in azimuth while retaining its elevation and
the renderer's ambient light. Collection validates
the ordinary B1 schema and excludes the fixture recordings from learning.
The evaluator expects two controlled cycles and an `out_of_domain` result
under the existing 45-degree rule. The two diagnostic resets share one
process and reuse the same physics CUDA Graph. Formal qualification uses a
fresh process for each trial and publishes its result through the usual path.

## Arm and Gripper

The ordered active right-arm joints are `RIGHT_SHOULDER_Y`, `RIGHT_SHOULDER_X`,
`RIGHT_SHOULDER_Z`, `RIGHT_ELBOW_Y`, `RIGHT_WRIST_Z`, `RIGHT_WRIST_X`, and
`RIGHT_GRIPPER_Y`. The last is a revolute arm joint, not finger opening.
Commissioning parks each arm with shoulder X at +/-0.35 rad and elbow Y at
-1 rad. The common benchmark ready/parked and fixed neck setup is frozen in
`configs/purdue_synthetic_probe.json`; its qualification belongs to Phase 4.1.
`NECK_Z` and `NECK_Y` are separate vision degrees of freedom.

Use the WSG 32-068 with the existing UMI v1 finger geometry. Hold both grippers
closed for the complete episode: normalized opening `0.0`, mapped by
`alex_purdue_wsg32_targets`, gives a leader target of `0 m`; retain its mimic
follower. Do not add gripper opening to A1-A4. Closed is a benchmark choice that
provides a compact push tool without grasping; it is not a measured hardware
operating recommendation. The model's closed inner gap is approximately 2 mm.

The authorized pushing surfaces are the forward distal ends of the
`WSG32_NEGATIVE_UMI_V1_CONTACT_CONVEX` and
`WSG32_POSITIVE_UMI_V1_CONTACT_CONVEX` colliders under
`right_WSG32_NEGATIVE_UMI_V1_FINGER_LINK` and
`right_WSG32_POSITIVE_UMI_V1_FINGER_LINK`, respectively. Collider names repeat
on both hands, so resolve their owning right-hand links rather than names alone.
Finger mounts, jaw/body metal, wrist, arm, left gripper, and pedestal are not
alternative push tools. Preserve their collisions and detect forbidden contact;
do not filter contact away to make a rollout valid.

## Operational Push Frame

Define the task-owned frame `right_push_tip` relative to
`right_WSG32_BASE_LINK`, with the same axes: +X is approach/push, +Y is the
closing axis, and +Z completes the right-handed frame. Its origin is the center
of the two forward support extrema of the closed finger contact hulls.

Read-only URDF/STL inspection on 2026-09-22 gives translation approximately
`(0.202000, 0, 0) m`, with identity rotation. The reference lies between two
distal contact supports; it is not a material point in the gap between the fingers.
The consumer computes it from the canonical collision geometry;
use the current geometry as the authority if the external asset changes.

The current forward mesh extrema are horizontal ridges, not flat pads.
`PushGeometry.distal_faces` retains those extrema for the expert and historical
qualification. Observed contact queries instead use `contact_covers()`: clip each
canonical convex mesh within the existing 3 mm distal classification tolerance,
include intersected triangle edges, then project that band along +X onto its
forward support plane. The two nominal covers each span approximately 25.8 mm by
3.4 mm, with distinct lateral/vertical offsets in `right_push_tip`. They cover
possible contact within that band without enlarging it to the whole finger or
treating the empty gap as contact. Convexity is justified by the canonical contact
hulls. This is a conservative geometric projection, not measured contact area,
finger deformation or proof of two loaded contacts. Model/alignment uncertainty,
allowed penetration and physical response remain D checks. No robot asset,
calibration, expert point, collider authorization or physical gate changes.

The existing `RIGHT_WSG32_TCP_FRAME` is at `(0.1305, 0, 0) m` in the same base
frame. It is 71.5 mm behind this push reference and must not silently replace it.
The mount from `RIGHT_GRIPPER_Y_LINK` to `right_WSG32_BASE_LINK` is already
provided by Alex; reuse it rather than introducing another wrist offset.

For expert contact/push/hold, keep +X directed into the panel, normal to its
surface, and +Z upward. This keeps the closing axis horizontal and the same
distal surfaces facing the panel as it rotates. Blend orientation during
approach/release. Use finite position/orientation tolerances, frozen on synthetic
doors, rather than exact algebraic equality. Full pose control must actuate
rotation through the seven-joint chain. Learned actions must produce their own
motion; an adapter must not secretly restore this orientation using simulator
door state.

The consumer derives this frame from the canonical meshes and resolves their
actual rigid owners after fixed-link merging. In the imported model the fingers
belong to the corresponding jaw bodies, and positive/negative meshes can reuse
leaf names. Full ownership paths, local convex geometry, a 3 mm distal tolerance
and a normal-direction check determine authorized contact. A body-name-only or
single-force filter is insufficient.

The gate distinguishes speculative points with zero load from loaded contacts.
A forbidden pair triggers an abort only when its separation is nonpositive or
its normal force is nonzero (either actor sign). Separated, zero-force candidates
remain in raw diagnostics; touching or penetrating fixed obstacles still abort even at zero force.
Task normal force sums authorized contacts once; forbidden and structural-support
records remain separate. Tangential friction is not included in the reported force.

## Robot feedback and observed D4 release

The [Alex003 Usage Guide](https://docs.google.com/document/d/17QtexPK_RqmfRammA7CsvEUuhFZFkicrJfONCdlWkEg/edit),
reviewed October 1, 2026, documents `q`, `qd` and `tau` in `OneDOFJointState` on
`rt/alex_state`. `tau` is described as current joint torque in N m. The SCS2 joint
charts include desired/measured torque and motor current/voltage. This establishes
documented feedback availability, not its accuracy or joint-output sensor origin.

`OneDOFJointCommand` exposes desired position/velocity/torque, stiffness, damping,
maximum position/velocity error and maximum torque. The guide says
`joint_control_type` is currently unused: all setpoints are resolved through a PD
law into desired motor torque. Do not assume a separate selectable effort mode.
`ft_states`/`number_of_fts` support force/torque devices in the message schema but
do not establish that a wrist sensor is installed on this Alex003 configuration.
The guide's EZGripper effort signal is not a WSG32 pushing-force measurement.

`scripts/isaac-run.sh scripts/collect_d4_observed.py` runs one declared D4-N
condition with the prepared panel, an explicit setup, protocol and operational
recipe, and a fresh output directory. The operational recipe supplies the
calibration recordings, model locations, residual wrench and stop bounds.
The launcher uses Steel and records one 60 Hz controller target across four
240 Hz steps. The bounded held-surface correction uses the last completed
native interval. The teacher proposes poses, then observed RGB-D, joint state,
load, and full-robot sweep and stop evidence determine admission.

Steel's joint-output feedback projects solved parent wrenches about the
link center of mass onto the corresponding joint motion axes. The residual
model compares those wrenches against rigid-link inverse dynamics. The
Newton mass matrix includes joint armature, so subtract its acceleration
contribution in the residual only. Keep the full matrix for predicting
the controller response and keep armature active in the solver. Contact
partner, normal force and separation remain independent evaluator channels.

The 5° orientation check protects aligned interaction while acquiring or
maintaining support. During delivered elastic unloading, an over-limit
orientation can reach the dispatcher only after current geometry and load
checks. The dispatcher requires the pose request, change from the previous
delivered target, and measured-to-bounded-target displacement to withdraw
both distal covers throughout the supported normal cone. The cone includes
the published observation age and the material angular allowance for the
next interval. The same admission covers collision, response, and stopping.
The actual target and completed interval must match the receipt.
Free release and stabilization require positive observed separation for both
fingers across the control interval. Spend the calibrated joint travel and
material motion against each finger's observed gap. The independent
whole-robot collision and stop proof still applies, as does the separate
position support check for the calibrated stop-load domain. A fresh positive
gap can overlap zero again as the observation ages. In that case the monitor
can continue a verified withdrawal from the previous delivered command.
The reduced-opening component rotates at its withdrawn position to exercise
this transition when the ordinary cycle's preceding open retreat pose is absent.

Independent stop verification latches the measured joint anchor, applies
the configured stop impedance, and advances the solver with gravity
compensation. The 10 N stop calibration domain bounds use of the calibrated
model. The setup's physical force limit is checked separately. These
simulation checks leave hardware feedback and stop response unqualified.

## Collisions, Limits, and Physical Approximation

Reuse the `full_convex` self-collision configuration, existing explicit pair
filters, WSG mimic constraints, and UMI contact material. Keep the pedestal,
door frame, panel, and non-actuated handles collidable. The package's joint
position/velocity/effort limits are authoritative; do not widen them or carry
over the old six-joint gains without validation. Task-level speed/force limits
and joint margins may be stricter and are fixed in Phase 4 before asset collection.

The package currently uses rigid UMI hulls, model-reference friction, and an
unmeasured conservative finger mass. Its WSG mount/TCP/contact parameters are
simulation references, not a physically calibrated soft-finger digital twin.
The WSG jaw effort limit is not a door-pushing force safety threshold.

## Pedestal and Placement

Reuse `load_purdue_alex003_pedestal_spec` and
`make_purdue_alex003_pedestal_cfg` from
`ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal`.
The reference composition is in `ihmc_alex_isaaclab.scenes.purdue_alex003`;
its default robot is SAKE, so explicitly compose/replace it with the WSG factory.

`~/Desktop/Alex/measurements.yaml` records:

| Measured item | Value |
|---|---|
| Mounting plane above the floor | 0.900 m |
| Right shoulder Y-axis above the floor | 1.266 m |
| Lower base dimensions (X, Y, Z) | (0.565, 0.600, 0.120) m |
| Column dimensions | (0.175, 0.160, 0.760) m |
| Upper mount dimensions | (0.220, 0.300, 0.020) m |

The robot root height is derived from the measured shoulder height and URDF,
currently approximately 0.883856 m; it is not the 0.900 m mounting-plane height.
Move the robot and pedestal together in floor X/Y and yaw. Fix height, roll, and
pitch. Check the complete assembly against the door and its swept volume.

## Head RGB-D

Reuse `make_zed_x_mini_cfgs` and `author_alex_purdue_zed_x_mini_mount` from
`ihmc_alex_isaaclab.sensors.zed_x_mini`, including the pinned official ZED X Mini
Wide asset. Start with its local SVGA acquisition configuration (960 x 600).
Policy resizing and capture cadence are later shared protocol choices.

`HEAD_ZED_X_MINI_JOINT` fixes the URDF camera reference to `HEAD_LINK` at
`xyz=(0.11603, 0.009965, -0.02983) m`, `rpy=(0, 0.3633, 0) rad`.
The package supplies the additional official-model alignment and left optical
camera transform. Do not treat the URDF mounting reference as the optical center.
Document GMSL2 as the physical interface; no GMSL transport emulation is required.

B1 observations use synchronized left rectified RGB, metric left-aligned depth,
valid-depth mask, and robot proprioception including the seven arm joints and
the neck state. No wrist camera is required in the main benchmark. All eight
model/representation cells receive the same observation channels and history.

The local left camera provides `rgb` and `distance_to_image_plane`; preserve
optical-axis depth in meters, rather than substituting Euclidean range. Define
validity from finite, positive, in-range sensor depth. The mask must not encode
door segmentation or other simulator labels. The real SDK also provides depth
aligned with the left image ([Stereolabs depth API](https://www.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/using-the-api)).

The supported Lab/Fabric runtime does not propagate the attached ZED camera's
optical descendant pose. Before acquisition, the consumer composes the current
physical ZED rigid-body pose with the canonical optical transform and uses the
public camera pose setter. Independent head/mount checks and rendered targets
verify this synchronization across neck motion and reset. No door state enters
this bridge. Reset settles renderer history at an unchanged physics state before
publishing a fresh sample. Each acquisition copies image and proprioception at
one physics state.

Rendered depth is an ideal geometric sensor approximation. It does not reproduce
stereo matching failures, confidence, material-dependent holes, or calibrated
noise. Keep that limit explicit; do not introduce an unvalidated stereo-noise
project as a prerequisite to B1. RGB-D is selected because the intended physical
camera provides depth and it supports metric manipulation perception.

## Sources and Validation Boundary

External source facts were checked in Alex's `README.md`, `measurements.yaml`, the
Purdue WSG and standalone WSG URDFs, `robots/alex_purdue.py`,
`robots/purdue_physics.py`, `robots/purdue_frames.py`, `end_effectors/weiss_wsg32.py`,
`platforms/purdue_alex003_pedestal.py`, `sensors/zed_x_mini.py`, and its dependency
record. No Alex files were changed. Consumer commissioning uses the GPU; no hardware
trial was performed. Operational pose tolerances are 10 mm / 5 degrees sustained
for 0.5 s. Phase 4.1 records the frozen common setup, probe/contact criteria and
force limits, with repeated four-door evidence. These are controller-qualified
simulation limits, not global kinematic bounds or hardware-safety limits.
