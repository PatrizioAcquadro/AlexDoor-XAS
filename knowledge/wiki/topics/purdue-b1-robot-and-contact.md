# Purdue B1 Robot and Contact Contract

This is the implemented Purdue operational contract for Subphases 4.0–4.1. The B0
Alex V2 runtime has been retired. The common synthetic-door setup is GPU-qualified;
learned B1 integration remains later work. Current evidence is maintained in
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]].

## Reuse and Ownership

Reuse the editable `~/Desktop/Alex` package. It owns robot/gripper assets,
actuators, collision geometry and filters, sensor mounts, and measured pedestal
geometry. DoorManipulation owns the task contact frame, control, sensing
contracts, scene placement, and benchmark qualification. Do not copy Alex assets
or generic integration code into this repository or Isaac Lab.

Use `ihmc_alex_isaaclab.robots.alex_purdue.make_alex_purdue_cfg` with
`fix_base=True`, `variant="full_convex"`, and `end_effector="wsg32_umi_v1"`.
The source is
`~/Desktop/Alex/assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf`.
The factory defaults to SAKE when the end effector is omitted; select WSG explicitly.

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
`(0.202000, 0, 0) m`, with identity rotation. The reference is the center of a
finite two-finger footprint, not a material point in the gap between the fingers.
The consumer computes it from the canonical collision geometry;
use the current geometry as the authority if the external asset changes.

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
Task normal force sums authorized contacts once; forbidden and structural-support
records remain separate. Tangential friction is not included in the reported force.

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
