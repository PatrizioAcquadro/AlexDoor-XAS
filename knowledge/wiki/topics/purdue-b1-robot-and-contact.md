# Purdue B1 Robot and Contact Contract

This is the implemented Purdue operational contract for Subphases 4.0–4.1. The B0
Alex V2 runtime has been retired. Historical synthetic-door qualification used the earlier controller; the current
benchmark controller has local robot-only 6.0D evidence. Contact qualification does
not transfer automatically. B1 learned-action integration remains open in Phase 6.1. Current runtime evidence is maintained in
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]].

## Reuse and Ownership

Reuse the editable `~/Desktop/Alex` package. It owns robot/gripper assets,
actuators, collision geometry and filters, sensor mounts, and measured pedestal
geometry. AlexDoor-XAS owns the task contact frame, control, sensing
contracts, scene placement, and benchmark qualification. Do not copy Alex assets
or generic integration code into this repository or Isaac Lab.

Use `ihmc_alex_isaaclab.robots.alex_purdue.make_alex_purdue_cfg` with
`fix_base=True`, `variant="full_convex"`, and `end_effector="wsg32_umi_v1"`.
The source is
`~/Desktop/Alex/assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf`.
The factory defaults to SAKE when the end effector is omitted; select WSG explicitly.

`DoorPushPurdueEnvCfg.urdf_cache_dir` optionally routes robot and pedestal URDF
conversion into separate `robot/` and `pedestal/` subdirectories of a writable
project cache. `pilot_physics.py --urdf-cache-dir PATH` supplies it for a workload.
The default remains the upstream converter path. This avoids a shared, unwritable
`/tmp/IsaacLab` on Blackwell without modifying Isaac Lab or Alex, changing assets,
or changing control/physics. `TMPDIR` alone does not redirect the hardcoded path
in the inspected converter version. See the actual compatibility boundary in
[[experiments/act-a1-complete-cycle-collection|the complete-cycle campaign]].

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

## Common benchmark controller and stop

`DoorPushPurdueEnv` keeps the shared factory's explicit `IdealPDActuator` and
physical effort/velocity/position limits. Task-local right-arm impedance is common
to A1 joint targets and the pose controller used by A2–A4. The independent factory
copy is changed only inside this project; left arm, neck, fingers and the external
Alex package retain their settings.

An optional `pose_target_domain_rad` supplies seven explicit joint-position
intervals for pose-command generation. The controller intersects them with the
original physical limits before applying `pose_target_margin_rad`; neither input
changes physical bounds, A1 joint-delta limits, load calibration or stopping.
Invalid/empty intersections refuse. A domain outside the next rate-limited target
step also refuses instead of forcing a target jump. Without the optional domain,
existing target construction remains unchanged. This configuration constrains
proposals and does not certify their actual movement/contact; observed admission,
model applicability and full stop proofs remain independently required.


The optional `pose_feedback_time_constant_s` changes the outer pose controller
only. At each measured state, it adds `dt / tau` times the damped pose correction
to the previous delivered joint target, retaining nullspace centering and the
same rate, reserved-target and physical bounds. `tau` must be finite and at least
one control interval; the previous target must be available and finite. Only the
bounded delivered target is retained, so clipping creates no hidden accumulated
integral. Repeated previews are pure, and zero measured pose error can retain a
nonzero target offset that balances elastic load. Reset and the independent
latched stop retain their original behavior. The option is disabled by default,
preserving the original absolute IK request. This software capability changes
neither inner actuator impedance nor a physical/load/observation acceptance gate;
complete native motion, related-condition and stopping qualification remain
required before collection adoption.

With feedback enabled, optional `pose_reference_feedforward` also tracks the
increment between the previous dispatched world-pose goal and the current goal.
For `alpha = dt / tau`, the task-space step is `alpha * error + (1 - alpha) *
reference_delta` before the existing damped solve, centering and bounds. The
current error already includes the next-tick reference increment; paying only
its remaining fraction avoids a one-tick lead. No future goal or evaluator input
is used. Previews never commit reference history. A delivered pose command saves
its goal; a joint command, reset or independent stop clears it. Cold start uses
feedback alone. This option is false by default and requires feedback mode.
It addresses moving-reference lag while retaining the balancing offset under
elastic load; software tests do not qualify contact or native stopping.



| Right joint, canonical order | Previous Kp | Current Kp, N m/rad | Current Kd, N m s/rad |
|---|---:|---:|---:|
| SHOULDER_Y | 26.78 | 320 | 8 |
| SHOULDER_X | 26.78 | 26.78 | 8 |
| SHOULDER_Z | 23.76 | 160 | 4 |
| ELBOW_Y | 23.76 | 160 | 4 |
| WRIST_Z | 5 | 20 | 1 |
| WRIST_X | 5 | 10 | 1 |
| GRIPPER_Y | 2 | 50 | 1 |

The ready-pose 4 N diagnostic reflects approximately 1.7 N m at shoulder Y,
0.7 N m at shoulder Z/elbow and 0.22 N m at terminal Y. The selected stiffness
allocates about 0.005 rad static deflection to those joints, with additional wrist
rotation restraint; shoulder X already meets the local budget. Terminal damping
increases from 0.5 to 1. No integral or frozen disturbance torque is introduced.
These values are simulated benchmark settings, not hardware calibration or a
validated contact impedance across the workspace.

Every physics substep commands gravity compensation plus the explicit position
spring and velocity damping, clipped by the existing effort limits. PhysX drive
stiffness/damping are zero for this explicit path: read the actuator object's
resolved gains. `computed_torque` is the total pre-clipping actuator command;
`applied_torque` is the clipped effort sent to the solver. Solver projected joint
output is a separate feedback quantity and is never replaced by either command.

`PurdueIO.stop()` clears admission and calls `env.stop_motion()`. The environment
clears pending commands, latches measured joint positions once, zeroes velocity
targets and retains state-dependent gravity compensation. The previous spring
compression is removed; the new position spring supports subsequent disturbance
relative to that measured anchor. Repeated stops preserve the original anchor.
A1, pose/A3, neck, nonzero Gym and IO commands refuse with `motion_stopped` before
advancing physics. Zero Gym actions can advance stopped physics and telemetry;
only an environment reset re-enables motion. This removes inherited pushing effort
without disabling support. Free-space load removal does not prove unloading in
frictional door contact; physical contact stopping remains unqualified.

`verify_stop_continuation` records a fresh measured `anchor_joint_position`
for each diagnostic segment, while `stopped_targets` retain the one original
stop latch until reset. Calibrated radial/directional stop covers use the original
stop-onset position and link axes throughout loaded, held and load-removed
continuations. Verify identical targets and the concatenated tail against that
latch; recentering the calibrated cover at a later loaded pose changes the
question. Per-segment travel diagnostics and their original limits still apply.

The [[../experiments/b1-approach-controller-compatibility|unchanged-controller approach comparison]]
adds unloaded trajectory evidence: right tracking passes, left transient tracking
fails despite static IK support, and actual speed reaches 0.527 rad/s under the
0.5 rad/s target-step cap. Two-second moving stops at the prespecified 2.1 s points
travel 2.57/4.62 mm. Neither the later fast state nor loaded/contact stopping is
qualified, and the original ready-pose residual domain does not cover these paths.

The current resumed pilot adds a targeted reference-speed headroom correction.
A learned ACT A2 run reaches 0.51826 rad/s under targets bounded at 0.5 rad/s,
violating the unchanged 0.51 rad/s physical criterion. The shared A1/A2 target
cap now uses 90% of the configured motion ceiling: 0.45 rad/s for this setup.
IK, joint centering, gains, joint position limits, stop behavior, door dynamics
and the physical speed criterion are unchanged. Preview and execution share the
same cap; raw requests and applied targets remain separately recorded.

A fixed robot-only replay reproduces the original 0.51829 rad/s peak. The same
requests with headroom peak at 0.47323 rad/s with zero speed violations. Load
bounds, the complete continuous stop cover and the unchanged 13 mm / 40 mrad
response limits pass. All 17,985 saved teacher target transitions are below
0.312 rad/s, so the new cap is inactive on the validated teacher trajectories.
This finite controller prerequisite is not learned-policy success or a global
hardware velocity guarantee. Current learned execution remains in
[[../experiments/b1-end-to-end-pilot-resume|the end-to-end pilot]].

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

## Documented robot feedback and planned contact monitoring

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

The current B1 recorder/IO carries only joint position/velocity in policy observations.
6.0A adds `RobotFeedback` and `PurdueIO.robot_feedback()` as a separate monitor-only
interface in canonical seven-right-arm joint order, N m and positive joint-coordinate
sign. Simulation reports the solver joint-output projection at each 120 Hz physics
sample; hardware acquisition and sensor accuracy remain unverified. Old recordings
without torque remain unavailable for loaded qualification. Commands and privileged
contact data never synthesize measured feedback. Freshness, known source/semantics, bounded
robot-only load evidence and verified stop/contact behavior are separate admission
requirements. B1 local contact has bounded simulation evidence and real learned
contact under the [[../experiments/b1-end-to-end-pilot-resume|resumed pilot]].
Complete full-door A3/A4 operational qualification remains separate. The current
main gripper acceleration bound is 13 rad/s², independently checked under known
8/10 N wrenches with unchanged error coefficients; geometric motion bounds, 1 N
weak-load admission and 10 N stop-force domain are unchanged. This is a finite
simulation recipe, not hardware or arbitrary-motion qualification.
A separately checked `weak_load_q_min` allows a lower elbow position of −2.33 rad
only at force upper ≤1 N and inside the original main acceleration envelope.
Independent 0/1 N known-wrench samples pass unchanged error, response and all
14-shape stop bounds; the unconditional stronger-load extension fails and remains
disabled. No combined weak-position/extended-acceleration domain is admitted.
The motion producer additionally requires the complete proposed-motion/stop
volume to remain contact-free whenever the conservative joint sweep/stop can
enter this extension.
It spends the declared nonzero target translation and angular rates through the
full stop duration, in addition to capture age and control-interval growth. A
small current force alone cannot authorize future contact outside the strong
position envelope. All main position bounds and complete teacher outputs remain
unchanged. A narrower strong extension also fails its 8 N lateral stop. The final
panel keeps the original main position recipe; the optional weak-position recipe
is inactive. Actual new-checkpoint task success remains pending.
### October 19 calibration and stopping correction

The current correction keeps empirical position coverage separate from physical
limits. `LoadCalibration.position_support()` accepts scalar legacy or lower/upper
per-joint force caps. Future motion uses the required complete stop-force domain,
not current weak force. A side qualified at that force may spend its supported
position range; weaker/exhausted coverage still requires independently observed
contact-free motion and stopping. Actual feedback is never clipped. Unsupported
position, speed, acceleration, load and combined domains remain refusals.
For a position side already qualified at the required stop force, forecast
comparison permits one outward representable float32 boundary step. Forecasts
combine native float32 joints with double-precision stopping arithmetic; a
retained D2-H- refusal differed by only 1.6e-12 rad. This numerical comparison
does not enlarge physical travel, change force caps or clip actual feedback.
An unqualified side receives no credit; a further representable exit still refuses.

The separate local recipe in
`outputs/evidence/act-a1-calibration-stop-20261019/runtime-recipe.json` supports
8.5 mrad at lower RIGHT_WRIST_Z and upper RIGHT_SHOULDER_Z with loaded 10 N stop
verification; other original weak sides remain at 1 N. This is one full 60 Hz
control interval of position support at the existing 0.51 rad/s bound, not sensor
precision. The residual transfer includes the independently checked reference
point uncertainty once. Main acceleration and combined-domain admission are
unchanged. A 10.044817 rad/s² unload transient slightly exceeds the nominal
10 rad/s² control prediction; an independent complete-curve cover verifies that
it remains inside the existing stop volumes. The control bound is not enlarged
and the empirical excursion is not called a physical safety violation.

The corrected stop-only profile has right-arm Kp
`[640, 640, 320, 320, 40, 80, 100]` N m/rad and unchanged damping ratios, measured
anchor latch, gravity compensation and physical effort limits. The environment
validates the seven positive gains; the pilot CLI takes them from the supplied
recipe, and runtime refuses an explicit controller mismatch. Reset restores all
normal gains. The default profile preserves the earlier stop behavior for existing
recipes; normal ACT motion gains never change.

Actual robot sweeps already include motion/acceleration/stopping. Geometry no
longer adds the 13 mm / 40 mrad tracking allowance as a second robot displacement,
or spends target translation twice. Target translation/rotation growth still
covers the complete two-second stop. Existing radial/joint/rotation stop bounds
are retained; the explicit new recipe's directional boxes conservatively cover
the changed stop direction. These empirical predictions are distinct from the
unchanged physical stop requirements. Calibration/stopping evidence is finite
simulation support, not global all-speed/workspace, hardware or learned task
qualification. Teacher joint-trajectory similarity remains diagnostic only.
See [[../experiments/b1-act-a1-calibration-stopping|validation and failures]].

The approved
[[../implementation_phases/phase-6-0-operational-perception-and-contact|operational plan]]
assigns actual torque/device acquisition and signal verification to 6.0D, without
changing policy features or old recordings. External-load inference needs a
robot-only model, verified feedback semantics and bounded model/friction/timing
error. An anomaly detector does not certify exact contact location or all force
criteria. Simulated observations must represent the robot sensor, not expose
evaluator contact pairs or perfect external forces.

The guide warns that disabling actuators can let the arms fall. The implemented
measured-anchor stop retains arm support while physics continues, as described
above. Clearing the command queue alone remains insufficient. Hardware safety and
physical unloading in contact remain unvalidated.

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


The resumed pilot exposes a stop-domain limit at extended arm poses: retained
4 N load causes 24.56/26.16 mm travel despite a fixed measured anchor. A prospective
stop-only correction raises right shoulder-X stiffness from 26.78 to 320 N m/rad
and scales its damping by `sqrt(320/26.78)` (8 to 27.65 N m s/rad), preserving its
damping ratio. Other gains, physical limits, gravity compensation and latch
semantics are unchanged; reset restores motion impedance. Across both measured
paths, linearized 4 N normal-load equilibrium displacement is at most 7.26 mm
versus 37.24 mm with motion gains. This is a numerical design bound for the tested
load directions, not a dynamic stop certificate or an arbitrary-direction bound.
Case 15 verifies twenty fixed 0/4 N stops with maximum travel 9.68 mm and
three native-peak fault stops. Its initial moving speeds reach only 0.0353 rad/s;
actual approach/task-speed coverage remains under independent verification.
The user subsequently removes the prerequisite case-count ceiling; resource
budgets stay fixed.


## Pressure feedback during qualified teacher collection

The guarded pressure mode uses the completed native interval when its below-target
force contradicts an above-target qualification window. Previously this branch
returned the target itself and froze force-error correction, even for a zero-load
interval. It now retains the bounded positive admittance response; gain, rate,
extent, physical force limits and qualification windows stay unchanged. Other
intervals use the qualification-window feedback, avoiding pressure withdrawal in
response to a single impact above a still-low supported mean. Historical explicit
native-interval recipes remain readable and retain their original behavior.
The October31 D2-RP diagnostic reproduced the refusal at152.0s and retained
intermittent zero native means with a0.03396N final physical window. The separate
observed lower bound crossed zero. This resolves the missing-feedback question;
that unchanged diagnostic alone does not qualify a full source. Subsequent distinct
pressure-v2 D2-RP and D1-H- sources independently pass their complete native,
variation/recovery, stop, visual and temporal reviews. Transfer beyond those
conditions remains a separate requirement.

An optional teacher-only `normal_force_lead_s` now forms a pressure-control
signal from the current completed native interval and its preceding sample:
`F_control = F + T * (F - F_previous) / elapsed`. The first sample uses its
measured force unchanged, acquisition clears the history, and episode/stop
lifetime prevents cross-episode reuse. Positive lead requires explicit
`native_interval_mean` feedback; zero preserves the prior controller. The
signal is recorded separately from measured force and never enters load or
contact qualification. Existing rate/extent limits still bound its command.

For an ideal first-order pose servo, choosing the force lead equal to its time
constant factors a nominal scalar characteristic. That model omits the actual
whole-reference feedforward, which also anticipates pressure corrections. It
cannot establish stability or transfer for the combined runtime. All four D4
N/H+/H-/P native lead qualifiers now fail early push contact continuity; the lead
is unadopted. Original recordings, independent lower-force failures, later
empirical withdrawal-domain exits and passing physical stops remain preserved.

A distinct recipe uses the existing `native_interval_mean` feedback with zero
lead and pending-motion compensation. This selects continuous gap-error
correction below the target, subtracting normal advance already commanded.
All four D4 cases still lose early push contact; the recipe is unadopted. Its
passing physical stops do not establish task contact or full-source success.

Pressure and motion requests must also be physically compatible. In the intended
normal-only rotational model, steady hinge resistance requires normal force
`damping * reference angular speed / normal torque lever`. For the original D4
geometry and unchanged dynamics, 1 degree/s requires about 0.354 N; 0.1 N instead
balances about 0.282 degree/s. A prospective shared recipe uses the existing
explicit `push_force_n` field for this derived request while retaining 0.1 N
acquisition/hold. No force/load acceptance limit, gain, rate, geometry or physical
dynamics changes. Native inertia/free-coast checks support the parameter reading;
startup acceleration, friction and full nonlinear contact still require complete
related-condition qualification. Privileged native normal feedback never
qualifies an observed source.

The prospective guarded controller reserves one-shot geometric gap recovery for
zero native force with separation beyond the setup's existing contact-gap
tolerance. Inside that tolerance, ordinary force-error admittance remains active;
target force, gain, rate/extent bounds and all qualification thresholds are
unchanged. This avoids treating a sub-tolerance gap as a geometric loss solely
because one native interval has zero force. The D2-H+ negative contains a10.726N
native impulse after such a recovery and correctly refuses its12.847N estimated
upper bound against the unchanged10N stopping model. Related qualified D2 sources
also contain sub-tolerance recoveries, so saved correlation does not establish
causality or controller transfer. Source `2cd8697` now passes the complete native
D2 nominal/H+ pair: motion, contact, hold, release, stabilization and original
six-second stops for all 14 shapes, with all 48,584 task-native load estimates
inside the original 10 N domain. The measured hold-height difference is 19.981 mm
against the original 20 mm variation. A distinct guarded observed H+ source is
running after this paired component qualification. Privileged native door
normals do not qualify observed geometry, identity, material or source admission;
transfer beyond this native pair remains a separate requirement.



A complete-collection protocol may explicitly enable `completion_opening_handoff`
with its bounded task-goal hold. The teacher then starts holding after the current
continuous supported window reaches the unchanged required opening and any
injected recovery has finished. This avoids continuing the maximum-reach expert
probe beyond an already achieved task goal. It does not admit that window as a
complete demonstration: the full original hold minimum, release, stabilization,
continuity, force, joint, perception and independent stop checks still apply.
Protocols without the option retain their original maximum-reach behavior.

The distinct D2-H- and D2-DR task-goal sources independently pass their full
native cycles, unchanged3s holding, original measured variations,2+4s stops,
real-image review and GPU temporal preparation. This qualifies those two
recordings; it does not change earlier maximum-reach outcomes or qualify D3/D4.
