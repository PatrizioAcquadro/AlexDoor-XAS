# Phase 6.0 — Operational Perception and Observed Contact

> Current boundary, October 2, 2026: 6.0A contracts and 6.0B static scans are
> maintained. The custom dynamic/material trackers, full-state replay command and
> SAM3 video comparison are retired. The CAD-free Point2Pose 6.0C prototype is
> implemented but unqualified; 6.0D-H remain planned. Historical failures remain failures.

## Objective and current boundary

Deliver observed geometry for contact-rich door manipulation while preserving
A4's full-pose approach/contact/push/hold/release semantics. Unknown total dimensions
are optional when they do not affect a proposed action. Static visibility,
whole-object ownership, original-material identity and motion admission are distinct.
See [[../topics/shared-door-perception|maintained interfaces]] and
[[../experiments/b1-perception-findings|results and retirement decisions]].

The current `GeometryProvider` exposes static candidates through `scan_state` and
returns invalid `DoorEstimate` values without policy encoding; the optional
Point2Pose adapter supplies independently supported local diagnostic zones. GroundingDINO,
native SAM3 and DINOv3 stay frozen for 6.0B. No qualified release exists; the legacy
6.1 release interfaces are retained without an early schema migration.

Future work starts with the two train pilots: left `door-2738468b94d74c5f` and right
`animated-door-1-88abf40`, each nominal/light. All 50 engineering-v2 episodes remain
available for later train/development qualification. Use one common recipe, one
simulator at a time and CUDA for model/simulator work. No new training, data
collection, corpus change, sealed-test access or push belongs to this cleanup.

## Approved operational protocol

### Inputs, ownership and common execution

The geometric provider receives only calibrated RGB-D, validity, timestamps/frame
counters, joint state and camera poses derived from calibration/FK. Exclude asset
identity, handedness labels, split, teacher commands/poses, recorded phases and
simulator annotations. Evaluators may use those labels for scoring and partition
enforcement; they cannot influence runtime acceptance or action choice.

Add a separate, timestamped robot-feedback channel for reported joint torque and
device health, initially consumed only by the common monitor. Preserve the policy's
current visual/proprioceptive feature schema; do not silently append torque features.
Declare joint order, units, signs, measurement semantics and unavailable/stale status.
An old recording without torque remains usable for geometric replay, but cannot
qualify loaded control. Never synthesize measured torque from recorded commands.

In simulation, emulate a physically realizable robot feedback channel matching the
documented actuator signal. Applied actuator torque, commanded torque and external
contact wrench are different quantities. Do not feed exact contact pairs, leaf
identities, door parameters or perfect external wrenches to this channel. Robot
models/calibration are allowed; true door state and physical-contact scoring remain
in the separate evaluator. Sensor/control approximations must be stated and tested.

Perception estimates; the policy chooses actions; the adapter transforms them;
the low-level controller executes them with common gains/limits; the monitor may
stop them. The diagnostic controller is an explicitly separate action source.
No monitor/adapter may choose a better point, switch the hinge side, compensate
opening progress, insert missing stages, add an expert push or complete release.

### Geometric state and action admission

Maintain a leaf object with associated surfaces, fixed surrounding geometry and
competing hinge hypotheses. Retain support and uncertainty for each required field.
Use acquisition, ambiguity, tracking, loss and reacquisition states. Extend existing
contracts rather than create independent replay and live estimators. Keep the old
complete-state validator for the legacy profile; route the new explicit validators
through every observer/adapter consumer without changing old validity silently.

| Information | Requirement and lifetime |
|---|---|
| Leaf identity and usable surface | Distinguish leaf from frame/wall/handle; retain a local 3D patch, full contact orientation and sufficient support for the two-finger footprint. |
| Hinge and closed reference | Axis position/direction, floor-anchored origin and observed closed orientation; initially one or more supported hypotheses. A leaf edge is not automatically the physical hinge axis. |
| Dynamic state | Signed angle, panel pose, patch pose and motion support, with acquisition/availability times and uncertainty. |
| Relevant occupied/free space | Robot/hand and leaf swept volume for the proposed segment and stop response; explicitly retain unknown regions and assumptions. |
| Total dimensions | Optional operational diagnostics unless the proposed action or collision envelope actually depends on them. Missing values remain missing, not nominal completions. |

The task supplies a closed, unlatched push-door initial condition, not a door pose.
Observe the closed reference. Infer hinge side from available views and metric cues;
left/right in an image alone is not a general push/pull rule. Express both hands in
one tested signed-angle convention; intersect the axis with the calibrated floor,
not an assumed lower leaf edge.

Separate three decisions:

1. **Geometric qualification:** a unique, supported operational state meeting the
   scoring profile below. It does not, alone, establish loaded-contact safety.
2. **Provisional action admission:** a specific bounded diagnostic action can be
   admissible despite an unresolved hinge, when its local material contact, world
   path and stopping margins remain valid across the bounded possible responses.
   An axis-dependent command also requires compatible supported hypotheses. This is
   not a qualified estimate or permission for arbitrary actions.
3. **Loaded-control admission:** fresh, validated robot feedback and the observed
   contact/load monitor are also available. An anomaly detector alone is insufficient
   evidence for all loaded-contact/force requirements.

For provisional admission, retain an explicit supported reference for interpreting
A4. The action source selects one physical patch/motion; express that same patch
consistently in each hypothesis and bound the possible leaf response and relative
contact error for the proposed world trajectory. Reject when the spread consumes
the contact/collision margin. Do not reinterpret identical local coordinates under
another hinge and thereby command a different point. Execute one admitted command,
not a mixture of competing trajectories.
No arbitrary canonical hinge or highest-score-only acceptance is allowed. If no
identifiable leaf patch exists, observe further within the common scan or stop.
An unobserved axis blocks A4 initialization, not the development of a local
Cartesian diagnostic probe designed to estimate it. That probe needs an explicit
pre-articulation state/admission path; the current `validate_reference`/`admit_action`
requires supported hinge hypotheses even for A2, so it cannot yet admit such a probe.
Do not insert an invented hinge or relax the existing validator. The policy input
can be well-formed while geometry remains
provisional; never set the existing complete-state `valid` flag just to get past
the current observer. Contract, consumer and evidence changes must land together.

### Surface memory, timing and uncertainty

Fuse the seven-view scan chronologically in the calibrated world frame. Preserve
different colors, relief faces and boundaries belonging to the same rigid leaf;
keep fixed frame/wall support separate. Joint reprojection, depth consistency,
appearance and later motion determine ownership, not plane area or mask confidence
alone. Human labels A/C=fixed and B=leaf bottom apply only to the reviewed images,
not as runtime rules; other doors can have a bottom frame.

Static observations remain useful after leaving the image. Do not reset hinge or
upper-edge knowledge merely because it is clipped. Update dynamic pose using
currently supported leaf features; propagate uncertainty through temporary gaps.
Rigid-body and local surface-continuity assumptions are allowed when bounded and
consistent with observations. An unobserved volume is neither automatically free
nor a reason to reject unrelated local motion. Check only the relevant swept volume
and stop envelope; do not require an exact whole-room reconstruction.

The maintained static scan targets semantic inference at 5 Hz. Future 6.0C
tracking must operate causally between results. Retain the 150 ms maximum age for dynamically supported estimates; old
static geometry needs consistency, not a new segmentation timestamp. Prediction
alone cannot refresh an observation or hide a lost identity. Replay/live share
completion events, bounded queues, generation cancellation, reset semantics and
no backdating. A result unavailable at a scored instant cannot repair that instant.
Force/proprioceptive checks run at the control cadence, independently of the visual
worker; determine their timeout from the stopping budget, not the semantic rate.

### Contact choice and exploratory interaction

The new diagnostic contact rule chooses an observed leaf patch with adequate
footprint/uncertainty clearance, reachable arm posture and separation from frame,
handle and discontinuities. Prefer useful distance from the hinge when resolved;
do not require a particular fraction of unknown width. Use one deterministic rule
on both hands/conditions, not per-door tuning. Retain the expert's 0.295 width
fraction / 1.09 m rule and all recorded labels unchanged. A policy's chosen point
is validated, never projected onto a better point by the adapter.

For an admitted exploratory action, use a short motion compatible with the remaining
hypotheses, with prespecified displacement, duration, velocity and load limits.
The action source chooses it. Track actual leaf motion; hand FK alone is not leaf
motion because pushing can slip. Reject insufficient rotation/ill-conditioned hinge
fits rather than claim centimeter precision from a barely visible displacement.

When the axis is unavailable, initialize an observed material patch in world
coordinates with its plane normal/tangent orientation. A diagnostic action source
may propose a bounded Cartesian displacement through the existing A2 pose executor;
it must not encode that probe as a guessed A4 arc. Precise axis location, hinge side,
closed angle, total dimensions and complete scene ownership are not prerequisites
unless that particular command uses them. The probe still needs a justified coarse
articulation/response envelope including no motion, slip and incompatible response,
finite contact/model uncertainty, continuous robot/tool and relevant leaf collision
cover, valid load feedback and a verified stop. A small hand displacement alone
does not bound a slipping or freely moving leaf. If no conservative response/stop
bound can be established, stop at observation and report that specific missing bound.
The probe must produce motion distinguishable from geometric/tracking uncertainty;
insufficient response leaves the axis unavailable. C/D must implement this explicit
local admission path and E must assess provisional evidence before F; existing
qualification gates and adapter semantics remain unchanged.

Coherent leaf motion refines the hypothesis. Rising load without leaf motion stops
the action; it does not prove the opposite hinge side. Hand-only motion may indicate
slip or missed contact. Unexpected articulation, loss or excessive uncertainty stops
and discards queued actions. Never automatically try the opposite side.

Exploration belongs to the diagnostic controller here. In a later learned benchmark,
any exploratory push must be chosen by the policy, consume the episode budget and
remain inside its A1-A4 sequence. No shared pre-policy push may perform part of A4's
task. Whether existing demonstrations teach exploration is a later 6.2 question,
not permission to train or collect data during 6.0.

### Load, compliance and stopping

Start with reported joint torque and a robot-only dynamics/friction model. Bound
the residual error before interpreting it as external load. Check actuator-side
versus joint-output semantics, gravity compensation, saturation, sample timing,
singular configurations and multiple-contact ambiguity. Distinguish gross anomaly
detection from a usable contact-load estimate. Do not assume a wrist sensor exists.

Reuse the robot's documented PD/impedance interface, physical limits and known
tool geometry. Select common gains, speed, duration and stopping margins from the
existing setup and bounded numerical/simulation checks. Lower stiffness or slow
motion alone does not prove a contact-force bound. Account for tool/gripper mass,
model error and latency; do not derive safe force by equating a joint torque limit
with newtons at the fingers. No new force regressor/training is prescribed.

The existing physical evaluator retains its authorized surfaces, force limit,
controlled-contact and collision rules. In `configs/purdue_synthetic_probe.json`,
80 N is the force limit, while 0.02 N is the loaded-contact threshold with a
0.1 s window and 0.0001 m gap tolerance. These are evaluator criteria, not a demand
that motor feedback resolve 0.02 N or measure a 0.1 mm gap. Operational detection
needs a justified uncertainty margin and may require a clearly larger detectable
load within the unchanged physical limit. If it cannot reliably distinguish the
required states, loaded control remains unqualified; do not use evaluator feedback
to fill the gap or silently relax physical criteria.

Keep low-level feedback tracking distinct from task correction. Normal replanning
occurs at explicit action/segment boundaries while state remains admissible. A
contradictory hinge/identity update must not silently reinterpret an executing A4
segment. Safety/loss stops latch, discard pending commands and require explicit
reset; reacquisition does not resume execution. Do not add automatic mid-segment
policy recovery in this revision.

Specify a robot-state/load-based stop response that removes continued drive into
contact while supporting the arm. Validate it with physics still advancing after
the stop. Clearing a queue, retaining a penetrating target, freezing simulation
or disabling motor power is not proof of safe stopping. A stop may terminate the
task, not add a successful release or helpful opening. If no suitable stop can be
verified, do not admit loaded tests or claim hardware safety.

## Scoring and progression gates

### Two explicit profiles

Keep the reusable full-state metrics and historical results unchanged under
the **legacy full-state profile**: dimensions, hinge, signed angle and prescribed
contact, 1 cm / 5 degrees, 95% coverage and 95% joint accepted-state precision.
The retired replay command is no longer maintained. Future evaluators may report
this profile separately; a missing
dimension is still a failure there. Historical failures do not become passes.

Implement a separately named **operational-v1 profile**, the approved future 6.0
qualification target. Required accepted state: unique leaf/hinge identity, supported
hinge origin/frame and closed reference, signed angle, panel rotation, selected
contact pose in local/world coordinates and current dynamic support. Keep 1 cm for
hinge origin and local/world contact positions; 5 degrees for hinge, panel and
local/world contact orientations and wrapped signed angle. Require every accepted
error p95 within its limit. Total dimensions are not mandatory in this profile,
but their errors/support and any action dependence remain visible in the report.
No scalar confidence or correct plane alone replaces this operational state.

Also evaluate the actual proposed short trajectory and full contact footprint:
propagate hinge/angle/patch uncertainty, current robot error, latency and stop travel
against available surface/collision clearance. The 1 cm / 5 degree component gates
are necessary qualification checks, not proof that every action has enough margin.
Prespecify the common uncertainty/admission recipe in 6.0A; calibrate it on train
pilots, then fix it before wider evaluation. Numerical gate values are not tuning
parameters. If a necessary bound cannot be justified, report the missing evidence.

The evaluator must score the **chosen physical contact**, not its distance from
the expert's point. At first selection, use evaluator-only true leaf surface and
pose to associate the submitted observed contact/footprint with a material location;
retain that correspondence through tracking. Compare local/world pose at that
location and reject fixed/handle/missing/ambiguous support. If a contact target
changes explicitly, score the new selection and its transition rather than snapping
each predicted point onto whichever surface minimizes error. Use an independent
surface query, not the predictor's own plane. Reuse prepared leaf geometry for
physical contact and visual geometry for visible-surface diagnostics, labeling any
disagreement. The old prescribed-point annotations cannot score arbitrary new
targets alone. Truth and reference queries stay entirely in the evaluator.

### Denominators and reports

For each door during contact/push/hold, coverage is qualified operational states
divided by all scheduled observations; joint precision is accepted states meeting
all operational limits divided by accepted states. Both must be at least 95%; an
empty accepted set fails. Provisional states, exploration, losses and ambiguity
remain in the coverage denominator and are not qualified accepted estimates.
Report their finite errors and action-admission outcomes separately. Predictive
bounds that admit unsafe trajectories fail regardless of aggregate precision.

Replay uses all chronological observations, distinguishing HDF5 rows and frame
counters. Phases for scoring are evaluator labels, never perception inputs.
In dynamic trials, failures before contact count as failed starts/trials; report
zero coverage for a trial that never reaches manipulation. Do not report a pass
from an empty or successful-only subset. Predeclare the timed diagnostic schedule;
early stops retain scheduled remaining manipulation ticks as unavailable. Include
all attempts, elapsed/initialization time, provisional duration and sustained
controlled progress. No free exploratory opening or successful restart selection.

Report per door, condition and phase: p50/p95/max for accepted, rejected finite and
provisional states; coverage, joint precision, rejection causes, inference/availability
latency, time to usable hinge, loss/recovery and physical validity. Highlight worst
left/right cases and preserve a few targeted association/geometry images. Separate
surface reacquisition from qualified state recovery. If a human image decision is
needed, show the exact ambiguity and dependent decision, then await the answer.

### Staged admission

1. **Pilot replay gate (6.0E):** both train pilots and both conditions pass
   operational-v1 with one recipe; main ownership/tracking causes are resolved.
   Document initialization observability and provisional candidate admission.
   Recorded teacher motion only tests estimation; it does not prove the controller
   could cause that motion. Legacy full-state results remain separate.
2. **Bounded pilot dynamics (6.0F):** allowed only after 6.0E and the feedback/stop
   checks in 6.0D. This is an explicit diagnostic exception to the old rule requiring
   all train/development offline passes before any door dynamics. It is limited to
   the two train pilots and never constitutes release qualification. If either
   prerequisite fails, report not run and return to that cause.
3. **Extended replay (6.0G):** resume only after the common two-pilot correction and
   bounded control checks pass. Freeze one recipe; evaluate all 50 engineering-v2
   episodes, including pilot failures and every train/development door. No per-door
   retuning, sealed test or overwrite of the stopped campaign.
4. **Qualification dynamics (6.0H):** all train/development operational offline gates
   must pass first. Test the two train pilots and all six development doors under
   the matrix below. Release only after offline and dynamic gates both pass. A
   failure remains a failure, not permission to change a threshold or drop a door.

This revision changes the future qualification target and diagnostic sequencing
explicitly. Passing operational-v1 would qualify bounded operational use, not full
door reconstruction or real hardware safety. Keep diagnostic recipes separate from
`PerceptionBinding`; extend release/profile compatibility deliberately at closeout,
never forge existing `offline_passed`/`dynamic_passed` flags for pilot work.

## Work packages and exit checks

Code, tests and measured evidence determine completion. The local TODO tracks an
implementation assignment; the wiki records its durable boundaries. Use one agent
unless delegation is explicitly requested, serialize GPU workloads and create small
validated local commits. External Alex/Isaac packages are outside this work.

| Package | Maintained or future outcome | Exit/prerequisite |
|---|---|---|
| 6.0A — maintained | Field support, geometry/action/load admission, explicit contact transitions and consumer compatibility | Numerical contracts; no physical qualification |
| 6.0B — maintained | Calibrated static fusion, original observation references, ownership alternatives, static hinge hypotheses, finite two-finger support and relevant-space queries | Bounded four-pilot scan diagnosis; unresolved ownership/axis remain explicit |
| 6.0C — implemented prototype, unqualified | CAD-free Point2Pose tracking and original-material pose/recovery | Supported identity, pose, loss/reacquisition and informative articulation on both pilots |
| 6.0D — future | Robot feedback, common compliant control and latched physical stop | Measured/justified signal, load, timeout, gain and stop bounds before loaded interaction |
| 6.0E — future | Independent operational-v1 evaluator and chronological two-pilot replay | All four pilot conditions pass one recipe; no truth enters inference |
| 6.0F — future | Bounded diagnostic interaction on both train pilots | D/E gates, physically valid approach/contact/push/hold/release and stop/reset checks |
| 6.0G — future | One common offline recipe on all 50 engineering-v2 episodes | F passes; every train/development door passes, including failures and both conditions |
| 6.0H — future | Dynamic qualification, evidence-backed freeze and 6.1 handoff | G and D pass; complete baseline/fault matrix below |

### 6.0C — CAD-free Point2Pose direction

Reuse the [official model-free Point2Pose components](https://github.com/tzuyuan/point-to-pose)
initially, rather than build another custom point tracker. Distributed visual
references and metric RGB-D must identify the same rigid leaf through occlusion
and reacquisition. The prototype, isolated dependencies, official checkpoints and
smoke/replay/observer entry points are implemented. Runtime initialization works
on CUDA; all four frozen pilot replays and eight fresh-process observer cases
fail useful availability at 0%. Live complete latency p95 is 0.784–1.755 s;
some raw accuracy also fails. These results leave 6.0C qualification open.

Use calibrated camera kinematics at each acquisition time to separate head motion
from leaf motion. Transform the recovered object pose into the repository's world
frame and update an explicitly selected push zone fixed to the panel. Preserve
original-material identity, independent visual/contact support, acquisition and
availability times, episode generations and uncertainty. A static fit or raw-depth
visibility cannot refresh lost material support.

The initial pose/reference must not use CAD, prepared-door pose, asset identity or
simulator annotations. Retain competing static objects until observed evidence
resolves them. A pose tracker does not establish a physical hinge, loaded-contact
safety or the validity of an entire hand/arm path. Infer articulation only from
informative observed leaf motion; an unobserved axis stays unavailable. Preserve
6.0B's local two-finger, collision and unknown-space queries and its effective
geometric covers. The covers are neither flat pads nor measured compliance.

Validate common capture/completion timing, camera motion with a stationary panel,
leaf motion with a stationary camera, combined motion, occlusion/reacquisition,
reset and uncertainty. Keep the selected material point distinct from the visual
reference and never silently reselect it. Loaded or axis-free diagnostic action
admission remains 6.0D work; do not invent a hinge to pass the existing validators.

#### Implemented recipe and remaining acceptance work

Pinned upstream revision: `51856226610df75e5c06e8de545bd27f7c4ba99c`.
Keep BootsTAPIR, SAM2 large, SuperPoint, registration and graph at the official
paper recipe. Derive TSDF radius/volume from filtered measured candidate geometry,
using one common truncation/uncertainty margin and 5 mm voxels. The 25 cm filter,
one-time volume initialization and unenforced YAML extent/voxel limits are
code-proven incompatibilities with door scale. Expand by replaying retained
official keyframes while preserving the sparse map/graph/object. Refuse actual
host/device memory overflow explicitly. Apply calibrated metric camera depth
limits consistently; filled depth never refreshes measured support.

Early automatic-candidate smoke and concurrent Isaac smoke precede extending the
adapter. Report resident GPU memory, sampled process peak, exact PyTorch allocator
peak, whole latency, queue waiting and useful outputs separately. CUDA is required
for both models and fusion. Do not repair latency by loosening the 150 ms gate.

Initialization keeps the source image, mask, depth, intrinsics, joints and time
together, including delayed masks. Validate SAM2 source-component interiors;
initialization mismatch and tracking loss are distinct. Preserve camera Ct,
initial map M, observed object O and fixed zone Z and the composition
`W_Ct * Ct_M * M_O * O_Z`. Numeric round trips and independent motion tests must
catch missing inversions or double alignment. Replay/live share a bounded provider;
complete episode reset recreates both workers before new acquisitions and clears
all Point2Pose temporal state. Frame gaps invalidate semantic requests without
model reloads. The stateless 6.0B worker is released after the immutable automatic
seed is ready. Same-episode native reacquisition preserves the original identity/zone.

Confirm zone ownership separately from candidate tracking. Once established,
transport the zone from panel pose without requiring features in the uniform
contact face. Keep current local plane/discontinuity, both finite footprints,
obstacles and relevant unknown-space checks separate from zone localization.
Compute physical slide from the measured region eroded by footprints and each
primitive uncertainty once. Include angular displacement to actual fingers;
retain holes and the immutable anchor. Calibration/FK and temporal bounds that
have not been measured stay missing. The 1 cm/5 degree gates remain perception
qualification thresholds.

| Diagnostic case | Required outcome |
|---|---|
| Camera motion / fixed panel | Stable world candidate and zone; no false articulation. |
| Panel motion / fixed camera | Coherent world motion and zone transport. |
| Combined motion | FK compensation and correct residual panel motion. |
| Uniform or covered zone, other references visible | Useful zone localization from the panel pose; contact geometry may be unavailable. |
| Total occlusion | No invented support refresh; expiry/loss within freshness/bound limits. |
| Reappearance | Same explicit identity/zone, supported by fresh observations. |
| Tool slide | Separate allowed region, departure and tracking error; no inferred loaded contact. |
| Episode reset | No inherited references, results, selections, map or temporal model state. |

Use the four existing two-pilot nominal/light recordings and brief observer-only
Isaac cases in fresh processes. Keep all attempts, including failed ones, in new
output directories. In independently defined sufficient-reference intervals,
require useful availability at least 95% and p95 error within 1 cm/5 degrees.
Always unavailable fails. Keep full input denominators and complete official
qualification gates separate; do not discard losses retrospectively. Synthetic
visibility faults and ideal rendered depth do not validate hardware occlusion or
loaded response. No test in this revision enables loaded contact.

Implementation details are canonical in
[[../topics/shared-door-perception|Shared Door Perception]]; measured results and
unvalidated limits are in [[../experiments/b1-perception-findings|Perception Findings]].

The separate `point2pose-offline` evaluator processes every 60 Hz source frame
serially and scores native and integration results at acquisition time. It runs
the four complete pilots plus two additional fresh initialization attempts per
recording. Latency is diagnostic only in this mode. Mask rejection stays latched
per candidate while native diagnosis may continue; operational guards and the
150 ms deadline above remain unchanged. Relative seed alignment, ambiguous
ownership and full missing/lost/unprocessed denominators remain explicit. These
results do not evaluate official release gates or admit contact; see the canonical
[[../topics/shared-door-perception|offline evaluation contract]].

The first 60 Hz campaign was interrupted by the user after 4,216 frames in its
first recording. The preserved 4–10 s audit finds correct acquisition-time
matching and geometric composition, but a frozen native transform with
`lost=False` while the camera moves. One authorized 361-frame CUDA diagnostic
repeat confirms incorrect point correspondences, unflagged no-cluster fallback
and a separate accepted SDF pose that is not returned. Both defects are corrected;
one subsequent 361-frame CUDA verification confirms native loss and pose/statistics
consistency. The primary still has only 59 supported rows and 301 preserved lost
poses, so correspondence failure and long-gap recovery remain open. The full
campaign remains stopped. See
[[../experiments/b1-perception-findings|partial offline results and drift limits]].

### 6.0D — Feedback, compliance and stopping

Map documented `tau` to a declared observable; actuator feedback, commanded effort
and exact external contact wrench are different signals. Unknown origin/precision
stays unknown. Develop robot-only load residuals with uncertainty and test unloaded
motion, declared known loads, saturation, friction/model error, timing gaps and
ill-conditioned postures. No hardware data collection is authorized here.

Validate finite contact/model assumptions, control/feedback timeouts and a latched
stop while physics continues. Missing/stale feedback blocks loaded execution even
with perfect geometry. Test a declared simulated error model when hardware noise
is unmeasured and limit conclusions to it. Preserve all task physical limits;
RGB-D/FK cannot become a calibrated force sensor.

### 6.0E-F — Independent scoring and bounded pilot interaction

The future evaluator scores the explicitly chosen physical contact with independent
surface queries, full footprint/trajectory errors and both named profiles. Test
fixed-surface acceptance, cancelling hinge/contact errors, empty/provisional/missing
sets, early stops, both hands and non-teacher contacts. The retained numerical
metrics are not an operational evaluator or a release mechanism.

After pilot replay and feedback/stop gates, an explicit diagnostic source may use
scan, approach, contact, admitted short push, hold and release. Prespecify duration,
displacement, velocity and load bounds; preserve every failed attempt. Verify actual
leaf response, contact/load consistency, no hypothesis flip or helpful correction,
at least one same-process reset and one loss/latched-stop case per pilot. Opening
must exceed measurement uncertainty with the existing 0.5 s sustained controlled
contact window; there is no fixed success angle or teacher pre-opening.

### 6.0G-H — Common qualification and release

Freeze one recipe before extended replay in a fresh directory. Report every
train/development door, condition/phase, rejected/provisional errors, latency and
worst handedness cases. A recipe change invalidates its qualification; retain
failures rather than selectively assemble favorable attempts. Training requires a
separate learnable-error diagnosis and authorization.

After all offline gates pass, run both train pilots and all six development doors.
Baseline uses nominal/light; faults use nominal unless evidence requires light.
Prespecify these cases without a Cartesian product of unrelated faults:

- Full baseline sequence and two same-process resets.
- 0.5 s depth loss and 1 s RGB-D occlusion, each during scan and push.
- 50/100/250 ms result delay, out-of-order completion and nonmonotonic timestamps.
- Torque-feedback loss/staleness and inconsistent load with declared fault injection.
- Latched stop, cleared queues, bounded physical stop and no automatic restart.

Every baseline must satisfy physical validity and the diagnostic sequence. Every
fault must satisfy its expected reset/stop/queue invariants and physical limits.
Failed starts and scheduled unavailable ticks remain accounted for. Fault-stop
success is separate from baseline task success; thresholds never change after
seeing a result. Reuse prior complete evidence only for the identical recipe/runtime.

Release requires both offline and dynamic gates, an explicitly bounded supported
domain and deliberate release/profile compatibility. No diagnostic may forge
`offline_passed`, `dynamic_passed` or `frozen`. Hand off to 6.1 for all eight action
paths, raw/live encoding parity, physical replay and observed-geometry integration.
6.2 policy data and Phase 7 training remain separate milestones.
