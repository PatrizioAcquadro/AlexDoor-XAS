# Phase 6.0 — Operational Perception and Observed Contact

> Approved design: October 1, 2026. 6.0A contracts and numerical consumer integration
> and 6.0B static object fusion/queries with bounded pilot diagnostics are implemented;
> 6.0C-H remain pending. Static identity/hinge ambiguities remain explicit.
> This plan supersedes the September prototype's future requirements, not its
> measurements or qualification results. Design baseline: `main` at `809f575`;
> 6.0A implementation baseline: `main` at `655d26b`;
> 6.0B implementation baseline: `main` at `cf83f90`.

## Purpose and execution boundary

Deliver the geometric state needed for reliable door manipulation, preserving
A4's hinge-anchored, full-pose approach/contact/push/hold/release semantics.
Reconstruct total dimensions when supported; do not make an irrelevant hidden
extent a universal prerequisite for a local action. See the parent
[[phase-6-perception-actions-and-demonstrations|Phase 6 plan]],
[[../decisions/visuoproprioceptive-generalization-benchmark|benchmark decision]],
[[../topics/shared-door-perception|implemented perception boundary]] and
[[../topics/purdue-b1-robot-and-contact|robot/contact contract]].

The completed assignments implement 6.0A contracts/admission and 6.0B static
object memory, geometric queries and four scan-only diagnostics with a conditional
SAM3 video comparison. The remaining work packages below are
instructions for later assignments, not a request to start agents, simulations or
campaigns now. Keep the existing 6.1 action-path and 6.2 dataset numbering.
No new training, demonstration/data
collection, corpus changes, sealed-test access or push belongs to 6.0. Diagnostic
test traces are evidence, not an expansion of the learning corpus. Preserve every
completed/partial run and the stopped `geometric-evaluation-02` directory.

Use the two train pilots first: left `door-2738468b94d74c5f` and right
`animated-door-1-88abf40`, each nominal/light. Extended evaluation remains stopped
until common corrections and the pilot gates below have passed. Use local frozen
weights, CUDA for model inference and GPU-capable simulation, and one simulator
at a time. Do not change shared Isaac/PyTorch installations. Keep replay/live on
one causal provider and preserve calibration, FK, IK and Purdue IO.

## Baseline evidence and current limits

- At the 6.0A baseline the provider selected/fused plane support; it did not reliably
  assemble all faces of one leaf or exclude fixed frame support. 6.0B now preserves
  object candidates, observed seams, fixed/unresolved support and static hinge
  alternatives. Its four scans still retain ambiguous ownership and no supported
  physical axis. Historical corrected pilot-06 still
  accepts no complete states. See [[../experiments/b1-perception-findings|evidence]].
- `action/b1.py` uses hinge frame, signed angle and a full local target for A4;
  total dimensions do not enter its transformation. A3 rotates free delta vectors,
  so its frame origin does not affect that delta transformation.
- At baseline, `DoorEstimate`, `B1Observer` and adapters required complete geometry,
  including dimensions. 6.0A preserves that legacy profile and adds explicit
  operational/provisional interfaces; the current provider still emits legacy states.
- `recording/b1.py` and `PurdueIO.observe()` expose RGB-D and joint position/velocity,
  not torque. `ObservedControlChecks` rejects loaded phases with
  `force_feedback_unavailable`; `PurdueIO.stop()` clears commands but does not
  establish a physical unloading/stop response.
- The Alex guide documents joint `q`, `qd`, `tau`, PD/impedance gains and command
  limits. It does not establish the accuracy/origin of `tau` or an installed wrist
  force/torque sensor. Hardware feedback and control facts, source links and
  limitations are recorded in [[../topics/purdue-b1-robot-and-contact|the robot topic]].

These source/code checks settle the design direction. 6.0A numerical regressions
verify interfaces. 6.0B diagnoses static observed geometry, without qualifying
runtime identity, feedback acquisition or physical behavior.

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

Keep semantic inference at the initial 5 Hz target and geometric tracking between
results. Retain the 150 ms maximum age for dynamically supported estimates; old
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

Keep the existing full-state evaluator and all historical reports unchanged under
the **legacy full-state profile**: dimensions, hinge, signed angle and prescribed
contact, 1 cm / 5 degrees, 95% coverage and 95% joint accepted-state precision.
Report new results for this profile as diagnostics where applicable; a missing
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

## Implementation work packages

Every package starts by reading the current parent plan, this protocol, applicable
AGENTS.md and the previous package's handoff. Reuse maintained tests; add only the
essential regressions listed. Update canonical status and make small validated
local commits. No push. Code, tests and measured results determine completion.
6.0A is complete as an interface milestone and 6.0B as a static implementation
and bounded diagnostic milestone. Packages 6.0C-H retain their unperformed exit checks.

### 6.0A — Contracts and measurable admission rules

- [x] Define explicit geometric qualification, provisional action and load-admission
  interfaces; stable/dynamic field support, timestamps, failure reasons and resets.
- [x] Specify one action-margin/uncertainty recipe, contact selection/evaluation
  correspondence and torque-signal semantics. Unmeasured force/gain/stop parameters
  are measurement tasks in 6.0D, not invented constants or optional hidden defaults.
- [x] Map all consumers: `perception/contracts.py`, `policies/observations.py`,
  `policies/rollout.py`, `policies/common/b1_contract.py`, recording/replay and IO.
  Preserve old full-state/schema behavior with explicit profile compatibility.
- [x] Verify essential contracts: incomplete/provisional is not qualified; missing
  torque stays missing; no truth reaches runtime; no helpful adapter correction.

**Exit:** reviewed executable interfaces and numerical regressions; no campaign.

**Implemented handoff:** additive `DoorEstimate.operational`, per-field support,
explicit geometry profiles, action-specific conservative admission, identified
contact transitions and evaluator-only material correspondence interfaces. Observer,
adapter/runner, dataset compilation, recording/replay and observed IO/monitor share
the compatibility rules; v2 releases stay legacy and diagnostic recipes opt in.
The current provider emits no operational state and actual torque remains absent.
No force/gain/timeout/stop values were invented. The common numerical recipe and
producer measurement responsibilities are in
[[../topics/shared-door-perception|implemented perception contracts]]. 6.0B uses
these interfaces; 6.0D may proceed independently. 6.0E scoring and all physical gates remain
unperformed. Phase 6.0 is unqualified.

### 6.0B — Object-level scan fusion and static hinge hypotheses

**Depends on 6.0A.** Primary surfaces: `perception/geometry.py`, `provider.py`,
`tracking.py`, existing model/worker interfaces and geometric tests.

- [x] Assemble leaf surfaces across views, retain observed edges before sampling,
  and separate fixed frame/wall support without a universal bottom-frame rule.
- [x] Fit static hinge candidates from observed hardware/borders/depth; preserve
  alternatives and distinguish supported axis from an assumed panel edge.
- [x] Produce observed contact patches, required footprint clearance and relevant
  collision/unknown-space support without nominal dimensions.
- [x] Regress multiview clipping versus absent evidence, leaf relief versus fixed
  coplanar/parallel frame, both bottom-frame cases and calibrated reprojection.

**Exit:** targeted diagnostics on both pilots explaining ownership and static
observability; no broad replay. Keep GroundingDINO/SAM3/DINOv3 frozen. Native SAM3
video prompting is a conditional association experiment, not a mandatory rewrite.

**Implemented handoff:** `Surface` retains per-observation membership, calibration,
dense silhouette/extrema, edge status and material IDs. `GeometryProvider.scan_state`
exposes leaf candidates, attached relief, fixed/unresolved support, static
`HingeHypothesis` alternatives and observed patches. Reprojection plus measured
internal seams establish associations; indistinguishable ownership remains ambiguous.
Panel-edge alternatives never become physical axes without supported hardware fits.
Both actual distal faces and caller-supplied volume covers have geometric queries;
response, stopping and load parameters remain unavailable. Tracker transforms retain
original material/observation references; legacy recipes remain supported.

All four chronological 0–25 s train scans completed in `operational-scan-02`,
including pending image results released at measured availability without later RGB-D.
The initial scans remain ambiguous, with no uniquely assigned fixed support or
physical hinge axis.
The single conditional automatic-box SAM3 forward-video configuration completed on
the same captured frames in `operational-scan-video-01`; it retained zero masks.
At that stage no prompting/threshold search or provider replacement followed.
Interrupted/failed
attempts are preserved. Evidence, limits and local volume probes are summarized in
`operational-scan-review-01` and
[[../experiments/b1-perception-findings|the canonical findings]].

The targeted re-audit in `operational-scan-diagnosis-01` corrects measured-support loss,
partial-view border contraction, first-component-only seam association, unrelated
perimeter requirements and invalid finite-face claims. Enclosed mask omissions need
valid plane depth; residual fitting keeps existing support/tolerance requirements.
Box-only video tracks were removed by native hotstart; the shared `door` concept
retains masks on both 20-frame nominal prefixes without changing the model or filters.
These masks do not establish material role. See the canonical findings for the
bounded results and the invalidated historical contact counts.

6.0C owns causal material association, articulation and field lifetimes, preserving
alternatives and uncertainty. Motion is informative only after an admitted action;
it is not assumed to repair B's evidence loss or guarantee identity. The first push
requires an identifiable local leaf patch and a reference appropriate to the action.
The follow-up `operational-contact-readiness-01` verifies a local patch on each pilot,
both conditions, and derives finite geometric covers from the unchanged distal mesh
bands. Human visual confirmation establishes the selected local material role,
not motion identity. C may start local tracking/field lifetimes while other scene
roles and the axis remain unresolved. D still owns physical contact/model error,
feedback/load and response/stop margins. Unrelated total dimensions remain optional.
No static
edge guess may stand in for an unobserved axis. The image/model availability events
and retrospective video evidence are distinct; neither establishes production timing.
Ideal RGB-D/calibration, resolved pixel support and predominantly vertical-axis
approximations remain unqualified for hardware. No dynamics, training, collection,
extended replay or sealed test ran. Phase 6.0 and every release flag remain unqualified.

### 6.0C — Causal tracking, articulation and field lifetimes

**Depends on 6.0B.** Primary surfaces: `perception/tracking.py`, `geometry.py`,
`provider.py`, `visual_worker.py` and the existing timing/replay regressions.

- [ ] Track verified leaf features independently of fixed surfaces; retain original
  references, handle slip/occlusion and reject degenerate rigid/hinge fits.
- [ ] Initialize the reviewed local material anchors without demanding a complete
  object assignment or precise axis. Reacquire the white upper-inset patch before
  interaction: its last full-cover observation is about 13 s, not the fused plane's
  later observation near 25 s. Keep patch/identity and plane timestamps distinct.
- [ ] Recover signed angle/axis uncertainty for both hands; preserve stable geometry
  while dynamically supported pose remains fresh, including partial-field loss.
- [ ] Exercise reset, episode generations, out-of-order/late completion, nonmonotonic
  time and replay/live equivalence with identical supplied availability events.
- [ ] Retain the existing tracker unless isolated evidence identifies it as the
  remaining cause; only then compare a causal pretrained alternative on the pilots.

**Exit:** bounded pilot traces with valid identity/motion and honest uncertainty;
model/tracker changes require actual CUDA smoke, not a repeated model campaign.

### 6.0D — Robot feedback, common compliance and physical stop

**Depends on 6.0A; may develop independently of 6.0B/C with numerical fixtures.**
Primary surfaces: Purdue IO/environment, `perception/control.py`, robot model/FK
consumers and focused control tests. Do not edit the external Alex/Isaac packages.

- [ ] Map documented `tau` to a declared simulated observable; distinguish actual
  actuator feedback from commands/contact truth. Check any available existing SDK
  or logs for semantics; unavailable precision evidence remains an explicit limit.
- [ ] Implement robot-only load residuals and uncertainty. Check unloaded motion,
  known-load numerical fixtures, saturation, friction/model error, timing gaps and
  ill-conditioned configurations; no new hardware collection is authorized.
- [ ] Implement common compliant execution and a latched stop with bounded residual
  motion/load; validate behavior while physics continues. Missing or stale feedback
  must block loaded execution, even when geometric estimates are perfect.
- [ ] Verify finite two-finger contact assumptions and that a force/load inference
  cannot identify an authorized surface by itself. The nominal geometric covers are
  implemented; orientation/model/contact error and loaded validity still need checks.
  Keep task physical thresholds.

**Exit:** justified signal/timeout/gain/stop bounds and essential numerical plus
bounded CUDA physics evidence before door interaction. Ideal simulated feedback
does not qualify hardware; unknown signal semantics cannot be replaced by truth.
Unmeasured hardware noise does not require a new collection campaign: test an
explicitly declared simulated feedback/error model and limit conclusions to that
model. Record what still needs SDK/hardware confirmation before deployment.

### 6.0E — Dual-profile evaluator and two-pilot replay

**Depends on 6.0A-C.** Primary surfaces: `perception/evaluation.py`, evaluator-only
prepared/visual surface queries, `scripts/perception.py` and geometric tests.

- [ ] Implement operational-v1 and unchanged legacy reports, selected-point material
  correspondence, denominators and independent full contact/trajectory errors.
- [ ] Regress wrong fixed-surface acceptance, cancellation of hinge/contact errors,
  empty accepted sets, provisional/missing samples, early-stop accounting and truth
  isolation. Test both hands and a selected point different from the teacher.
- [ ] Replay both pilots/conditions chronologically with one recipe; preserve new
  evidence separately. Diagnose worst errors and runtime reasons before changing
  common code; never accept a correct mask/plane as the required door state.

**Exit:** pilot replay gate passed, or a cause-first failure report and no dynamics.

### 6.0F — Bounded diagnostic action on the two pilots

**Depends on 6.0D/E.** Primary surfaces: explicit diagnostic action source, common
Purdue IO/adapter/monitor path and a separate physical evaluator.

- [ ] Execute scan, observed-patch approach/contact, short admitted push, hold and
  explicit release. Include provisional exploration only when its admission checks
  pass; limit duration/displacement/load in the shared recipe before execution.
- [ ] Confirm actual leaf response, hinge refinement, contact/load consistency,
  safe stopping and no automatic hypothesis flip or helpful adapter correction.
- [ ] Verify at least one same-process reset and one loss/latched-stop case on each
  pilot before considering extended replay. Preserve all failed attempts.

**Exit:** both pilot controllers physically valid under the operational contract;
demonstrate actual opening beyond measurement uncertainty and the existing 0.5 s
sustained controlled-contact window, without a fixed success angle. No learned-policy,
full-corpus, release or hardware claim. If initialization cannot be admitted, report
that failure; never pre-open using the teacher.

### 6.0G — Common train/development offline qualification

**Depends on 6.0F.** Freeze the shared recipe, then evaluate all 50 existing episodes
in a fresh output directory. Require every train/development door to meet the
operational gate; include condition/phase breakdowns, rejected/provisional errors,
legacy reconstruction diagnostics, latency and worst cases by handedness. A recipe
change invalidates qualification for that recipe; rerun affected evidence explicitly,
not selectively retained favorable results. The sealed test remains closed.

**Exit:** all offline gates passed, or a complete failure report with dynamics
qualification not run. Training is considered only if residual errors identify a
specific learnable task, separate from observability, calibration and latency.

### 6.0H — Dynamic qualification, freeze and 6.1 handoff

**Depends on 6.0G and 6.0D.** On the two train pilots and all six development doors,
one simulator at a time, use the same admitted diagnostic recipe. Run baseline in
nominal/light; exercise each fault in nominal unless evidence requires a light
case. Freeze this allocation before the qualification run:

- [ ] Baseline full sequence; two resets within the same process.
- [ ] Depth loss for 0.5 s and RGB-D occlusion for 1 s, each during scan and push.
- [ ] Result delays of 50, 100 and 250 ms, out-of-order completion and nonmonotonic
  timestamps; account for actual availability and do not accept stale results.
- [ ] Stop before the next prohibited command, clear pending actions, verify physical
  stop behavior and no automatic restart after reacquisition. Include torque-feedback
  loss/staleness and an inconsistent-load case using a declared fault injection.
- [ ] Score unchanged physical criteria independently, include failed starts/stops,
  and distinguish expected injected-fault termination from baseline task success.

Every scheduled baseline must satisfy physical validity and the declared diagnostic
sequence; every fault case must satisfy its expected reset/stop/queue invariants
and physical limits. Any failing case prevents dynamic qualification; no best-attempt
selection or tolerance changes after seeing the result.

Run each required case without an unnecessary Cartesian product of unrelated
faults. Fault-test success means the specified bounded stop, not continued opening;
it does not remove fault intervals from any reported coverage denominator. Baseline
qualification and fault-response results remain separate. Reuse still-applicable
6.0F evidence only when recipe/runtime are identical and the case is complete.

**Exit:** both operational offline and dynamic gates passed; diagnostic/legacy
results retained; qualified profile and supported domain explicitly bound into
release compatibility. No release flag until actual evidence exists. Hand off to
6.1 for all eight action paths, raw/live encoding, physical replay and integration;
6.2 policy data/training remains a separate authorization and milestone.

## Delegation and verification discipline

Delegate one package with its prerequisites, owned files, expected output and exit
checks, not this entire plan as unconstrained parallel work. 6.0B/C own geometric
inference; 6.0D owns feedback/control; 6.0E owns truth-bearing scoring. Agree the
6.0A interfaces first and serialize changes to shared observation/runner contracts.
Only explicitly requested future delegation may create subagents. Serialize all
GPU-heavy jobs and simulator processes on the workstation.

Each handoff states: implemented behavior, exact bounded checks/evidence, failures,
remaining assumptions, local commits and the next eligible package. Update the local
ignored TODO for the assigned work; never mark a package complete from documentation
or a smoke test alone. Use relevant existing tests before adding new ones. Broaden
testing only for changed interfaces or unresolved failures; no training regression,
full replay or simulation is required for this documentation-only revision.
