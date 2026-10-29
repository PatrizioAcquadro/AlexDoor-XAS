# ACT A1 Four-Door Expansion

## Question and outcome

The October 19, 2026 staged study tested teacher/data coverage beyond the original
two doors and one initial posture. Two new nominal recordings, a qualified left
hold/release continuation and one separately qualified initial-posture variation
pass their declared scopes. These are simulation teacher results, not learned
completion, a continuous posture/placement range or held-out generalization.

The later [[act-a1-diverse-collection|112-cell campaign]] completes its original
scopes; its later continuity limitations are reviewed by
[[act-a1-complete-demonstration-audit]]. The current complete-cycle campaign and
human pause belong to [[act-a1-complete-cycle-collection]] and [[status]]. Historical
screening proposals and estimates do not authorize further acquisition.

## Geometry and first complete episodes

The four identities below are all in the frozen **train** inventory. Actual
composed USD colliders, contact surfaces, hinges and inertia were read, not just
filenames. At fraction .295 / height 1.09 m all nine queried points in a 40×40 mm
contact neighborhood have leaf support. This establishes a local surface, not
robot clearance, contact qualification or an entire reachable trajectory.
[Geometry, saved margins and cost audit](../../../outputs/evidence/act-a1-consumption-20261019/geometry.json).

| Door / frozen group | Relevant geometry and modeled dynamics | Learning coverage / proposed variation | Teacher/controller and acceptance | Original planning estimate (measured totals below) |
|---|---|---|---|---|
| Retain right `animated-door-1-88abf40` / `animated-doors` | Width/height/thickness .839/1.946/.029 m; contact X/Y −.0146/−.1806 m; hinge-to-contact radius .249 m; panel Izz 1.469 kg m² | Thin-leaf baseline; existing nominal plus qualified hold-start C. Use existing acquisition, push-entry and release windows now. Later P or O variation only for a demonstrated missing cell. | Existing shared setup; frozen required opening 41.008°; retain qualified release and ordinary tail. No nominal recollection. | Zero for existing records/views. Later full cycle approximately 9.1 GiB raw+prepared and 6.2–7.3 min whole launch. |
| Retain left `door-2738468b94d74c5f` / `djmaesen` | .945/2.038/.050 m; X/Y −.0243/+.1952 m; radius .288 m; Izz 1.866 | Wider opposite motion; left approach/limit and release coverage. A separately qualified left hold-start now adds release coverage without another nominal push. | Same controller, fraction/height; required opening 42.029°. Full left path reaches a joint boundary; no symmetric physical pose range is assumed safe. | Existing views zero. A C-length continuation would be about .55 GiB raw+prepared / .6 min; allow qualification/conditioning overhead separately. |
| Added right `psx-front-005-ee7d5c6` / `psx-residential` | .816/1.969/.092 m; collidable glazing inset plus outer leaf; X/Y +.0021/−.1703 m; radius .246 m; Izz 1.406 | Approximately 3× thickness and a 16.7 mm different push surface versus current right, with similar lever length. Tests geometry/appearance dependence without changing opening direction. Nominal complete; any later P/O condition requires a demonstrated coverage gap. | Asset-specific `PreparedDoor.contact_pose`/handedness, unchanged fraction/height and shared control. Fixed external benchmark 59.713° (80%=47.771°); current-setup reach/contact/stop applicability passed before collection. No target refit. | First complete attempt: approximately 9–10 GiB and 6–8 min; qualification overhead/failures additional. |
| Added left `psx-interior-wood-002-02e442c` / `psx-residential` | .757/1.809/.086 m; different frame/leaf fit; X/Y −.0423/+.1552 m; radius .239 m; Izz 1.210 | Approximately 20% narrower than current left, 17% shorter contact radius and 35% lower modeled Izz. Tests approach/release on a different lever/surface rather than recoloring the wide door. Nominal complete; any later P/O condition requires a demonstrated coverage gap. | Same controller; correct asset geometry/sign, no mirrored arm logic. Fixed external benchmark 62.683° (80%=50.146°); current-setup applicability passed before collection. | Approximately 9–10 GiB and 6–8 min per full attempt, plus qualification/failures. |

All four USDs use the same synthetic 25 kg panel, angular damping 4 and material
friction .5; their geometry-derived inertias differ. These are modeling
assumptions, not identified real-door parameters. Keep mass, damping, friction,
controller gains, force/speed/stop requirements and contact thresholds fixed.
There is no justified nonzero dynamics-randomization range in this task. The
right `animated-door-2` has effectively the same dimensions as door 1 and adds
less geometric coverage; it is not selected merely for a different filename.
The two added PSX doors share one frozen group and do not add two independent
families. The four-door proposal covers three train groups.

Use `configs/b1_shared_pilot_setup.json`: floor (−.437, .482327, 0), contact
fraction .295 / height 1.09 m, 6 s approach, 3 s acquisition, 1°/s push,
.3° lead, 3 mm compression, existing qualified hold, 6 s release, .4 s real
nominal pre-roll and 2–6 s ordinary stabilization. Geometry changes parameterize
the existing teacher; they do not justify changing the controller or its limits.
Full acceptance retains opening >=80% of the prospectively frozen reference,
required valid hold, actual withdrawal/release, ordinary stabilization, original
contact/footprint/material/speed/force checks and independent healthy stop.
The 80 N hard limit, .5 rad/s command limit with existing numerical allowance,
tracking bounds and independent operational-admission requirements remain.
Data acceptance separately requires integrity/terminal support, safe reviewed
commands, causal labels/masks, all required phases/transitions and genuine tail
support. Partial candidates have a separate explicit admission, not a relaxed
full-demonstration gate.


## First bounded execution: fixed requirement applicability

The September PSX references use floor Y=.38 rather than the current .482327 m
and a 3 s release rather than 6 s; they also predate current hold support and
ordinary stabilization. Their prepared qualification flags do not establish
current setup applicability. Keep their 80% requirements fixed at 47.770638°
right / 50.146446° left, without treating the old maxima as current achievable
maxima. Before scored collection, one dedicated current-setup cycle per door
must pass that requirement and the unchanged physical/release/tail/independent
stop gates. This establishes applicability of an externally fixed benchmark; it
does not publish a new maximum reference or replace the corpus paired-reference
procedure. A failure blocks scored collection on that door.

`pilot_physics.py qualify` performs this non-supervising check with the shared
setup and inspection/pre-roll. `collect` and `qualify` first check prescribed
footprint support, obstructions and pedestal intersection, then retain all
existing every-row reach/contact/speed/force checks. Detached independent stop
verification remains separate. The left existing-source continuation qualifies
reconstruction and sustained loaded pre-roll within its single physical launch,
then records release and ordinary stabilization; it is not a closed-door opening.

The prospective protocol and resource reconciliation are in
`outputs/evidence/act-a1-four-door-20261019/declaration.md` and `protocol.json`.
Five launches were prospectively planned, with one diagnosed-correction reserve; maximum six
launches, two GPU workload hours and 100 GiB additional evidence, within project
limits. Per-case 1,200 wall s / 210 sim s / 12 GiB RSS / 20 GiB total GPU ceilings
remain. No later posture/placement tranche is launched here.

## Measured first expansion and usable handoff

The separate fixed-requirement checks pass at 51.281° right / 54.371° left,
including valid hold, ordinary release/stabilization and detached independent
stop. Fixed collection requirements stay 47.770638° / 50.146446°; no requirement
or corpus maximum reference is refitted. Scored collection is separate.

| New source | Physical result | Usable 12-step windows |
|---|---|---:|
| New right PSX nominal | 51.281° complete opening; hold/release/2 s ordinary tail/stop pass | 7,891 |
| New left PSX nominal | 54.371° complete opening; hold/release/2 s ordinary tail/stop pass | 9,517 |
| Existing-left hold/release continuation | Reconstruction, fresh guarded loaded hold, release/2 s ordinary tail/stop pass; not closed-door task success | 518 |

All three recordings pass native integrity, real terminal support, scoped phase
coverage and reviewed command/window admission. Raw RGB-D, requested/delivered
commands, native measured simulated torque, clocks, masks and prior ambiguous
samples remain. Left H uses the exact qualified original-left source, six real
conditioning ticks, fresh generation/local t=0 reference and genuine one-second
pre-roll. Its successful continuation never becomes full closed-door opening.
Eighteen H commands remain diagnostic; four otherwise reviewed short-prefix
commands cannot form a valid 12-action chunk across the internal invalid interval.
All 518 usable starts retain real history and terminal padding.
New-left requests are limited/clipped on 64 hold and 216 release rows (maximum
.036347 rad request/target difference); labels use actual delivery minus measured
pre-state. None of the three sources has a refused command.

The combined six-record train handoff contains 33,987 complete eligible
windows, alongside 126 overlapping phase-focused source views. It represents
four independent nominal motions across three train families and two hold
reconstruction derivatives. One initialized CUDA checkpoint/shared train-only
normalization consumes nominal and intermediate records; masks, causal joins,
actual labels and genuine tail padding pass with unchanged weights and zero
optimizer updates. Focused event anchors do not replace background/crossing
windows or count as independent demonstrations.

Five physical launches, no failed/refused/interrupted attempt or retry, consume
1837.91 s including startup/exit; CUDA consumption adds
54.44 s. Additional retained evidence is 21.79 GiB,
within the six-launch/two-hour/100-GiB allocation and remaining project budgets.
The 10.056 N acquisition peak is retained below the unchanged 80 N hard limit;
no empirical load envelope is expanded. Zero AABB clearance lower bound is not
positive placement slack. Measured healthy post-cycle stops do not establish
global loaded emergency-stop coverage or learned operational admission.

[Complete outcomes, costs and recommendation](../../../outputs/evidence/act-a1-four-door-20261019/report.md),
[explicit source handoff](../../../outputs/evidence/act-a1-four-door-20261019/handoff.json)
and [scoped quality/CUDA audit](../../../outputs/evidence/act-a1-four-door-20261019/verification.json).
The prior three-source handoff and every historical failed/interrupted outcome
remain intact. No ACT learned task result is added; the eventual criterion is
physical completion from its own observations, not teacher trajectory similarity.


## Single initial-posture pilot — completed

The single candidate on existing left train `door-2738468b94d74c5f` passes as a
teacher demonstration: initial right-arm `RIGHT_SHOULDER_Y` −.05 rad, seed 6200,
all other actual setup fields/light/robot/controller unchanged. Sustained opening
52.615259° exceeds the fixed 42.0288232010161° requirement; valid hold, actual
release, 2 s ordinary stabilization and separate healthy 2 s stop pass. There are no
physical violations, refusals, interruptions or retries. This qualifies only the
measured single teacher condition, not learned ACT success or a posture interval.

`pilot_physics.py collect --screen-initial-state` audits parked runtime hard/soft
limits, actual tool pose, imported clearance, all-body contacts and the existing
visibility predicate before arm motion. This candidate has .215851 rad initial
joint margin, 2.368 mm AABB lower bound, no contacts and visibility 25/8/28. A detached
unloaded 2 s stop passes; the maintained producer then resets the complete configured
state and ordinary impedance before recording inspection onward. The screen is
for closed-door collection, not loaded reconstruction. Whole-path AABB lower bound
remains zero, and the initial/final stops do not prove all loaded-state stops.
Every-row checks and numerical/empirical tolerance handling remain unchanged.

### Measured coverage

The perturbation survives inspection: at first policy-ready t=25.2 s the measured
joint difference is .05 rad, tool displacement 21.772 mm/2.865° and maximum native
solver-feedback difference .709 N m. Mid-approach tool displacement is 4.403 mm;
by approach end it is .00435 mm, with maximum joint residual .000965 rad. Both
sources first qualify measured contact at 33.333333 s (associated command row
33.316667 s) and have 65/180 loaded/contact-valid acquisition samples. Across the
3 s acquisition, maximum joint difference is .001001 rad and tool median/max
distance .00485/.03228 mm. This adds observed early approach/posture variation,
not a distinct acquisition geometry, contact location or corrective behavior.

Later motion retains measured differences: push is 4.45 s shorter; equal elapsed
push times differ by up to .108503 rad/7.969 mm because door progress differs.
Matching measured pre-command door angles over .693–51.933° (6,784 pairs,
maximum angle mismatch .008736°) gives tool median/max .02560/.17837 mm, joint
RMS/max .005436/.017662 rad, native tau RMS difference .029292 N m and delivered
target maximum difference .017813 rad. Thus contact/tool geometry converges,
while residual joint/feedback and timing variation remains. Event-aligned
hold/release tool differences stay below .395/.338 mm. These are descriptions,
not new similarity acceptance thresholds or causal robustness evidence.

Selected original RGB-D pairs confirm early gripper/arm variation and converged
contact context. Full-image depth mean absolute difference falls from 14.394 mm at
readiness to .279 mm at first valid contact; median is zero at both, since most
context is shared. Contact remains partly self-occluded. Pixel differences do not
certify contact precision. Saved measured traces, clocks, requested/delivered
actions and the full comparison are in
[the pilot report](../../../outputs/evidence/act-a1-posture-20261019/report.md).

### Usability, costs and next decision

The complete recording has 9,242 states/9,241 commands, 1,541 native RGB-D captures
and 18,483 healthy native feedback samples. Reviewed commands/usable starts total
7,729: inspect 12, approach 360, contact 180, push 6517, hold 180, release 360,
stabilize 120. All six phase transitions retain 11 crossing starts; 25 focused views
retain real preceding history without copying training images. At first readiness
all 23 image / 241 robot-history slots are supported by real inspection, back to 21.2 s.
The final 11 padded targets describe only the true terminal tail.

Five limited approach requests (maximum .000149727 rad) remain recorded, versus
nominal 36/.001537085 rad. Every label equals delivered target minus its measured
pre-state, including all limited rows. The existing initialized shared checkpoint
with its frozen six-source train-only normalization consumes 32 new windows and
one two-row backward; finite output/gradients, exact unchanged weights, zero
optimizer updates. No new model behavior benchmark or learned evaluation runs.
The seven-record handoff now has 41,716 usable starts and 151 overlapping focused
views on four door identities/three families; five complete reset recordings
include this same-door condition, alongside two hold derivatives.

One of at most two launches used; correction reserve unused. Physical whole wall
434.659 s + CUDA preparation/consumption 15.150 s =449.809 s (7.497 min), conservative
charge 459.809 s including allocation tails. Accounted simulation 158.017 s includes
both detached stops. New evidence about 9.49/30 GiB, peak process-tree RSS 7.760 GiB
and total GPU 6.539 GiB. Project counters 22/24 attempts, conservative 18.661711/40 h,
about 560.50/1024 GiB logical evidence. All original ceilings hold; failures and
prior payloads remain preserved. Exact costs/paths are in
[resource summary](../../../outputs/evidence/act-a1-posture-20261019/resource-summary.json)
and [handoff](../../../outputs/evidence/act-a1-posture-20261019/handoff.json).


## Initial bounded physical/data tranche and handoff

[Declaration](../../../outputs/evidence/act-a1-teacher-temporal-20261019/declaration.md),
[all attempts and commands](../../../outputs/evidence/act-a1-teacher-temporal-20261019/report.md),
[actual-data verification](../../../outputs/evidence/act-a1-teacher-temporal-20261019/verification-final/report.json)
and [ready inputs](../../../outputs/evidence/act-a1-teacher-temporal-20261019/handoff.json).
The fixed train identities are right `animated-door-1-88abf40` and left
`door-2738468b94d74c5f`, nominal seed 6200. Required opening retains 80% of the
frozen matched references (51.260252/52.536029 deg), never a refit on these results.

| Retained attempt | Physical outcome | States/commands | Healthy native tau | Data outcome |
|---|---|---:|---:|---|
| First right | 51.11 deg, release + 2 s ordinary tail; stop passes | 8,850/8,849 | 17,699 | Incomplete inspect→approach coverage; 7,337 partial candidates, separate admission required |
| Corrected right | 51.75 deg, release + 2 s ordinary tail; stop passes | 9,101/9,100 | 18,201 | Qualified; 7,537 training windows |
| Left | 52.55 deg, release + 2 s ordinary tail; stop passes | 9,509/9,508 | 19,017 | Qualified; 7,996 training windows |
| Physical reset C | Fresh loaded hold, release + 2 s ordinary tail; stop passes | 541/540 | 1,081 | Qualified continuation; 528 training windows |
| Controlled interruption D | Delivered command 60, interrupted before post-state; stop passes | 60/60 | 119 | One missing terminal state; recovered, zero expert windows |

The first right exposed a real readiness seam: approach began at 25 s while
inputs first became ready at 25.2 s. One affected repeat followed the explicit
.4 s ordinary pre-roll correction. The original outcome/report remain; its early
demonstration flag predates the completed window gate and does not qualify it
for handoff. No unchanged favorable retry occurred. All present transitions in
the three admitted files retain 11 crossing windows. These overlapping windows
and one reset derivative are not independent demonstrations or a success-rate
estimate. Source/family train grouping remains frozen; C stays with its source.

All 56,117 native torque vectors across five attempts are finite/healthy under
the simulator API contract; none is missing, and none of the seven channels
reaches >=99% of configured effort limits. No saturation flag is exposed: this
comparison is diagnostic, not sensor saturation certification.
[Native event-centered traces](../../../outputs/evidence/act-a1-teacher-temporal-20261019/torque-event-traces.json)
and figures retain all seven channels around contact, hold and real release.
Actual prepared batches pass finite CUDA forward/backward with image, temporal,
measured-state and torque gradients, exact unchanged weights and zero optimizer
updates. Live/offline representative snapshots include true terminal/tail states;
A1 labels and causal previous-target support agree exactly. Valid N m changes
reach ACT tensors; invalid/missing tau is masked. D reopens through ACT preparation
with the real prefix and zero eligible windows. Focused numerical tests pass.

Five whole launches consume 1,229.03 s (20.48 min); final CUDA verification takes
17.75 s, plus a 5.51 s exact CUDA-tensor reach check. Peak simulation process-tree RSS is 7.73 GiB and total GPU memory is
6.87 GiB including desktop graphics. Final validation uses 1.96 GiB process RSS
and peaks at 1.65/3.67 GiB CUDA allocated/reserved. About 28.8 GiB of tranche data
is retained, within the declared 80 GiB; all per-case/cumulative budgets hold.
No remote allocation, optimizer update, final-test access or learned evaluation
is added. This closes the collection tranche, not ACT physical task success.



## Evidence and limits

The original declarations, measured geometry, source ancestry, failures and costs
remain under `outputs/evidence/act-a1-four-door-20261019/`,
`act-a1-posture-20261019/` and `act-a1-teacher-temporal-20261019/`.
Source-row views duplicate no training images; overlapping windows and reset
continuations do not add independent demonstrations. Initial posture differences
largely converge before contact, so their early visual diversity does not prove
new acquisition geometry. All prospective screening ranges and ordinary narrative
remain in Git at `4d21837`; none is a currently qualified operating envelope.


## Historical loaded-reset measurement

The original right reset changes generation 1 to generation 2. Reconstruction
errors are 0.566 mrad q, 38.655 mrad/s dq, 0.643 mrad door angle, 0.259 mm tool
position and 0.482 mrad tool rotation, within its declared state bounds and hard
limits. Fresh pre-roll has 58/60 valid contact samples. Its first snapshot has
2/241 q/dq/tau slots, one previous target and 2/23 images; first eligible decisions
occur at 0.2 s. This demonstrates that measured reset's scope, not a generic
loaded-state producer or hardware reset.


### Source decisions and remaining limits

[StageACT](https://arxiv.org/html/2509.13200v2) supports proprioceptive/torque review
of contact transitions and identifies observational phase ambiguity; its
privileged stage input/architecture is not adopted.
[DoorBot](https://arxiv.org/html/2504.09358v1) supports closed-loop haptic
verification and explicit failures; its sensors/planner are not imported.
[MimicGen](https://mimicgen.github.io/docs/tutorials/datagen_custom.html) supports
task-specific subtask end conditions and diagnostic annotations; its framework
and randomized offsets are not adopted. These sources do not qualify this reset
or Alex hardware feedback. The
[existing hardware audit](../../../outputs/evidence/act-a1-observation-design-20261019/analysis.md)
continues to leave source/scale/sign/noise/filtering, clocks/latency and installed
F-T/tactile availability unverified. Historical
[[../experiments/b1-data-pipeline|data infrastructure]],
[[../experiments/b1-act-a1-two-door-diagnostic|learned failures]] and
[[../experiments/b1-act-a1-calibration-stopping|calibration/stopping]] remain unchanged.
