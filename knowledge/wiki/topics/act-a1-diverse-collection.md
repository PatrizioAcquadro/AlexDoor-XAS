# Diverse bounded ACT A1 collection campaign

**Status: scoped resumption, goal unmet.** The original six physical attempts produced three individually
usable episodes and three failures. Only 81/112 target cells pass; no policy training
or optimizer update occurred. The resumed runtime decision below permits bounded
collection in qualified scopes; unsupported variations retain independent gates.

## Prospective declaration

Baseline `a3c7550`, branch `PatrizioAcquadro/b1-policy-pilot`, clean tracked tree.
The objective is new, varied temporal demonstrations, with no policy training,
optimizer update, learned evaluation, RL, architecture comparison or final-test access.
Prior recordings remain controls; clips and windows never count as new episodes.
The campaign supersedes the obsolete 24-attempt cap prospectively only.
[[act-a1-observations-and-time]] owns observation/action semantics;
[[act-a1-four-door-expansion]] retains the original results and opening references.

| Split / door | A acquisition | B initial opening | C push to hold | D release to settle | Complete |
|---|---:|---:|---:|---:|---:|
| Train animated-door-1-88abf40 | 6 | 6 | 6 | 6 | 2 |
| Train door-2738468b94d74c5f | 6 | 6 | 6 | 6 | 2 |
| Train psx-front-005-ee7d5c6 | 6 | 6 | 6 | 6 | 2 |
| Train psx-interior-wood-002-02e442c | 6 | 6 | 6 | 6 | 2 |
| Development modern-door-2fb8d024 | 1 targeted A/B | — | — | 1 | 2 |
| Development psx-industrial-001-4f5561b | 1 targeted A/B | — | — | 1 | 2 |

Each train family has two pose, two dynamics and two correction cells, exactly one
primary category per new physical execution. Total: 96 short train, eight complete
train and eight development episodes. No repeated deterministic replay fills a cell.
The frozen source groups and splits stay unchanged. Development cannot fit weights
or normalization. Source continuations retain ancestry and are not independent.

## Conditions and qualification order

These are candidate simulated conditions requiring physical qualification, not
validated hardware ranges. Use task-space targets and existing feasible IK/control.
First check individual factors, then the declared combinations, without a Cartesian sweep.

- Pose high/low for complete and late C/D episodes: contact height 1.11/1.07 m
  (nominal 1.09 m), with unchanged
  material-relative normal/orientation. Screen reach, finite footprint and clearance.
  Complete train episodes use these two conditions per door. Initial unloaded
  approach additionally investigates 20–50 mm normal distance and 20 mm lateral
  error / 5 degree yaw; correction may converge to the same final contact.
- Dynamics light: uniform panel density scale .7 (17.5 kg), inertia scaled .7,
  unchanged center of mass, geometry and damping 4. Dynamics resistant: density
  scale 1.3 (32.5 kg), inertia scaled 1.3 and damping 6 after separate factor checks.
  These bracket the synthetic 25 kg, damping 4 model; they are assumptions, not
  identified real-door properties. Read actual composed and active PhysX values.
  Compare measured angle/speed/torque/contact response to saved nominal control.
  Contact friction remains .5 unless an interaction requires a separately declared check.
- Correction cases A-1/A-2: physically executed lateral/yaw misalignment, or a stopped-short
  normal approach, followed by measured realignment/acquisition.
- Correction cases B-1/B-2: insufficient normal contact, or a bounded weakening pulse during
  initial opening, followed by force-supported approach/recontact when recoverable.
- Correction cases C-1/C-2: a weakening normal-contact pulse, or residual motion induced by
  a brief commanded push-rate increase, followed by maintained support and settling.
- Correction cases D-1/D-2: incomplete initial withdrawal (8 mm), or a brief residual-contact
  pause, followed by ordinary complete unloading/separation and stabilization.

Disturbances use executed commands/forces, never mid-episode teleportation or
edited/spliced states. Exact pulse duration/amplitude is frozen in the per-case
protocol before each factor check; unsupported values block their dependent cells.
Injected erroneous actions and unsafe/ambiguous rows remain causal history but
are excluded from action targets; whole chunks crossing them are excluded. Corrective
commands require physical and phase validity. Hard safety stops remain latched.
Historical lost-contact failures on left train and industrial development prioritize
contact weakening/reacquisition; they never become successful complete attempts.

## Independent endpoints and history

A requires at least .5 s contiguous actual valid loaded contact, finite footprint,
material/orientation support and no violation. B additionally requires at least
1 degree actual controlled opening beyond acquisition with sustained valid contact.
C requires real late push, maintained support and at least .5 s qualified hold after
controlled settling. D requires actual unloaded separation >=10 mm and the unchanged
ordinary stabilization tail. Every short includes safe ordinary withdrawal/tail and
its independent continuing-physics stop; the stop cannot provide the expert tail.
Full episodes retain unchanged frozen opening/hold/release/stabilization criteria.
These endpoints are measured from physical records, independently of phase flags.

Phase starts must restore robot/door q/dq, targets, gains and controller state and
rebuild contact with fresh physics under the actual placement/dynamics. An existing
hold state cannot stand in for early contact or late push. Obtain real preceding
movement before critical events; unavailable history stays masked, with no borrowed
source prefix or repeated frames presented as movement. Keep source ancestry.

## Resources, admission and stopping

Verified baseline: 18.661711 conservative local GPU hours, 560.493949 GiB logical
retained evidence, 22 historical physical attempts, approximately 1.1 TiB free disk.
The original160-attempt campaign ceiling was explicitly raised to220 total
campaign attempts on October20, including every already charged attempt.
No counter or baseline resets. Current ceilings:220 campaign attempts,12 additional GPU workload wall-hours,
400 GiB additional retained evidence; cumulative 40 hours and 1 TiB remain binding.
Count each episode separately even in a reused simulator. Include all qualification,
failed/interrupted attempts, startup/shutdown, preparation, stop continuations and
CUDA validation. Reserve 60 s GPU and at least 12 GiB free headroom for interruptions.
Only one GPU simulation runs at a time; preserve unrelated workloads.

Initial guards: complete/source qualification 1200 s wall / 240 s simulated;
early short 600 s / 90 s; late short 600 s / 60 s, including conditioning/tail/stop.
Guards are resource terminations, never shortened success sequences. Use the
existing native RGB-D/control/feedback rates and bounded one-frame stores/batches.
Expected main payload is approximately 200–300 GiB with short states; measure the
qualification tranche before committing remaining launches. No costly favorable retry:
a replacement needs a diagnosed mechanism and recorded change, preserving denominators.
Unsupported required conditions, structural defects or exhausted budgets stop affected
launches and leave the goal unmet. Every category must show observed variation.

All 112 counted recordings must yield admissible temporal samples. Check native
alignment/clocks/feedback provenance, delivered-minus-prestate A1 labels, intervention
masks, transitions, limited actions and true tails. Check offline/online agreement
across all families and directions. Fit normalization on admitted train sources only.
Bounded CUDA forward/backward checks must be finite with unchanged weights and zero
optimizer updates. The final handoff must enumerate exact train/development sources,
ancestry, masks, outcomes and all resource costs. None of these gates is passed yet.

Local payload and ledger: `outputs/evidence/act-a1-diverse-20261019/` (ignored;
Git cannot recover it). This page owns campaign state; original evidence is preserved.


## Implemented interfaces and numerical policy

The parked screen applies one outward representable float32 bound step to both
hard and soft reset limits. Original signed margins remain in the audit. Focused
checks cover exact bounds, one-step acceptance, two-step/real excursion rejection,
NaN and both hard/soft limits. This is quantization allowance; physical limits,
planning margins, empirical calibration support and stop criteria are unchanged.

The existing producer accepts declared A/B/C/D scopes. Early shorts acquire a
fresh t=0 camera reference and real approach history. No source prefix is borrowed.
A/B post-endpoint hold/withdrawal/tails are diagnostic cleanup, excluded from expert
targets so short collection does not teach early release for the full opening goal. B requires
its measured sustained initial-opening endpoint, not the full probe's final hold
window. The full 3 s cleanup hold still executes with unchanged physical/contact-loss
guards; full/C/D end-hold criteria are unchanged. One retained low-front B attempt
passed its endpoint and every physical/release/tail/stop gate but was refused by
the extra cleanup predicate: 178/180 hold rows valid, a 33 ms gap near the end.
Its original failed report and raw outcome remain immutable. An explicit
metadata-only HDF5 view links the same native streams and episode identity for
independent B-scope review, with zero new physical execution. Temporal/CUDA and
pose-diversity review must pass before that view fills its single original cell;
`B-cleanup-scope-review.json` owns the decision. No in-episode safety stop is cleared.
Native roles distinguish expert (0), injected excluded (1) and corrective (2).
Delivered injected targets stay in causal history; chunks crossing them are excluded.
Phase/role/truth/outcome fields never enter policy observations.

C/D candidates use actual preceding-push states, measured exit guards and the
original 3 s hold. Extending an already completed hold to obtain history is rejected.
New loaded starts require restored q/dq, targets, impedance and controller state,
fresh contact, an independent stop, reset/conditioning and actual preceding movement.
Older native ledgers may supply reset states, with explicit command-row ancestry;
these views are not new episodes. Source placement/contact/dynamics/controller fields
must match; only prospective phase durations may differ for the short scope.
Experimental source-controller conditioning is opt-in; existing non-campaign hold
continuations retain their legacy conditioning and exact setup checks.

Short B/C has an explicit, at-most-one-second bounded recontact path only while
physical/material/force checks remain valid. It never clears a stop latch and does
not change complete-task loss criteria. Broader recovery paths and new variation starts require their own
physical qualification. Uniform density authoring scales panel/handle mass and
inertia together, preserves geometry/COM/axes, and saves active PhysX values. The
first-door light density/damping 4 early A/B scope is now qualified; full/late
dynamics and other doors retain their own gates.

Before the main matrix, unlaunched early A/B pose-low candidates were specified as
20 mm robot-base retreat along world X, 50 mm precontact distance and nominal contact
height. Late/complete pose-low retains 1.07 m height. All eight early A/B pose-low cells now pass their measured 20 mm base/precontact
variation and independent physical/data gates.
Static-reference times are per episode; model history/resolution/chunk/execution stay
4 s / 240×384 / 12 / 6, visual reuse off. Raw stream clocks are actual local episode
acquisition/availability. Probe trace/endpoint support times use a separate zero-based
completed-control-row clock; join by the aligned annotation rows, not identical values.

## Initial six physical outcomes and fixed denominator

| Attempt | Outcome and independently measured limitation |
|---|---|
| train-1-complete-pose-high | Pass: 52.885° sustained opening versus unchanged 41.008° requirement, valid hold, release, ordinary tail and stop; 7,980 usable starts. |
| train-1-A-pose-high | Pass: >=.5 s physical loaded/material/footprint support, actual withdrawal/tail/stop; 517 usable starts. |
| train-1-D-pose-high | Qualification refusal before recording: static joint-anchor conditioning loses moving contact; q-velocity reconstruction error .05712 versus .05 rad/s. Final stop passes. This empirical mismatch alone is not a hard physical violation. |
| train-1-D-pose-high-conditioned | Source-controller conditioning restores contact, but extra hold consumes the original release reserve. At 2.1 s, joint speed .70034 exceeds the unchanged criterion; subsequent stop peaks at 4.47000 rad/s. Failed 127-command recording, zero expert windows. |
| train-1-A-correction-1 | Pass: actual bounded precontact misalignment followed by measured residual realignment/acquisition, withdrawal/tail/stop; 507 usable starts. Scope below. |
| train-1-D-pose-high-from-push | Earlier genuine push state and existing validated stop recipe still fail start-stop settling: final speed .32121 versus .01 rad/s; continuing check reaches 4.46986. No recording starts. |

All three D attempts address the same required cell and retain their diagnosed
changes; none is an unchanged favorable retry. Six attempts mean three passed
individual scopes and three failures, including two qualification refusals before
recording. No interruption, budget termination or recorded command refusal occurred.
Missing recordings are unavailable, never reconstructed from a source prefix.
Conditioning and continuing-stop traces remain preserved for those attempts.

The last failed stop has constant latched joint targets, zero native normal contact
force, alternating velocity every 1/120 s and approximately 1.019 amplitude growth
per native step near 2 s. This establishes unresolved stop/runtime instability at
the visited posture; it does not identify its internal numerical cause. Profiles
and states differ between attempts, so this is not a paired controller comparison.
The legacy and separately qualified stop profiles have their original scope; neither
passes every new visited condition. No new gain or physical threshold is adopted.
[Measured stop figure](../../../outputs/evidence/act-a1-diverse-20261019/stop-instability.png).

The initial physical launcher was closed by `campaign-stop.json`, verified without
starting a simulator. The resumed qualification decision below now permits scoped
collection; all original stop outcomes remain unchanged.

## Initial closeout coverage and data usability

| Train door | A qualified/planned | B | C | D | Complete |
|---|---:|---:|---:|---:|---:|
| animated-door-1-88abf40 | 2/6 | 0/6 | 0/6 | 0/6 | 1/2 |
| door-2738468b94d74c5f | 0/6 | 0/6 | 0/6 | 0/6 | 0/2 |
| psx-front-005-ee7d5c6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/2 |
| psx-interior-wood-002-02e442c | 0/6 | 0/6 | 0/6 | 0/6 | 0/2 |

Each frozen development identity (`modern-door-2fb8d024`,
`psx-industrial-001-4f5561b`) remains 0/2 complete, 0/1 acquisition/opening and
0/1 release/stabilization: 0/8 development total. Short train categories are
pose 1/32, dynamics 0/32 and correction 1/32. Complete train is 1/8. There are
109 missing target cells, including the failed D cell; 108 cells were never launched.

Higher-contact complete/short records retain approximately 19.99 mm contact-Z
change versus nominal, with 65 physically valid acquisition rows each. The `A-correction-1` pulse
produces measured 19.86 mm lateral /4.99° deviation. Its excluded taper removes part
of the error: eligible teacher correction begins at 5.22 mm /1.75°, reaching about
.18 mm /.052° after .5 s and then valid contact. This qualifies bounded residual
realignment; it is not 20 mm exclusively expert recovery or general lost-contact
coverage. New source groups/door identities are not created. Native selected images
show changed gripper/arm context with partial contact occlusion; pixels do not prove
millimetric contact precision. Existing posture-convergence evidence remains unchanged.

The three new admitted recordings provide **9,004 usable windows**. Combined with
the unchanged seven reviewed controls, the partial inventory has ten train records
and 50,720 starts: six complete motions on four doors, two historical hold derivatives
and two new early shorts. These correlated views/conditions are not independent
families or a success-rate estimate. All failed-source targets remain excluded.

Native integrity, delivered-minus-prestate A1 labels, clipping evidence, supervision
masks, causal joins, right-side offline/online phase snapshots, recovery-history
views and true-tail support pass. The existing initialized checkpoint consumes
representative actual data with finite CUDA forward/backward, exact unchanged
weights and zero optimizer updates. Normalization remains the original six-train-
source fit, used only for compatibility; no new normalization or development fit
occurs. Original raw/prepared controls and checkpoint files remain unchanged.
Both-direction/new short C/D agreement and the 112-source gate are **not validated**.
Simulated torque retains its solver joint-output provenance, not direct contact force.

## Initial closeout costs and handoff

Six whole physical launches: 580.418 s; complete CUDA preparation/consumption:
19.556 s. Total GPU workload **599.974 s (10.00 min)**, conservative charge
634.974 s including allocation tails. Additional retained evidence approximately
12.34 GiB; reviewed-scope project totals approximately **18.838/40 GPU hours** and
**572.84/1024 GiB**, with approximately 1080 GiB actual free disk. All campaign
limits hold: 6/160 attempts, .1764/12 charged GPU hours, 12.34/400 GiB.
Startup, shutdown, failures, qualification and preparation are included in whole
wall costs. Accounted episode/conditioning/stop simulation is 218.933 s; runtime
initialization ticks were not separately instrumented. Sampled physical peaks are
7.730 GiB process-tree RSS and 6.487 GiB total GPU. CUDA validation uses 2.076 GiB
RSS and .498/.688 GiB allocated/reserved. CPU-only audits use no GPU allocation.

- [All attempts and reasons](../../../outputs/evidence/act-a1-diverse-20261019/attempts.json).
- [Every planned cell and gap](../../../outputs/evidence/act-a1-diverse-20261019/coverage.json).
- [Exact partial train inventory, masks and ancestry](../../../outputs/evidence/act-a1-diverse-20261019/handoff.json).
- [Actual-data and CUDA validation](../../../outputs/evidence/act-a1-diverse-20261019/verification.json).
- [Measured variation and response](../../../outputs/evidence/act-a1-diverse-20261019/measured-coverage.json).
- [Resource ledger](../../../outputs/evidence/act-a1-diverse-20261019/resource-summary.json).
- [Local commits](../../../outputs/evidence/act-a1-diverse-20261019/local-commits.json).

The completion gates do not all pass: A has 3/112 coverage; B has unresolved loaded
start/stop validity; C validates only the admitted subset. D delivers an honest
partial inventory, costs, documentation and local commits, while the requested
training-ready handoff remains unavailable. The overall goal is unmet. The frozen final test
remains closed. No policy training, optimizer update, learned evaluation, RL,
architecture comparison, remote allocation or push occurs.

Next: diagnose and independently qualify stop stability at the actual postures,
then qualify reusable C/D starts and remaining correction/dynamics cases. Only after
that may the declared missing cells and both-direction/development data gates resume.
The resumed investigation below continues from this preserved closeout. Once the full corpus is qualified,
a bounded first ACT experiment should test whether causal history improves contact
acquisition, phase handoff and unloading, and whether genuine observed variation
improves frozen-development completion without increased refusals/unsafe acceptance.
This partial corpus cannot test the requested broad recovery/generalization claim.

## Resumed qualification and active collection (20 October)

The original objective and original cumulative resource baseline continue. Current
admission after pose/acquisition correction and initial dynamics/recontact qualification
is **51/112**: 43/96 short train, 8/8 complete train and 0/8 development;
**61 cells remain**. The exact live
inventory is `handoff.json` / `coverage.json`; every physical attempt, including
qualification and failures, remains in `attempts.json` and the resource ledger.
The corpus is not ready for the requested training experiment. Training, optimizer
updates, learned evaluation, RL, architecture comparison and final-test access
remain closed.

### Restoration and numerical stopping cause

The checked and recording starts now both use `continue_controller=True` for the
campaign. This corrects a later orchestration mismatch, not the earlier instability.
The preparation regression exercises both actual call paths with the intervening
stop/reset. The first matched C/D starts agree exactly in full robot q/dq, door
q/dq, delivered targets and reconstructed contact. Across eight admitted right/left
loaded starts, subsequent feedback can differ slightly despite identical inputs:
maximum joint-position difference 12 microrad and velocity difference .00286 rad/s,
within the original reset bounds. Initial target and recorded controller/gain inputs
match exactly; six audits include direct gain readback and two earlier ones do not.
Post-conditioning contact/target differences remain diagnostic, with independent
physical contact/phase/stopping admission; bitwise equality is not a physical gate.
`restoration-path-comparison.json` retains this review and the initial overly strict
derived comparison. Fresh preceding push,
not extension of a completed hold, supplies temporal history.

The actual IdealPD model computes stiffness times position error plus damping times
velocity error and one gravity feedforward term each native physics step. Recorded
computed effort matches that formula to about 1e-6 N m; applied effort matches
physical effort clipping exactly. The failed push stop has fixed targets and zero
normal contact force. Using its actual mass matrix and existing .01 joint armature,
the local arm integration model predicts an alternating multiplier **-1.01896**;
the measured multiplier is **-1.01899**, with .9999998 velocity-mode alignment.
This establishes a local discrete explicit-PD/physics instability. The 4.47 rad/s
plateau is the unchanged shoulder-Z velocity ceiling, not stable settling.

Only the physics timestep changes in the matched comparison. At 120 Hz the push
stop reaches .32121 rad/s at 2 s and later 4.47 rad/s. At 240 Hz it ends at .000216
rad/s and remains stable through 6 s. The identical failed ordinary-hold target
replay reaches .70034 rad/s at 120 Hz; at 240 Hz it peaks at .47379 rad/s, then
passes the two-second stop and four-second continuation. Fresh push, the original
hold, release and ordinary stabilization also pass in both opening directions.
Six right/left loaded 10 N normal/cone and unloaded controls pass the unchanged
residual, joint, held-joint, geometric stop-envelope and rotation bounds. No load
calibration fit, bound increase, mass/inertia/friction/armature change or solver
migration occurs. Compatibility analysis uses the fourth native sample for the
60 Hz pose join and independent stop clocks to exclude callback rows with stale
motion/reference annotations after force removal; raw files are unchanged. See
`runtime-compatibility/measurement-notes.json`. TGS/backend settings are retained. The local linear model omits
nonlinear contact/jaw constraints; sampled qualification is not a global theorem.

The opt-in recipe uses four 240 Hz physics steps per unchanged 60 Hz command/render
boundary and unchanged 10 Hz native RGB-D. Every actual native feedback sample is
retained. Four-second history, 240×384 model images, chunk 12, execution 6 and visual
reuse off remain unchanged. Existing 120 Hz recordings retain their demonstrated
physical/data scopes; their cadence is declared per source. New conditions retain
independent physical and diversity admission. Every resumed collection records the
unchanged 2 s stopping gate followed by 4 s diagnostic continuation with the same
latched target. Newly screened/restored starts also receive their own 2+4 s check
before a fresh recording begins. [Matched native figure](../../../outputs/evidence/act-a1-diverse-20261019/runtime-before-after.png),
`controller-mechanism.json` and `runtime-qualification-decision.json` preserve proof
and scope.

### Ordinary contact support and actual corrections

The left high-contact full attempt acquired contact, then lost it for .267 s as a
roughly 31 µm separating gap opened while the panel briefly coasted ahead of the
reference clock. No hard physical violation or stopping instability occurred. A
.1 s gap response reduced the gap but still failed the unchanged loss deadline;
that attempt remains failed. Correcting the measured displacement at the next
60 Hz boundary, within the existing 6 mm/s normal rate and displacement bounds,
reduces the maximum early gap to 9.88 µm and maximum unloaded interval to .10 s.
The complete condition then passes **53.54 degrees versus 42.03 required**, hold,
release, ordinary stabilization and the separate 2+4 s stop observations. Excess
force retains unloading priority and the push reference never chases a coasting
panel. Gap feedback is disabled during excluded injected pulses. Both restoration
paths use the same support logic. `left-contact-causal-comparison.json` retains
all three conditions and their original outcomes.

Future disturbance targets ramp into the error and retain it until handoff. Eligible
teacher correction starts from the measured tool pose and executes a reported .5 s
cosine return through ordinary IK and all physical gates. The first new normal-error
case transfers **14.95 mm** of its requested 15 mm to eligible supervision, then
measures 9.20 mm after .25 s, 2.95 mm after .5 s, .377 mm after 1 s and .284 mm after
1.5 s. All 119 injected commands stay excluded while delivered actions and their
actual consequences remain causal history. The original 5.22 mm example is unchanged
and retains that honest scope. Subsequent A cells measure the actual injected peak
and retained eligible residual; physical acquisition, withdrawal, stabilization and
stopping determine task success, not reproduction of one exact error curve.
Left acquisition corrections transfer 20.04 mm / 5.01 degrees and 14.96 mm to
eligible teacher control, retaining 99.98% and 99.72% of their stationary injected
errors. At .5 s the measured residuals are 1.53 mm and 1.08 mm. The normal-error
retention review excludes moving-approach lag from its peak denominator; its initial
derived review is preserved, with no new simulation or altered physical outcome.
B/C/D corrections and remaining dynamics stay gated by their own physical/response checks.
The first B2 physically succeeds but does not fill its cell: nine bounded-recontact
compression increments shrink the native gap from 1.83 to .65 mm during excluded
injection; its first 14 actual corrective closing commands are also masked.
First usable supervision starts at only .062 mm normal separation, although the
adaptive-reference tracking error is 4.26 mm; those are different measurements.
The repair removes that independent compression catch-up and explicitly admits
only physically bounded, at-most-one-second corrective recontact after prior valid
contact. Injection remains excluded. The original measured-origin transition and
ordinary contact feedback perform the correction. Actual loaded support still
controls correction completion and the task endpoint. The optional evaluator
`recovery_valid` flag never enters policy observations; legacy files retain their
original masks. The matched repair physically succeeds, restores contact and includes all 48
corrective commands from its first row; its 0.983 mm native initial gap remains
below the prospectively declared 1 mm coverage floor, so it does not fill the cell.
The input audit finds 0.33–0.38mm ordinary forward lead, leaving only 1.12–1.17mm
requested separation from a 1.5 mm relative action. B2 now smoothly targets the
original 1.5 mm separation from the current material plane, with unchanged amplitude,
stage timing, coverage floor and physical limits. Physical qualification of that
input realization passes: first eligible supervision starts at **1.241 mm native
contact gap**, with zero masked corrective commands and 18 independently checked
bounded-recontact annotations. At .5 s the native gap is .000145 mm with actual
loaded contact (3.21 N); the original task endpoint, ordinary tail and 2+4 s stop
also pass. Both earlier physical successes and limited scopes remain unchanged.
`B-recontact-causal-comparison.json` separates native gap, material separation and
adaptive-reference tracking error; they are not interchangeable recovery magnitudes. New correction traces retain the ordinary teacher
pose and measured pre-action error before injection or corrective blending. That
reference includes existing normal/tangential support; it is an evaluator diagnostic,
not a counterfactual trajectory or policy input. This permits first-eligible-row
recovery measurements even when phase validity excludes some corrective commands.
Bounded recontact feedback during injection must be checked for premature error
cancellation before the B/C conditions are multiplied. A retained-trace C2 audit
finds that multiplying the ordinary reference clock by 1.5 changes no request in
826 visited C states because the lead cap is active. The corrected mechanism
accumulates half the positive ordinary reference progression during the pulse
and physically executes that extra material-angle advance as an excluded action.
The ordinary clock/cap and all physical limits remain unchanged. No taper removes
the accumulated offset before measured-origin corrective control. A regression
exercises the saturated-cap path, actual requested progression and retained handoff
error. This is implementation only; C2 remains unqualified until actual motion,
contact, corrective labels, tail and stopping demonstrate its declared scope.
`pace-pulse-cap-audit.json` and prospective protocol preserve the distinction.

A subsequent right late-C1 and left B2 prototype fail required sustained contact
without physical limit violations; both independent stops pass. Their original
recordings/outcomes remain failures. Native traces show contact-force and panel-speed
overshoot after recontact, distinct from the repaired alternating PD instability.
In C1 the ordinary normal target advances approximately 1.2 mm during the first
half of its measured-origin corrective return. Gap-position feedback now waits
until that 0.5 s return finishes; force admittance/unloading remains active.
A focused regression crosses push into hold and checks both completion and pending
physical recovery. This is a causal hypothesis under matched physical comparison,
not a qualified cure. Same-source C1, opposite-direction B2 and a nominal-direction
B2 compatibility control are prospectively declared; criteria, pulses, gains,
physics rates and source states remain unchanged.

The matched no-overlap C1 attempt also fails, now at the unchanged one-second
recontact deadline: 0.352 mm gap remains while the 0.5 s return is still active.
Its first corrective row has 0.933 mm normal-reference residual and only 0.25 s
of remaining loss allowance. C1 alone now uses a 0.25 s measured-origin cosine
return; the resulting requested normal peak is 5.86 mm/s, within the existing
6 mm/s gap catch-up rate. Other correction transitions remain 0.5 s. This repairs
a pulse/return timing incompatibility without shortening the pulse or relaxing a
physical/task/loss criterion. Physical qualification remains pending; the failed
matched attempt and its ordinary cleanup/stable stop are preserved.

The shorter C1 return restores contact but fails the subsequent near-reach hold;
a genuine earlier source also fails. These are retained physical failures, not
qualified examples. The angular-reference audit rejects windup: lead remains
exactly 0.3 degree throughout. Continuing the position return/positive opening
lead after contact is already restored produces excessive panel momentum and
exhausts reach during hold. The next bounded test makes one explicit achieved-contact
handoff from position correction to ordinary force holding. At actual valid physical
contact it expresses the achieved tool position in the current material frame,
within unchanged normal/tangential offset bounds, then begins the complete 3 s hold
with zero additional opening lead. Physics, gains, velocities and robot/door states
are never reset. Recovery still needs sustained real contact; all original hold,
release, stabilization and independent stop criteria remain. This is teacher
implementation pending physical qualification, not a demonstrated cure.

The opposite-direction B2 matched no-overlap test passes: first-used native gap
is 1.415 mm, all 29 injected actions excluded, zero corrective actions masked, actual
contact restored by .5 s and sustained thereafter. Its task, ordinary tail, 2+4 s
stops and unchanged-weight CUDA validation pass. The original failed B2 has higher
recovery door speed (.0475 versus .0292 rad/s) and remains a failure. This supports
the overlap correction in B2; it does not qualify late C holding.

The matched first-door C1 supported-hold check passes physical, data and unchanged-weight
CUDA gates. First usable supervision retains 0.695 mm native gap, 0.822 mm material
separation and 0.983 mm adaptive-reference error (distinct quantities); all 47 injected
commands remain excluded and all 76 corrective commands are usable. One actual-contact
handoff is followed by the full 180-row hold at 51.91 degrees, release, ordinary
stabilization and 2+4 s stop checks. A transient contact reloss is retained; recovery
is not claimed monotonic. The reference changes at the reported handoff, so its later
tracking error is not a fixed-goal recovery metric. Native gap and actual supported
settling establish the task result. Other C1 doors still require their own gates.
`C1-causal-qualification-comparison.json` preserves all four earlier failures and
rejected explanations. The same supported-hold mechanism is implemented for C2's
pace-to-settle task; actual delivered pace, residual motion and settling remain
unqualified until its bounded physical prototype passes.

The first physical C2 input executes only 52/59 injected commands before the unchanged
predictive guard selects hold; it loses hold contact and remains failed. An earlier
25 s native source permits all 59 commands and one supported-hold handoff, with no
physical violation and successful release/stabilization/independent stopping. It still
fails the C endpoint: the fixed 3 s hold ends at 0.90 degree/s without satisfying
the complete .5 s support/speed/tool-travel window. The configured speed bound is
1 degree/s, with at most2.5mm tool travel; a terminal value alone is insufficient. C2 now retains the full 3 s minimum and continues
ordinary supported holding only until the current final .5 s meets the original
criteria, bounded at 6 s. This is controlled settling, not additional temporal
history: the first correction already follows 5.2 s genuine motion. All speed,
contact, drift, physical and stop criteria are unchanged. The matched physical check passes after3.2s with all59 injected actions excluded,
all41 corrective actions usable, one supported-hold handoff, the original tail and
independent2+4s stops. All C collection holds now use this completion-driven rule;
non-C holds retain their existing duration. Earlier passing3s C records retain
scope; this never extends a completed hold for temporal history.

Before multiplying D corrections, the implementation now prevents a moving panel
from automatically cancelling the declared release error. D1 smoothly targets the
original8mm separation from the current material frame, not a stale release-origin
world pose. D2 continues the already qualified material-following force hold during
its excluded pause, rather than letting the panel coast away from a fixed tool pose.
Pulse times/amplitudes, injected-command masks, ordinary corrective blend, release,
tail and stop criteria remain unchanged. A moving-material regression exercises
both paths and their real corrective handoff. After a failure, further error injection is suppressed during ordinary cleanup;
the failure latch and original outcome remain. D1's physical prototype passes with7.33mm first-supervised material separation,
20.85mm ordinary-reference error, all59 injected actions excluded and all37
corrective actions usable. It exceeds10mm separation within.25s and reduces
reference error to.92mm by1.5s; ordinary tail and2+4s stops pass.
D2's continued-contact pause instead reaches.533rad/s before correction, with its
independent stop stable. The inherited positive opening lead continues driving the
door near the reach boundary. The matched candidate captures achieved supported
offsets once and removes this angular opening lead, while continuing ordinary
bounded force support. This is an excluded intentional contact pause, not another
opening push. Physical qualification remains pending; original failed states and
native oscillation-free stopping remain preserved.

The light model's full source and C/D targets pass their individual gates. The
first resistant full source instead loses ordinary contact near36 degrees; its
independent stop passes and all raw outcomes remain failed. Native traces show
4–6 micrometre geometric gaps with force near zero for the unchanged .25 s loss
window. The controller selects max(gap catch-up, force admittance), so the larger
geometric term masks the positive pressure increment needed beyond first touch.
The corrected composition adds the two distinct terms while retaining the same
admittance gain, 6 mm/s rate bound, offset bounds, force-unloading priority and all
physical/task/loss criteria. A focused regression exercises this measured weak-load
condition. The full-opening report now also preserves the primary physical failure instead
of overwriting lost_contact with required_opening_not_reached. Old reports remain
unchanged; their original teacher result retains the cause. Matched initial robot/tool/door states and setup are exactly equal. The corrected
resistant full source passes51.53 degrees versus41.01 required, including hold,
release, stabilization and independent2+4s stopping. All31 observed control rows
just beyond the former failure angle retain valid contact. The derived causal
comparison is pressure-composition-qualification.json. Its C/D continuations also
pass individually. This contact-feedback defect is separate from the already
corrected PD stopping pole; other postures/models retain their own gates.

C pose-high records 4.4 s genuine push before hold and 20.11 mm actual height
separation from nominal. D records 6.12 s genuine pre-release history, the original
3 s hold and 20.12 mm height separation. B initial opening has 19.97 mm separation.
The lower-height complete right episode passes 49.96 degrees versus 41.01 required
and has -19.95 mm actual hold-height change. The base-retreat acquisition condition
passes its measured 20 mm root/precontact-gap variation. Both third-door complete
height conditions pass: 52.86/49.84 degrees held versus 47.77 required, with separate
ordinary tails, 2+4 s stops, temporal-data and unchanged-weight CUDA checks. Source
continuations retain ancestry and do not create independent door groups; extracted states/windows never
fill cells. All four short families now have admitted examples in both opening
directions. Third/fourth-door pose cells continue under the same per-case gates.

All eight declared complete-training cells are individually qualified. Held openings
are compared with the unchanged per-door minimum:

| Train door | High / low held opening | Required minimum |
|---|---:|---:|
| animated-door-1 | 52.89 / 49.96 degrees | 41.01 degrees |
| door-2738468 | 53.54 / 50.95 degrees | 42.03 degrees |
| PSX front | 52.86 / 49.84 degrees | 47.77 degrees |
| PSX wood | 55.95 / 52.84 degrees | 50.15 degrees |

The last low-wood push ends at its existing predictive tracking-margin guard,
then qualifies hold, release and stabilization without a physical violation.
All new complete episodes pass separate 2+4 s stop observations and actual-data
CUDA checks. The original high-right 120 Hz episode retains its original scope.
Observed high/low contact-height changes are approximately +20/-20 mm; the eight
complete episodes do not fill any of the distinct short-family cells.

### Temporal usability and continuing limits

The first resumed 240 Hz recording exposed an ingress capacity/pruning mismatch at
4.03 s, before disturbance execution. Its failed prefix has zero expert windows;
its stops pass. Buffer capacity now covers the same unchanged history/carry-in
horizon as pruning. A six-second 240 Hz regression verifies bounded retention and
exact offline/online observations. The native-feedback denominator now uses recorded
`sim_dt` with the legacy 120 Hz fallback. Earlier raw outcomes/reports remain intact;
`native-quality-rate-review.json` corrects their derived count text. Admitted 240 Hz
sources have every expected native sample.

Actual-data masks, delivered-minus-prestate labels, causal joins, phase transitions,
tails, native provenance and representative CUDA forward/backward pass for admitted
sources. Weights remain exactly unchanged with zero optimizer updates. The original
six-train-source normalization remains frozen for compatibility; development never
fits it. More windows and correlated source continuations do not establish diversity.

All 32 train pose cells and all eight A and eight B correction cells are admitted, retaining
the historical 5.22 mm scope of the original A1 example. The first light-dynamics
A/B cells pass active .7 mass/inertia scaling with damping 4; over a matched 56-row
initial-opening interval, mean control-tick native peak force is .433 N versus .712 N nominal
(39% lower). Motion/endpoint similarity does not erase that measured dynamic
response. Earlier unequal-window analysis is retained, and no threshold or
physical outcome is changed. Remaining dynamics still require their factor/source
qualifications.

Each wave stops for diagnosis on any failed gate. New dynamics require
matched nominal/mass-only initial-opening checks, active PhysX mass/inertia/damping
readback, observed response differences, and full same-dynamics sources before C/D.
The original sixteen-control plan reuses three already qualified nominal B2
pre-pulse prefixes (all contact and first 61 push rows are ordinary expert actions).
It therefore needs thirteen auxiliary training factor/full-source executions;
no additional target cells arise. The first mass-only control is physically safe
but its 4.1% contact-force change is below the target diversity criterion; this
negative result is preserved. It remains a separate-factor control, never a diverse
target. Resistant target admission still requires the original measured criterion.
Resistant full sources prospectively allow 225 s push and 300 s total simulation,
scaling the original 150 s push budget by viscous damping 6/4. The 1200 s wall guard,
all task/physical criteria, teacher speed, sensor/control rates and ACT configuration
remain unchanged. This permits slower physical progress; it adds no policy information
per second. Every execution consumes the original campaign ceilings. The prospective
matched-response review requires at least 10% change and a material absolute effect
(0.1 degree angle, 0.1 degree/s speed, 0.02 N mean force or 0.02 N s impulse). These
are coverage criteria; physical safety and task predicates remain unchanged.
Late-dynamics admission additionally requires cadence-correct native force
integrals or measured kinematic response. A mean of per-control native maxima is
only a peak-envelope metric; it cannot alone establish a force difference between
120 and240Hz recordings. The first light C/D and resistant C/D cells meet this
additional review, including their complete ordinary hold. The observed effects
are conditional responses of the actual model and reconstructed posture, not an
isolated local mass estimate; matched early factor controls address that question.

The resistant A example at the original .1N acquisition target remains unqualified
for diversity. Its whole-contact native mean changes .1247 to .1224N; the final
window changes .1685 to .1807N, below the declared material criterion. A large
single native maximum does not rescue it. Before spending an A attempt in another
unchanged context, its actual already-executed B acquisition prefix is reviewed.
This can reject a redundant launch, never fill a second cell. Qualified B contact
also establishes prerequisite phase physics for an independent later C/D launch;
a missing A diversity cell does not invalidate safe contact already executed.

The final handoff must still satisfy every original coverage, physical, data and resource gate.

The left C1 prototype recontacts with a5.08N impact, then loses support after the
first-touch handoff; speed and its independent stop remain safe. Retained older
no-handoff right runs establish sustained contact before their later failure.
New C handoffs therefore require the original.5s contiguous valid-contact window,
including reset on intervening loss, before switching to achieved force hold.
A first collision sample cannot cancel the ongoing corrective return. This changes
no supervision validity, contact-loss deadline, pulse or physical threshold.
Matched left qualification remains pending; earlier qualified right sources keep
their demonstrated scope.

The D2 zero-opening-lead trial still exceeds speed before correction (.522rad/s),
with stable stopping. Its push-exit prediction reserves the ordinary hold and
release but omits the intentional1s loaded pause and complete.5s corrective blend.
The next matched candidate includes this declared1.5s support duration in the same
conditioning and endpoint-reach projections. It does not extend hold/history or
change any rate or physical criterion. Removing lead alone is explicitly
insufficient; the new supported-start qualification remains pending.

Sustained support alone does not fix left C1: the matched trial still loses hold
contact after.317s, with stable stopping. At the recovered-contact handoff, the
.1s qualification window retains about.606N while the latest native intervals
already have zero force. This stale impact load suppresses positive geometric
catch-up until the tool has recoiled away. Only the new recovered C force-hold mode
now uses the preceding native interval's average force for pressure feedback.
The original rolling window still decides loaded qualification; targets,
admittance, bounds, contact-loss/hold/stop criteria and all rates are unchanged.
Ordinary nominal/pose/dynamics holds retain their previous controller. The matched
left candidate remains unqualified until its physical/data/recovery gates pass.
Both prior failed handoffs are preserved; no old valid recording is relabeled.

The matched current-native-force C1 trial passes its original3s hold, recovery,
ordinary tail, native/data/CUDA and2+4s stopping gates. Before handoff the source
and corrective residual are unchanged:1.835mm reference error and.490mm native
contact gap at the first usable corrective action, with47 injected actions excluded
and no corrective labels masked. Recontact is measured directly; a changed
post-handoff reference is not claimed as fixed-goal error reduction. The initial
review-command import failure was repaired using project Python, without another
physical or CUDA run.

D2's extra1.5s linear reach reserve still fails at.530rad/s; all59 injected actions
execute, but corrective supervision has not begun. Native RIGHT_SHOULDER_Y velocity
is same-sign within physics intervals, with60Hz target-tracking lag at Jacobian
sigma.00136. This differs from the corrected alternating stopping pole; fixed-target
stopping remains stable. The next candidate keeps the exact saved start and
finishes pushing after the original supported task-opening goal and sufficient
real pre-release history, rather than pushing onward to the reach boundary. It
still physically executes the original3s hold once,1s loaded pause,.5s corrective
transition, full ordinary tail and2+4s stopping. The held opening must also meet
the original full-task minimum. This is an explicit task-based phase-planning
correction, not a claim that the previously failed maximum-opening posture is
qualified. Its physical result remains pending.

The task-goal D2 trial passes from the same saved start; its first84 push rows are
physically exact against the failed reserve-only trial. Held opening is50.53 degrees,
above the unchanged41.01-degree task minimum. The original3s hold and4.4s actual
pre-release context yield complete eligible windows without extending a completed
hold. Expert control begins with19.07mm reference error and real loaded contact
(native gap1.26micrometres); all59 injected actions are excluded and no corrective
actions are masked. Peak actual joint speed is.498rad/s versus.530 before; full
ordinary stabilization and2+4s stopping pass. The old maximum-opening pause posture
is still unsupported. Other D2 candidates use the first actual supported window
meeting their unchanged full-task opening goal; each must pass fresh reconstruction
and its own complete physical/data/recovery gates.

The remaining right A resistant cell passes at a prospectively matched1N teacher
acquisition target: .55 versus.71degree opening response (23% difference), with
unchanged physical/task criteria. Both pair members use the same excitation; the
original.1N target negative remains. This is a declared teacher context change,
not a measured hardware parameter. Direct true-native-force/kinematic light-versus-
resistant audits additionally pass all ten currently available dynamics pairs;
six late pairs remain pending. Conditional late-state differences are not claimed
as isolated local mass estimates.

The original A1 example remains usable for its measured5.22mm correction. A single
full-transfer follow-up is declared to close that original20mm target in full;
it must deliver at least90% of both20mm and5degrees to actual usable corrective
supervision. Only after it passes will the old narrower example move to a separate
retained usable-scope inventory. Its original physical/data outcomes and recordings
stay unchanged, and only one episode can fill the target cell.
