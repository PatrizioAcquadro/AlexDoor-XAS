# Diverse bounded ACT A1 collection campaign

**Status: scoped resumption, goal unmet.** The original six physical attempts produced three individually
usable episodes and three failures. Only 7/112 target cells pass; no policy training
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
Campaign ceilings: 160 new attempts, 12 additional GPU workload wall-hours,
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
targets so short collection does not teach early release for the full opening goal.
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
not change complete-task loss criteria. This recovery path and C/D starts remain
**physically unqualified**. Uniform density authoring scales panel/handle mass and
inertia together, preserves geometry/COM/axes, and saves active PhysX values. No
new dynamics range has been physically qualified in this halted campaign.

Before the main matrix, unlaunched early A/B pose-low candidates were specified as
20 mm robot-base retreat along world X, 50 mm precontact distance and nominal contact
height. Late/complete pose-low retains 1.07 m height. These remain untested candidates.
Static-reference times are per episode; model history/resolution/chunk/execution stay
4 s / 240×384 / 12 / 6, visual reuse off. Raw stream clocks are actual local episode
acquisition/availability. Probe trace/endpoint support times use a separate zero-based
completed-control-row clock; join by the aligned annotation rows, not identical values.

## All physical outcomes and fixed denominator

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

The physical launcher is closed by `campaign-stop.json`, verified without starting
a simulator. No remaining matrix or development attempt runs after the stop gate.

## Actual coverage and data usability

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

## Costs, handoff and required next work

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
No new physical attempt follows this closeout. Once the full corpus is qualified,
a bounded first ACT experiment should test whether causal history improves contact
acquisition, phase handoff and unloading, and whether genuine observed variation
improves frozen-development completion without increased refusals/unsafe acceptance.
This partial corpus cannot test the requested broad recovery/generalization claim.

## Resumed investigation (20 October)

The original 112-cell objective and original cumulative resource baseline continue.
Bulk collection remains halted while bounded matched-state diagnostics investigate
ordinary and stopping control. The checked start and recording start now both opt
into the same source-controller reconstruction (robot/door state, targets, impedance,
controller state and fresh contact). A focused preparation regression exercises both
restorations with the intervening stop/reset. This fixes a later orchestration mismatch;
it does not explain or qualify the earlier recorded oscillation. Existing three usable
recordings retain their demonstrated scopes pending runtime compatibility assessment.

Future correction pulses now ramp into the error and retain its full target until
handoff. Eligible correction starts from the measured tool pose and returns to the
ordinary task target over a reported .5 s cosine transition through existing IK,
physical speed/force/contact gates and delivered-action accounting. Start/latest
position and orientation residuals are evaluator metadata. Injected commands remain
excluded; the old 5.22 mm example is unchanged. This mechanism passes focused target
and mask checks but requires physical qualification before multiplying its cells.

Matched native diagnostics reproduce both failures at 120 Hz: the earlier push stop
ends at .32121 rad/s after 2 s and grows to the 4.47 rad/s configured ceiling; the
retained ordinary-hold target replay reproduces .70034 rad/s before stopping.
Changing only integration to 240 Hz (four native steps per unchanged 60 Hz command)
stabilizes both: push final speed .000216 rad/s; ordinary replay peak .47379 rad/s,
stop final .000791 rad/s, with no delayed growth through 6 s. At the push posture,
the right-arm explicit-PD/semi-implicit integration linearization using actual mass
and existing armature predicts -1.01896 per step, matching the observed -1.01899.
The alternating numerical mode explains this stopping failure; the velocity ceiling
is saturation, not stability. No gains, mass, inertia, armature or criteria changed.
The local calculation omits jaw constraints/nonlinear contact and is not a global
stability proof. Fresh right-side push/hold/release/ordinary-tail progression also
passes; both-direction loaded and nearby controls remain under qualification.

An opt-in `physics_substeps=4` recipe refines physics to 240 Hz while retaining
60 Hz control/render boundaries and 10 Hz native RGB-D. Default recipes retain
120 Hz. Native feedback retains every actual substep; history, image size, action
chunk and execution length are unchanged. Old 120 Hz calibration/stop scope is not
automatically transferred. Diagnostic records live under `runtime-diagnostics/`;
`runtime-diagnostic-plan.json` and `runtime-diagnostic-attempts.json` retain all cases.

The 240 Hz compatibility check now passes all six right/left 10 N normal/cone and
unloaded controls against the unchanged residual, joint, held-joint, shape-stop and
rotation bounds. No calibration fit or bound increase occurred. Analysis explicitly
aligns the fourth native substep with the first 60 Hz pose; motion excludes the
independently timed stop/removal rows. Raw diagnostics retain the original phase
annotation limitation; the corrected joins are in `analyze_compatibility.py`.
Both opening directions pass fresh push, original hold, release and ordinary
stabilization, with separate 2 s stops and 4 s fixed-anchor continuations. These
are qualification executions without recordings; they do not fill target cells.

`runtime-qualification-decision.json` admits scoped collection. The first resumed
cell is A correction-2, to physically qualify the measured-origin corrective handoff.
Each new variation retains task/physical/diversity admission. Every resumed collection
also records a separate delayed stop continuation after the unchanged 2 s gate.
Native load monitoring checks every configured 120/240 Hz substep, including peak
load between unchanged 60 Hz decisions. The legacy 120 Hz inputs remain supported.
Nineteen campaign attempts have been charged before this first resumed collection:
six original, six matched/integration diagnostics (one wrapper failure), six
compatibility controls and one left integration. Exact current totals are in the
resource ledger; all original costs and failed outcomes remain in the denominator.

The first resumed correction-2 attempt exposed a temporal ingress capacity defect
at 4.03 s, before disturbance execution. Its prefix remains failed with zero expert
windows; the two-second stop and four-second continuation pass. Capacity had used
the shorter numeric gap while pruning retained the longer image/numeric carry-in
horizon. Capacity now matches that unchanged horizon at the declared native rate;
a six-second 240 Hz regression checks bounded retention and exact offline/online
observations. This changes neither sample rates, four-second history nor model input.

The corrected acquisition attempt passes physical, temporal and measured-diversity
admission: first eligible prestate 14.95 mm, 9.20 mm after .25 s, 2.95 mm after .5 s,
.377 mm after 1 s and .284 mm after 1.5 s. 119 injected commands stay excluded, while
actual delivered actions remain in history. CUDA consumption/gradients are finite
with unchanged weights and zero optimizer updates. Current coverage is 4/112:
3/96 short train, 1/8 complete train, 0/8 development; 108 cells remain.
[Matched native before/after](../../../outputs/evidence/act-a1-diverse-20261019/runtime-before-after.png)
shows both diagnosed conditions with unchanged physical/task criteria.

The first C pose-high source now passes all three admission layers. It contains
4.4 s of genuine fresh push before the original hold,180 supported hold rows and
20.11 mm actual tool-height separation from the nominal control. Checked/recording
reconstruction has zero measured difference in full robot q/dq, door q/dq and
joint targets, with identical .18468 N reconstructed normal contact. Separate
2 s stopping and4 s continuation pass. Native240Hz observations, phase seams,
causal labels/masks and unchanged-weight CUDA forward/backward pass. Coverage is
5/112 (4 short train,1 complete train,0 development),107 missing. New diagnostic
executions do not count as target cells. The corrected D recording follows the
same qualified runtime with its own genuine preceding push and original3s hold.

The initial resumed short-family tranche now admits B, C and D pose-high plus the
15 mm acquisition recovery. B demonstrates sustained >=1 degree initial opening
with19.97mm observed height separation. D has6.12s genuine pre-release history,
the original3s hold and20.12mm observed height separation. Both pass physical,
data/mask, diversity and delayed-stop checks. Coverage is7/112:6 short train,
1 complete train and0 development;105 cells remain. The declared complete-pose
matrix continues. Earlier states and failure outcomes remain unchanged.

Native computed effort agrees with the explicit PD plus one gravity feedforward
term to about1e-6Nm; applied effort matches the physical effort clip exactly.
Fixed targets have zero native variation. The observed unstable velocity vector
aligns with the predicted120Hz local eigenmode at .9999998 cosine similarity.
Unchanged TGS/backend settings, actual mass matrices and existing .01 armature
are retained in `controller-mechanism.json`; no parameter was invented to suppress
oscillation. Actual gain readback and controller inputs now accompany each new
reconstruction audit, alongside native q/dq, targets, door state and contact.
