# Resumed End-to-End A1/A2 Pilot

Baseline `4ec7571`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The user explicitly resumes Goal 1: correct the actual contact-monitor/teacher
problems and continue through real demonstrations, short ACT/Diffusion × A1/A2
training and learned closed-loop simulation. Preserve all previous work/failures,
fixed physical criteria and sealed test doors. No production campaign or robot work.

## Left position-domain prerequisite

The original ACT A1 left refusal is a joint-position envelope exit: elbow
−2.305783 rad versus the calibrated lower bound −2.305511, while velocity and
acceleration remain inside their bounds. An independent robot-only known-wrench
experiment tests a prospective outer lower bound −2.35, with unchanged error,
speed, force and stop criteria. The 0 N cohort completes 1,680 native samples
and passes joint/scalar error bounds and the original continuous 14-shape stop
boxes (minimum directional margin 0.235 mm). These are robot CAD/stop margins,
not vision accuracy requirements.

The following 8 N /−0.4 rad lateral-wrench cohort fails its loaded stop:
24.921 mm travel, 0.09150 rad rotation and 0.59608 rad/s peak joint speed. Its
failure-unload tail is retained too. The unconditional position-domain candidate
remains disabled; no operational calibration or controller is changed. All
3,120 raw samples, both stops and 40.742 charged GPU seconds are preserved in
`robot-left-elbow-domain`.

A separate test at the existing 1 N weak-load cap completes three wrench yaws
and 5,040 native samples, reusing the completed 0 N cohort. All unchanged torque,
scalar-normal and response gates pass: minimum weak scalar margin 0.00501 N,
maximum tool response 3.850 mm /0.02321 rad against 13 mm /0.04 rad. All three
loaded stops satisfy the original continuous 14-shape/held/acceleration boxes;
minimum radial/directional margins are 0.996/0.229 mm. Every cohort excites elbow
positions below −2.34, beyond the preregistered operational lower bound −2.33.
The source-path error in the CPU validator is preserved and corrected without
another physical allocation. This second allocation charges 50.905 GPU seconds.

`LoadCalibration.weak_load_q_min` supplies only a lower-position extension. It
requires the existing force upper cap ≤1 N and original main acceleration; an
untested combination with the extended weak acceleration domain is rejected.
Original main position bounds, stronger loads, upper position limits, speed,
error coefficients, geometry, response and stop margins are unchanged. Both
complete teacher paths remain mathematically identical because every sample is
inside the original position envelope. The actual original left refusal passes
the new predicate with force upper 0.49918 N; this saved-state check uses arbitrary
unit normals solely for the force-independent envelope predicate and is not a
new operational rollout. This is a current-load check, not authorization for a future stronger load.
The maintained motion producer therefore requires observed separation for every
proved robot volume throughout the proposed motion and full stop whenever the
conservative joint sweep/stop can enter the weak position extension. The proof deducts nonzero declared
translation and angular-rate growth through the entire stop, beyond the growth
already paid for camera age and the control interval. Unknown, zero or negative
separation refuses. The original left snapshot is not claimed to pass this
additional motion guard. All four models use the same guarded recipe in the next
common panel version. Actual integrated behavior remains pending.

A narrower strong-load check centers the elbow at −2.30, using the same waveform,
seed, 0/8/10 N levels and three lateral directions. Its prospective operational
lower edge is −2.315, with required independent excitation below −2.3175.
It also fails at 8 N /−0.4 rad: 25 mm travel and 0.594 rad/s peak speed. Both
complete raw stops and the failure-unload continuation are retained; 35.617 GPU
seconds are charged. The candidate remains disabled. The next final panel keeps
the original qualified main position calibration and main-13 gripper bound for
all four models. The optional weak-position recipe is inactive, so its narrower
future-load support cannot be mistaken for strong contact qualification.

## Original-checkpoint full-v7 outcomes

Four completed outcomes are retained separately: ACT A1 right/nominal executes
1,878 commands, reaches 18.790 degrees controlled opening (36.6568% reference), then
stops at true contact loss; the same checkpoint left/nominal executes 193 before
load-envelope refusal. ACT A2 right/nominal executes 239 before complete-sweep/stop
refusal, and Diffusion A1 right/nominal 371 before observed orientation refusal.
All four input/command/clock/continuing-physics stop audits pass. None succeeds,
completes hold or releases. All 20 unallocated slots remain explicitly unevaluated
in this retired version; they are never combined with another checkpoint version.

The right ACT A1 contact monitor is available at all 1,879 operational endpoints.
Its first confirmation follows instantaneous native contact by 100 ms. Independent
joins find 16 confirmed ticks without instantaneous force presence (maximum 83.3 ms),
three below unchanged physical Loaded, and five physical Loaded ticks without
confirmation. These are measured discrepancies, not zero-error qualification.
The evaluator keeps its unchanged instantaneous physical requirements and excludes
uncontrolled intervals from controlled opening. The left upstream load refusal
has one missing monitor decision out of 194 endpoints, explicitly unavailable.
See `learned-contact-metrics-v7.json` for the separate denominators and all four
cases; no independent evaluator stream enters runtime admission or policy inputs.

## Interrupted continuation and exact cache repair

Job `11974622` times out after completing ACT A1 and Diffusion A1 at 4,400
updates and saving ACT A2 at 3,200. Diffusion A2 remains at its original 3,000.
The complete parent/step accounting charges591GPU seconds, including external
cleanup, bringing cumulative remote allocation to3,334/3,600. All partial states,
optimizer/RNG/EMA, final A1 checkpoints and logs are retained. Checkpointed
cumulative updates are15,164including infrastructure; one possible interrupted
100-update epoch is conservatively charged, giving an upper count15,264.
The remaining2,600usable updates would finish at17,764checkpointed /at most
17,864charged updates. Partial training is not complete.

Actual epochs take about3.5seconds per100updates. Cold/shared-file reads of the
13.8GB flat cache dominate the failed allocation. The resume path now validates
completed reports before data access and skips complete action representations.
Pure CPU/I/O repair preserves the original remote cache and repacks the same two
train episodes into self-contained HDF5 virtual observations. Exact visual-prefix
reuse reduces physical storage to0.620GB; all original observation/label/time
bytes, attributes and metadata match. All14,987observation rows and14,985action
rows remain, with2,498unique visual-prefix tensors and unchanged current q/dq.
The existing HDF5 reader supports this layout without policy or dataset changes.

The full local data-reader/statistics check takes12.16seconds plus6.98seconds
for factory validation and exactly matches frozen source normalization. It
constructs no model and performs no optimizer work. No development payload leaves
the workstation. The repaired GPU submission must resume saved states with the
unchanged protocol, seed and target, inside266remaining allocated GPU seconds;
job `11974876` is queued without a pinned node, using the corrected Slurm CPU
count, one Python/CUDA process, and a 200-second owned task timeout. Its maximum
260-second additional allocation reservation fits the 266 seconds left. Queue
waiting is uncharged. Successful completion and returned checkpoint verification
remain pending.

## Matched training continuation

The 3,000-update ACT A1 checkpoint reaches 18.790 degrees under full-v7, then
loses real contact before hold/release; the identical checkpoint also fails on
the left. All four initial loss curves still decrease at 3,000 updates. The next
bounded experiment, submitted as Gilbreth job `11974622`, extends all four identically to 4,400 updates (14 additional
100-update epochs each, 5,600 additional updates; 17,764 cumulative including
infrastructure). No architecture, data, seed, sampler, normalization, controller,
replanning or physical criterion changes. Improvement remains unproven.

`pilot_offline.py train --root NEW_ROOT --protocol EXTENDED_PROTOCOL
--continue-from COMPLETED_ROOT/training` requires a new destination and all four
completed source cells. The protocol may differ only in its increased epoch count.
Source model/optimizer/RNG/EMA state and train-only statistics are checked and
retained; the original directory is never edited. Epoch-boundary state in the new
destination supports the existing `--resume` command after interruption. Source
states retain the original protocol rather than being relabeled with the extended
one. Use the prepared train-only cache on Gilbreth; no repeated raw preparation or
development transfer. The existing 3,000-update panel remains a separate frozen
version until a new common checkpoint version is explicitly declared.

## Current resumption

The October 17 authorization raises the cumulative workstation ceiling to 15 GPU
hours and removes numerical submission/prerequisite case ceilings. The remote
3,600 allocated GPU-second, 18,000 optimizer-update, 300 GiB storage and six fresh
demonstration ceilings remain cumulative. At resume, 2.989649 local GPU hours,
70 remote GPU seconds and 164 infrastructure updates are consumed; all six fresh
attempts remain counted. The initial inspection found no subsequent source changes after reviewed commit `a9462e9`.

The initial repair job fails before updates on missing SciPy, then a second fails
on the relocated URDF's parent-relative fingertip STL. Both are diagnosed, repaired
and charged. Job `11970038` completes all four fixed cells on the same corrected
right-dim/left-bright recordings, seed 17 and source `605730e`: 3,000 updates per
cell, 12,000 genuine updates total, plus the prior 164 infrastructure updates.
The full training state is resumable at epoch boundaries. Completed parent/step
Slurm accounting conservatively includes external-step epilogs: 2,743 allocated
GPU seconds across all five jobs, leaving 857/3,600. The real train preprocessing
and remote filesystem reads dominate the allocation; 100-update epochs take about
3.3–3.6 seconds. All four remote checkpoint round trips have zero prediction error.
The returned final files match the remote SHA-256 values. All four load on the
workstation RTX 4090 and evaluate the two local development recordings using the
unchanged checkpoint weights and train-only statistics; neither is recomputed or
updated during inference. Each uses the same 256 sampled chunks /3,064 valid
actions, no clipping, and no model selection. ACT A2 mean/p95 translation error is
0.595/2.850 mm; Diffusion A2 is 0.820/2.494 mm. These are offline imitation errors,
not closed-loop task success. `pilot_offline.py evaluate --checkpoints-root` is
the maintained local inference command after `prepare --reference-raw`.

The original load-model q/dq/ddq envelope contains all 17,458 right and 18,512 left
corrected native samples. Both development records have 556 unavailable samples
after 124.6167 s from joint-position envelope exits; no bound is expanded or
availability fabricated. Complete load/contact evidence remains a separate gate. No development recording was
transferred; remote development metrics remain explicitly unavailable.

Both actual-contact stops pass with continuing physics. Corrected right native
tracking also completes the full 145.48-second recording: all 7,218 post-warmup
60 Hz local queries retain the same supported identity with nonzero between-capture
temporal growth. Normal position bounds span about 4.94–31.71 mm, and full pose
bounds remain distinct. This is an observed component result, not complete contact,
sweep/admission qualification or learned execution. Earlier failures and historical
budget proposals below remain preserved.

## Current observed-runtime integration

The complete first nominal comparison now retains five full-v6 failures including
the original ACT A2 dim slot. Diffusion A1 executes 359 commands and stops at
observed contact orientation; Diffusion A2 executes 242 and exits the load domain
before contact. All five consumed-input and continuing-physics stop audits pass;
no physical violation occurs in this version. Scheduling is held between slots to
resolve the measured ACT A1 stronger-load gripper dependency, with all five results
and the full original denominator preserved.

A separate robot-only known-wrench verification keeps every error-model coefficient
fixed and tests the failed robot state under 8 N and 10 N loads. The abrupt first
8 N lateral preload causes wrist-X speed 0.5774 rad/s and fails; raw data and the
missing old failure-stop evidence remain explicit. Its completed zero-load cohort
(1,056 samples) passes and is reused. Ramping load application/removal over one
second avoids that fixture-induced speed excursion while preserving final force,
force directions, gripper excitation, error bounds and speed criterion. All 4,956
new native transit/ramp/preload/motion/settling/removal samples pass unchanged
joint-error and force bounds, with peak speed 0.3484 rad/s. The 8/10 N cohorts
reach gripper accelerations 29.633/29.651 rad/s², exceeding the predeclared 14.3
coverage requirement for the prospective 13 domain; no parameter is fitted.
An independently reconstructed completion validates every raw sample and stop file:
the original wrapper missed its marker after Isaac shutdown and is preserved as
an infrastructure failure, without a physical rerun.

`qualified-loaded-gripper-domain-calibration.json` changes only main gripper
acceleration from 10.574482 to 13 rad/s², already the weak and geometric bound.
Both teachers remain inside the old main domain, so every complete v42 load,
contact and admission value is unchanged. The real model now accepts the rejected
ACT A1 native sample with force upper 3.762 N (independent instantaneous normal
force 3.121 N; episode peak 7.095 N is separate). Its actual continuing-physics
stop fits all original 14-shape directional/radial bounds, with minimum directional
margin 0.235 mm and held joints contained. Main elbow, all other joint domains,
1 N weak gate, 10 N stop domain, response/geometry/clock bounds, 80 N physical
criterion, teacher, controller and four checkpoints remain unchanged. Accelerations
above 13 in the verification are stress evidence for the error model, not broader
operational geometry permission. Full-v7 will verify the identical ACT A1 checkpoint
on both nominal train doors before completing the remaining frozen panel; all 24
outcomes under that final version and the original success rule remain required.

The next full-v6 slot, ACT A1 right/nominal, demonstrates the first actual learned
contact and controlled opening: 18.815704 degrees /36.7062% of the 51.260252-degree
reference over 1,875 learned commands. It stops at native 56.45 s when gripper
acceleration reaches 12.997291 rad/s², above the 10.574482 stronger-load limit but
inside the predeclared weak 13 rad/s² domain. The concurrent stronger load prevents
using that weak domain. No calibration bound is expanded to admit it. The native
force peak is 7.0954 N, no physical violation or contact loss is recorded, and the
continuing-physics stop travels 0.231 mm /0.730 mrad. Hold, release and task success
remain false. Its complete consumed-input/command audit passes; the fixed panel
continues with ACT A2 right/nominal.

Independent post-run contact joins retain all 1,876 operational control endpoints:
1,875 monitor decisions are available and one is missing after the upstream load
refusal (99.9467% availability). Confirmation takes 100 ms with six initially
unconfirmed physical Loaded ticks. There are 13 confirmed ticks with instantaneous
zero force, at most 83.3 ms continuously, but zero confirmed ticks below the unchanged
physical Loaded criterion. These different denominators are explicit in
`learned-contact-metrics-v6.json`; finite uncertainty, indirect contact, missing
endpoints and the bounded force window are not reported as perfect sensing.
Inspection without a contact receipt is outside this monitor denominator.

The full-v5 ACT A2 right/dim outcome executes 232 genuine learned commands
and stops before contact at a finite-material action-margin refusal. Independent
scoring reports no physical violation and no controlled opening, hold or release;
the continuing-physics stop passes. Its consumed-input audit passes. The runner
now preserves an explicit producer refusal before adapter validation, avoiding the
misleading global-state error for a rejected local receipt. The rejection and stop
remain unchanged. A saved causal RGB-D/proprioception replay is diagnosing the
specific margin. It reproduces all 232 actions within 5.93e-6 per component and
the rejection: 114.74 mm finite clearance versus 115.45 mm combined allowance.
The tracker already includes 6.667 mm of linear capture-age growth in its current
pose bound; admission charged it again. An explicit numerical prepaid-growth
term now credits only that already-paid translation. The same saved proposal has
a 5.959 mm margin after correction. Acquisition/support/availability clocks remain
29.0 s at the 29.0667 s query; next-tick motion still costs 1.667 mm, and both
13 mm response terms, 40 mrad rotations and 20 mm stopping allowance are unchanged.
Existing complete-path margins remain accepted by the nonnegative-credit
inequality; load/contact, geometry and swept-stop values are unchanged. The first real full-v6 execution (`d8de2f0`) passes the prior refusal and
executes 239 learned commands. It then refuses the contact monitor's additional
3 mm model-discrepancy allowance (`contact_uncertainty_exceeds_admitted_margin`).
That allowance was not paid by the local receipt and remains required. This is a
retained failed outcome, not evidence to relax finite support. No physical
violation, contact, opening, hold or release occurs; the audited 2 s stop travels
3.533 mm /6.467 mrad and passes. The 23 remaining frozen full-v6 slots are queued
sequentially with unchanged checkpoints, protocol and cumulative guards. Current
cost is 5.5322/15 local GPU hours, 2,743/3,600 remote GPU seconds, 12,164/18,000
updates and 242.946/300 GiB retained. All 24 outcomes and identical-checkpoint
success on both doors remain required; previous versions remain separate.


The common background now includes explicit 85-degree initial-pose calibration
surveys for right, left and development doors. Each preserves the original passive
hinge dynamics and rollout starts; no calibration motion becomes imitation data.
The current target is excluded independently before historical static free volumes
are reused. Complete convex ray volumes and cached certificates retain their
original acquisition clocks and charge every new position/radius error.

The observed handle/relief envelope has a finite tangent extent. Outside its
entire enlarged extent, the independently audited one-sided Panel front bound is
0.5 mm. This is a physical model bound, not visual accuracy or whole-Panel
coplanarity: the latter is rejected by inward recesses of12.52/66.87mm. Actual
finger CAD ordering, finite opaque material, complete14-shape sweep, response,
full2s stopping and self-clearance remain separate required proofs.

Free-motion receipts explicitly prepay normal uncertainty growth and relative
motion over the control interval; post-step checks spend any excess. Raw RGB-D acquisition and control run at60Hz, native feedback at120Hz, and
policy/geometry consume images at10Hz. Actual observation ages remain explicit.
Certain observed separation ends pending acquisition bookkeeping while preserving
identity, load history and latched stops. Conservative possible contact still has
the original3s/30mm acquisition limit. The5-degree contact orientation criterion
applies when contact is possible, rather than during certainly separated approach.
Every operational contact result, including acquisition state, is now retained
separately in `operational-contact.jsonl`.

The first complete replay stops at61.0167s on the right door: conservative normal
force lower bound18.73mN versus the operational20mN maintenance cutoff, despite
native intended contact. The stale acquisition deadline also overwrites the real
loss reason. Both defects are preserved. The corrected operational contract uses
a strictly positive force lower bound to maintain previously confirmed contact;
new acquisition still requires30mN for100ms. This detects contact presence rather
than duplicating the independent evaluator's unchanged20mN loaded-task criterion.
Zero/negative bounds still stop, and completed acquisition clears only its own
bookkeeping; lost/unavailable states cannot be relabeled acquisition failures.
Both complete retained replays and their independent detection metrics now pass
the bounded operational-path check below; learned closed-loop execution remains
a separate requirement.

Complete before/after-command replay now passes both corrected train paths through
their final endpoints, using actual proposed commands, calibrated native load,
current local geometry, full14-shape sweep, self-clearance and stop/response
proofs. There are no latched losses or admission refusals. The evidence is bounded
to retained right-dim/left-bright trajectories; new learned states still receive
current operational checks and may refuse.

| Metric | Right | Left |
| --- | ---: | ---: |
| Complete samples | 7230 | 7757 |
| Available monitor samples | 7224 | 7751 |
| Confirmed-contact samples | 6391 | 6921 |
| Initial confirmation misses | 6 | 6 |
| Instantaneous-force mismatches | 6 | 7 |
| Contact decisions outside physical loaded criterion | 1 | 0 |
| Confirmation delay | 100ms | 100ms |
| Longest instantaneous-force mismatch | 50ms | 50ms |

The six unavailable samples per door are explicit native-identity warmup. The
independent join uses actual HDF5 transition endpoints because teacher-relative
time starts25.0167s after the simulation clock origin; one pre-teacher endpoint
per door stays truth-unavailable. Brief zero-force push gaps remain physically
loaded under the unchanged causal-window rule. The single contact decision below
the physical loaded criterion occurs during right release. Every discrepancy is
retained in `operational-contact-full-v41/contact-metrics.json`; instantaneous
contact, bounded-window operational contact and physical scoring stay distinct.
These are retained-input checks, separate from learned execution.

The first full-protocol learned outcome is now retained: ACT A2 right/dim stops
on `observed_contact_orientation` at28.7667s after214 actual nonzero deltas. The
robot moves0.2414m;213 applied-target changes and all input/control/native/evaluator
joins pass. The independent surface gap is31.78mm and true orientation error1.80
degrees, while operational orientation uncertainty exhausts the unchanged margin.
There is no physical contact, controlled opening, hold or release. Physics continues
for2s after stopping:2.78mm maximum tool travel, no violations. This is a complete
failed panel outcome, not an initialization refusal or a successful task.

The fixed panel is frozen at source `2df0e54`, all four returned checkpoint hashes,
the unchanged9000-tick protocol and original expert references. The first failure
counts in its original slot; five initial outcomes now demonstrate all4trained combinations through the real
runner. ACT A1/A2 right-nominal issue209commands each before orientation stops;
Diffusion A1/A2 issue72/122commands before measured acceleration exits from the
verified load domain. All five outcomes have0controlled opening and passing
retention/continuing-physics-stop audits. The remaining19slots are held at an
unallocated case boundary for diagnosis; no failure is removed or retried.
All24final-version outcomes and an identical-checkpoint two-door success remain
required. There is still no learned contact/opening/hold/release success.

The saved ACT RGB-D replay identifies the orientation refusal precisely: current
normal alignment23.05mrad plus observed normal uncertainty26.12mrad plus the
40mrad **future controller-response** bound exceeds the unchanged5-degree limit.
The tool pose in this check is already measured by current proprioceptive FK;
its frozen calibration bound is only26.99microrad. Independent actual tool states
across all five learned runs give maxima0.821micrometer/1.680microrad, inside the
preexisting bounds. The corrected check uses that measured FK contribution and
preserves the13mm/40mrad future response bounds in admission. Missing FK bounds
retain the previous conservative fallback; physical criteria are unchanged.
The replay reproduces the old refusal at28.7667s and no corrected orientation
refusal in the retained prefix. The proposed temporal-normal cache is not adopted.

The corrected full-v2 source `18dd332` passes the previous orientation refusal,
then stops at28.9167s after223 actual learned commands: elbow acceleration
6.3549rad/s² exceeds the original5.5411rad/s² load envelope. There is still no
contact or opening. Its consumed-input audit and2s continuing-physics stop pass.
This remains one failed full-v2 outcome; the next final version still requires
all24outcomes. The five full-v1 failures remain separate, never mixed with
favorable later-version results. All checkpoints remain the genuine3000-update
files; no weights, training data, normalization, seed or controller changes.

The first complete allocation is331.13s including conservative tail, with0.365GiB
retained. Its measured policy cycle mean/p95 is245.26/826.38ms. An all-early-stop
scenario would cost2.21GPUh for24outcomes; extrapolating this limited service to
all full horizons gives16.54h, exceeding the remaining budget. Neither scenario
is a guarantee or measured loaded/left runtime. The prior8.64h estimate is
superseded; actual cumulative guards apply before every case and during its
bounded allocation. Full protocol horizon and failure accounting are unchanged.

Closed allocations through the first full-v2 outcome total5.09503/15 local GPU hours, including every failure
and the observation replay whose independent USD join was repaired afterward on
CPU. Remote consumption remains2743/3600GPU-seconds and optimizer updates12164/18000.
Active allocations are provisional in the live ledger. Consumer-audited removal of regenerable flattened
visual features reclaimed24.13GiB. Exact labels/masks, raw complete recordings,
checkpoints, statistics, failures and reports remain retained; current whole-pilot
storage is241.32/300GiB after a further11.48GiB lossless
container repack of one older failed teacher recording. Every dataset byte and
attribute is verified before atomic replacement; old/new file hashes remain in
`preserved-failure-lossless-repack.json`. No raw sample, failure or eligibility
changes. The feature-cache audit and regeneration command are retained separately
in `derived-feature-cache-removal-audit.json`. No GPU allocation was used for
these CPU numerical integration checks or storage repairs.

The six retained learned prefixes fit the original joint-residual and weak
normal-error coefficients under the earlier frozen stop-acceleration vector.
They also expose three stopping acceleration exits and a24.33mrad head movement
beyond the old held-joint assumption. More conservative interpolation acceleration
and held-joint allowances contain all six stop tails with the original radial and
directional boxes. Six fixed independent robot-only coupled-pulse stops also fit
those unchanged boxes. All18sampled actual teacher proposals pass the larger
uncertainty cover; minimum self-clearance/sweep margins are0.358/0.034mm.

The new robot pulses expose an additional defect: the original joint-torque error
floor misses up to1.15mNm at interior states. A floor fitted only to those pulses
still fails preserved loaded robot traces; a floor fitted only to the original
robot traces still fails five interior pulse trials. Both failed candidates remain
preserved. The conservative union uses the original robot-only29-stop data and
six interior pulse calibrations, retaining25% plus10micro-Nm reserve. It increases
declared uncertainty, without changing dynamic coefficients or physical criteria.
All six actual learned prefixes independently fit this floor. No pulse is claimed
as independent load validation after it becomes calibration data; its stopping
holdout remains independent of the stop-box fit.

The optional acceleration extension is permitted only when the original joint-box
wrench calculation gives a force upper bound at most the existing1N weak-load
ceiling. Stronger bounded loads retain the original acceleration envelope. Source
`98e27d5` includes behavioral tests for this distinction. The revised finite-domain
load/stop recipe passes both complete causal paths in `operational-contact-conservative-domain-v42`:
right7230samples/7224available, left7757/7751, with only the same six identity
warmup queries unavailable per path and no refusal. Force upper bounds peak at
9.99397/7.97662N, inside the unchanged10N stop domain. Independent native joins
retain100ms confirmation, six initial misses per door,5/7instantaneous-force
mismatches lasting at most33.3/50ms, and zero detections outside physical Loaded.
One pre-teacher endpoint has unknown independent truth per path. These remain
finite retained-path component results, not learned success or global guarantees.
Full-v3 is frozen and queued behind shared GPU availability;
no rollout allocation begins on missing/failed qualification. Teacher/controller,
all four genuine3000-update checkpoints, and physical task criteria remain unchanged.
The robot-only holdout costs35.65allocated GPU-seconds including tail, bringing
local consumption to5.10494/15GPUh; remote and optimizer totals are unchanged.

The first full-v3 ACT A2 outcome executes230commands, then stops at29.0333s
before contact. At29.025s, elbow acceleration7.1064rad/s² is outside the original
main domain, while force upper1.01175N exceeds the unchanged1N weak branch.
Independent physical force remains zero; that truth is used only after execution.
Retention and the2s stop pass, including the existing continuous stop boxes.
Charge331.04GPU-s and0.368GiB retained. Mean reached-state cycle237.06ms gives
15.40GPUh for23full horizons as a scenario, not a budget guarantee.

Previously excluded independent loaded robot samples fit the unchanged conservative
joint-error floor:13extended strong-load samples, including three elbow samples
up to10.08398rad/s². A targeted main elbow bound12.8rad/s² adds25% plus0.1rad/s²,
rounded upward0.1. No force limit or error coefficient changes. All other joints
retain their main limits and existing weak branch. The seven actual learned
prefixes and stops pass. Both full teacher recordings remain wholly inside the
original domain, proving every v42 computed load/contact/admission signal unchanged;
the geometry acceleration cover was already14rad/s² at the elbow. Completed full
checks are reused. Full-v4 freezes this one-value calibration correction at the
same source/checkpoint/protocol and queues a new first integrated outcome. The
one full-v3 failure remains separate; its23unallocated slots are held.

The full-v4 ACT A2 outcome executes 231 commands, passes the previous load
refusal, then violates actual joint speed at native 29.0417 s: elbow 0.51826 rad/s
against the unchanged 0.51 limit. No contact or opening occurs. The complete
failed recording and two-second effective stop remain retained. This is a real
physical criterion violation, not just unavailable calibration evidence.

The selected IK/gains are preserved; a targeted 10% reference-speed headroom
corrects the demonstrated target-rate/tracking mismatch. Fixed causal robot-only
replay reproduces 0.51829 rad/s at the old reference cap and measures 0.47323 rad/s
at the new 0.45 cap, with zero candidate speed violations. The same error floor,
full-link stop boxes, held-joint bounds and 13 mm / 40 mrad response limits pass;
measured response maxima are 2.162 mm / 13.608 mrad. The correction is inactive
on all 8,729 right and 9,256 left teacher target transitions, whose peaks are
0.31110 / 0.27169 rad/s. The complete teacher-path evidence is reused without
new demonstrations or repeated geometry studies. A new real learned integration
and all 24 final-version outcomes remain required; replay is not learned behavior.

The baseline prerequisite finishes physics before a NumPy count serialization
error. Isaac shutdown returns process exit zero despite that error; its helper
incorrectly writes a completion marker without a report. The false marker is
rejected and preserved, and no qualification is accepted from it. Explicit integer
conversion and a required completed-report check repair the infrastructure. The
already completed baseline is recovered from raw signals on CPU; only the
unexecuted candidate runs afterward. Both allocations are charged and all failed
source/log/raw artifacts remain under `robot-reference-headroom`.

Earlier investigations below preserve their original decisions and failures.
Their pending-status statements are superseded by the current integration above.

## Finite material component

The current diagnostic model retains a connected measured RGB-D material mask,
its complete normal-depth interval and known holes/disconnected background.
Every metric cell projects its entire normal column through each real inspection
camera; this spends local parallax instead of attributing a distant grazing
view's worst error to the entire leaf. The measured planar core conditions
orientation at the existing1mm point-error resolution; the wider relief/geometry
tolerance remains unchanged. This is a bounded shallow opaque relief model,
not a claim that an occluded interface is directly measured or solid everywhere.
Its contact interval remains separate from physical scoring and full robot space.

The1mm computational grid preserves positive original full action-margin bounds
at all7224 right and7751 left valid retained queries, including between captures:
minimum right3.20mm and left99.61mm. The20mm stop reserve and physical evaluator
are unchanged. The preceding2mm grid retains three right refusals; all earlier
failed material variants remain preserved. Native tracking is unchanged. Known
holes and separate background, complete depth-column parallax and future
inspection clocks have essential regression coverage. This is numerical component
evidence. The maintained provider now retains only declared inspection captures
and their actual publication clocks, attaches the frozen cover to the observed
candidate, and passes its finite support/depth interval to the actual-motion and
contact producers. The complete normal interval also reduces free-motion
separation; it cannot silently become a zero-thickness plane. This path is opt-in
through the operational geometry recipe; loaded whole-space/contact qualification
is still pending.

Unassociated straight-edge and upright-edge trials worsen local rotation and
are rejected; neither remains active. A tighter6.5mm isotropic stop candidate
also fails the complete29-stop continuous audit, including a proposed slower
Cartesian-speed subset. No global stop bound is reduced. A separate calibration
of complete finite-finger tangential and inward travel is retained for subsequent
qualification; it does not yet authorize contact admission.

## Directional stopping component

A complete link-local stop box is calibrated from all29 robot-only stops, using
the same continuous per-native-interval velocity/acceleration construction and
all eight shape corners. The existing radial bound remains; their intersection
is covered adaptively without discarding physical motion. All four previously
frozen moving-door holdouts pass the boxes independently. The producer spends
possible link rotation between the current control state and stop onset.
Whole-volume, missing-interior and source-query regressions pass.

This covers the four original unsupported volumes using existing static surveys.
The complete retained sweep exposes further near-contact volumes, so whole-path
qualification remains open. The left30s numerical state admits after8367 cover
queries, while contact35s still refuses; this is not learned execution. Historical
raw RGB-D offers background certificates for remaining examples, but those are
not current clearance: current moving-foreground exclusion and designated distal
contact remain necessary. The parked-arm camera preflight does not diagnose
an occluding robot at those ray centers, so no new survey allocation is launched.

## Observed open-pose background calibration

The right35s failed loaded-sweep sphere is not certified even at radius zero by
any earlier right-scene view, although its center is269mm from the independently
measured static frame. An explicit85-degree initial-pose survey exposes the
background using the same head ZED and unchanged passive hinge dynamics. The
measured10.162mm sphere has107.6mm free clearance; the original radius and all
camera/depth bounds remain unchanged. The25s survey keeps zero door drift and
passes the2s continuing-physics stop. Its96.82s workload plus conservative tail
is charged; no demonstration slot or policy outcome is claimed.

`scripts/pilot_workspace_survey.py --asset-root <prepared-root> --asset-id
<declared-train-or-development-id> --view front --initial-opening-deg 85 --output
<new-directory>` reproduces the calibration. Inspection measures drift from its
declared initial pose; closed-start teacher/runtime behavior is unchanged.
Historical background groups may name per-record `sample_times_s`, preserving
short survey captures and longer episodes without inventing or truncating clocks.
The candidate right calibration is merged only into its corresponding calibration
condition; all conditions must still support each background volume. New sparse
right-path samples through120s admit; later rows reveal a separate floating-point
schedule comparison at130/140s. The existing1ns representational clock tolerance
now also applies when constructing a local receipt; an extra control tick still
refuses. Previous numerical refusals remain retained. No final panel has started.

## Static ray-query cost

Historical background can now certify a complete convex stop cover against real
free ray pyramids. It checks every half-space at every vertex and erodes the
whole volume for robot/model/camera errors; a separately proved current moving
target exclusion remains mandatory. Reused pyramids retain their original depths
and clocks and pay the new complete vertex/radius camera lever on every query.
This avoids replacing thin oriented volumes by large spheres. Missing interior
pixels, finite view bounds and unknown space still refuse. The numerical right30s
nonzero-motion prefix improves from35s to about7.8s per adjacent control tick;
that remains too costly for the full panel. Sparse loaded-path checks still find
unsupported volumes on both train doors, so these components do not establish a
qualified operational path. Static-frame diagnostics are independent evaluator
evidence only and do not enter admission.


Owned static workspace returns now have an immutable spatial index. Occupancy
queries retain the original projected pixel rectangle, inclusive Euclidean test,
free-ray margins, unknown handling and source clocks. Current captures keep the
direct path. Fixed-input random/boundary comparisons are identical. Retained26s
admission calls reduce from0.263/0.368s to0.112/0.169s for right/left;27s calls
remain0.702/0.878s. These exclude index construction and are numerical service
profiles, not end-to-end or real-time qualification.

## Current observed-space integration

Continuous recorded robot sweeps use calibrated CAD cells, joint-axis cell levers
and independently measured stop excursions. Static workcell rays keep their original
capture/publication clocks. A smaller ray pyramid may certify a larger requested
sphere only by spending its measured radial clearance on the complete radius growth;
unknown or occupied volume is never filled. Regression checks cover raster-border
inconclusiveness, real unknown holes and occupancy. This corrects a constructive
coverage failure but does not qualify the remaining unsupported fingertip sweep.
An adaptive union of free covering balls now also certifies the complete original
sphere when one constant-depth pyramid is inconclusive beside an oblique surface.
Every intersecting octree cell must be certified; missing interior rays, occupancy
and bounded-search exhaustion still refuse. The original requested ball and source
clocks are retained in the cache. A retained failing 36.5 mm sphere is covered by
41 subqueries without reducing its stop envelope; full recorded-path validation
remains separate.
Front, back and one lateral survey are retained; the lateral trial did not resolve
that gap and its unadopted source remains with the evidence.

## Corrected boundary and implemented contract

The [[../implementation_phases/phase-6-0-operational-perception-and-contact|existing phase 6.0 contract]] already distinguishes physical 0.1 mm /0.02 N scoring from
operational detection. The earlier monitor instead demanded tool-center/plane
agreement plus >=4 mm uncertainty within 0.1 mm. That requirement was erroneous.
The [[b1-observed-contact-certificate|preserved component study]] measures a useful
geometry discrepancy and metric mismatch; it does not prove operational contact
infeasible or require a 0.1 mm depth sensor.

`ObservedContactMonitor` now uses two calibrated fingertip geometries, explicit
separation intervals and uncertainty, verified local identity/footprints/sweep and
robot-only load, plus causal confirmation/loss and proposed-motion consistency.
The monitor may stop; it never supplies task motion. A commanded withdrawal permits
expected unloading; unexpected load loss or contradictory support latches a stop.
An unresolved recipe, load model, identity or relevant space still refuses.
Native 120 Hz normal-load intervals now support a completely covered causal
contact window, while instantaneous upper-force protection remains separate.
Missing intervals or inadequate history do not establish contact. Whole-arm
velocity/acceleration terms can express calibrated residual uncertainty without
forcing the moving-state error onto stationary contact; no new calibration or
verified flag is inferred from this interface.
The software contract and consumers are checked numerically; calibration and
independent false-admission/miss/delay/coverage measurements remain pending.

The maintained tracker also keeps two distinct statements: the original finite
material cover transported by current rigid tracking and its complete uncertainty
budget, versus current direct depth compatibility at the contact pixels. Occlusion
no longer erases the former or gets presented as directly measured contact. Both
statements are explicit in diagnostics. Missing original coverage, pose/identity
bounds or relevant swept-space evidence still refuses; force/contact admission is
separate. No hidden interface shape is reconstructed or certified from depth.
No physical threshold, gain, door dynamics or qualification flag changes here.

A local action may carry a separately eroded whole-robot sweep/stop residual.
Its positive finite margin remains independent of the finger material margin:
material pose, response and contact-model uncertainty spend only finite material
support, while the robot certificate must already include its own motion, scene
and stop errors. Missing, nonpositive or unknown sweep evidence still refuses.
This interface does not create a qualified producer or establish live admission.

Continuing-physics stop sidecars now retain the measured full-robot anchor and
control states, absolute tool poses, original native joint/torque/model/Jacobian
streams and separately named native evaluator force/gap/forbidden streams. Earlier
sidecars retained only aggregate travel/speed/force and cannot establish per-link
stop motion from actual-door checks. This retention correction changes no stop
command, duration or physical acceptance. Physical validation of the new sidecar
remains pending.

The maintained robot geometry now supplies a complete coupled-velocity sweep:
measured joint velocity defines the nominal curve, endpoint-chord curvature is
bounded analytically, and an explicit acceleration box covers departure over each
control interval. One interval's FK is reused across adaptive cells. Stops and
scene/model uncertainty remain separate; no current-frame bound is extended by
assuming zero motion. Dense coupled-rotation/acceleration regressions pass; actual
operational admission and full-path support remain pending.

`RobotSelfClearance` evaluates full authored convex meshes and exact sphere
support. Shared ancestor motion cancels from relative pair bounds; nonshared
motion and a supplied model reserve erode separation. Existing importer pair
filters apply to owning rigid bodies and all fixed descendants, matching the
retained authored USD. The earlier diagnostic applied them only to named links,
leaving a spurious nanometer-scale cross-jaw assembly constraint. No simulation
filter or physical criterion changes. Full recorded-path verification with the
maintained producer remains separate from the numerical tests.

A complete proposed-footprint audit reveals that the single native seed view is
insufficient even where earlier inspection views measured the same coplanar material.
A local-only material surface now reuses same-identity coplanar observations and
includes their measured plane disagreement. Native/A3-A4 seed geometry remains
unchanged. Finite footprints still require actual observed pixels; holes and gaps
between measured regions remain unsupported. Packed-mask ROI queries preserve the
same raster proof without repeatedly decoding full images.

The development stream also exposes a current-plane association failure: the
largest visible face can contradict the original material while a smaller face
remains consistent. The local constraint now applies its existing material-depth
consistency check before area ranking. Missing support and equal compatible-area
ambiguity still refuse; native poses and physical thresholds remain unchanged.
Numerical coverage/identity/ambiguity regressions pass. All fixed native streams
must be replayed through the corrected local path before operational qualification.

## Independent learned-rollout scoring

`pilot_physics.py rollout --expert-reference <frozen JSON>` now scores through an
independent evaluator. The reference and prepared-door geometry enter only the
post-execution scoring/recording callback. Policy observations, robot load,
admission and deployed control receive none of those values. The recorder retains
FK of the actually submitted joint targets for A1 and A2, as well as requested
commands, measured tool/robot state and native feedback with their original clocks.

The scorer reuses the unchanged 0.02 N /0.1 mm physical contact window, 0.5 s
controlled sustain, 10 mm material-position and 5 degree orientation criteria,
finite distal footprint and native safety limits. A complete three-second active
loaded-contact window above 80% of the frozen matched reference establishes hold;
commanded withdrawal after hold plus unloaded >=10 mm separation and valid actual
target tracking establishes release. Raw angle, controlled angle, hold, release,
unexpected contact loss and actual learned commands remain separate. Missing native
samples are explicit errors. Uncommanded/unloaded coasting and a latched stop cannot
supply success. The corrected teacher references reproduce both physical loaded
streams exactly and pass these gates when their executed commands are supplied for
a reference-only scorer audit; this is not learned execution or a panel outcome.
The live recorder and operational producer remain pending.

## Scope and budgets

A3/A4 operational hinge/bootstrap/release qualification is deferred. Before new
recording, resolve only indispensable shared retention: synchronized raw RGB-D,
calibration, proprioception, actual/commanded poses and timing, native feedback,
episode boundaries and supervisor-only geometric/phase annotations. Keep raw
reuse, reconstructible supervision and qualified observed execution distinct;
this pilot is not the final matched A1-A4 dataset.

The recorder now also retains synchronized actual calibrated tool-link pose and complete measured
robot joint positions/velocities in a separate `robot_state` stream, including
inspection/reset/terminal rows. This closes the explicit-pose/full-proprioception
retention omission without changing the observed-only policy schema. Existing raw
RGB-D, calibration, command poses/targets/timestamps, native 120 Hz feedback and
supervisor-only phase/material annotations are retained. Numerical stream and
dataset checks pass; the next physical diagnostic exercises this addition.

To respect the 300 GiB allocation while retaining all raw data, `pilot_offline.py
prepare --reference-raw` now exports only learned views with explicit original raw
paths after checking episode identity. The default still copies every raw group,
including `robot_state`. Reference exports are not self-contained archives: keep
the original synchronized recordings separately. Training/loading uses only the
observed features/actions and does not require privileged raw groups on Gilbreth.

Six of twelve prerequisite cases are consumed, six remain. Prior monitored
workload spans plus conservative tails total 0.164 one-GPU hours; charge an
additional 120 s reserve for brief unmonitored numerical preparation. Approximately
2.803 of the three local GPU-hours remain under this conservative accounting,
excluding idle preflights. The earlier 2.17 hours was elapsed conversation wall,
not measured GPU use. Reference candidate 2 is already included in the encoder
checks monitor. Preserve contaminated/failed work in accounting and all outcomes.

Other unchanged ceilings: six fresh demonstration episodes /300 GiB; 17,836
remaining optimizer updates after 164 infrastructure updates; one remaining
Gilbreth job and about 0.983 allocated GPU-hours; planned learned panel of
2 doors ×3 predefined conditions ×4 models, currently unexecuted. Evidence-driven
iterations are allowed within the remaining allocations; no artificial per-candidate
one-attempt restriction. Each new physical case is counted. Request a small exact
extension only when a necessary next experiment exceeds the remaining budget.

## Current execution plan

Use saved teacher traces to test compression/reference/door-motion/hold hypotheses.
Implement the smallest supported common correction. Verify actual tracking, load,
joint speed, controlled sustain, hold, release and continuing-physics stopping.
Establish matched-condition expert references independently before learned scoring;
historical failures retain their original denominators and scores. A new reference
cannot turn invalid contact/hold/release into success.

Then collect valid bounded demonstrations, retain failed episodes separately,
train the four real shared-encoder cells on immutable source/data using Gilbreth,
and execute the fixed learned panel through maintained observations/admission/IO.
Synthetic checkpoint loading and pre-actuation refusal remain infrastructure/safety
evidence, not learned task execution or pilot completion.

Ignored `outputs/evidence/b1-shared-learned-pilot-20261016/end-to-end-resume/`
retains the explicit authorization, baseline TODO, budget reconciliation and
software validation. Active implementation dependencies remain in `TODO.md`.


## Teacher-control development

The preserved contact-ramp traces show that the first loaded tick is already a
brief 7.7–9.7 N impact. Continuing the ramp then maintains 3–6 N, while the old
reference advances to each coasting measured angle. Across the three identities,
descriptive force/normal servo-deflection ratios reach 2,574 N/m. These are
closed-loop descriptive values, not a calibrated material stiffness.

Case 7 tests stopping that ramp at first load, removing excess compression,
reference-clock advancement and a captured hold anchor. It reduces peak door
speed to 1.26 deg/s, but the inherited normalized compression gain becomes an
aggressive bidirectional controller: contact chatters and no 0.5 s valid sustained
window forms. Measured wrist-X speed reaches 0.521 rad/s and the new actual-speed
check stops immediately; release is correctly unvalidated. Preserve the failed
53.10 degree raw opening without calling it controlled progress or changing its
historical reference. Continuing-physics stopping settles after the latch.

Case 8 changes only the force-admittance gain to 0.0005 m/(N s), retaining the
6 mm/s cap, 0.1 N target and all physical criteria. A 3,000 N/m design envelope
above the saved descriptive maximum and the 100 ms feedback window give a
lag/gain product 0.15 rather than 18 near the previous setpoint. This motivates
numerical stability; physical contact, holding, release and speed still require
the new case. It is not a learned-policy trial or expert reference.


Case 8 establishes 51.74 degrees of physically valid sustained push on the right
train identity (84.28% of the preserved historical denominator), with actual
joint-speed peak 0.320 rad/s and valid release. The complete teacher still fails:
its fixed hold target loses contact while the door coasts. This is not a successful
teacher or policy and does not justify a new smaller denominator. Eight of twelve
prerequisite cases are consumed; four remain. No new demonstration or real training
has started.

Zero-force intervals in the three preserved failures independently fit passive
angular decay time constants of 1.556 /1.922 /1.466 s. A captured fixed pose cannot
brake a unilateral push interface. The next bounded correction will prescribe a
common 2 s decay of the angular reference from the measured transition speed,
without refreshing it from the subsequently coasting angle. This uses privileged
teacher development evidence only; unchanged loaded-contact/hold/release scores
must still pass. Door dynamics remain untouched.


The maintained observation/admission interface now distinguishes selected usable
material coverage from the actual tool's relative-footprint query. A parked arm
need not already touch the selected zone; every submitted trajectory still needs
its own actual footprint/sweep evidence. The runner's initial health check may
precede its first receipt, while loaded checks and dispatch remain receipt-gated.
Both 120 Hz robot velocity samples are checked, and a measured-anchor stop occurs
before perception cleanup, including cleanup failure. These are numerical contract
checks; no operational admission recipe is thereby qualified.


Case 9 keeps valid contact throughout nearly all of the planned deceleration,
reaching 53.18 degrees valid sustain, but a near-straight-elbow singularity drives
actual speed to 0.522 rad/s and correctly latches a stop before release. The
smallest Jacobian singular value drops from 0.0396 at 118 s to 0.00241 at 121.2 s;
pose reachability alone had not reserved dynamic conditioning. This failed cycle
remains excluded from imitation and cannot establish a new expert denominator.
Case 10 tests a common causal conditioning-trend reserve over the hold horizon,
using twice the unchanged DLS damping as the 80% weakest-task-gain boundary.
No door-angle cutoff or physical scoring threshold is changed.

The diagnostic raw file retains 7,274 synchronized RGB-D rows, 14,546 native
feedback samples and all 20 measured robot joints. Independent FK agrees with
actual tool-link pose within 0.94 micrometer. Images are visually inspected. The
17.70 GiB file retains inputs for supervisor reconstruction, with full A4 termination invalid because release is missing. It is a failed
teacher diagnostic, not an imitation demonstration or operational A3/A4 result.

The saved complete inspection prefixes also show 0.0755 rad of arm self-motion
under Cartesian hold, despite a stationary tool. Teacher and runner now preserve
the same configured joint anchor during the common scan. Numerical checks pass;
case 11 physically verifies the scan anchor (maximum actual arm speed 1.83e-7
rad/s, tool displacement 0.12 micrometer). The complete teacher under that exact
condition still requires valid release.
Keep all old Cartesian-inspection recordings unchanged and establish fresh matched
expert references before learned evaluation.


Case 10 confirms the conditioning reserve prevents the speed violation: measured
peak is 0.389 rad/s, release is valid and sustained controlled progress is 51.99
degrees. The full cycle still fails holding: the prescribed angular decay loses
loaded contact. A push-only interface needs pressure-consistent material tracking
rather than an angular decay independent of force. Ten prerequisite cases are
consumed, two remain. No demonstration, matched expert reference or learned
execution is claimed from these development failures.


The pressure-hold candidate preserves the material-relative goal lead and reduces
the common normal-force target from 0.1 to 0.05 N. Its angular following is only a
coordinate for maintaining light pressure on a unilateral interface; loss of
loaded contact beyond the same 0.25 s budget as push latches a stop. Independent
scores still require valid contact and the final contiguous hold window. Report
actual settling motion, rather than calling a free-coasting angle controlled.
The conditioning reserve now covers the complete remaining hold and release,
and case 11 includes the corrected full joint-parked inspection. All 6,270 push
ticks and the final contiguous hold window satisfy unchanged contact/pose criteria.
During the 3 s hold the door advances 1.75 degrees under positive load; final
mean normal force is 0.281 N rather than the 0.05 N command. This is contact
maintenance, not proof of stationary holding or target-force accuracy.

Case 11 fails release: the measured elbow peak is 0.526 rad/s at 0.167 s into
withdrawal. The latched stop is then followed by another release command, causing
`motion_stopped`; no valid release or continuing-physics stop verification exists
for that failed cycle. The release loop now exits immediately when stopped.
Case 12 doubles the common release duration to 6 s, retaining its geometric path
and zero-start cubic profile, and reserves the complete 3+6 s conditioning tail.
Its physical result is pending; no threshold, dynamics or gain changes are made.

The case-11 supervisor ended before its simulator child. Preserve the unmonitored
tail, charge its entire 553.67 s case wall plus startup/tail, and exclude it from
performance comparisons. The next supervisor is detached from the expiring
terminal session and retains all three shared locks, 5 s interference checks and
a finite 1,200 s workload ceiling. One launcher-only descriptor-forwarding error
is preserved before simulation creation; it consumes no physical case. Eleven
of twelve prerequisite cases are consumed before case 12 starts.


## Completed right-cycle verification and shared raw audit

Case 12 completes the right train cycle with unchanged physical scoring: 51.2603
degrees sustained controlled opening, valid final hold and release, 9.6977 N
native normal-force peak and 0.32018 rad/s measured joint-speed peak. Independent
saved-contact/actual-pose reconstruction agrees on every validity tick, with no
forbidden contact. Release clearance is 25.15 mm. The subsequent 2 s physics
continuation keeps stopped targets fixed, travels at most 0.0335 mm and ends at
0.000153 rad/s. The hold advances 1.55 degrees under maintained positive load;
stationary-angle holding and target-force accuracy are not claimed.

The 83.50% ratio to the historical 61.3931-degree reference is descriptive only.
A valid matched-condition expert pair is still required before learned evaluation.
This is a privileged teacher diagnostic, with zero learned commands and no
imitation promotion. The same solver, gains, placement and teacher apply to left.

The complete raw retains 8,730 synchronized observations, 17,458 native feedback
samples and all 20 measured joints in 20.97 GiB. Independent double-precision FK
agrees within 1.03 micrometers and 1.92e-6 rad. Supervisor-only A3 reconstruction
is exact to numerical precision; 311 A4 segments reproduce recorded goals within
0.999 mm /0.455 degrees. Inputs needed for later supervision are retained and
numerically reconstructible for this complete episode. Observed A3/A4 initialization
and admission remain unqualified; this is not the final matched dataset.

All twelve original prerequisite cases are consumed, preserving all failures. The
user authorizes the minimal extension to fourteen total cases: common left-cycle
verification and independent robot-only dynamic load/stop calibration. The local
three-GPU-hour ceiling is unchanged; about 0.7905 hours is charged before the
queued observed-only static replay and left verification. No fresh demonstration,
real training update or learned execution has occurred. The static replay uses
the existing full chronological scan, the maintained frontend and observation
fields only; it creates no additional Isaac case or A4 initialization requirement.


The current full right raw also updates the campaign cost proposal. Its 147.48 s
simulation takes 570.64 s case wall and 20.97 GiB raw. A single-condition
extrapolation gives 41.85 workstation hours /5.41 TiB raw for 264 production
episodes, and 133.15 hours /17.20 TiB for the future 840-outcome sealed evaluation.
These are projections from one valid right cycle, not corpus measurements or
permission to run either campaign. The provisional reserve becomes 220 workstation
hours and 30 TiB archive (including dense views and margin); 64 allocated A100
hours remains provisional pending real-data pilot throughput. No storage purchase
or allocation is made. The original historical cost proposal remains preserved.


## Common left verification and dynamic load-domain result

Case 13 independently validates the identical corrected teacher on the left train
identity: 52.5360 degrees controlled sustain, valid final hold and release,
7.6819 N native normal-force peak and 0.28440 rad/s actual joint-speed peak.
Every independently reconstructed validity tick agrees; forbidden contacts are
absent. Hold advances 0.709 degrees under maintained load. Release gap is
25.89 mm; the subsequent 2 s continuing-physics stop travels 0.0531 mm with
unchanged targets and final speed 0.000153 rad/s. The 90.83% historical-reference
ratio is descriptive; matched-condition expert pairs remain pending.

Its 9,257 synchronized observations and 18,512 native samples retain all 20
joints in 24.69 GiB. Independent FK agrees within 0.793 micrometers /1.72e-6 rad;
A3 reconstruction is exact and 1,358 A4 supervisor segments reproduce goals
within 1.000 mm /0.498 degrees. Both engineering cycles remain excluded from
imitation targets and do not qualify A3/A4 observed execution.

Case 14 completes the declared robot-only calibration/verification sequence:
100,848 native samples /840.4 simulated seconds, about 700 s case wall. Fitting
uses only unloaded paths, followed by independently timed paths, weak loads,
tangential loads, native peaks and four moving stops. The candidate covers
70,150/70,158 verification samples; unavailable samples and uncertainty remain
explicit. No calibration or stop qualification flags are promoted. Unloaded
stops travel 2.15/2.13 mm; retained 4 N loaded stops travel 24.56/26.16 mm and
fail the unchanged 10 mm criterion despite low final speed and fixed anchors.
Robot Jacobians and the existing impedance predict 23.58/24.74 mm equilibrium
deflection at those states, close to measured 23.28/24.80 mm. This supports a
compliance diagnosis, rather than reference drift or a successful stop claim.

All fourteen authorized prerequisite cases are consumed. Reconciled monitored
workloads charge 1.421 local GPU-hours before the next observed-only replay; the
three-hour local ceiling remains unchanged. No fresh demonstrations, real training
updates or learned commands exist. Saved-data analysis and numerical corrections
continue within budget; any further physical prerequisite needs an explicit
extension. Existing completed teacher, raw, encoder and transfer work is retained.


## Stop correction and continued authorization

The user removes the per-case count ceiling and authorizes necessary prerequisites,
including twenty or more cases, without new per-case approval. GPU, data and
training budgets remain unchanged. Counters and every failed outcome remain.

The stop-only shoulder-X correction in `26aa0de` passes case 15's twenty fixed
0/4 N tests across five states on each path: maximum stop travel 9.6815 mm,
rotation 0.870 degrees, final joint speed 0.000855 rad/s and return error after
load removal 0.0168 mm. Actual complete-trace speed peaks at 0.25755 rad/s, with
no actuator clipping. All three native 10 N envelope-exit pulses stop within one
control interval and travel at most 1.403 mm. Of 34,696 native estimates, 32,706
are available and all available bounds cover the independent reference; 1,990
envelope exits remain explicit refusals. No bound is refitted on these tests.

These moving-stop initial speeds reach only 0.0353 rad/s, so they do not certify
the full measured approach/task-speed domain. Case 16 prospectively replays the
same two command paths to their independently selected actual approach and
push/hold speed maxima, then tests the same stop without changing trajectory,
load calibration or physical criteria. Global qualification is not inferred
from a low-speed stop pass.

The observed-only right replay now initializes at 25 s and completes all 1,455
10 Hz captures through 145.4 s in 611.15 s model wall, without ground-truth inputs
or another Isaac case. Independent evaluation corrects the trace's separate
25 s inspection clock before scoring; the initial incorrect-clock audit is
preserved as an analysis error, not a physical failure. Native tracking alone
does not establish admission. Saved-data audits expose an unsupported arbitrary
material anchor, then excessive conservative frame errors at a supported anchor.
Finite-reference selection and separate frame/contact uncertainty accounting
are implemented and numerically validated; all missing bounds/space support
remain explicit. The same observed recipe is now being checked on the saved left
cycle. Full A4 initialization is not added as a prerequisite.

The case-16 preflight initially starves on repeated 12% utilization spikes from
whitelisted desktop graphics, with no compute or unknown graphics processes.
Its pre-physics wait is preserved and restarted without another physical case.
All three leases and foreign-process monitoring stay intact; the corrected idle
test requires 180 s without compute/unknown graphics, at least 8 GiB free, and
both current and 180 s mean utilization at most 10%. Sustained contention and
every foreign compute process still block. Desktop activity is not a GPU job
failure, and no other process is terminated.

Receipt revalidation previously rejected any uncertainty increase or local
clearance decrease, including 0.1 mm updates against ample remaining margin.
The same original finite-error calculation now rechecks the remaining interval
and spends positive error increases plus clearance loss against that margin.
Numerical checks accept bounded updates, reject exhausted/missing support and
retain the response/identity stops. This does not establish physical admission.

Case 16 preserves `stop_verified:false` and `motion_coverage:false`: all four
physical stops pass but the loaded reference replay fails its original speed
coverage. No corrected metric relabels that run. Case 17 tests a prospective
common 0.05 rad half-sine calibration excitation after 4 N preload at the same
source peak conditions. Both coverage checks and stops pass: right/left initial
speed 0.0727/0.0831 rad/s, maximum travel 4.598/4.506 mm, final speed below
0.000342 rad/s. Fourteen envelope exits remain explicit unavailable estimates;
the calibration flags stay false pending bounded operational qualification.

Case 18 is the first prospectively declared fresh demonstration, right nominal
seed 6200 with frozen source `722185e`. Independent reconstruction finds zero
physical-validity mismatches or forbidden contacts, 51.2603 degrees controlled
opening, valid final hold and 25.15 mm release separation. Peak measured joint
speed is 0.3202 rad/s; continuing-physics stop travel is 0.02185 mm over two seconds.
The 20.967 GiB raw record retains 8,730 observations, 17,458 native feedback
samples and all 20 robot joints. Clock joins and independent FK pass; A3/A4
supervisor reconstruction passes existing tolerances. This establishes raw reuse
and reconstructible supervision, not executable observed A3/A4 initialization.
The unchanged paired-reference rule combines the prior valid nominal diagnostic
and this fresh episode, yielding right matched reference 51.2603 degrees with
zero repeat spread. Historical 61.3931-degree reference/scores remain unchanged.

Case 19 acquires a separate 25 s front workspace survey using the same head ZED;
robot/door motion remains negligible and the two-second stop passes. Its actual
RGB-D rays cover every external covering ball of the first saved right approach
command, with a minimum certified free margin of 16.01 mm before the additional
operational uncertainty budget. Interior balls are separated only by calibrated
robot geometry. This audit uses unloaded measured stop bounds and does not
qualify the complete task or dynamic-door/self-collision coverage. The same front
survey is queued on the other train/development doors; a second observer view is
not required by this first result. Original sensor clocks remain retained.

The resource reconciliation now separates physical prerequisite cases from fresh
collection attempts, while charging every monitored workload. Case-count ceiling
is removed by the user's instruction; six fresh demonstrations, 300 GiB storage,
three local GPU hours and the prior remote/training ceilings remain unchanged.

The saved observed/load component audit exposed another error-accounting overlap:
local patch publication supplied already uncertainty-eroded clearances to a
consumer that separately charges the complete pose/calibration errors. Local
patches now publish the measured finite material model; the original eroded slip
query remains diagnostic and the actual action proof retains all uncertainty
terms. Numerical tests retain finite geometry, missing-support and narrow-margin
refusals; physical qualification is still pending.

Saved right native contacts expose about 83 ms of zero force at the hold handoff,
including one unloaded physical-score tick; the earlier final-half-second hold
criterion still passes and is not relabeled. The first hold endpoint equals the
last completed push endpoint instead of the next interval's endpoint. A common
correction carries the last commanded angular increment into the initial hold
lead, bounded by the existing 0.3-degree limit. This tests a concrete reference
continuity cause, without raising force targets or changing physics/acceptance.
Numerical continuity and mechanical-stop checks pass; an independent physical
case is required before treating the correction as demonstrated.

A componentwise residual-bound candidate undercovers independent loaded/moving
robot samples and is rejected. A coupled wrench-interval calculation passes its
screening but barely changes task hold availability, so it is not implemented.
A direct robot-tool-normal error bound is under investigation: fit uses only
unloaded robot calibration; independent known normal/tangential forces remain
separate. No candidate calibration or operational qualification flags are promoted.

The scalar normal-error candidate materially improves observed right push/hold
availability without changing force thresholds, but its unrestricted loaded-speed
calibration fails 34/3,204 available case-17 samples (maximum 0.0572 N undercoverage).
It is not adopted as a global bound. The implementation therefore supports an
explicit weak-load force-upper domain and retains the original conservative bound
above it. The prospective 5 mN scalar floor comes from case-14 zero/tangential
development noise, with speed/acceleration terms fitted only on unloaded robot
motion. Known-force independent verification and the new actual contact handoff
case must establish the final bounded recipe. Missing robot FK or partial optional
configuration refuses; neither model/stop qualification flag is promoted.

The fresh nominal left case independently passes: 52.5360 degrees controlled
opening, all 180 hold ticks physically valid, 25.889 mm release separation and
0.2844 rad/s actual joint peak. Two-second stop travel is 0.02076 mm. The 24.696 GiB
record retains 9,257 synchronized observations, 18,512 native feedback samples
and all 20 joints; independent FK and A3/A4 supervisor reconstruction pass. Its
unchanged paired-reference rule yields 52.5360 degrees with zero repeat spread;
the historical 57.8391-degree denominator/scores remain unchanged.

Both additional common front workspace surveys complete with no manipulation.
The observed-only reader now intersects all three raw ray banks without reading
asset identities or evaluator annotations and keeps their original capture clocks.
A coarse sphere per robot convex shape certifies eleven shapes at the first right
approach condition, while three shoulder shapes remain unknown. No action is
admitted by this result; refinement, dynamic-object exclusion, actual response and
self-collision remain separate requirements.

Source-level verification of the candidate weak-normal/strong-load fallback now
covers all 123,529 available independent native samples in cases 14–17, with zero
false contacts among 108,817 zero/tangential samples. All 2,017 envelope exits
remain unavailable and counted. The total-force upper bound is unchanged. These
are robot-domain load results, not physical contact-monitor precision/delay/coverage
or permission to execute through missing geometric/stop evidence.

The fixed 24-outcome learned panel alone projects 8.64 further local GPU allocation
hours using measured sensing and physics service costs at the 175 s maximum
inspection-plus-policy horizon. A concrete proposal requests 12 total local hours
instead of three; it is pending user approval, and current work remains within the
existing ceiling. Fresh collection, storage, optimizer and Gilbreth ceilings are
unchanged. The user removed only the prerequisite case-count ceiling so far.

Case 23 (`dae8e77`, physics-only privileged diagnostic) confirms the handoff cause:
all approach/push commands and prior motion match the baseline up to holding.
All 180 hold ticks now pass, with minimum native 0.1 s force mean 0.08217 N; the
14 isolated zero-force substeps never exceed 8.33 ms consecutively. The initial
analysis incorrectly required every substep to be positive; that extra requirement
is rejected in favor of the unchanged physical window. Independent physical
reconstruction has zero validity mismatches/forbidden contacts, valid release and
0.02056 mm two-second stop travel. Opening is 51.1111 degrees versus the unchanged
frozen right reference 51.2603; no smaller denominator is introduced.

Remaining demonstration collection uses this common corrected teacher. The two
already collected nominal records remain valid physical references/evidence but
are explicitly excluded from imitation because of the demonstrated supervisor
handoff pause. Their original scores and eligibility audit are preserved; the
new protocol selection is explicit. The four remaining prescribed lighting/seed
conditions and all 24 learned panel entries stay unchanged; no favorable retry
or evaluation-denominator reduction is used.

The pilot RTX camera publishes SDK intrinsics with half-integer sample centers.
Geometry now converts those intrinsics once at session ingress to integer pixel
indices; new calibration metadata declares `[0.5, 0.5]`, and the explicit pilot
recipe provides the convention for retained legacy recordings. Conflicting
metadata refuses. Raw SDK intrinsics, RGB-D, clocks and learned image preprocessing
remain unchanged. Static workspace maps are rebuilt from normalized rays; prior
native 3D tracking outputs are not relabeled as corrected.

Independent saved-data checks explain the systematic bias: four visible-floor
ROIs fall from up to 6.160 mm error to below 1.57 micrometers. Five other fixed
ROIs include occluders and do not establish floor accuracy. On the fixed sparse
train contact-plane samples, normalized geometry gives maximum plane errors
0.06844/0.00199 mm and normal errors 0.000268/0.000266 rad (right/left).
These are diagnostic numerical results, not a complete sensor uncertainty bound,
hardware qualification or observed-contact admission.

The corrected fresh right dim and left bright episodes independently pass the
physical and raw-reuse audits. Controlled openings are 51.1111/52.5461 degrees;
all 180 hold ticks pass on each door. Actual joint peaks are 0.3202/0.2844 rad/s
and two-second stop travel is 0.02056/0.02077 mm. Both retain complete native
feedback and 20-joint states with reconstructible supervisor-only A3/A4 labels.
The fixed nominal expert references remain 51.2603/52.5360 degrees.

Saved observed plane/edge constraints with all measured native inliers reduce
maximum translation residuals to 28.11/11.14 mm on the fixed right/left sparse
samples; unrestricted native pairs contain extreme outliers and are retained as
a rejected diagnostic. Conservative conditioned bounds and missing-edge refusal
are implemented only as an opt-in local path. Old native maps still use the old
pixel convention and cannot qualify corrected operational tracking.

A separate evaluator-only ray/triangle audit now checks the reconstructed points
against authored visible Panel/Handle/Frame meshes. An initial evaluator sign
conversion error is preserved and explicitly superseded. With the correct native
angle, 335,999 of 336,000 selected points have independent reference hits; maximum
range errors are 0.02931/0.00636 mm. One right boundary ray has no reference hit
and remains explicit. This supports a bounded ideal-simulation point-model study;
it does not establish full scene coverage, collision-model equivalence, hardware
accuracy or contact-monitor precision/recall.

All six prospectively declared fresh attempts complete. The four corrected-teacher
records (right dim, left bright, development dim/bright) independently pass physical
and raw-reuse checks; development opening is 42.2334 degrees in each lighting
condition, all 180 hold ticks valid, 22.994 mm release separation, 0.3498 rad/s peak
and 0.01924 mm stop travel. Development remains excluded from weights/statistics.
The two nominal records remain preserved and explicitly excluded from imitation.

Case 28 tests the actual right first-contact stop with the same teacher/controller.
A common robot-feedback normal-lower threshold of 1 N triggers at a measured
9.6977 N physical peak and 0.04449 rad/s native joint speed. Continuing-physics
stop travel is 0.07063 mm, final speed 0.0001763 rad/s and no violations. This
extends actual-contact stop evidence on the right; the corresponding left check
and operational monitor/sweep qualification remain open. The 45.923 s workload
is charged. No per-case count ceiling is reinstated.

The local allocation is now 2.98965 of 3 GPU hours (37.26 s remain). The existing
12-total-hour request remains pending. Automatic review separately rejects the
specific development-record transfer despite a verified synthetic/redistributable
origin and validation-only consumer path; explicit user authorization has been
requested. Train-only remote preparation/training is a safe independent path;
missing development metrics stay explicit and must be completed separately.

Runtime wiring exposes lazy initial model startup to avoid duplicate CUDA loads
from unused observer/runner constructor resets. Numerical lifecycle checks verify
zero model boots before the first capture, one actual first boot and full worker
recreation after a used episode. Default existing callers remain eager.

The first real train-only Slurm submission, 11968272 (`de2d3e6`), fails before
Python/CUDA: `SLURM_CPUS_PER_TASK=5` conflicts with step TRES `cpu=4`. Preserve
its eight allocated GPU seconds and zero updates; two jobs/70 GPU seconds are now
consumed. The descriptor repair unsets stale step TRES and uses the actual granted
CPU count. Shell/scheduling dry runs pass without allocating a replacement. One
extra submission (three jobs total) is requested while retaining the 3,600 GPU-second
ceiling and all 12,000 planned real updates; no automatic resubmission occurs.

A CPU lossless recording benchmark on 50 fixed right captures projects 51.54 GiB
for full-frame temporal gzip chunks, reduced to 37.60 GiB with 10x32x64 capture
chunks. Every raw byte round-trips, including depth float bits. This covers the
10 Hz RGB-D actually consumed by the encoder/frontend in all 24 maximum-horizon
outcomes; full-rate proprioception/commands/native feedback and all other evidence
still need separate retention/budget checks. Demonstration recordings are unchanged.

The local contact patch can now publish independently propagated plane-normal
position and direction errors at its material anchor. They include the explicit
metric point/model terms, observed conditioning, angular lever, robot/camera
calibration and temporal growth; a fit residual alone never supplies them. Full
material-pose uncertainty still protects finite footprints and sweeps. The contact
monitor uses these current normal bounds, when present, for fingertip separation
intervals and separately calibrated actual-tool FK errors rather than commanded
tracking error. Missing normal evidence keeps the conservative full-pose fallback.

Causal three-dimensional consistency retains the complete pose bound. Acquisition
travel now accumulates measured world tool path length, so material-coordinate
jitter cannot masquerade as robot travel and an oscillation cannot evade the
existing travel ceiling by returning to its anchor. Identity, sweep receipts,
verified load, confirmation/loss, stop behavior and independent physical thresholds
remain unchanged. Numerical contact/no-contact/loss/ambiguity cases pass; complete
physical false-admission/miss/delay/coverage measurements remain pending.

A saved-input wiring audit identifies a separate initialization defect in the old
ad-hoc component replay: it passed rank-one distal extrema to a finite-footprint
query. The existing maintained `PushGeometry.contact_covers()` supplies rank-two
covers; without changing masks, zones or observations, the two saved train seeds
then each provide a valid finite zone and 13 static material witnesses. This is
an input-geometry correction, not qualified ownership or a new native result.
Corrected-convention component replay is prepared with finite covers, full observed
robot-return exclusion and explicit diagnostic selection; it awaits local budget.

Current-capture normal bounds need no assumed future speed. Without a measured
speed model they are available at the exact source timestamp and become unavailable
between captures; no zero-motion bound is fabricated. The preserved whole-finite
model discrepancy reaches 1.973 mm, so the existing 3 mm reserve is retained.
A temporary unqualified 1 mm reserve proposal is preserved as incomplete and was
never used for physical admission. Complete calibration/contact/sweep validation
remains required.


## Separate-clock rollout recording

The maintained rollout entry point now records consumed images, full-rate robot
state, actual submitted/applied commands and native feedback using the lossless
layout tested in the storage projection. Explicit image indices preserve the
static/three-history input join. Independent contact records and door state stay
in evaluator channels. The command snapshot precedes a measured-response stop,
so a stop anchor cannot overwrite the original applied target in evidence.
A saved two-second prefix retains 121 control states, 21 images and 240 native
feedback rows with unchanged source clocks; raw bytes and commands round-trip.
Streaming write plus saved-input reads take 10.46 s (not live runtime cost).
Initial benchmark column/scalar-check errors and their artifacts are preserved.
Failed and interrupted boundaries are tested. This is numerical/retention
validation only: no additional physical run, real training update or learned
command was executed, and the current admission source remains incomplete.
See [[../topics/episode-and-dataset-contracts|recording contract]].


The refreshed subsequent-campaign proposal uses all four valid corrected-teacher
records: mean workload 600.75 s and 22.59 GiB raw. Holding the earlier unapproved
264-collection/840-evaluation assumptions fixed projects 44.06 collection
workstation hours /5.82 TiB raw and 302.39 evaluation hours from separately measured
physics/frontend service. The lossless 10 Hz image layout projects 1,315.9 GiB for
840 evaluation outcomes before other retained streams. Provisional reserves are
500 workstation hours, 64 allocated A100 hours and 20 TB archive plus models/logs.
The completed real two-episode A100 pilot now measures steady100-update epochs
at3.29–3.50s. Five seeds per four cells at100kupdates project18.66A100GPUh for
that measured training loop, excluding preparation, larger-data I/O, selection and
simulation. The64A100h reserve remains provisional. Loaded final-version learned
runtime, production data throughput and verified variation ranges remain open;
the earlier302.4h service extrapolation is not a measured evaluation budget.
The local evidence directory retains `campaign-budget-proposal-revised.json` and
`.md`, plus `campaign-budget-proposal-real-training-v2.json`/`.md`, all original
proposals and measured basis. No campaign is authorized;
deferred observed A4 initialization is not an A1/A2 prerequisite.

Self-clearance now requires explicit travel bounds for every ancestor joint of the relevant active-arm/body pairs, including held head, left arm and jaws. Missing held-joint evidence cannot silently become zero motion. Shared rigid ancestors cancel; existing body-pair filters still inherit to fixed collision descendants. Full recorded-motion qualification remains pending.

The material replay exposed worker-estimated seed-observation availability surviving beside correctly timed map observations. The session now assigns seed observations their measured publication time before local material adoption, including synchronous simulation. Original acquisition/support clocks remain unchanged. This closes a concrete initialization failure; it does not backdate asynchronous results.

`MaterialCover` provides conservative finite support across overlapping inspection views. Each metric planar cell must fit completely inside one actual measured mask with projected-cell/raster reserve; only those cells are unioned. Distance to unknown cells then spends grid and depth tolerance. Common holes remain unknown. The action producer must still propagate identity, current pose, calibration, temporal growth, robot response and stopping uncertainty. This addresses single-view boundary loss without inventing unseen support or changing physical contact criteria.

The first full local-geometry replay after material fusion is rejected: the similarity-only plane union admitted tilted relief, inflated uncertainty and introduced independent normal-bound violations. The repaired union requires every measured point of an additional component to lie within the existing reference-plane tolerance envelope. The failed replay remains under `local-geometry-current-right-publication`; no physical rollout used it. Full replay of the correction is pending.

The repaired material atlas now preserves the original seed plane and registration geometry. Additional strictly coplanar observations enlarge only finite coverage; averaging their relief normals had demonstrably biased the local frame, so that rejected behavior is removed. Native tracking and original pose bounds remain unchanged. Current measured planes still constrain normal position/orientation separately.

A1/A2 unloaded admission now accepts an independently eroded full-robot free-space/stop proof plus current-material separation. An irrelevant projected contact footprint is not required while the complete robot remains separated. Loaded proposals cannot use this branch. The receipt retains current identity, causal supports, normal uncertainty and the entire interval's motion bound; growth can exhaust the original margin. Actual producer/live verification remains pending.

The maintained rollout CLI now requires `--operational-recipe` and builds the real session provider, causal robot exclusion, verified robot load source, contact monitor, full-robot self/static sweep and actual proposed-motion admission. No door asset/evaluator enters these producers. Native substep load projections additionally spend the declared material-normal rate across the different clocks. Actual proposed/applied commands and operational admission/load streams are retained. Both saved train prefixes at25.2/26/27s pass complete numerical free-motion proofs; this is not learned execution or contact qualification. Continuous whole-link stopping is independently checked against all four moving-door holdouts. Earlier insufficient speed coverage is retained and distinguished from its completed correction.

The first real ACT A2 integration dispatches two learned commands through real observed geometry, load and full-volume admission. Inputs, applied targets and native/evaluator streams are retained; the continuing-physics stop passes. It is a declared120-command prerequisite, not a final-panel outcome. The receipt schedule then falsely expires because25.216666666666665+1/60 rounds a few femtoseconds below the simulator clock25.233333333333334. The existing one-nanosecond clock-roundoff allowance now applies to receipt endpoints; an extra control tick remains rejected. The generic safety label additionally retains its exact underlying exception for diagnosis. Earlier missing-worktree-USD allocation and this stopped result remain charged and preserved.

The direct SIFT registration diagnostic is rejected with its fixed minimum12 pairs and unchanged ratio/reprojection gates: both retained seeds lack sufficient textured feature support. A separate dense RGB-D connectivity diagnostic finds237,666 observed right material pixels connected to114,222 root-plane seed pixels, versus115,462 seed pixels alone, with measured12.1mm depth relief. The existing plane-only material representation omits this measured interior relief. This is diagnostic evidence, not yet a qualified material-volume or collision certificate.


The clock correction completes the declared120-command ACT A2 prerequisite.
All120 nonzero learned requests reach the controller; image/feature clocks,
full20-joint state, actual/applied targets and3264 native samples join causally.
The two-second continuing-physics stop passes. This is two seconds of free
motion, with no contact/opening/hold/release success and no final-panel outcome.
Its295.47s allocation is charged; the recorder retains323,964,793 bytes.
Admission service has mean185.5ms and p95508.5ms, so real-time operation is not
established. The complete retained failure before the clock correction remains
in the denominator of integration attempts.

Static sweep queries now retain the complete adaptive sphere partition. Each
translated/scaled leaf spends its old free-volume margin on center/radius and
calibration-lever growth; unknown or occupied new volume still refuses. Static
source clocks remain original. A retained-prefix numerical profile reduces
right26s admission from0.859s to0.263s and left26s from1.777s to0.368s; other
states remain costly, including left27s1.976s. This is a bounded query reuse
improvement, not a real-time or full-episode cost claim.

Historical static-background queries now retain each source recording and its original acquisition/publication clocks. They require a separate positive current-foreground exclusion margin and reject ordinary static-query access. Each declared calibration condition must supply a real free-ray witness; a past moving door or robot return is not asserted to remain occupied. This primitive is not enabled in the rollout recipe: the complete current target/contact exclusion producer is still pending. It does not qualify contact or convert historical free space into current free space.


The bounded rigid-interface component is now independently audited on the two
train records and two local development records. Positive-force native contacts
share a planar interface; the largest measured indentation is1.38 micrometers.
The frozen model retains a0.5mm planarity span and0.11mm indentation bound, with
all original physical evaluator thresholds unchanged. These are contact-model
bounds, not claims of submillimeter vision accuracy. Absolute visual interface
offset remains uncertain and is shared by distal/body ordering; current identity,
normal uncertainty, finite material coverage, load and complete static/moving
sweeps remain separate requirements. CAD ordering uses actual distal extrema,
including finite lines, and complete clipped mesh interiors under a normal cone;
it never substitutes the projected finite footprint for a physical contact face.

The four moving-door stop tails validate nonzero material motion within the
existing0.04rad response bound; their positive contact force samples are zero.
They therefore do not qualify strong-load contact stopping. The two independently
retained first-contact stops remain the relevant strong-load stop evidence. The
new interface/current-target/background producer is still a development candidate
pending full sweep and contact-monitor qualification; no new learned or final
panel outcome is inferred. Planar-only visual-core alternatives fail the unchanged
finite footprint margins and remain preserved and unadopted.

Observed target-front envelopes now retain non-robot RGB-D components connected
in3D to the measured root inside its finite tangent extent. This includes
protrusions excluded by planar material extraction and rejects unrelated
foreground merely overlapping the leaf image. The existing25mm association
radius and4mm voxel/depth tolerance remain unchanged. A25% reserve plus10mm and
the depth tolerance was frozen before independent collider validation. The two
observed bounds are55.06/128.80mm; independent complete moving-collider extents
are32.85/91.55mm. All moving collider vertices fit with22.21/37.25mm margins.
Evaluator geometry is used only in that independent audit, never by the producer.
The bound is a model qualified for these fixed pilot targets and observed setup;
it does not certify arbitrary hidden attachments or new hardware.

Finite material queries now support complete projected sphere footprints. Normal
position and physical contact remain separate: a positive tangent-circle margin
alone cannot admit contact. The candidate producer combines the observed front
envelope, actual CAD contact order, current rigid free volumes and independent
static-background support. Whole-path and live contact qualification remain open.

Historical background queries keep the largest valid free-ray margin within each
condition, then intersect the conditions. Every chosen margin is a real complete
volume certificate; occupied historical foreground never becomes a fabricated
current-free claim. Larger valid margins reduce avoidable cache exhaustion.

Complete static free-ball certificates can now be reused across neighboring CAD
and stop-cell keys through a bounded256-entry cache. Containment pays center,
radius and calibration growth; a volume outside all certified balls still needs
new observed support. Camera rotation uncertainty includes the farthest point
of the whole sphere, and the same radius factor is spent on cache/seed growth.
This prevents redundant geometric reconstruction without reducing the sweep,
stopping envelope or physical thresholds. The previous12000 computational
query cap is insufficient for the right30s numerical state; a separately declared
60000-query development cap completes15241queries over2323cells. This is one
retained-state component result, with zero learned commands and no whole-path or
real-time qualification.
