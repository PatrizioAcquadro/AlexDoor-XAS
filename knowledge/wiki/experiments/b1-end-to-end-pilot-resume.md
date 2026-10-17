# Resumed End-to-End A1/A2 Pilot

Baseline `4ec7571`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The user explicitly resumes Goal 1: correct the actual contact-monitor/teacher
problems and continue through real demonstrations, short ACT/Diffusion × A1/A2
training and learned closed-loop simulation. Preserve all previous work/failures,
fixed physical criteria and sealed test doors. No production campaign or robot work.

## Current resumption

The October 17 authorization raises the cumulative workstation ceiling to 15 GPU
hours and removes numerical submission/prerequisite case ceilings. The remote
3,600 allocated GPU-second, 18,000 optimizer-update, 300 GiB storage and six fresh
demonstration ceilings remain cumulative. At resume, 2.989649 local GPU hours,
70 remote GPU seconds and 164 infrastructure updates are consumed; all six fresh
attempts remain counted. No source changes follow reviewed commit `a9462e9`.

Repaired train-only job `11969867` is submitted after a Slurm resource dry run and
recording/robot metadata checks. It reserves the unchanged four 3,000-update cells
on corrected right-dim and left-bright recordings, seed 17, source `de2d3e6`.
The CPU environment mismatch is repaired by removing inherited
`SLURM_TRES_PER_TASK` before explicit `srun --cpus-per-task`. Queue waiting is
uncharged. No development recordings are transferred; remote development metrics
remain unavailable. The pending left actual-contact stop and corrected native
component checks use all three shared leases and the existing idle preflight.
Training and end-to-end execution remain uncompleted until their actual evidence
returns. Earlier budget proposals below record their historical authorization.

## Current observed-space integration

Continuous recorded robot sweeps use calibrated CAD cells, joint-axis cell levers
and independently measured stop excursions. Static workcell rays keep their original
capture/publication clocks. A smaller ray pyramid may certify a larger requested
sphere only by spending its measured radial clearance on the complete radius growth;
unknown or occupied volume is never filled. Regression checks cover raster-border
inconclusiveness, real unknown holes and occupancy. This corrects a constructive
coverage failure but does not qualify the remaining unsupported fingertip sweep.
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
These supersede the earlier single-cycle/historical cost projections; real A100
throughput, live evaluation overhead and verified variation ranges remain open.
The local evidence directory retains `campaign-budget-proposal-revised.json` and
`.md`, all original proposals and the measured basis. No campaign is authorized;
deferred observed A4 initialization is not an A1/A2 prerequisite.
