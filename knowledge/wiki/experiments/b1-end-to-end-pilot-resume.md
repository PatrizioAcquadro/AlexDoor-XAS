# Resumed End-to-End A1/A2 Pilot

Baseline `4ec7571`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The user explicitly resumes Goal 1: correct the actual contact-monitor/teacher
problems and continue through real demonstrations, short ACT/Diffusion × A1/A2
training and learned closed-loop simulation. Preserve all previous work/failures,
fixed physical criteria and sealed test doors. No production campaign or robot work.

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
