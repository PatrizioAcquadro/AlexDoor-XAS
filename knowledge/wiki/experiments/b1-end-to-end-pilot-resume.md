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
The software contract and consumers are checked numerically; calibration and
independent false-admission/miss/delay/coverage measurements remain pending.
No physical threshold, gain, door dynamics or qualification flag changes here.

## Scope and budgets

A3/A4 operational hinge/bootstrap/release qualification is deferred. Before new
recording, resolve only indispensable shared retention: synchronized raw RGB-D,
calibration, proprioception, actual/commanded poses and timing, native feedback,
episode boundaries and supervisor-only geometric/phase annotations. Keep raw
reuse, reconstructible supervision and qualified observed execution distinct;
this pilot is not the final matched A1-A4 dataset.

The recorder now also retains synchronized actual tool FK and complete measured
robot joint positions/velocities in a separate `robot_state` stream, including
inspection/reset/terminal rows. This closes the explicit-pose/full-proprioception
retention omission without changing the observed-only policy schema. Existing raw
RGB-D, calibration, command poses/targets/timestamps, native 120 Hz feedback and
supervisor-only phase/material annotations are retained. Numerical stream and
dataset checks pass; a fresh physical recording has not yet exercised this addition.

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
