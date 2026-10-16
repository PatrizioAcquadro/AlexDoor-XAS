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
