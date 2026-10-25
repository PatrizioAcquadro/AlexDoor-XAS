# Approach-to-contact corrections and guards

Status: qualification in progress from `d798998` on `PatrizioAcquadro/b1-policy-pilot`.
Only expert approach/contact corrections and their operational guards are in scope.
No training, policy inference, checkpoint comparison, new door, development panel
or final test is authorized. The prior learned result remains 0/22.
See [[act-a1-failure-diagnosis]] for the completed diagnosis, which is not repeated.

## Plan and acceptance

Reuse the 114 train sources, retaining their original outcomes and frozen
112-source normalization. The focused numeric coverage audit finds 64 sources
with admitted approach/contact rows and 2,703 corrective commands in 13 sources;
zero injected-error commands are admitted. Nearby postures alone do not establish
joint posture/velocity coverage of ACT's later states.

Eight preregistered short A collections cover both current training directions:
shifted initial shoulder posture, an early pause, a late pause, and combined
normal/tangent/orientation error with a measured-origin return. Precontact ranges
25–35 mm, contact intervals 3–4 seconds, with the already supported slow-tail
entry. Pauses and injected offsets remain causal history, excluded from positive
supervision; corrective commands begin from the actual measured state. These
are expert subtask demonstrations, not learned completion or a replacement for
the broad corpus. Existing settling and physical endpoint criteria remain.

Require sustained native contact, ordinary withdrawal/stabilization, no physical
violation, native force within the unchanged 10 N stop domain, independent 2+4 s
stop and maintained temporal recording qualification. Every attempted case and
original outcome remains; no favorable retry or retrospective relabeling.

Three additional slots reserve independent guard/load qualification. A left
position extension needs motion and maintained 10 N stopping evidence, not an
unloaded stop. The physical 80 N limit and operational 10 N scope are distinct.
No threshold is increased to make a recorded attempt pass.

The complete reservation is 11 × 1,265 GPU seconds (3.866 h) and 48 GiB, including
startup, execution and shutdown. Baseline costs are 42.733604 local/31.823056
remote GPUh, 996.104 GiB retained and 198,773 cumulative updates. New updates and
remote allocations remain zero. The 48/32 GPUh and 1126.4 GiB caps remain.

Evidence: `outputs/evidence/act-a1-first-completion-20261022/approach-contact-20261025/`.
`scope.json`, `coverage.json` and `native-plan.json` hold exact cases and reserves.
Refinement readiness is pending physical and temporal qualification.

## Demonstrated expert timing defects and intervention

The original shifted-posture pair fails the unchanged sustained contact endpoint.
Right has 29 valid ticks (0.467 s span), below the required 0.5 s; the fixed 3 s
contact phase ends before the required proof. Both physical tails/stops pass.
The original early right pause fails the 0.51 rad/s physical speed guard: the
nominal path advances during the pause, leaving a measured 61.2 mm backlog for
a fixed 0.5 s return; native speed reaches 0.524655 rad/s. All three original
attempts remain negative and excluded from supervision.

The separately declared timing intervention reserves 1.5 s for approach pause
recovery. A short A contact keeps its ramp unchanged and can hold for at most
`sustain_s + contact_force_window_s` beyond its nominal interval, until the
original physical sustained endpoint is demonstrated. This changes demonstrated
expert behavior and phase budgeting, not acceptance thresholds or prior outcomes.
The first new right early-pause case passes with 4.734 N peak, 0.466 rad/s maximum
native joint speed and independent 2+4 s stop. Remaining cases are pending.

`native-plan-v2.json` preserves the predecessor and exact intervention cases.
The revised complete reservation covers 11 collection attempts plus three guard
slots: 14 × 1,265 GPU seconds (4.919 h), still inside the original 48 h local cap
from the unchanged baseline; storage reserve remains 48 GiB. The three original
failures are included, never overwritten or selected away.

A candidate pause-aware seeking clock is implemented but not yet operationally
qualified. It suspends time only when both submitted distal motion and measured
distal motion lie within the independently verified FK resolution, while contact
load is not confirmed. It retains cumulative travel, sample/identity checks,
force/contact confirmation and the independent motion/stop guard. The wider
material uncertainty is not used as a standstill tolerance. Existing recipes keep
the prior clock unless explicitly enabled after native verification.

## Fixed-target pause qualification boundary

Eight revised demonstrations pass, with 3.590–6.675 N peak native forces and
successful independent stops. A coverage check places their pause handoffs at
42 mm and 130–138 mm. Two additional declared near-contact returns address the
22–26 mm ACT neighborhood using a held 10 mm/4 degree pose error near 21 mm and
a 1 s measured-origin return. Both pass (4.666 N right / 3.537 N left).

Three guard trials remain negative: the first clock mistakenly charges the prior
pause at resumption; the corrected clock with zero A1 deltas permits servo drift;
and a fixed target still fails when measured settling is compared to FK precision.
All have zero learned commands, no physical violations and passing 2+4 s stops.
They are qualification failures, not favorable retries or model evaluations.

The maintained rule now compares target change with FK resolution and bounds
settling with the independently qualified 13 mm/0.04 rad response envelope.
It retains the 3 s active-seek and 30 mm travel budgets. Fixed-target diagnostic
replay derives requests from current observed joints without loading a policy.
Native qualification of this final rule remains pending; prior left control
allocations started before this revision and retain their own implementation.

Four independent robot-only stop cases pass, including three maintained 10 N
loads and neighboring shoulder postures/load directions. The existing joint
stopping bounds cover every measured stop/continuation/removal sample; the frozen
residual model has zero exceedances on the motion samples. This supports the
matched stopping conditions, not a broader combined posture/acceleration or
contact-impact domain. Operational calibration remains unchanged.


The next native right fixed-target trial exposes a repeated-clock integration
defect: its after-check erases the submitted interval's active intent. It reaches
the unchanged total-travel cap (30.003 mm) with zero contact and a safe stop;
this is preserved as another failed qualification. The corrected implementation
charges each completed interval using its previously submitted intent. New tests
cover actual before/after callback order, tiny resumption after long holds,
continuous target creep and independent active time/travel rejection.

The optional mode now applies one rule to both 3 s/30 mm acquisition budgets:
charge active commanded search, while bounded settling of an unchanged target
uses the independently verified command-response envelope. Total measured travel
and elapsed time remain logged. This operational interpretation changes only the
explicit candidate recipe; all force, physical, observed-space and stop gates
remain, and final native qualification is still required.
