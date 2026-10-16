# Joint-Limit-Aware Pose IK

Baseline `7e02074`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The user explicitly authorizes one bounded IK cycle after
[[b1-shared-learned-pilot|two rejected reference-only candidates]]. The one
correction is **rejected** after complete physical measurement.
The original active pose solver and tests are restored; rejected sources remain
in local evidence, without another variant or completed-case retry.

## Demonstrated issue and rejected correction

The old pose solver computes a damped Cartesian correction plus exact-nullspace
centering, clips individual physical position limits, then scales the whole target
step against the previous command. The failed original left approach has 129
position-clipped and 100 rate-scaled targets; the rejected FK-reference candidate
still has 184 position-clipped targets with no rate scaling. Independent coordinate
clipping cannot redistribute the lost task correction over remaining free joints.
Existing static IK support does not prove a feasible continuous trajectory.

Keep the existing primary correction, nullspace centering and whole-vector target-
speed scaling. When a physical position limit is active, solve only the weighted
projection of the original proposal `p` onto the physical joint box:

`min_x ||J (x-p)||² + damping² ||x-p||², lower <= x <= upper`.

This uses the same mixed-unit Cartesian metric and damping as the old solve,
without another task weight or joint margin. An active-set algorithm solves the
free variables and releases constraints with invalid multiplier signs. The seven-
variable conditional solve uses float64 on the same device, then returns the input
dtype. Thirty-two iterations are a finite numerical cap; lack of convergence
refuses explicitly. An already feasible proposal is exactly unchanged. A1 and
stop/latch behavior are unchanged; no task trajectory, teacher fallback or hidden
door/evaluator input is added.

## Validation protocol fixed before execution

- Independent mathematical checks: near-limit redundant compensation, physical
  bounds, unchanged unconstrained proposals, original nullspace/rate regressions,
  and agreement with SciPy BVLS on deterministic coupled fixtures. These are
  numerical unit checks, not CPU simulation or policy/model training.
- CUDA fixed-state comparison: all 720 original saved train approach states,
  original Jacobians, previous targets and Cartesian errors. Reconstruct the old
  applied targets before comparing; retain every new target and error. A better
  saved-state target is not evidence of better closed-loop tracking.
- Four new physical cases maximum: both original complete six-second Cartesian
  approaches plus two-second endpoint holds/stops, and both original moving-stop
  prefixes at 2.1 s plus two seconds continuing physics. Reuse the existing
  [[b1-approach-controller-compatibility|accepted baseline cases]], without rerun.
- Keep original source initial states, robot/floor, fixed base, damping 0.01,
  centering gain 2, 0.5 rad/s target-step budget, actuator gains/effort limits,
  reference time/poses and tracking/hold/stop criteria. No corpus door, external
  force or contact; abort any unexpected robot contact.
- Require all three shared GPU locks and the existing 180-second idle protocol.
  Preserve interruptions and resume only incomplete cases with unchanged source;
  no favorable completed-case retry, reference retiming or further IK variant.

Acceptance remains poststep p95/endpoint <=10 mm /5 deg, endpoint-hold joint error
<=0.02 rad, continuing-physics stop <=10 mm/3 deg and final speed <=0.01 rad/s.
Report measured speed against 0.51 rad/s, physical limits, clipping, timing and both
120 Hz native feedback samples. The original residual calibration is unchanged;
all out-of-domain estimates remain unavailable. Neither complete contact nor
hardware qualification follows from these unloaded cases.

## Numerical and physical outcome

The independent numerical gates pass, including coupled bounded least-squares
agreement and exact preservation of unconstrained proposals. The 720 saved-state
comparison reconstructs old targets within 4.77e-7 rad. Right has no position-bound
changes; left changes 129 steps and reduces saved-state target position/rotation
p95 from 30.068 mm / 7.715 deg to 28.891 mm / 7.310 deg. These are hypothetical
targets at fixed old states, not closed-loop behavior.

Exactly four new CUDA cases complete in one uninterrupted attempt. Original
Cartesian references, source initial joints/previous targets, runtime configuration,
gains, calibration and 60/120 Hz clocks match the accepted baseline. The two right
physical traces are bit-identical to baseline. No unexpected contact occurs.

| Complete approach | Original right | Candidate right | Original left | Candidate left |
|---|---:|---:|---:|---:|
| Poststep translation p95, mm | 0.673 | 0.673 | 28.783 | **21.363** |
| Poststep rotation p95, degrees | 0.216 | 0.216 | 7.873 | 2.607 |
| Actual peak joint speed, rad/s | 0.154 | 0.154 | 0.527 | **0.529** |
| Rate-scaled / raw position-limited steps | 0 / 0 | 0 / 0 | 100 / 129 | 108 / 117 |
| Original complete-path criteria | Pass | Pass | Fail | **Fail** |

The left rotational criterion now passes, but translation remains above 10 mm
and measured speed exceeds 0.51 rad/s at 94 physics samples. The target speed
bound does not guarantee the measured servo speed. Endpoint and endpoint-hold
checks pass. No measured joint exceeds physical limits by more than the unchanged
1e-4 rad tolerance, and no effort clipping occurs. Static IK supports
all 360 reference poses from each original seed set; global unreachability is not shown.

Both moving stops pass: right 2.573 mm / 0.284 deg, left 4.589 mm / 0.791 deg;
final speeds are 0.000152 / 0.000257 rad/s after two seconds. Full-case endpoint
stops also pass. Targets latch at measured joints and physics keeps running.
All 3,384 native feedback samples are finite and available, but only 108 residual
estimates remain inside the original calibration domain. Missing load support
is preserved rather than promoted by the complete native feedback.

The cases contain 28.2 simulated seconds and 26.77 measured case wall seconds;
case wall time includes instrumented stepping and static IK checks, while shared-
GPU waiting, process startup/teardown and final trace serialization are excluded. This is unloaded,
camera-disabled robot evidence, not full-pipeline real-time or contact readiness.

## Decision and preserved evidence

Reject this single correction under the frozen criteria. Keep the existing pose
solver, teacher and adapters. The authorized one-cycle/four-case ceiling is
exhausted; no additional seed, IK formulation, reference retiming, changed threshold
or favorable retry is run. The learned pilot remains incomplete with no new
physical demonstration or learned command.

`outputs/evidence/b1-shared-learned-pilot-20261016/ik-limit-cycle/` preserves user
authorization, previous TODO, frozen protocol, exact rejected source and tests,
patch against `7e02074`, saved-state outputs, four complete traces, GPU monitoring,
comparison, verification and `outcome.json`. The `accepted/` links mean acquisition
completed without interference, not controller acceptance. Active source and tests
are restored byte-for-byte to baseline before the outcome is committed.

The smallest next decision is a separately bounded execution-feasibility study.
Initial robot placement is already within the goal's train/development domain
question; inspect that evidence before selecting one practical pre-episode
placement correction. Keep the base fixed throughout each episode, retain both
selected doors and original task/acceptance rules. Do not assume placement will
solve the failure or begin another physical search under this exhausted cycle.
Observed local sweep/response, load support and the first-command A3/A4 dependency
still require completion before genuine collection and learning.
