# Selected-Candidate Registration with Preserved Alternatives

## Question and method

Can registering the selected candidate every frame and alternatives periodically reduce cost while
retaining complete-sequence quality?

Historical `a8d0470` registers alternatives every nine processed frames, expanding on selected
support failure. Deferred poses are frozen/unavailable and do not grow maps or refresh support. Four
601-frame pilots precede four complete 2,858-frame CUDA comparisons. Original initialization,
graph-off/partial-on controls and five inliers/4 mm remain fixed.

## Result and decision

Pilots have 600 correct nonseed poses each and median request savings of 37.66% light / 36.01%
nominal. Complete attempts retain all 11,432 scheduled frames and reproduce the all-registration
reference exactly.

| Condition / schedule | Correct / wrong accepted | Accepted precision | Correct availability | Position p95 (mm) | Rotation p95 (degrees) | Longest correct-pose gap (s) |
|---|---:|---:|---:|---:|---:|---:|
| Light / all | 2,450 / 407 | 85.75% | 85.72% | 11.89 | 0.835 | 0.250 |
| Light / selected | 2,752 / 105 | 96.32% | 96.29% | 9.05 | 0.349 | 0.067 |
| Nominal / all | 2,484 / 373 | 86.94% | 86.91% | 12.30 | 0.476 | 0.317 |
| Nominal / selected | 1,785 / 1,072 | 62.48% | 62.46% | 13.70 | 0.329 | 1.683 |

The full nominal tail from 70 s regresses 325→30 correct poses out of 518 despite a lower position
maximum. Scheduling changes shared RNG, reference/map histories and later support; alternatives are
not unchanged backup estimators. **Reject common adoption and retire the schedule.** Current runtime
rejects its activation.

## Limits and evidence

Correctness is relative zone position ≤1 cm and full rotation ≤5 degrees; the seed stays in the
2,858-row availability denominator but is excluded from errors. Nine processed frames represent 150
ms only in dense 60 Hz replay. No sparse/live scheduling, material recovery, A3/A4 or contact
qualification is established.

Evidence: `outputs/evidence/b1/perception/p2p-selected-registration-01/`, original protocols/traces,
per-candidate audits and `full-comparison.png`. Git: `a8d0470` /`57ad483`, retirement `c01c3a1`. See
[[p2p-observed-geometry-and-sdf-jacobian]].
