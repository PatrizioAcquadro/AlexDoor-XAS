# B1 Perception — Complete-State Fitting Check

## Question and fixed scope

Can the corrected estimator fit both the operational contact and its articulated
components on the same two train doors, without the compensations exposed by
[[experiments/b1-perception-fit-check-01|fit-check-01]]?

Implementation baseline is `8692a64`; the tested correction is `1bd633e`.
`articulated-state-v3` gives all nine geometry errors equal tolerance-scaled
supervision, uses geodesic angular loss and well-conditioned rotation/angle
initialization, and supervises confidence against complete-state correctness.
The canonical loss, evaluation and checkpoint contracts are in
[[topics/shared-door-perception|Shared Door Perception]].

The subset is unchanged: left `door-2738468b94d74c5f`, right
`animated-door-1-88abf40`, both nominal/light episodes, 2,330 cached frames and
2,318 causal windows. Seed 6100, AdamW learning rate, batch size, recording and
feature caches, DINOv2 backbone and contact gates are unchanged. The estimator is
freshly initialized; no earlier optimizer or fitted checkpoint is resumed.
Normalization, sampling, selection and evaluation use only these two train doors.

Before optimization, the engineering component budgets were fixed at 1 cm for
Euclidean length/vector errors and 5 degrees for angular errors, matching the
existing contact resolution. Every component p95 must pass. At least 95% of
manipulation frames must meet all nine limits simultaneously, and confidence-valid
coverage must also be at least 95%. Original contact thresholds remain required.
These are additional state-fit checks, not a claim of hardware tolerances or a
relaxation of any gate.

## Result

The RTX 4090 run stops with **`train_state_passed` at epoch 17**, after
**1,241 optimizer steps and 139.42 seconds (2.32 minutes)**. Best and last
checkpoints are epoch 17. This is the first complete-state pass on the fixed
train subset; it is not development qualification or completion of Subphase 6.0.

| Metric, manipulation frames | Left | Right |
|---|---:|---:|
| Contact position p95 | 0.872 cm | 0.964 cm |
| Contact orientation p95 | 1.256 degrees | 2.749 degrees |
| Hinge origin p95 | 0.239 cm | 0.509 cm |
| Hinge orientation p95 | 0.106 degrees | 0.069 degrees |
| Dimensions vector error p95 | 0.349 cm | 0.456 cm |
| Local contact position p95 | 0.515 cm | 0.560 cm |
| Local contact orientation p95 | 0.100 degrees | 0.143 degrees |
| Panel orientation p95 | 1.250 degrees | 2.861 degrees |
| Signed articulation p95 | 1.187 degrees | 2.832 degrees |
| Jointly within all nine tolerances | 98.25% | 96.70% |
| Confidence-valid coverage | 100% | 100% |
| Correct complete state among accepted estimates | 98.25% | 96.70% |

The hinge error is now millimetric, rather than the approximately one-meter error
in fit-check-01; hinge/local rotations are no longer reversed. Contact errors
remain within their original limits. This demonstrates success of the revised
recipe on the fitting subset, not an ablation isolating which individual change
caused the improvement. Training curves remain nonmonotonic; the run stops at the
first jointly passing checkpoint and does not establish repeatability over seeds.

## Verification

Regression checks reject a deliberately shifted/rotated hinge whose contact pose
is preserved by compensating outputs. Its confidence loss now pushes confidence
down. Other checks cover near-180-degree and zero-error angular gradients,
well-conditioned initialization, detached confidence gradients, missing-input
behavior, joint state coverage, train-only scope and checkpoint contracts.

A no-update GPU reload reproduces the complete saved per-door report exactly.
Four recorded RGB-D histories, one deterministic middle-push window per selected
episode, also pass through `ObservedEstimator` with the frozen backbone. All four
produce valid estimates; the maximum difference between live preprocessing and
cached-feature outputs is 7.75e-7. These are replays of recorded observations,
not a fresh simulator run or a closed-loop validation. The verification confirms
unchanged estimator weights and a frozen backbone.

Checkpoint format `b1.perception.checkpoint.v2` records
`confidence_scope=articulated-state-v1`. Older weights remain readable for error
diagnostics, but their contact-only confidence cannot validate a complete state
or enter `ObservedEstimator`. Optimizer resume across recipes/scopes is rejected.

Local evidence remains in `outputs/b1/perception/fit-check-02/`: `summary.json`,
`metrics.jsonl`, `best.pt`, `last.pt`, `verification.json` and the no-update
`verify.py`. The training console is
`outputs/b1/perception/fit-check-02.console.log`. These ignored payloads and all
previous attempts are preserved; only source, tests and durable documentation are
committed.

## Next action and limits

The diagnosed loss, state-validation and confidence issues are corrected, and
the bounded two-door check passes. The next experiment is a separately authorized
full-corpus estimator run in a new output directory, using the current recordings
and features and reporting the original contact gates plus the complete-state
checks on development doors. Do not resume fit-check-02 as full-corpus training.

No full-corpus run, development/test inference, overnight extension, additional
collection or policy training was performed here. Generalization, confidence
calibration on unseen doors, loss/reacquisition and dynamic usability remain
unverified. Subphase 6.0 remains open.
