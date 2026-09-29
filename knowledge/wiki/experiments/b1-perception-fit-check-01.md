# B1 Perception — Corrected Loss and Two-Door Fitting Check

## Question and scope

Can the existing observed-only estimator fit its operational contact targets on
one train door per handedness after correcting the loss imbalance identified in
[[experiments/b1-perception-run-01|run-01]]?

Implementation baseline is `e160f1f`; the tested recipe and command are committed
in `796797c`. The loss and inference contracts are in
[[topics/shared-door-perception|Shared Door Perception]]. This is a trainability
check, not a development experiment or a completed Subphase 6.0 gate.

The fixed subset is the first train identity of each handedness in the feature
index: left `door-2738468b94d74c5f`, right `animated-door-1-88abf40`. Both existing
nominal/light episodes are included: four episodes, 2,330 cached frames and
2,318 causal windows. Selection preceded optimization and was not based on fit
outcomes. Normalization, balanced sampling, selection and evaluation use only
these two identities. No development/test inference or additional collection was
performed. The frozen backbone/cache, gates, split and run-01 remain unchanged.

The estimator starts fresh with seed 6100, the existing AdamW learning rate and
batch size, and the revised tolerance-scaled objective. Confidence receives
only detached recurrent features and cannot affect geometry gradient clipping.
The six auxiliary geometry contributions share weight 0.1. The diagnostic has a
ten-minute loop limit, the existing stagnation rule and a stop at the first
checkpoint passing both per-door operational geometric thresholds.

## Result

The RTX 4090 run stopped at **epoch 30 / 2,190 optimizer steps**, after
**226.04 seconds (3.77 minutes)**, with `reason=train_geometry_passed`.
The best and final checkpoints are the same epoch. Read-only checkpoint replay
reproduces the reported positional p95 values.

| Train door | Contact position p95 | Contact orientation p95 | Confidence-valid coverage | Within both geometric tolerances |
|---|---:|---:|---:|---:|
| Left | 0.838 cm | 1.232 degrees | 100% | 98.35% |
| Right | 0.792 cm | 1.219 degrees | 100% | 99.36% |

These contact/push/hold values satisfy the unchanged 1-cm, 5-degree and 95%-coverage
thresholds **on the fitting subset**. The summary's `fit_check_passed=true` and
`best_train.offline_passed=true` have only that scope. They are not evidence of
held-out accuracy, calibrated confidence or safe dynamic use. The two-door
experiment also differs from run-01 in training scope, so it is not a controlled
full-corpus comparison demonstrating that loss changes alone explain the gain.

## Remaining geometric failure

Operational contact precision hides incorrect primitive outputs:

| Primitive error p95 | Left | Right |
|---|---:|---:|
| Hinge origin | 1.021 m | 1.008 m |
| Hinge orientation | 171.65 degrees | 175.46 degrees |
| Contact position in panel frame | 1.243 m | 1.276 m |
| Contact orientation in panel frame | 176.78 degrees | 179.48 degrees |
| Signed articulation | 177.67 degrees | 97.46 degrees |
| Dimensions, vector norm | 2.45 cm | 2.23 cm |

The decoder composes hinge origin, hinge rotation, articulation and local contact
geometry into the world contact pose. Incorrect component predictions can
compensate in that composition; the measured operational fit does not establish
that the components have their intended physical meaning. Auxiliary supervision
is present, but this recipe and contact-based stopping rule do not enforce its
accuracy. In particular, rotation chord loss has weak angular sensitivity near
180 degrees. This is a representation/optimization concern, not a newly lowered
or tightened acceptance gate. The resulting frame is not qualified for A3 or
other consumers of the articulated state.

Read-only autograd on 64 evenly spaced subset windows measures shared fusion
weight gradient norms of 28.341 for contact position, 3.668 for contact rotation,
0.019 for hinge rotation and **0 for confidence**. Confidence still has a nonzero
output-head gradient (1.248), confirming that its classifier can learn while
remaining isolated. The confidence interference is fixed; the primitive-state
problem remains. This diagnostic does not determine whether reweighting,
initialization or a more constrained parameterization is the best correction.

## Alignment and preservation checks

All 2,330 selected cached timestamps, frames and target arrays match their source
recordings exactly. Reconstruction from annotated primitive targets agrees with
annotated contact positions within 1.96e-8 m and rotation elements within
2.98e-8. These checks find no target/cache alignment mismatch in this subset;
they do not prove that every state component is observable from RGB-D.

The analysis uses no optimizer and confirms unchanged estimator weights. Loading
run-01's best checkpoint with the revised forward path changes inference by at
most 1.2e-7 on the diagnostic batch. Old checkpoint inference remains compatible;
optimizer resume across loss recipes or train/evaluation scopes is rejected.

Local evidence is preserved under `outputs/b1/perception/fit-check-01/`:
`summary.json`, `metrics.jsonl`, `best.pt`, `last.pt`, `diagnosis.json` and the
read-only `analyze.py`. The persistent training console is
`outputs/b1/perception/fit-check-01.console.log`. These payloads are ignored by Git;
this page preserves the durable result and its implementation reference.

## Next action and limits

Parts 1–2 are executed: a corrected objective is implemented and a bounded
train-only test has run. Contact targets fit, but the complete articulated state
does not. Resolve the primitive-state fit through a bounded loss/initialization/
representation investigation, then repeat the same two-door check before a
full-corpus run. Preserve this attempt; do not change gates, lower the confidence
threshold, collect more data or extend training merely to hide the mismatch.

No new full-corpus run, overnight extension, policy training, development/test
claim or dynamic validation is part of this result. Subphase 6.0 remains open.
