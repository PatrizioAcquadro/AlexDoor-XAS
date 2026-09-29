# B1 Perception Corrections Before Retraining

Baseline: `main` at `a48e06f`; no pre-existing tracked changes. The user authorized
the four interventions from [[b1-perception-input-diagnosis|the input diagnosis]],
including a common ZED mount study after neck-only coverage proved insufficient.
The scope is implementation and bounded validation, not a new full training run.
All earlier recordings, caches and checkpoints remain preserved.

## Observation and static memory

`configs/perception_inspection.json` defines one 25-second trajectory and a common
10-degree upward camera-mount rotation, with a compensated final neck pitch.
URDF limits remain unchanged. Seven observations, at 4/7/10/15/18/21/25 seconds,
are retained separately from the recent four-frame window. RGB-D/camera FK and
proprioception are observed inputs; truth only supervises or audits the result.
The arm holds its parked tool pose during inspection. Continuous timestamps and
explicit neck commands remain in the recording. The mount is a simulated hardware
proposal, not a claim that the real robot has been modified or validated.

Two nominal train pilots in `datasets/b1/perception/inspection-pilot-01` passed
inspection, expert manipulation, hold and release in fresh RTX 4090 processes:

| Door | Hand | Hold angle | Observed top boundary | Observed bottom boundary |
|---|---|---:|---:|---:|
| door-2738468b94d74c5f | Left | 57.99° | 57.4% | 52.5% |
| animated-door-1-88abf40 | Right | 61.32° | 71.3% | 54.5% |

Boundary fractions refer to 101 uniformly spaced points along the prepared
front-face top/bottom bounds, accumulated over the seven views. A projected point
must also agree with a recorded depth in a 5×5 neighborhood within 3 cm. This is
a geometric observability diagnostic, not a new estimator acceptance threshold.
The montage was visually inspected. Portions of both boundaries are visible;
the entire silhouette is not claimed. Camera FK discrepancy stayed below 0.001 mm
and 0.000002 rad. Maximum parked-tool displacement was 0.151 mm, door motion below
0.000102 rad, and neck tracking error below 0.073 rad during motion. Neck targets
remain inside physical limits; measured upward pitch reaches the existing stop.

The tallest development door (`void-frame`) also passes the separate
`--inspection-only` scan: 24.8% of its top boundary and 56.4% of its bottom
boundary meet the same frustum/depth diagnostic. The recorded montage confirms
the previously missing top edge is now visible. This file cannot enter training
because it has no expert hold/release.
Evidence: `outputs/b1/perception/inspection-study/`, with the original neck/mount
frustum preflight, recorded RGB-D audit and image montages. No test door was opened.

## Metric estimator and confidence

`configs/perception_metric.json` selects the separate RGB/metric encoders, calibrated
world XYZ, an observed centroid plus learned hinge residual, static inspection
memory and recent articulation updates. Metric depth is filtered to 3 m before
32×32 pooling. The same preprocessing and context selection apply to cached and
live inputs. Missing inspection or inadequate depth cannot grant online validity.
Legacy heads/caches remain explicitly loadable; new defaults target separate
`engineering-v2`/`features-v2` paths and reject old fixed-view data.

Geometry optimization and checkpoint selection exclude confidence. `refine` uses
a fresh smaller-step optimizer on train only. `calibrate` fits confidence after
freezing geometry, verifies unchanged geometric weights, and qualifies the result
on the complete development set using the unchanged confidence threshold and
geometric gates, plus at least 95% precision among accepted states per door.
Every geometry update invalidates qualification. Legacy or failed qualification
cannot produce a valid online estimate, even at high raw confidence.

The complete observation/cache/model pilot contains 1,677 cached frames and
1,169 causal training windows. `metric-fit-01` was interrupted after exposing
repeated scattered HDF5 reads of the same inspection context. Keeping those
fixed inputs in memory and reading recent frames contiguously reduced the first
epoch from 99.3 to 5.8 seconds. Both attempts remain preserved.

`metric-fit-02`, with a constant 3e-4 step, stopped for train stagnation at epoch
18; its best per-door contact p95 was 14.9/21.9 mm. A standard common
`ReduceLROnPlateau` schedule now reduces the step by 0.3 after four evaluations
without 0.5% relative geometric improvement, down to 3e-6. The original patience,
physical gates and time limits remain unchanged. Scheduler settings/state are
recorded and restored with checkpoints.

With that schedule, `metric-fit-03` passed at epoch 26 / 962 updates / 142.1 seconds,
using the same two nominal train episodes and seed:

| Door | Contact p95 | Orientation p95 | Joint geometric coverage |
|---|---:|---:|---:|
| animated-door-1-88abf40 | 7.41 mm | 1.30° | 98.88% |
| door-2738468b94d74c5f | 9.93 mm | 1.70° | 95.08% |

All nine component p95 checks pass. Confidence stayed frozen and unqualified;
this is a geometry fit, not an offline confidence or development pass. No further
updates were made to obtain more favorable margins after the stop. GPU regressions
also establish metric translation behavior, depth sensitivity, static-memory
retention, reset/missing-context rejection and legacy checkpoint boundaries.
Raw RGB-D replay through `ObservedEstimator` reproduces all 1,169 cached-window
predictions: maximum contact difference 0.0071 mm and rotation difference 0.0014°.
Weights stay unchanged and all online outputs remain invalid because confidence
is unqualified/below threshold. Evidence is in
`outputs/b1/perception/metric-validation/replay.json`. A fixed +1% scale intervention on metric XYZ (128 manipulation windows, both
train doors, context and recent views) changes predicted contact by p50 4.12 mm /
p95 4.95 mm. This confirms numerical depth use, not robustness or held-out accuracy.
Independent confidence qualification is a separate check.

## Current validation boundary

The first legacy refinement (`refine-01`) passed all 19 train geometry gates after
one epoch/96.5 seconds. Its initial implementation recorded an incorrect top-level
`development_evaluated=true`; actual scope, loader and evaluation were train-only.
The implementation now reports that boundary correctly and preserves confidence
weights exactly, including protection from optimizer weight decay. That original
attempt is preserved rather than rewritten. `refine-02` verifies the final
implementation: one epoch / 782 updates / 100.9 seconds, all 19 train doors
passing, contact p95 2.29–4.00 mm and worst orientation p95 0.77°. It evaluates
train only and does not establish development accuracy.

`confidence-check-01` fits only the legacy refined model's confidence row for
2,967 updates within a three-minute update budget (192.3 seconds including final
evaluation). The geometric weights remain bit-identical. Both label classes occur:
83,220 accurate and 11,670 inaccurate/missing examples. All six development doors
still fail geometry: contact p95 is 7.41–32.32 cm, raw confidence accepts every
manipulation sample and accepted-state precision is 0%. Qualification correctly
remains false. Separate confidence fitting alone does not repair this legacy
model's generalization; this is a validated rejection, not a successful classifier.
The new metric model has only a train pilot fit and remains unqualified as well.

Independent artifact verification confirms original run-02 checkpoint hashes,
unchanged confidence weights during refinement, unchanged geometry during
confidence fitting, qualification rejection and restoration of the reduced LR and
scheduler state. Evidence: `outputs/b1/perception/metric-validation/artifacts.json`.
Targeted model/data/recording/lifecycle/documentation regressions pass on the
supported runtime, with model checks on the actual RTX 4090.

No new full-corpus training, full refreshed recording campaign, sealed-test
inspection, learned-policy rollout or hardware trial has been performed. The
next step is a separate refreshed 50-episode engineering campaign with the common
inspection, feature preparation and no-update readiness check. Only then should a
new full training run measure development accuracy and qualify confidence. The
old recordings cannot supply the missing top-boundary views. Subphase 6.0 remains
open; dynamic loss/reacquisition and hardware mounting/calibration are unvalidated.
