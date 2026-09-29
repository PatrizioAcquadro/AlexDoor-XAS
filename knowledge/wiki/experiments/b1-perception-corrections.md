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

The metric model and confidence corrections are tracked in the next implementation
milestone. This observation study does not establish estimator accuracy.
