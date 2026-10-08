# Residuals of the Selected Renewal-Filtered Trajectory

Baseline: `main` at `cf08b5f`, clean tracked checkout, only local branch. This is
step 1 of the requested sequential investigation. Diagnose the **new**
[[p2p-sam3-selected-development|selected depth-filtered trajectory]], rather than
reuse causal conclusions from its preceding unfiltered run. Keep bounded SAM3
OFF, one SAM/one original object, TAPIR/SuperPoint, graph off, partial batch on,
rollback off, 120 active references and five inliers/4 mm. Ground truth remains
exclusively in saved-data evaluators. Original recordings, baseline, previous
attempts and the selected recipe are preserved.

## Evidence and coordinate checks

New ignored evidence root: `outputs/b1/perception/p2p-sam3-residual-02/`.
`diagnose.py`, `audit_sources.py`, `split_geometry.py`, `summary.json` and
`source-audit.json` retain the computations. `evaluation-02/` contains paired
point records, births/promotions and complete stage timelines. Inputs are the
selected `p2p-sam3-selected-01/quality-full-{light,nominal}/attempt-1/` traces and
original HDF5 rows 1860–4717, 2,858 captures per condition, 31–78.6166667 s.
Acquisition frame equals HDF5 row + 8; native index equals row - 1860. Verify these
identities and original capture timestamps on every row; never join by rounded time.

Trace publication exactly matches each saved frame pose. Registration inputs
exactly match the previously stored references for the same stable track IDs;
there are zero post-frontend reference changes on selected pairs and zero graph
updates. Reconstruct birth and current points with native **integer-truncated**
pixel sampling and measured optical-axis depth/intrinsics. Birth storage uses
the native rigid inverse (rotation transpose), not a general matrix inverse of a
slightly nonorthogonal float32 rotation. Both lifted-point checks agree below
4e-16 m. The first diagnostic evaluator's 3.54-micrometer convention mismatch and
partial output are preserved; the corrected evaluator does not change inference
or relax any tracking gate. This is an evaluator repair, not an inference retry.

Let `C_t` be camera-to-world, `D_t = P_t * inverse(P_seed)` the evaluator leaf
motion, and `T_t` the published camera-from-map. For a measured birth point `x_b`:

- `q_b = inverse(C_seed) * inverse(D_b) * C_b * x_b` is its observed birth transported
  into the seed map using evaluator motion.
- `r_b = inverse_native(T_b) * x_b` is the actually stored birth reference.
- `r_t` is the conserved/promoted reference read by current registration.
- `H_t = inverse(C_t) * D_t * C_seed` transports seed-map points into current camera.
- `c_t = inverse(H_t) * x_t` backtransports the current TAPIR RGB-D correspondence.

Pair the same IDs: birth bias `r_b-q_b`, promotion change `r_t-r_b`, total reference
bias `r_t-q_b`, and current correspondence displacement `c_t-q_b`. These are
**vectors**; their scalar norms cannot be added as a causal error budget. Current
TAPIR displacement includes integer sampling/depth effects. The measured-birth
rigid transport is an explanatory proxy, not independently annotated material
identity or sensor-noise ground truth.

Approximate exclusive prepared collision-hull membership uses the established
4 mm evaluator margin, with overlap/unclassified retained separately. A moving
transport residual for a fixed-born point is not TAPIR drift: static births are
also evaluated against static-world transport. Nothing from these hulls, motions,
or oracle fits enters workers.

## Light tail: reference birth, promotion and correspondence

Audit all 301 final-five-second captures, plus a 25-frame neighborhood centered
on 75.9333 s. The tail has 3,145 moving-born published-inlier observations over
58 unique IDs; these repeated samples are not independent trials. Published
support ranges from seven to 19 pairs. Fixed-only inliers occur in 279 tail frames,
but are absent from the exact light position/rotation peak.

| Moving-born tail inliers | Median / p95 / maximum norm (mm) |
|---|---:|
| Bias already in reference at birth | 10.98 / 19.34 / 21.43 |
| Reference change during promotion | 3.46 / 11.68 / 13.03 |
| Total current reference bias | 13.63 / 24.03 / 25.79 |
| Current correspondence displacement | 3.59 / 8.95 / 19.34 |

Median continuous-pixel displacement is 1.45 px. A same-point vector split using
native sampled rays gives median lateral/depth components 3.63/1.26 mm; the depth
component includes surface slope, occlusion and measurement error, not noise alone.
The larger lateral component does not independently label the true pixel identity.

All 354 new light references change exactly once, at promotion; the original
30 never change. There is no continuous mutation of an individual frozen reference.
Every one of 1,062 recorded promotion support observations is paired with its
original native index/current measured point. Their native pose backprojection
agrees within 6.13e-8 m, and the float32 coordinate-wise median reproduces stored
promotion exactly. For 338 moving-born promoted references, promotion increases
the birth-transport bias in 243 and reduces it in 95. Birth/promoted median biases
are 7.03/8.08 mm across the complete history, not just the tail. This demonstrates
feedback of estimated pose and tracked observations into new references; it does
not identify promotion as the unique origin of error.

Frozen unweighted SVD fits retain the **same moving-born published inlier IDs**
at every tail frame. They do not rerun native weighted RANSAC, change selection or
prove a deployable oracle correction:

| Frozen tail fit | Primary position median / p95 / maximum (mm) |
|---|---:|
| Stored reference + current measured correspondence | 12.30 / 14.63 / 24.76 |
| Original stored birth reference + current correspondence | 10.85 / 13.89 / 24.16 |
| Evaluator-transported observed birth + current correspondence | 4.31 / 6.93 / 20.22 |
| Stored reference + rigid birth-transported current point | 13.27 / 15.42 / 15.89 |

Correcting reference coordinates in this local counterfactual removes much of the
persistent tail position bias but leaves a large peak. Removing only promotion
changes gives a modest gain. Perfecting only current correspondence transport
still leaves biased references. The two mechanisms interact with the sampled
geometry and target lever arm; a point-count or frame-only explanation is insufficient.
The oracle-birth/oracle-current fit returns numerical zero, checking transforms.

## Light peak at 75.9333 s

Primary error/rotation: previous pose evaluated at the current capture
19.09 mm/1.238 degrees; selected registration before SDF 23.63 mm/1.639 degrees;
after SDF and publication 23.82 mm/1.658 degrees. SDF adds only 0.19 mm/0.018 degrees
at this event. The five-inlier gate and graph are not responsible for a publication
switch here. All nine published inliers are moving-only at birth and currently.

| Track ID | Birth time (s) | Birth bias (mm) | Promotion change (mm) | Current total reference bias (mm) | Current correspondence displacement (mm) |
|---|---:|---:|---:|---:|---:|
| 266 | 65.8167 | 10.03 | 3.23 | 11.89 | 4.20 |
| 295 | 68.3000 | 9.92 | 2.62 | 10.07 | 7.35 |
| 315 | 69.6833 | 8.06 | 4.01 | 8.86 | 5.29 |
| 317 | 69.6833 | 7.29 | 4.41 | 10.94 | 4.95 |
| 323 | 70.2500 | 10.98 | 2.72 | 13.63 | 6.64 |
| 325 | 70.7667 | 10.42 | 3.02 | 13.03 | 6.88 |
| 327 | 71.3000 | 8.58 | 2.42 | 10.98 | 3.30 |
| 344 | 73.8667 | 12.25 | 1.91 | 12.96 | 6.35 |
| 353 | 75.0333 | 12.17 | 2.78 | 14.67 | 4.04 |

Their principal spatial RMS is 136.69/45.46/5.11 mm; second/first ratio is 0.333,
with planar hull area 0.04136 m². This is not the severe near-line geometry of the
older nominal peak. The primary target is 0.634 m from their transported centroid,
so rotational bias can produce a larger target error than pair displacement.
The unweighted fit is 24.76 mm/1.778 degrees; using evaluator-corrected observed
birth references still gives 20.22 mm/1.649 degrees. Using stored references with
perfect transported current points gives 12.21 mm/0.312 degrees. Thus the peak
contains correspondence/rotation effects as well as reference bias.

| Adjacent light capture (s) | Published primary (mm) | Rotation (degrees) | Moving/fixed-only inliers |
|---|---:|---:|---:|
| 75.9000 | 10.03 | 0.056 | 9/1 |
| 75.9167 | 19.08 | 1.238 | 9/0 |
| 75.9333 | 23.82 | 1.658 | 9/0 |
| 75.9500 | 9.76 | 0.306 | 7/1 |
| 75.9667 | 12.48 | 0.122 | 9/2 |

Different inlier subsets are selected in the neighboring frames. Other saved
clusters at the peak have evaluator primary errors 13.19 and 5.54 mm with five/six
inliers. This establishes a local cluster-choice contribution, not an observable
rule for selecting the correct cluster. Choosing by evaluator accuracy is forbidden.

## Nominal peak at 39.7 s

The previous pose evaluated at current capture is 1.71 mm/0.298 degrees. The
support guard replaces the initially selected cluster 2 (five inliers, evaluator
error 2.58 mm/0.410 degrees) with cluster 1 (nine pre-SDF inliers), under its retained
native support rule: maximum cluster support 11, required eight. The chosen seed
is 32.37 mm/2.764 degrees; SDF reduces it to 31.48 mm/2.574 degrees, with eight final
inliers, and that pose is published. Candidate metadata after refinement is mutable;
use the separately captured SDF seed for the selected pre-refinement pose.

Seven final pairs (IDs 44, 46, 49, 50, 51, 56, 58) are fixed-only at birth/current;
one (ID 2) is moving-only. Static-world birth transport gives the fixed pairs a
median 2.77 mm/2.60 px displacement, rather than the approximately 37 mm/30 px
obtained by incorrectly treating them as moving leaf points. They mostly continue
tracking the frame. The moving-only subset has one pair, insufficient for an
independent rigid fit. The all-eight unweighted fit remains 31.88 mm/3.161 degrees.
Support spans a second direction (RMS 406.86/74.02/0.22 mm), but geometric spread
of mixed physical bodies does not establish correct leaf identity.

| Adjacent nominal capture (s) | Published primary (mm) | Rotation (degrees) | Moving/fixed-only inliers |
|---|---:|---:|---:|
| 39.6667 | 2.16 | 0.290 | 12/1 |
| 39.6833 | 1.70 | 0.298 | 10/1 |
| 39.7000 | 31.48 | 2.574 | 1/7 |
| 39.7167 | 1.82 | 0.321 | 12/2 |
| 39.7333 | 2.38 | 0.392 | 12/0 |

This is an isolated selection of predominantly fixed support, not a graph jump or
an SDF-created error. It is a new-trajectory-specific finding, unlike the preceding
nominal 70.6 s spike with biased moving references and nearly line-like support.

## Angular deterioration and final support gate

Independently recomputed same-target metrics exactly reproduce all three saved
quality and memory-only target evaluations, including 10/15/20 mm plus 5 degrees,
availability, precision, all-finite/accepted errors, gaps and tails. There is no new
inference/timing measurement. Comparing only common accepted nonseed captures
(2,857 light, 2,799 nominal) avoids nominal coverage selection bias:

| Condition/window | Memory-only rotation median / p95 (degrees) | Selected depth-filtered median / p95 (degrees) |
|---|---:|---:|
| Light full common support | 0.472 / 0.688 | 0.649 / 1.100 |
| Light final 301 common captures | 0.474 / 0.625 | 0.188 / 0.528 |
| Nominal full common support | 0.104 / 0.472 | 0.229 / 0.681 |
| Nominal final 244 common captures | 0.263 / 1.108 | 0.143 / 0.581 |

The bulk deterioration is concentrated earlier/mid-sequence, while tail angular
median/p95 improve in both conditions. The first panel-coordinate rotation-error
component dominates bulk p95; the third, along the evaluator hinge axis, remains
about 0.03–0.05 degrees at p95. Do not call full orientation error a measured
opening-angle error. Axis-resolved common-frame records are in `source-audit.json`.

On the selected trajectory, pre-SDF/after-SDF/published angular p95 is
1.134/1.100/1.100 degrees light and 0.954/0.670/0.682 degrees nominal. SDF improves
angular error on 2,097/2,857 light and 2,485/2,856 accepted nominal captures; local
worsening remains. This evidence does not support globally disabling SDF to repair
the rotation regression. The saved trace lacks historical TSDF voxel/gradient
snapshots, so dense-field bias versus optimizer conditioning is not fully separable.

Nominal's **published angular maximum is at 38.8833 s**, separate from its 39.7 s
position maximum: 12.23 mm/4.426 degrees. Its refined pose would score
6.22 mm/1.858 degrees, but has only three inliers. The unchanged final five-inlier
gate correctly falls back to the selected pre-SDF cluster, supported by 13 pairs
(ten moving/three fixed). The immediately preceding/following published frames
are 0.85 mm/0.376 degrees and 1.27 mm/0.109 degrees. Moving-only support is weak in
its second direction: principal RMS 430.87/19.25/4.30 mm, ratio 0.0447. Its frozen
moving-only fit remains 6.95 mm/2.625 degrees; correcting birth references alone
still gives 6.36 mm/3.060 degrees. Accepting the unsupported SDF pose or relaxing
the five-inlier gate is not a valid correction.

## Limits retained for the next decision

Light's per-call renewal-filter rejection trace is absent because that run's
observer compared acquisition ID with native index. All 354 admitted renewal
births remain independently checked against the configured gate, but rejected
candidate identities, counts and their spatial distribution are unavailable.
Do not infer the loss of useful geometric spread from admitted references alone.
Nominal has the complete rejection events; this does not fill the missing light
trace. No replay is repeated to improve this instrumentation.

The three inspected birth/current image pairs per condition show low-texture leaf
regions and fixed-frame tracking consistent with the numerical classification.
`same-id-pairs-{light,nominal}.png` and `new-trajectory-stages.png` are retained.
They are targeted visual checks, not independent true material-pixel annotations;
yellow transport markers use evaluator motion, and black crop padding is outside
the original image. Correspondence, pose-feedback and local cluster effects are
demonstrated; a unique observable common correction remains unestablished.
