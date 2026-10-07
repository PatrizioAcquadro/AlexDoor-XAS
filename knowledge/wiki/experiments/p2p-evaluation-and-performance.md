# Point2Pose Evaluation and Isolated Performance

20 preserved attempts; every original scheduled row and failed tail retained. Seed excluded from correctness and errors, retained in scheduled denominator. Each cell is availability % / accepted precision % / maximum correct-pose absence seconds. Rotation remains 5 degrees. This changes evaluation, not 4 mm registration or operational/contact bounds.

| Condition / preserved system | 10 mm | 15 mm | 20 mm | Accepted point p95/p99/max mm |
|---|---:|---:|---:|---:|
| light / original | 79.64 / 98.49 / 0.667 | 80.48 / 99.52 / 0.667 | 80.72 / 99.83 / 0.667 | 7.79/11.15/22.37 |
| light / unbounded_failed | 44.40 / 93.93 / 25.083 | 47.03 / 99.48 / 25.083 | 47.17 / 99.78 / 25.083 | 10.30/13.28/21.16 |
| light / bounded_graph_on | 78.73 / 80.62 / 0.550 | 92.27 / 94.48 / 0.367 | 96.57 / 98.89 / 0.333 | 15.43/20.05/45.17 |
| light / bounded_graph_off | 89.89 / 91.52 / 0.183 | 97.13 / 98.90 / 0.117 | 98.15 / 99.93 / 0.067 | 11.51/15.30/21.05 |
| light / partial_only | 85.72 / 85.75 / 0.250 | 98.53 / 98.56 / 0.067 | 99.93 / 99.96 / 0.017 | 11.89/15.77/22.17 |
| light / 6mm | 52.31 / 52.33 / 3.400 | 71.94 / 71.96 / 2.033 | 94.58 / 94.61 / 0.233 | 20.67/26.51/29.98 |
| light / promotion_geometry | 54.13 / 54.32 / 4.133 | 79.39 / 79.67 / 0.717 | 96.12 / 96.45 / 0.067 | 19.47/22.20/30.84 |
| light / rollback | 76.38 / 84.97 / 0.967 | 89.12 / 99.14 / 0.700 | 89.71 / 99.81 / 0.617 | 11.55/14.73/63.56 |
| light / reference_vectorized | 85.72 / 85.75 / 0.250 | 98.53 / 98.56 / 0.067 | 99.93 / 99.96 / 0.017 | 11.89/15.77/22.17 |
| light / selected_registration | 96.29 / 96.32 / 0.067 | 99.93 / 99.96 / 0.017 | 99.97 / 100.00 / 0.017 | 9.05/11.99/15.16 |
| nominal / original | 52.27 / 94.44 / 4.417 | 54.90 / 99.18 / 4.417 | 55.14 / 99.62 / 4.417 | 10.20/13.49/36.07 |
| nominal / unbounded_failed | 21.59 / 99.84 / 37.300 | 21.62 / 100.00 / 37.300 | 21.62 / 100.00 / 37.300 | 3.35/6.06/12.02 |
| nominal / bounded_graph_on | 76.38 / 95.66 / 2.633 | 78.76 / 98.64 / 2.633 | 79.50 / 99.56 / 1.633 | 9.28/15.97/178.61 |
| nominal / bounded_graph_off | 70.12 / 97.19 / 7.850 | 71.62 / 99.27 / 7.850 | 71.80 / 99.52 / 7.850 | 8.15/13.34/27.52 |
| nominal / partial_only | 86.91 / 86.94 / 0.317 | 98.53 / 98.56 / 0.033 | 99.69 / 99.72 / 0.017 | 12.30/15.83/22.92 |
| nominal / 6mm | 53.11 / 53.17 / 13.500 | 59.52 / 59.58 / 3.733 | 85.48 / 85.57 / 0.683 | 22.10/25.77/65.00 |
| nominal / promotion_geometry | 61.37 / 94.25 / 9.417 | 64.07 / 98.39 / 9.417 | 64.94 / 99.73 / 9.417 | 10.57/16.07/29.16 |
| nominal / rollback | 77.50 / 95.85 / 5.017 | 79.95 / 98.87 / 5.017 | 80.76 / 99.87 / 2.383 | 9.12/15.18/22.36 |
| nominal / reference_vectorized | 86.91 / 86.94 / 0.317 | 98.53 / 98.56 / 0.033 | 99.69 / 99.72 / 0.017 | 12.30/15.83/22.92 |
| nominal / selected_registration | 62.46 / 62.48 / 1.683 | 97.66 / 97.69 / 0.067 | 99.90 / 99.93 / 0.017 | 13.70/16.64/21.37 |

## Observed target transport

Three full-pose geometric contact-frame candidates use measured depth at the original component centre and two distributed original component pixels. The seed zone supplies the observed surface orientation; target coordinates are frozen in that observed zone. Evaluator-only motion supplies expected subsequent poses. Absolute calibration, membership, footprint/sweep, physical hinge, uncertainty and action admission are not supplied. No approach, motion, action, load or policy prediction is invented.

| Condition / reference target seed pixel | Lever mm | 10 mm availability/precision % | 15 mm availability/precision % | 20 mm availability/precision % | Point p95/max mm |
|---|---:|---:|---:|---:|---:|
| light / [185, 315] | 38.4 | 87.02/87.05 | 98.71/98.74 | 99.93/99.96 | 11.66/21.84 |
| light / [185, 145] | 131.3 | 79.60/79.63 | 98.11/98.14 | 99.90/99.93 | 12.75/22.57 |
| light / [185, 447] | 131.4 | 92.65/92.68 | 99.13/99.16 | 99.93/99.96 | 10.79/21.12 |
| nominal / [186, 318] | 58.6 | 88.28/88.31 | 98.78/98.81 | 99.86/99.89 | 11.97/22.61 |
| nominal / [186, 150] | 130.5 | 89.71/89.74 | 98.92/98.95 | 99.86/99.89 | 11.71/22.23 |
| nominal / [186, 448] | 145.2 | 86.18/86.21 | 98.46/98.49 | 99.69/99.72 | 12.39/23.03 |

## Paper, pinned implementation and live configuration

The [paper, section 4.4](https://arxiv.org/html/2604.10415v2) reports 2–10 Hz,
whereas the [project page](https://point2pose.github.io/) announces 30 Hz on an
RTX 4090 in September 2026. Neither specifies our five/six candidate, full-size
leaf, full observation/IPC/adapter scenario; neither is a local latency result.
The pinned official source `51856226610df75e5c06e8de545bd27f7c4ba99c` already has
`query_chunk_size`, `tapir_crop`, `num_pips_iter` and SAM2.1 Small support.
Its `configs/realsense/default.yaml` uses all queries per chunk, Small,
256×256 mask-centred crop, fewer TAPIR iterations and `svd_residual_outlier`.
The reference uses 64, Large, full-frame 480×480 and four iterations.
SuperPoint mask cropping, disabled image writing, BF16 autocast and TF32 are
already active in the reference; they are not new improvements.

The live YAML also changes adaptive MAD/outlier thresholds, minimum inliers,
map-growth and lost/jump guards. These are not imported. Diagnostic compute
controls change only named components on the preserved all-registration,
graph-off, partial-batch, rollback-off reference, five inliers and 4 mm, with
the previously validated vectorized SDF Jacobian. Simplified official SVD is
explicitly configured with `threshold_method=fixed`, `inlier_thres=0.004`,
`min_inliers=5`, five outlier iterations and no SDF refinement; unrelated
frontend guards remain unchanged. Nominally setting 4 mm with MAD would not
apply that bound. All performance controls require diagnostic mode and leave
normal worker/provider and policy defaults unchanged.

## Attribution and operational boundary

CUDA comparison proceeds through chunks, Small, crop at original resolution,
then smaller crop resolution, then reduced iterations, then simplified SVD.
Each causal pilot starts at the original observed 31 s initialization and
processes every original capture through 41 s, including accumulated queries,
reference replacement and keyframe/TSDF history. Full original openings are
reserved for promising variants. The fixed exploratory progression rule permits
at most two percentage points lower availability/accepted precision at 15/20 mm,
at most 150 ms longer correct-pose gaps and requires at least 10% median complete
latency reduction in both conditions. The historical 10 mm view remains reported
but is not the sole rejection rule. These are exploratory comparison rules,
not contact or operational qualification limits.

The saved selected-registration variant illustrates the changed interpretation:
nominal availability is 62.46% at 10 mm, 97.66% at 15 mm and 99.90% at 20 mm.
Its 10 mm regression does not erase its wider-tolerance benefit. Secondary
registration deferral and altered estimator history remain a separate compromise;
the requested compute reference keeps every candidate registration.

Evidence is in ignored `outputs/b1/perception/p2p-performance-01/`:
`saved-evaluation.json` retains every candidate, all thresholds, error histograms,
p99/peaks, original-time intervals, failures/tails and measured target definitions.
Preserved inputs/results are not overwritten. Initial evaluator serialization
failure and repaired evaluator log are both retained; inference was not repeated.
The first direct CUDA probe lacked the installed nvcc path before processing a
frame; its startup failure is preserved separately from the corrected launch.
Optimization measurements are in progress. A4 initialization/bootstrap,
movement/contact, training and policies are deferred and unchanged. Ground truth
remains evaluator-only. Dense replay quality does not establish live resampled
quality, 150 ms freshness, ownership or safe contact.
