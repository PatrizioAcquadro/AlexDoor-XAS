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
Its `configs/realsense/default.yaml` uses chunk size 2,048 (all of our at most 720 queries), Small,
256×256 mask-centred crop, one TAPIR iteration per level and `svd_residual_outlier`.
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
Per-request diagnostic time now includes measured trace binding/copies, promotion/prompt observation, frontend snapshots and export/resource inventory. Complete observation time retains this work; subtracting it is an arithmetic estimate, not a diagnostic-off live measurement. CUDA worker peak CPU RSS, CUDA allocations/resident sampling and crop windows are also saved. The official frontend developer timings may include asynchronous work and are reported with that attribution limit. Isolated pilots are complete; the two promising paths are being evaluated on both original complete openings. A4 initialization/bootstrap,
movement/contact, training and policies are deferred and unchanged. Ground truth
remains evaluator-only. Dense replay quality does not establish live resampled
quality, 150 ms freshness, ownership or safe contact.

### Completed chunk microcomparison

Both original 601-frame CUDA prefixes preserve every reference pose/native-loss
flag exactly while stateless paired 64/256/all-query predictions are inspected at
13 fixed frames, including up to 345/560 queries in mature light/nominal state.
Input causal tensors/query features remain unchanged; point axes/state keys are
preserved. The alternative prediction/returned state is discarded after scoring,
so it cannot enter the baseline's next frame.

| Condition / chunk | Prediction median / p95, ms | Maximum resized track delta, px | Every visibility identical |
|---|---:|---:|---|
| light / 64 | 115.40 / 169.93 | reference | reference |
| light / 256 | 39.94 / 65.94 | 1.219 | no |
| light / all | 37.43 / 41.96 | 3.578 | no |
| nominal / 64 | 115.74 / 253.21 | reference | reference |
| nominal / 256 | 41.16 / 94.20 | 0.734 | yes |
| nominal / all | 38.11 / 48.08 | 1.639 | yes |

These are not numerically equivalent optimizations: absolute causal-state deltas
reach 7.327/5.557 and uncertainties can change. Retain existing BF16/TF32 math;
causal trajectories must be evaluated separately before adoption. The microprobe
holds multiple returned states simultaneously for comparisons, so its raw CUDA
peak is not a deployable process peak. On mature nominal frames, peak-minus-start
allocation increments are 1.831/1.997/2.733 GiB for 64/256/all. Complete replay
process peaks are measured separately. No resolution/model/iteration change or
tracker reset is combined with this chunk control.

### Concrete A4 target transformation on existing observed fits

The same three measured seed targets are also expressed in the first **already
saved observed** hinge frame from `p2p-geometry-jacobian-01/geometry-*.jsonl`.
Keep each full-pose `target_panel` fixed and apply the existing pure `panel_pose`
math using subsequent saved observed origins/static rotations/current angles.
Evaluator-only truth transforms the **same numerical target**, rather than
choosing a better target. This measures the target component used at an A4
segment boundary; no segment phase, duration, hinge delta, tool pose, policy
prediction or admission is fabricated. No new axis fitting/bootstrap occurs.

| Condition / seed target pixel | Target p95/max mm | 10 mm available/precision % | 15 mm available/precision % | 20 mm available/precision % |
|---|---:|---:|---:|---:|
| light / [185, 315] | 3.61/9.01 | 80.09/100.00 | 80.09/100.00 | 80.09/100.00 |
| light / [185, 145] | 3.58/8.55 | 80.09/100.00 | 80.09/100.00 | 80.09/100.00 |
| light / [185, 447] | 3.64/9.50 | 80.09/100.00 | 80.09/100.00 | 80.09/100.00 |
| nominal / [186, 318] | 19.61/27.81 | 71.80/88.45 | 75.75/93.32 | 77.57/95.56 |
| nominal / [186, 150] | 19.58/27.77 | 72.04/88.75 | 75.79/93.36 | 77.57/95.56 |
| nominal / [186, 448] | 19.65/27.85 | 71.62/88.23 | 75.54/93.06 | 77.57/95.56 |

The original 2,858-frame denominator includes the missing early hinge:
first saved primary fit is 40.4833/39.9667 s. The JSON retains all 10/15/20 mm
continuity, distributions, peaks/tails and definitions. Light's approximately
80.1% availability/100% conditional precision does not mean uninterrupted A4.
Nominal target p95 is about 19.6 mm despite the better zone point p95 of 12.3 mm.
Missing complete calibration/FK/floor bounds, ownership, footprint/sweep/load,
predicted phase/duration and action admission still prevent an A4 execution score.
This reference-only consumer component analysis cannot be automatically applied
to new tracking trajectories whose articulation inputs were not saved.

### Resolution versus actual refinement passes

The official TAPIR resolution pyramid has two refinement levels at 480 and 384,
but one at 256. With `num_pips_iter=4`, the actual causal refinement passes are
8/8/4 respectively. Add 384×384 after the 480 crop to isolate reduced pixels
with eight passes retained; also inspect the official 256 path and name its
inherent pyramid reduction. Only afterwards reduce `num_pips_iter` to two,
separately at 384 (8→4 total passes) and 256 (4→2), then one at 256 (2→1) to match the official live setting. Resource inventories record
actual feature resolutions and causal levels, rather than inferring iteration
count from the configured key alone.

### Fresh reference and causal chunk pilots

Fresh automatic CUDA references complete 601/601 captures per condition and
reproduce **all candidate** preserved poses/native loss/integration decisions
exactly (maximum pose element delta zero). Source masks and original regenerated
SAM2 masks match. Light/nominal mature states reach 370/576 live queries and
872/1,446 historical references. No historical ID/history is discarded by a
reset. The official pinned live YAML also matches the remotely retrieved original
bytes exactly (`official-live-config.remote.yaml`); its actual chunk is 2,048 and
`num_pips_iter=1`, rather than relying on its introductory comments.

| Pilot / condition | Complete median / p95 ms | Median saving | 10/15/20 mm availability % | Extension eligible |
|---|---:|---:|---:|---|
| reference / light | 507.70 / 665.56 | reference | 99.83 / 99.83 / 99.83 | reference |
| chunk 256 / light | 463.38 / 558.09 | 8.73% | 99.83 / 99.83 / 99.83 | below fixed 10% cost criterion |
| all queries / light | 435.82 / 541.33 | 14.16% | 99.83 / 99.83 / 99.83 | yes |
| reference / nominal | 628.82 / 889.92 | reference | 99.83 / 99.83 / 99.83 | reference |
| chunk 256 / nominal | 557.11 / 749.32 | 11.40% | 99.83 / 99.83 / 99.83 | yes |
| all queries / nominal | 511.72 / 676.13 | 18.62% | 99.83 / 99.83 / 99.83 | yes |

All accepted nonseed primary poses satisfy all three bounds in these prefixes;
maximum correct gap is the 16.7 ms constructed-seed gap. The original 601-frame
denominator is retained. All-query replaces 256 for full extension as the faster
promising choice for the same control, with both pilots retained. This does not
establish late-tail quality; that requires the two original complete openings.
GPU sampled resident peaks are 6.807/9.605 GiB for the all-query pilots (not
an indefinitely bounded resource claim). Full resource inventories/secondary
candidate quality remain in the comparison JSON.

Complete measurement begins with an already calibrated saved RGB-D packet and
includes snapshot, IPC/native work and observational adapter. Physical sensor
acquisition, fresh live FK/calibration, queues, HDF5 input reads, rendering and
evaluator scoring are excluded. Initialization/startup is separate. Offline has
no acquisition queue wait; the IPC-minus-native difference includes serialization,
decode and process waits and cannot be called queue latency alone. No live
resampling/quality, operational/contact admission or 150 ms freshness is inferred.

### SAM2 Small: both completed pilots

Only the segmenter checkpoint/config changes. Primary accepted precision remains
100% at all three bounds in 600 nonseed observations (601 scheduled captures);
complete median improves 7.86%, below the fixed 10% extension cost rule.
Official developer segmenter median changes 22.99→13.23 ms; these spans retain
async-attribution limitations. GPU resident peak changes 6.193→6.076 GiB,
not the factor-of-five checkpoint-size reduction.

Initial Small/Large mask IoUs across five candidates are
0.924/0.967/0.942/0.078/0.077. This is agreement, not ownership accuracy.
Small triggers unchanged `sam2_initialization_candidate_mismatch` checks for
candidates 3/4, which have zero integrated acceptances rather than 600/600.
Candidate 3 also remains native lost; candidate 4 can produce native poses but
cannot bypass its initialization refusal. No gate is disabled to rescue Small.
Observed contours remain on different plausible leaf relief/edge regions;
no ownership mask truth exists. Preserve these failures and the source/returned
masks. `sam2-seed-masks.png` compares only the original 31 s images/masks.
Nominal median saves 22.44% (487.73 ms median / 644.02 ms p95), but primary
initialization fails the unchanged candidate-mismatch check: zero integration
acceptances and zero correct availability at **all** three bounds. Candidates
3/4/5 also have zero acceptances; only 1/2 retain 600. Primary initial Small/Large
IoU is 0.363. Completing native processing is not successful integration. Small
is not extended or combined; retain Large on this original initialization.
Mature mask agreement is saved independently; absent membership truth prevents
an absolute segmentation-accuracy score.

### Crop at original 480×480: completed light pilot

The actual recording is 960×600; the official crop window is a fixed 600×600,
initially at (81,0), while TAPIR remains 480×480 with eight refinement passes.
Original regenerated masks, initial maps/IDs/poses and every source timestamp are
unchanged. Primary accepted position p95 is 3.64 mm and maximum 9.10 mm; all
600 nonseed poses pass 10/15/20 mm and 5 degrees. Complete median increases
507.70→578.35 ms (+13.91%), p95 is 684.44 ms. Query peak rises 370→473 and
historical references 872→924; crop-only is not extended as a speed improvement.
This is a measured changed estimator trajectory, not an isolated pixel-count
saving. Subsequent 384/256 resolution trials retain their separate attribution.

### Reduced pixels with unchanged eight passes

Both 384×384 crop pilots complete 601/601 original captures and retain all
600 nonseed primary acceptances within 10/15/20 mm and 5 degrees. Actual feature
resolutions are `[256,256,384]`; eight causal levels are retained. Complete median
is 9.2% slower light / 15.9% slower nominal than the full-frame reference, versus
13.9% slower in both 480 crop pilots. Reducing only pixels at these two levels is
not enough to offset the altered trajectory/registration/history cost; 384 is
not extended as a common speed improvement. Full sampled component/resource
and conditional error distributions remain in `pilot-comparison.json`.

### Official 256 resolution/pyramid path

Both 256×256 crop pilots retain all 600 nonseed primary acceptances within
10/15/20 mm and 5 degrees, with four observed causal refinement levels. Light
complete median/p95 is 470.58/624.38 ms (7.31% median saving); nominal is
631.03/792.00 ms (0.35% slower median). These are compared against the full-frame
reference and against 384 with its extra refinement level; the gain cannot be
attributed to pixels alone. Light causal state peak drops 0.677→0.392 GiB while
query peak increases 370→428. The worker CPU RSS remains about 3.32 GiB.
Neither condition makes this path eligible as a common speed extension under
the fixed median rule. Explicit iteration reductions remain a separate test.

The official tracker uses a hardcoded factor of four in `construct_initial_causal_state`
even when the model requests two/one. The cold seed allocation can
therefore contain eight/four levels, then shrinks to the actual four/two/one
prediction levels after the first track. No inference fix is combined with this
comparison. Separate actual refinement passes per prediction from allocated
causal levels and retain the cold seed memory peak. Existing records permit
reconstruction from configured iteration count and observed feature levels.

### Explicit reduction to two iterations

Both 384×384/two-iteration pilots retain 600 correct nonseed primary
acceptances at every bound. Relative to the reference, median complete saving
is 7.94% light / 3.69% nominal, with four actual prediction passes. The separate
256×256/two-iteration pilots retain the same primary quality and use two actual
passes; saving is 13.69% light / 3.81% nominal. These are distinct from the
resolution/pyramid change and are not common full extensions under the preset
median criterion. Cold state allocation remains counted separately.

### One iteration and the latency-tail extension decision

The official one-iteration 256 path retains 600 correct nonseed primary poses
at 15/20 mm in each pilot; nominal has two accepted poses just above 10 mm.
Median saving is 12.79% light / 5.85% nominal, so the original median-only
exploratory rule remains **failed**. However, complete p95 improves about 20%
in both conditions (532.86/707.97 ms), with actual one-pass state and preserved
15/20 quality/continuity. Because the user objective is a 150 ms complete-latency
requirement, extend this separately as a p95-promising path (at least 15% p95
reduction in both pilots). This additional exploration decision is explicit in
`full-admission.json`, before any full attempt. Do not retroactively erase the
median-rule result, call it qualification, relax 4 mm/five inliers or re-run a
failed model for a favorable outcome. All-query remains the primary median-rule
extension. Small, 480/384 crops and the two-iteration paths are not combined.

### Simplified SVD: fixed gate verified, nominal continuity fails

Both CUDA prefixes finish with the official `svd_residual_outlier` path alone.
Audit every one of 3,000/3,600 registration calls: recomputed residuals agree
with the reported residuals and every final inlier mask equals residual ≤4 mm.
All 2,957/2,869 nonlost registrations have at least five inliers; maximum retained
inlier residual is 3.99978/3.999998 mm. MAD and demo gate relaxations remain disabled.

Complete median/p95 drops to 239.04/329.04 ms light and 305.57/407.36 ms nominal,
but nominal primary accepts only 483 of 600 nonseed observations. At 10 mm,
availability/accepted precision is 76.04%/94.62%; at 15/20 mm it is 80.37%/100%.
The maximum correct-pose gap is 1.383 s at every bound, versus 16.7 ms in the
reference prefix. There are 109 native-loss observations and a censored 10 mm
tail. Accepted nominal point p95/p99/max is 10.84/13.29/14.16 mm, while all finite
poses reach 46.45 mm. This conditional precision does not restore availability.
Light retains all 600 nonseed correct acceptances, but the nominal continuity
regression excludes SVD from full extension and from any combination. Preserve
all failures without changing five inliers, the 4 mm gate or upstream guards.
