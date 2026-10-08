# Point2Pose Evaluation and Isolated Performance

## Decision and evidence boundary

This historical multi-object study retained **all-query TAPIR, full-frame 480,
four iterations/eight passes and SuperPoint** for explicit diagnostic 15/20 mm
comparisons. Chunk 64 remains the operational/narrow-10 mm comparator. Larger
blocks are not numerically equivalent and did not meet 150 ms. SAM2 Small,
reduced crop/resolution/iteration paths and simplified SVD were rejected; their
active controls/tests were retired at `c01c3a1`, with source in Git `57ad483` and
all local evidence preserved. The later single-object SAM3 recipe is selected in
[[p2p-sam3-unified-frontend|the frontend comparison]] and
[[p2p-sam3-selected-development|selected development]]. Defaults are unchanged.

Evidence root: `outputs/b1/perception/p2p-performance-01/`. `saved-evaluation.json`
retains 20 prior attempts, every candidate, original-time gaps/tails, histograms,
peaks, source/configuration and target definitions. Readable counterparts are
`saved-evaluation.md`, `performance-comparison.md`, `full-target-report.md` and
`full-comparison.png`. Initial evaluator serialization and pre-frame nvcc startup
failures remain separately preserved; no favorable inference retry replaced them.
Source milestones: `8c33bdc`, `9e7f619`, `129be5e`, `154bfc6`, `518d162`.
**`full-target-evaluation.json` remains a dependency of later material/residual
analyses**, not an obsolete run artifact.

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

## Compute protocol and attribution

Pinned upstream: `51856226610df75e5c06e8de545bd27f7c4ba99c`. Its saved official
live YAML uses chunk 2,048, SAM2.1 Small, 256×256 mask-centered crop, one TAPIR
iteration and `svd_residual_outlier`; our reference uses chunk 64, Large,
full-frame 480/four iterations. The live YAML also relaxes other guards and
adaptive thresholds, which were **not imported**. SuperPoint cropping, BF16/TF32
and disabled image writing were already active. The [paper's](https://arxiv.org/html/2604.10415v2) 2–10 Hz and the
[project's](https://point2pose.github.io/) September 2026 30 Hz announcement do not describe this complete
five/six-candidate leaf/IPC/adapter scenario.

Every causal pilot starts at the original observed 31 s seed, includes all 601
captures through 41 s and retains accumulated query/reference/keyframe/TSDF state.
Fresh reference masks and all-candidate poses/loss/integration decisions exactly
reproduce the preserved reference; mature live queries reach 370/576 and historical
references 872/1,446. The fixed exploratory rule allowed at most 2 percentage
points less 15/20 mm availability/precision, 150 ms longer correct gaps and required
at least 10% complete median saving in both conditions. These were comparison
rules, not registration, contact or release gates.

All registrations, graph OFF, partial batches ON, rollback OFF, five inliers/4 mm
and vectorized SDF stay fixed unless the isolated named control changes them.
Simplified SVD used fixed thresholds, five outlier iterations and no SDF; merely
setting 4 mm with adaptive MAD would not enforce that bound. All compute controls
were diagnostic-only; only `query_chunk_size` is maintained now.

Complete latency starts with a calibrated saved packet and includes snapshots,
IPC/native inference, adapter and diagnostics. Sensor acquisition, fresh FK,
rendering, HDF5 reads, evaluator scoring and live queues are excluded. Startup/seed
are separate. IPC-minus-native is not queue delay alone; native developer spans
may be asynchronous. Subtracting diagnostic cost is arithmetic, not a measured
live diagnostic-off path. CPU RSS, sampled GPU residency, exact allocator and
simultaneous/process peaks retain their distinct meanings.

### Chunk microcomparison

At 13 fixed mature frames, unchanged input tensors/query features were scored
with 64/256/all-query blocks; alternate predictions and states were discarded.
Both underlying 601-frame reference trajectories remained exact. Mature counts
reached 345/560 queries in these samples.

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

### Isolated pilot decisions

All pilots preserve original timestamps, seed maps/IDs and source masks. Primary
600/600 nonseed acceptances pass all bounds unless specified. Full inventories,
secondary results and rejected masks remain in `pilot-comparison.json`.

| Control | Decisive outcome | Extension decision |
|---|---|---|
| Chunk 256 | Median savings 8.73/11.40% light/nominal. | Light misses fixed 10% rule; preserve pilot. |
| All queries | Median savings 14.16/18.62%, all primary poses correct. | Extend both complete openings. |
| SAM2 Small | Light saves 7.86% with secondary 3/4 initialization refusals; nominal saves 22.44% but primary mismatch yields zero integration availability (seed mask IoU 0.363). | Reject; native completion cannot bypass initialization. |
| Crop 480 | Fixed 600×600 crop from original 960×600 image; complete median ~13.9% slower both conditions, light query peak 370→473. | Reject speed extension. |
| Crop 384, four iterations | Eight actual passes retained; median 9.2/15.9% slower. | Pixel reduction alone insufficient. |
| Crop 256, four iterations | Four passes (pyramid changes too); savings 7.31% light, -0.35% nominal. | No common extension. |
| Crop 384/256, two iterations | Four/two actual passes; savings 7.94/3.69% and 13.69/3.81%. | No common extension. |
| Crop 256, one iteration | One pass; median savings 12.79/5.85%, so original rule fails. About 20% p95 saving in both; nominal two poses exceed 10 mm, all pass 15/20 mm. | Separate prespecified >=15% p95 extension recorded in `full-admission.json`; median failure remains. |
| Simplified SVD | Median/p95 239/329 ms light, 306/407 nominal. Nominal accepts 483/600, loses 109, correct gap 1.383 s versus 16.7 ms; 15/20 mm availability 80.37% despite 100% conditional precision. | Reject continuity regression; no full extension. |

The 480/384 TAPIR pyramids use two refinement levels, 256 one; `num_pips_iter=4`
therefore means eight/eight/four actual prediction passes. Cold causal allocation
uses a hardcoded factor four before shrinking on the first prediction; its peak
is retained separately. Reduced resolution, pyramid depth and explicit iterations
must not be conflated. Small's checkpoint reduction did not imply equivalent GPU
savings; light residency was 6.193→6.076 GiB, and mask IoU was agreement, not
ownership truth (`sam2-seed-masks.png`).

The SVD audit covers all 3,000/3,600 registration calls, verifies every residual
and exact <=4 mm inlier mask, and at least five inliers in all 2,957/2,869 nonlost
registrations. Nominal finite poses still reach 46.45 mm. Correct enforcement of
the fixed gate did not repair continuity, and no guard was relaxed.

### Complete all-query openings

Both original automatic initializations process 2,858/2,858 captures without
inference retry/reset or discarded history. Every source frame, mask, initial
map/ID/pose and candidate is accounted for. A4 remains deferred. Cells report
availability / accepted precision / maximum correct-pose gap in seconds.

| Condition | 10 mm | 15 mm | 20 mm | Accepted point p95/p99/max, mm | Complete median/p95/max, ms | GPU peak, GiB |
|---|---:|---:|---:|---:|---:|---:|
| light | 79.67 / 79.70 / 0.583 | 98.08 / 98.11 / 0.033 | 99.90 / 99.93 / 0.017 | 13.38/15.96/21.62 | 438.27/553.79/1544.39 | 9.855 |
| nominal | 69.80 / 69.83 / 0.817 | 98.15 / 98.18 / 0.050 | 99.83 / 99.86 / 0.017 | 13.18/16.40/23.24 | 402.26/644.39/4069.27 | 12.088 |

The 10 mm regression is substantial, especially in the final five seconds:
only 91/95 of 301 captures are correct light/nominal, while at 15 mm 294/294
remain correct and at 20 mm all 301/301. Median complete time improves
29.67%/38.34%; 15/20 availability declines less than 0.5 percentage points
and 15/20 mm correct gaps remain below 67 ms. This supports an explicit wider-tolerance
compute tradeoff, not numerical equivalence, a 10 mm replacement or live/contact
qualification. Both measured complete paths have zero observations ≤150 ms.
Measured diagnostic median/p95 is 44.02/47.52 ms light and 48.54/58.63 ms nominal;
arithmetic subtraction still leaves p95 far above 150 ms. CPU worker/GPU/TSDF,
history, component timing, distributions/peak times and all secondary candidates
remain in the JSON. The isolated crop/one-iteration full results below retain the nominal failure.

At complete maxima (76.15 s light/42.4 s nominal), native work is 1.531/4.051 s,
while diagnostic work is 45.8/61.3 ms and IPC outside native about 5.0/7.5 ms.
TSDF rebuild counts simultaneously rise 16→17/5→6 with keyframe promotion;
volumes grow 18.73→20.70/60.33→98.59 million voxels, near the retained 100-million
guard. This is observed coincidence, not exclusive kernel attribution. Active
reference bounds do not prove bounded historical maps/TSDF. The peaks cannot be
explained by an acquisition queue or export alone.

### Complete reduced-path openings and combination decision

Both original 2,858-frame sequences finish as processes. Light has 2,853 accepted
nonseed poses, four native losses and three native returns; correct availability
at 10/15/20 mm is 78.69%/97.76%/99.79%, with accepted precision
78.83%/97.93%/99.96%. Maximum correct gaps are 0.383/0.083/0.033 s. Complete
median/p95/max is 417.29/569.07/1,276.80 ms and GPU peak is 6.275 GiB.

Nominal has 2,061 accepted nonseed poses, **796 native losses**, 55 native
returns and zero accepted/correct poses in the last five seconds. Correct
availability is 63.19%/71.73%/72.04%, conditional accepted precision
87.63%/99.47%/99.90%, and maximum correct gap **7.817 s at every bound**.
Accepted point p95/p99/max is 11.59/14.03/20.80 mm; its apparently improved
conditional p95 excludes the lost tail. Complete median/p95/max is
401.38/669.87/4,264.38 ms, GPU peak 10.145 GiB and worker CPU RSS 6.634 GiB.
Process completion does not mean tracking success or a latency gain at equal quality.

The native prefix of every full path exactly reproduces its own 601-frame pilot
for every candidate pose, loss flag and acceptance. The failure therefore appears
in later causal state, without changing initialization or rerunning a failed model.
All 2,858 crop windows are saved per condition: 600×600 remains fixed, origins
move (81,0)→(248,0) light / (225,0) nominal, with 402/316 position changes. The
pinned crop code retains causal state when moving the window. This audits the
actual path; it does not isolate recentering, resolution or one iteration as the
sole cause of the late failure. No full eight-pass crop or four-pass 256 counterpart was admitted
by its initial cost evidence.

Do not combine the reduced path with larger chunks to rescue this failure.
`combined-admission.json` records that the second isolated full path is unsupported;
no combined pilot/full inference is launched. Small and simplified SVD are also
unsupported as common paths. All unsuccessful prefixes and full tails remain
available, together with their error/latency/resource records.

### Observed targets, reference compatibility and scope of adoption

On the all-query full path, the three same observed target poses have accepted
point p95 of 13.10–13.46 mm light / 12.85–13.79 mm nominal, versus
10.79–12.75 / 11.71–12.39 mm in the preserved reference. Target availability at
15 mm is 97.90–98.36% / 97.62–98.25%, with maximum correct gaps 33/50 ms;
at 20 mm it is about 99.90% / 99.79–99.83%, with 16.7 ms maximum gaps.
Peak target error reaches 22.04/23.53 mm. Every target still has its separate
10/15/20 mm precision, distribution, peak time, continuity and tail in
`full-target-evaluation.json`. No absent articulation input is invented for
these new trajectories; their target transport is not a complete A4 score.
The earlier pure A4 target transformation remains reference-only.

The original-reference compatibility audit uses evaluator-only motion to test
directly measured original-point inliers at 4 mm, separately from pose bounds.
Larger chunks have no primary native-loss return to test. Reduced light/nominal
have 3/55 returns, none with five compatible directly measured original inliers.
Original references are progressively replaced and are absent from the final
inlier sets: absence does not prove wrong material, but cannot establish strict
original-material recovery. Neither renewed references nor a native return is
physical identity/ownership evidence. All secondary candidates remain in the
JSON; their quality is weaker and uneven, so adoption concerns the original
primary diagnostic target only, not arbitrary candidates or held-out doors.

All-query adoption concerns the original primary diagnostic at 15/20 mm, not
arbitrary candidates, other doors or 10 mm numerical equivalence. Chunk 0 includes
all queries (official 2,048 also includes the observed <=720). Keep Large,
full-frame 480/four iterations/eight passes and all other fixed controls; do not
combine rejected paths. Current selected configuration remains opt-in.

Complete p95 553.79/644.39 ms remains 3.69×/4.30× the 150 ms deadline, with zero
timely complete observations. Arithmetic diagnostic subtraction still yields
507.16/584.65 ms. Visual/P2P startup was 9.63/9.70 s and 4.12/4.33 s, first complete
initialization 2.77/2.94 s. None measures live queue/resampling quality. Future
latency work must consider registration and promotion/TSDF spikes while preserving
quality; it is outside the October 8 cleanup. A4 inputs/admission, movement/contact,
training and policy changes remain outside this experiment. Recording completion
through 63.7209 degrees does not validate the configured 90.7-degree limit.
