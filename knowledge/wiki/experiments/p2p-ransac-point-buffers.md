# Point2Pose RANSAC Point Buffers

2026-10-15; baseline `fa36656` on `PatrizioAcquadro/b1-observed-a3a4`, initially
clean. **Reject default adoption for the current selected operational pipeline.**
Candidate `dd2f0ae` is numerically exact and measurably faster locally and in
fixed native service, but complete-provider benefit is not sustained across the
timed histories. Maintained and installed registration are restored to baseline.
SAM3 remains eager, and TSDF remains at the maintained baseline. All
[[sam3-official-compilation|rejected compilation evidence]] is preserved.

## Scope and preset gates

The active native `SVDClusterRANSACRegister._RANSAC` uses 100 sequential four-point
draws, weighted SVD, source/target degeneracy checks and first-wins equal-support
scores. Five inliers, inclusive 4 mm residuals, refit rejection, remaining-mask
mutation and exception handling are held fixed. Candidate selection, SDF,
renewal, cadence, models and the selected configuration are unchanged.

The single candidate precomputes the immutable remaining homogeneous source and
target pool once per call and reuses sample arrays already extracted for
degeneracy checks. The same 4×4 matrix product, norm and SVD execute in the same
order. The maintained `patch_tracking` setup/startup path at `dd2f0ae` reproduces it
and retains original source. `candidate.patch`, `patches-candidate.py` and the
preserved numerical sources reproduce the experiment after baseline restoration.
The evidence harness uses that saved patch module, with identical generated
candidate source verified independently of the current checkout.

Persistent evidence: `outputs/evidence/b1/perception/ransac-buffers-20261015/`.
`criteria.md` predates testing. Saved inputs cover 32 evenly spaced calls per
condition, extremes of support, residuals nearest 4 mm and every saved refusal.
Each is paired from RNG seeds 0 and 20261015, extracting all clusters and terminal
failure. This is a controlled isolated comparison, not historical RNG recovery.

The local performance gate requires both ≥10% median reduction and ≥2 ms paired
median saving per registration frame in both conditions, p95 regression ≤5%,
and improvement in at least six of seven repeat aggregates. One warmup and seven
alternating paired repeats retain setup/refit/failure costs and every timing.
The CPU budget is 20 minutes excluding preparation. Failure closes full replays.

Conditional GPU gates are both complete 3,217-input fixed native histories,
followed by four continuous timed runs. They use existing shared locks and idle/
foreign-compute checks; interrupted cells are preserved. Three physical targets,
10/15/20 mm and 5°, losses, wrong acceptances, gaps, memory, service and publication
age remain separate from timing-induced changes in processed history. No result
here qualifies 150 ms freshness, policies, controllers, A3/A4 or contact.

## Isolated correctness milestone

The selected saved corpus contains 92 light and 56 nominal inputs, including
39/1 saved refusals. All 296 paired comparisons and 32 adversarial cases exactly
match candidate fields, residuals, inlier/remaining masks, draw sequences,
degeneracy/fit/refit results, exceptions and final NumPy RNG state. No numerical
tolerance was needed; the declared float64 tolerance was 1e-12. Adversarial cases
cover insufficient support, source/target degeneracy, noncontiguous remaining
points, weighted/unweighted and zero weights, threshold neighbors, ties and
injected hypothesis/refit errors. Explicit unsupported refit keeps rejection
without changing the remaining mask. See `local.py` and `correctness.json`.

## Local performance

All 1,036 measured pairs are retained: seven repeats of 92 light / 56 nominal
registration inputs, including failed terminal calls. Per-frame medians across
repeats yield:

| Condition | Baseline p50 / p95 ms | Candidate p50 / p95 ms | Median reduction | Paired median saving |
|---|---:|---:|---:|---:|
| Light | 30.781 / 46.152 | 26.282 / 39.761 | 14.62% | 4.202 ms |
| Nominal | 19.855 / 32.109 | 16.842 / 27.673 | 15.18% | 2.896 ms |

Both preset benefit gates pass; every repeat aggregate improves. The corpus is
purposefully enriched with low support/refusals and boundary cases, so these
medians do not replace full-history costs. Timing uses native NumPy 2.1.3,
OpenBLAS one thread and OMP four threads. The summary writer failed on a NumPy
integer after all timings completed; the retained measurements were summarized
without rerunning the benchmark. Original script/error remain preserved.

A separate instrumented CPU profile attributes most remaining cluster work to
repeated degeneracy checks and weighted-fit preparation. Its timings include
profiler overhead and are not a second benefit measurement.

## Complete fixed-input CUDA comparison

All four native sessions complete 3,217 original inputs per path with saved eager
masks, identical RGB-D and initialization, seed 0 and the selected BF16/TF32
context. Order is light baseline/candidate and nominal candidate/baseline. Every
packet, native trace, inlier/decision, reference/history and final NumPy RNG state
matches exactly at every frame. Each condition compares 3,217 reference files;
33 light / 19 nominal audit rows also exactly match TSDF/color/weights and SDF
queries. The current native histories exactly match the saved historical traces,
poses and losses, permitting reuse of their unchanged geometry binding decisions.
No tolerance or numerical exception was needed.

The same three frozen physical targets are scored at 5 degrees on all 2,858
opening captures. Every count is unchanged; cells list correct 10 / 15 / 20 mm:

| Condition / target | Baseline and candidate | Wrong accepted at 20 mm |
|---|---:|---:|
| Light 0 | 2503 / 2788 / 2824 | 26 |
| Light 1 | 2473 / 2788 / 2822 | 28 |
| Light 2 | 2502 / 2783 / 2823 | 27 |
| Nominal 0 | 2808 / 2852 / 2858 | 0 |
| Nominal 1 | 2819 / 2853 / 2858 | 0 |
| Nominal 2 | 2790 / 2851 / 2857 | 1 |

Native losses remain zero. Across 3,216 nonseed outputs, geometry accepts
3,208 light / 3,216 nominal, retaining all eight light refusals. All thresholds,
precision, chronological gaps, terminal tails and original errors remain in
`reference-quality.json` and `fixed-summary.json`. This preserves the existing
quality and qualification limits; it does not improve geometric accuracy.

Ordinary steady frames exclude the initial five source seconds and TSDF updates
(2,901 light / 2,908 nominal paired frames). Values below are p50 / p95 ms:

| Condition / stage | Baseline | Candidate |
|---|---:|---:|
| Light RANSAC | 23.777 / 40.071 | 20.320 / 34.430 |
| Nominal RANSAC | 22.675 / 30.903 | 19.471 / 26.494 |
| Light native service | 111.985 / 142.602 | 108.189 / 136.995 |
| Nominal native service | 97.082 / 129.103 | 94.480 / 124.619 |

Median paired savings are 3.687 / 2.767 ms RANSAC and 3.939 / 2.800 ms native
service. These are measured native-pipeline effects, not sums of isolated model
percentiles. Total launcher duration also contains input reads, audit and trace
IO and is not used as the service benefit. Complete provider service/age remains
separate and is measured below.

Sampled compute-GPU peaks are byte-identical within both pairs: 2.662 GiB light /
6.705 GiB nominal. Joint coordinator/worker RSS peaks are 3.211 -> 3.185 GiB and
3.361 -> 3.295 GiB. Sampling is once per second and does not qualify every startup
spike or indefinite memory stability. No persistent buffer pool or history is added.

## Complete operational comparison

Four continuous recorded-provider sessions complete all 4,718 original arrivals
per run with eager SAM3, frozen selection, seed 0, thresholds and cadences.
Order is light baseline/candidate, nominal candidate/baseline. These are wall-clock
recorded-ingress tests, not concurrent Isaac rendering or physical execution.

Steady service and publication age are p50 / p95 ms; memory is sampled joint
GPU / coordinator-and-worker RSS GiB:

| Condition / mode | Provider service | Publication age | Hz | Skips | GPU / RSS |
|---|---:|---:|---:|---:|---:|
| Light baseline | 207.928 / 270.017 | 228.602 / 293.641 | 4.482 | 4458 | 7.820 / 5.613 |
| Light candidate | 222.019 / 277.074 | 242.694 / 294.479 | 4.322 | 4464 | 7.939 / 5.602 |
| Nominal baseline | 218.616 / 271.403 | 244.218 / 294.253 | 4.246 | 4469 | 9.707 / 5.581 |
| Nominal candidate | 217.064 / 272.353 | 243.109 / 294.399 | 4.361 | 4471 | 9.188 / 5.530 |

There are zero publications within 150 ms among **819 steady outputs** and zero
memory-sampler errors. Publication-interval p95 is 283.121 -> 283.822 ms light
and 282.959 -> 283.078 ms nominal. Maximum intervals fall 750.461 -> 500.982 ms
and 983.599 -> 467.537 ms, but the large nominal baseline service stall contains
846.0 ms of SAM3 time. Its internal cause is unisolated and no foreign compute
was observed in that valid run. RANSAC cannot be credited with removing that
unmatched SAM3 stall.

Native nonseed output counts, losses and geometric acceptances remain distinct:

| Condition / mode | Native outputs | Native losses | Geometry accepted |
|---|---:|---:|---:|
| Light baseline | 223 | 68 | 155 |
| Light candidate | 218 | 48 | 117 |
| Nominal baseline | 219 | 38 | 181 |
| Nominal candidate | 216 | 42 | 174 |

Light candidate also has 60 `degenerate_registration` refusals. Rejection reasons
can overlap; fewer native losses do not establish better geometry or admission.

The same three physical targets at **5 degrees**, correct **10 / 15 / 20 mm**
opening counts, retain every scheduled capture, including skips:

| Condition / target | Baseline correct | Candidate correct | Wrong accepted at 20 mm, baseline -> candidate |
|---|---:|---:|---:|
| Light 0 | 140 / 142 / 142 | 42 / 42 / 42 | 4 -> 64 |
| Light 1 | 138 / 141 / 142 | 42 / 42 / 42 | 4 -> 64 |
| Light 2 | 141 / 142 / 142 | 42 / 42 / 42 | 4 -> 64 |
| Nominal 0 | 157 / 162 / 163 | 159 / 163 / 165 | 2 -> 1 |
| Nominal 1 | 157 / 161 / 162 | 154 / 162 / 166 | 3 -> 0 |
| Nominal 2 | 156 / 162 / 163 | 159 / 163 / 165 | 2 -> 1 |

Every all-scheduled denominator is 2,858 opening captures; quality is evaluated
at capture time, not at the later publication time. At 20 mm, accepted precision
falls 97.26% -> 39.62% light; nominal is 98.18–98.79% -> 99.40–100%.
All-scheduled correct availability is 4.97% -> 1.47% light and 5.67–5.70% ->
5.77–5.81% nominal. Processed-only denominators, all three thresholds, precision,
error peaks, last-five-second values and detailed gaps remain separately in
`operational-summary.json` and `compact-summary.json`.

At 20 mm the maximum gap is the censored final tail for all targets: 4.950 ->
36.150 s light and 4.583 -> 3.183 s nominal. At 10 mm, baseline light target 1
has a 9.233 s final gap; other baseline light targets have 4.950 s. Every candidate
light threshold/target retains the 36.150 s tail. No failed tail is interpolated
away or counted as available.

## Numerical equivalence, history and adoption decision

Timed native seeds are **27.1667 -> 26.9333 s light** and **25.7167 -> 27.2000 s
nominal**, unlike the identical 25.0167 s dense seeds. Only 52 light / 31 nominal
published capture IDs are shared between timed paths, including inspection.
Different initialization and sparse histories begin before native RANSAC can
supply its later speedup. Scheduling, elapsed service and sparse tracking can
therefore change both the input history and resulting quality. The exact fixed
pairs establish no observed arithmetic/decision regression; these timed pairs
do not isolate the optimizer's speed as the sole cause of their quality change.

**Keep the baseline as the maintained default.** The local and fixed-native
performance gates pass, with real 14–15% RANSAC savings and about 3–4 ms lower
native service. Complete provider service/age does not improve consistently,
and the unmatched light history has much worse correct support, precision and
continuity. Default adoption is conservatively unsupported by this complete
comparison. This is not a failed numerical equivalence gate or evidence that
point-buffer precomputation is intrinsically inaccurate. No favorable repeat,
threshold change, model change or second optimization is used to obtain adoption.

The maintained patch module and installed registration are restored byte-for-byte
to `fa36656` / `register-baseline.py`; SAM3 eager, TSDF baseline and all rejected
compilation evidence remain intact. Candidate `dd2f0ae`, local-result commit
`b5d0f69`, setup patch and all executable harnesses remain reproducible. The saved
setup module regenerates exactly the evaluated candidate independently of the
restored checkout. `decision.json` and `README.md` are evidence entry points.

The next CPU cost is repeated degeneracy checks (`np.cross` dispatch) and
weighted-SVD preparation; the diagnostic profile supports that direction but
is not a new optimization. Sequential eager SAM3 (about 108 ms in these timed
runs) plus history-sensitive P2P still dominates complete service. A subsequent
operational adoption comparison needs a matched sparse initialization/history
control before interpreting scheduler-induced quality changes. No claim of
150 ms readiness, qualified tracking, policy, controller, A3/A4 or contact follows.

## Shared-GPU interruptions and budget

Existing locks and foreign-compute preflights protect all GPU cells. Two light
baseline prefixes (392 / 314 frames) and one nominal candidate prefix (2,798)
are preserved as interfered. The last interruption was foreign RustDesk compute;
only our process groups were stopped. Completed valid cells were reused and only
incomplete cells restarted. No other task or source was changed.

Valid full fixed sessions consume 27.15 minutes; four timed sessions consume
6.09 minutes. Interfered sampled active spans total 7.54 minutes. Total suite
wall time is 105.68 minutes, dominated by shared-GPU waiting; it is not service
latency. The preset 100-minute fixed and 12-minute timed active budgets hold,
also counting interrupted prefixes. All workers exit and both leases are released.
Sampling does not establish indefinite memory stability or startup-spike bounds.
