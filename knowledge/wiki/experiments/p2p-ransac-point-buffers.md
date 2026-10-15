# Point2Pose RANSAC Point Buffers

2026-10-15; baseline `fa36656` on `PatrizioAcquadro/b1-observed-a3a4`, initially
clean. Candidate `dd2f0ae` passes isolated correctness and local performance; full
GPU validation is pending. SAM3 remains eager and
TSDF remains at the maintained baseline; all
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
order. The maintained `patch_tracking` setup/startup path reproduces it and
retains original source; the installed native source is still baseline during
the isolated test.

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

Both full fixed chronological pairs and subsequent timed comparisons are now
admitted conditionally. Shared GPU preflight currently waits for foreign compute;
no peer process is interrupted. Integrated quality, memory, service and publication
age remain unvalidated. The installed source stays baseline until a leased test
selects its own variant; adoption has not been decided.
