# Point2Pose RANSAC Point Buffers

2026-10-15; baseline `fa36656` on `PatrizioAcquadro/b1-observed-a3a4`, initially
clean. One isolated CPU candidate is under evaluation. SAM3 remains eager and
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

Performance and integrated benefit are pending. No GPU replay has started.
