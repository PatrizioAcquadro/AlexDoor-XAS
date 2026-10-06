# Point2Pose Refit Hypothesis Preservation

Work starts from clean `main @9c8003c`, the only local branch. All earlier
recordings, baselines, failures, models and original initializations remain
preserved. The starting comparator is bounded renewal without the global graph,
with five inliers at 4 mm and 120 active references per candidate.

## Isolated implementation

`refit_seed_rollback=True` retains the winning RANSAC seed only when its own
weighted SVD refit loses the required support. The seed is revalidated on the
same remaining pool; pose, indices, count, mean residual and pool removal all
describe that returned transform. Valid refits are untouched. Sampling, tie
ordering, selection, SDF, final support, jump guards and renewal stay fixed.
The option is disabled by default and independent of partial batch admission.

Saved numerical diagnosis reproduces the observed seed-to-refit failure,
including nominal 63.3833 s and light 70.6500 s. Eleven failed pool winners in
the selected diagnostic cohort need an extra 0.035–1.488 mm for the fifth
refitted pair. Three seeds are already inaccurate at 10 mm/5 degrees.
These are material residual crossings, not floating-point comparisons.
This selected cohort does not estimate their population frequency.

No separate refit tolerance is justified by an independent sensor-noise bound.
Choosing it to retain these examples would tune acceptance to known failures;
declaring support at a different tolerance while later checks use 4 mm would
also break the support contract. No tolerance variant or four-inlier reduction
is implemented or tested. The rollback itself retains the unchanged 4 mm gate.

## Evaluation protocol

One fresh CUDA attempt per original light/nominal condition, from the original
31 s automatic initialization through all 2,858 original samples at 60 Hz.
Ground truth remains evaluator-only. Compare correct accepted poses within
10/15/20 mm and 5 degrees, wrong acceptances, unavailable intervals, conditional
and all-finite errors, outliers, and every candidate's memory/support behavior.
The recorded range ends at 78.6167 s / 63.7209 degrees. It does not validate
90.7-degree travel, absolute initialization, material identity, contact or
150 ms operational freshness. One attempt does not establish repeatability.

The combined partial-batch variant is conditional on the isolated result;
comparison must retain baseline, partial-only and rollback-only trajectories.
Jump-guard and late-correspondence diagnosis will propose the next intervention
without adding another correction in this phase.

Ignored evidence and scripts are under
`outputs/b1/perception/point2pose-refit-hypotheses-01/`; prior evidence remains
in [[point2pose-isolated-improvements|isolated comparisons]] and
[[point2pose-bounded-global-graph-ablation|the graph-off baseline]].
