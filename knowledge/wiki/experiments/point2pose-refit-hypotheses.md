# Point2Pose Refit Hypothesis Preservation

Historical experiment: rollback code was retired at `c01c3a1`; source remains
at `96c0943`/`57ad483`, with all local evidence preserved. Work started from clean `main @9c8003c`, the only local branch. Implementation
commit is `96c0943`. All earlier
recordings, baselines, failures, models and original initializations remain
preserved. The starting comparator is bounded renewal without the global graph,
with five inliers at 4 mm and 120 active references per candidate.

## Isolated implementation

`refit_seed_rollback=True` retains the winning RANSAC seed only when its own
weighted SVD refit loses the required support. The seed is revalidated on the
same remaining pool; pose, indices, count, mean residual and pool removal all
describe that returned transform. Valid refits are untouched. Sampling, tie
ordering, selection, SDF, final support, jump guards and renewal stay fixed.
The historical option was independent of partial batches and disabled by default.
Current code rejects activation while retaining the false compatibility keyword.

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
Ground truth remains evaluator-only. Compare correct accepted relative zone poses within
10/15/20 mm and 5 degrees, wrong acceptances, unavailable intervals, conditional
and all-finite errors, outliers, and every candidate's memory/support behavior.
The recorded range ends at 78.6167 s / 63.7209 degrees. It does not validate
90.7-degree travel, absolute initialization, material identity, contact or
150 ms operational freshness. One attempt does not establish repeatability.

The complete isolated light result fails the combination gate: it loses correct
availability at all three bounds, adds wrong 10 mm acceptances and increases
loss duration and peak error. Do not attempt rollback plus partial batches to
rescue this result. The separate saved partial-only comparison remains visible.
Both isolated attempts complete: 5,716/5,716 native CUDA rows, without missing
frames, process failure or inference retry. No combined variant is attempted.

## Complete light result

All 2,858 CUDA frames complete without failure/retry. Original initialization,
candidate selection, source times, unchanged numeric controls, zero graph state,
120-reference cap and actual unique CUDA storage counts pass verification.

| Variant | Correct 10/15/20 mm | Wrong accepted 10/15/20 mm | Native lost | Accepted point p50/p95/max, mm | GPU peak, GiB |
|---|---:|---:|---:|---:|---:|
| Graph-off baseline | 2,569 / 2,776 / 2,805 | 238 / 31 / 2 | 49 | 2.52 / 11.51 / 21.05 | 7.78 |
| Partial only, preserved | 2,450 / 2,816 / 2,856 | 407 / 41 / 1 | 0 | 5.16 / 11.89 / 22.17 | 8.41 |
| Rollback only | 2,183 / 2,547 / 2,564 | 386 / 22 / 5 | 280 | 5.07 / 11.55 / 63.56 | 7.33 |

All correctness columns also require full rotation within 5 degrees. The seed
is excluded from counts/errors; coverage retains the full scheduled denominator.
The rollback gains/loses 133/519 correct 10 mm rows, 31/260 at 15 mm and 14/255
at 20 mm. Accepted precision at 10 mm falls from 91.52% to 84.97%.
Longest correct-pose absence grows from 0.183 to 0.967 s. Accepted rotation
p95/max grows from 0.526/2.804 to 0.597/5.976 degrees; all-finite point p95 grows
from 11.59 to 12.28 mm. The 63.56 mm accepted peak occurs at 69.0333 s.
The favorable GPU decrease is not a tracking-quality gain.

The variant retains 348 primary rollback clusters, 140 inaccurate at 10 mm/5
degrees. A restored seed is selected on 45 rows; 43 are integration-accepted,
15 with inaccurate returned poses. These observed counts do not attribute every
later error solely to that selected cluster: changing retained pools also changes
shared draws, selection, reference/map and TSDF history. First primary published
pose divergence is 31.6 s, despite the first restored cluster at 31.1 s.
All 2,582 nonempty-cluster returned poses have recomputed residuals/inlier masks
matching the returned transform. The 275 no-cluster calls and five jump refusals
account for the 280 primary native losses. All five refused candidates fail
5 degrees, including those with small zone-point error; the guard is protective
in these observed light cases.

Native returns are 72, correct returns 38, strict historical-anchor returns zero.
Neither a native return nor self-consensus establishes material recovery.
Secondary correct/wrong/lost candidate-frame sums change from 4,258/4,296/2,818
to 4,229/3,991/3,139; they are separate candidates, not extra unique physical leaves.
Their individual errors, gaps and three-bound counts stay in `comparison.json`.

## Complete nominal result

The same initialization/source/configuration/CUDA storage checks pass. Five
inliers and 4 mm are unchanged in every native runtime; rollback is the only
semantic native configuration delta. Partial batch admission is false.

| Variant | Correct 10/15/20 mm | Wrong accepted 10/15/20 mm | Native lost | Accepted point p50/p95/max, mm | GPU peak, GiB |
|---|---:|---:|---:|---:|---:|
| Graph-off baseline | 2,004 / 2,047 / 2,052 | 58 / 15 / 10 | 794 | 2.46 / 8.15 / 27.52 | 9.61 |
| Partial only, preserved | 2,484 / 2,816 / 2,849 | 373 / 41 / 8 | 0 | 6.27 / 12.30 / 22.92 | 10.41 |
| Rollback only | 2,215 / 2,285 / 2,308 | 96 / 26 / 3 | 546 | 3.19 / 9.12 / 22.36 | 8.70 |

Rollback gains/loses 245/34 correct 10 mm poses, 259/21 at 15 mm and 274/18 at
20 mm. Accepted precision falls from 97.19% to 95.85%, while correct availability
rises from 70.12% to 77.50% of the full schedule. The longest correct-pose absence
falls from 7.850 to 5.017 s at 10/15 mm, and to 2.383 s at 20 mm. Native absence
falls to 2.300 s; the last 139 frames remain lost from 76.3167 s (2.317 sampled
seconds). The baseline has 472 lost final frames from 70.7667 s.
Accepted rotation p95/max improves from 0.729/10.720 to 0.590/2.515 degrees.
All-finite point p95/max improves from 100.30/115.94 to 22.34/38.16 mm.
This is a measured nominal benefit with remaining accuracy and support failures.

The variant retains 258 primary rollback clusters, 118 inaccurate at 10 mm/5
degrees. A restored seed is selected and accepted on 43 rows, 19 inaccurate.
All 2,311 nonempty-cluster returned transforms match their recomputed residuals
and strict 4 mm masks. All 546 native losses are no-cluster calls; no jump guard
refusal occurs. Native/correct/strict historical-anchor returns are 48/25/1.
Secondary correct/wrong/lost sums change from 3,304/1,127/7,770 to
3,697/1,256/7,522. Material recovery and ownership remain unqualified.

Neither accepted error peak is a directly restored seed: light 69.0333 s and
nominal 68.7167 s select ordinary refitted clusters with five/six pairs, already
wrong before SDF. SDF reduces support to zero/three; the existing final gate
correctly restores the selected cluster. In light no correct cluster exists at
that call and the translation jump is only 64.33 mm, below 100 mm. Thus strict
support and a working final fallback can still return a wrong pose. The phase
does not add a selection, SDF or jump correction to conceal this outcome.

## Memory and continuity accounting

GPU resident peaks include models/allocator effects; Torch and component peaks
are separate maxima and must not be summed as simultaneous usage. CPU RSS is
not measured. Historical maps, keyframes and retained masks remain outside the
120-reference active-state bound.

| Condition/variant | Torch peak, GiB | Max queries | Causal state, GiB | Masks, GiB | TSDF, GiB | Historical references |
|---|---:|---:|---:|---:|---:|---:|
| light / baseline | 5.27 | 520 | 0.952 | 0.923 | 0.583 | 2,670 |
| light / partial only | 6.06 | 599 | 1.097 | 1.202 | 0.602 | 2,831 |
| light / rollback | 5.19 | 499 | 0.914 | 0.848 | 0.518 | 2,400 |
| nominal / baseline | 5.74 | 630 | 1.154 | 0.798 | 1.832 | 1,890 |
| nominal / partial only | 7.15 | 720 | 1.318 | 1.687 | 1.400 | 2,939 |
| nominal / rollback | 5.63 | 581 | 1.064 | 0.811 | 0.981 | 1,890 |

All candidates obey the actual unique CUDA storage cap, retain stable IDs and
zero global-graph state. Recovered absences end at the first qualifying sample;
terminal absences are censored at the last original sample. Both observed spans
and sampled durations, all intervals, p50/75/90/95/99/max errors and peak times
remain in `comparison.json`. The plot includes every finite lost/rejected pose;
its accepted point CDF alone does not impose the separate 5-degree criterion.

## Saved recovery guard and original nominal tail

The unchanged graph-off nominal baseline has 31 primary jump refusals: 24
candidate transforms satisfy 10 mm/5 degrees, and 29 satisfy 15 or 20 mm/5
degrees. Twenty-nine refusals occur while already lost. These are evaluations
of the actually returned candidate before the guard, not a guard-off replay or
proof that its admission would preserve subsequent reference/map quality.

At 64.4667 s a supported correct candidate has 5.99 mm/0.714 degrees and five
4 mm pairs. Its camera-map translation delta is 151.66 mm from the previous
wrong pose, 27.14 mm/10.69 degrees with four current pairs. The 100 mm jump test,
combined with a single hypothesis below eight inliers, republishes that previous
pose and declares loss. Translation at the map origin is not the zone-point
error. The previous pose also becomes stale through subsequent loss.
The light rollback's five actually wrong jump candidates demonstrate that
blanket guard removal would discard a useful rejection mechanism.

The original nominal terminal loss starts at 70.7667 s. The exact expected
transform supports zero pairs there, but enumerating all 20,475 ascending
four-index subsets finds two supported correct fits; one is 1.854 mm/0.582
degrees with five 4 mm pairs. A rare unsampled fit still exists at the first
lost sample. This is evaluator-only existence evidence, not a runtime proposal.

At 74.2 s/end the same complete ascending-subset enumeration tests 10,626/1,365
fits and finds no five-inlier hypothesis, correct or wrong. Maximum support
among fits within each 10/15/20 mm and 5-degree bound is three/one. This finite
family does not prove the absence of every arbitrary rigid transform, but its
failure differs from the sampled-hypothesis or jump-refusal cases.

Original active references stay at 92, while extracted confirmed pairs fall
from 42 at 63.4667 s to 28/24/15 at 70.7667/74.2/end. Native visible counts fall
from 45 to 30/25/15; almost all depth samples remain valid, and no tentative
fallback points are used. Exact-expected fifth residual grows from 3.614 to
5.582/6.894/20.943 mm. Restoring finite depth alone cannot supply valid support.

Separately reconstructing each reference's measured first-observation material
anchor gives current pair median residuals 12.02/18.27/18.39/25.61 mm at those
four samples, versus current-map/birth-anchor median differences
4.86/5.11/5.35/5.49 mm. The degradation remains with the map contribution
removed. Both coordinate bias and accumulated current correspondence error are
present; these vectors do not add as scalar medians. Anchors assume leaf-rigid
membership and do not independently certify ownership or pixel identity.

The changed nominal trajectory is also diagnosed independently. At the same
70.7667/74.2/end samples it has 36/31/24 pairs and exact-expected 4 mm support
2/0/0. Complete ascending-subset enumeration finds zero supported fits at
70.7667 s, 45 at 74.2 s (zero within 10 mm, 18 within 15 mm, 42 within 20 mm,
all also requiring 5 degrees), and zero at the end. At 74.2 s the retained finite
pose happens to be correct at 9.65 mm/1.36 degrees, but has only one current pair;
it is lost and unavailable. Geometric correctness of a fallback is not support.
Birth-material pair median residuals at these samples are 20.68/17.17/24.18 mm,
with map/birth-anchor differences 6.11/6.41/6.87 mm. Ninety-one active references
still defer a full batch of 30. These histories differ from the original 92-
reference trajectory; no saved-pair enumeration predicts a combined replay.

## Decision and unimplemented recovery hypothesis

Keep the bounded graph-off baseline as the comparison reference, five inliers,
4 mm, original models/prompts/seeds, and the independently tested partial-only
diagnostic. Own-seed rollback is now historical source; its mechanical support
correction and real nominal benefit remain documented.
Reject its common/default adoption: light loses correct availability at every
bound, adds inaccurate acceptances, increases absences and develops a 63.56 mm
peak. Fewer nominal losses and lower memory do not offset that common-quality
failure. No graph-on combination or operational default is qualified here.

Reject an uncalibrated refit tolerance; no threshold widening or four-inlier
variant is tested. Do not add partial batches to rescue the failed isolated
common result. Preserve all earlier results and the separate partial-only
precision/coverage tradeoff. No further correction is implemented in this phase.

The experiment proposed, but did not implement, an isolated recovery check on the
original graph-off baseline:
evaluate current-pair support of the previous pose before treating its jump as
a veto, and require observed temporal/material consistency for a candidate
when the previous pose is lost/unsupported. Preserve the normal jump guard and
five-inlier/4 mm gates. In the saved baseline this lost/unsupported condition
contains 22 correct 10 mm refusals but also five inaccurate ones, so simply
bypassing the jump test on that condition is not justified. This proposed check
cannot repair late pairs that lack compatible support; reference/correspondence
renewal remains a separate subsequent problem. No P2P replacement, policy edit,
parameter search, acquisition, training or full campaign follows.

Numerical/contract regressions pass (42 passed; the sandbox CUDA storage test
is skipped, then actual storages are verified across both full GPU runs).
Lint, whitespace, canonical links and visual plot review pass. A saved-tail
evaluator initially lacked its native `_min_var` helper attribute; its failed
log and evaluator-only repair are retained, without an inference retry.

Ignored evidence and scripts are under
`outputs/b1/perception/point2pose-refit-hypotheses-01/`; prior evidence remains
in [[point2pose-isolated-improvements|isolated comparisons]] and
[[point2pose-bounded-global-graph-ablation|the graph-off baseline]].
