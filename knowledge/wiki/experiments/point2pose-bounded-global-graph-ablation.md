# Point2Pose Bounded Global Graph Ablation

## Question and method

Does disabling global optimized map/keyframe publication remove the observed spike without losing
complete-opening quality?

One original-seed CUDA attempt per light/nominal condition processes every 31–78.6167 s capture
(2,858 each). Graph-off retains keyframe creation, pending promotion, bounded replacement and TSDF
rebuilds. Original models, five-inlier/4 mm gates and the 120-reference active cap stay fixed.

## Result and decision

The nominal graph-publication spike is removed, but complete outcomes are mixed:

| Condition/system | Correct accepted poses | Inaccurate acceptances | Native-lost rows | Accepted point p95 / max, mm | Normal p95 / max, degrees | Conditional target p95 / max, mm |
|---|---:|---:|---:|---:|---:|---:|
| light / baseline | 2,276 | 35 | 531 | 7.79 / 22.37 | 1.18 / 6.22 | 15.38 / 48.28 |
| light / bounded | 2,250 | 541 | 65 | 15.43 / 45.17 | 1.20 / 5.17 | 16.66 / 23.18 |
| light / graph-off | 2,569 | 238 | 49 | 11.51 / 21.05 | 0.05 / 2.77 | 13.95 / 25.39 |
| nominal / baseline | 1,494 | 88 | 1,274 | 10.20 / 36.07 | 1.18 / 9.06 | 11.91 / 69.01 |
| nominal / bounded | 2,183 | 99 | 568 | 9.28 / 178.61 | 0.40 / 29.81 | 7.28 / 116.23 |
| nominal / graph-off | 2,004 | 58 | 794 | 8.15 / 27.52 | 0.05 / 10.71 | 10.54 / 80.78 |

Graph-off primary light quality improves while nominal terminal loss and conditional target tails
worsen. All 32 light / 50 nominal flag returns fail strict historical material-plus-pose recovery.
Light secondary inaccurate acceptances rise 3,355→4,534. No accepted graph-off pose has fewer than
five recomputed metric inliers, yet inaccurate poses remain.

**Retain this as an independent comparison, not a common replacement.** A constant voxel count does
not establish unchanged fused geometry; publication/map consistency and late correspondence/capacity
history remain separate mechanisms.

## Limits and evidence

Actual CUDA active-state storage obeys the cap, but retained maps/keyframes/TSDF are separate costs;
native GPU peaks remain 7.78/9.61 GiB. Diagnostic p95 711/894 ms is not operational freshness.
Different lighting seeds, one camera/door and unrecorded travel beyond 63.7209 degrees prevent
broader qualification.

Evidence: `outputs/evidence/b1/perception/point2pose-bounded-no-graph-01/`; `controls.json`,
`comparison.json`, `impact-summary.json`, `history.json`, original traces and event/tail image
sheets. Git baseline `3430808`, ablation `1689623`.
