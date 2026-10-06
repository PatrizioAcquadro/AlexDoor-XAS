# Point2Pose Consumer Impact and Two Failure Windows

Saved-data diagnosis from clean `main @3430808` on 2026-10-06; evaluator
implementation `28626c1`. The baseline (`59abf31`) and bounded renewal (`2b127b4`)
retain their original 31–78.6167 s recordings, 2,858 rows per condition, automatic
candidate-0, seed pixels, models, thresholds and 120-reference bounded budget.
All five light and six nominal candidates are evaluated. No inference, acquisition,
policy change, ablation, reset, favorable retry or campaign occurs in this diagnosis.
Previous failed variants and missing rows remain preserved. Latency is separate.
Evidence and reproducible analysis scripts are in
`outputs/b1/perception/point2pose-consumer-diagnosis-01/`; payloads are ignored,
not recoverable from Git. See [[b1-perception-findings|prior tracking comparisons]].

## Consumer meaning and measurable quantities

The published frame is an observed material-zone candidate, not a complete door
reference. Let `Z0` be its original measured seed frame, `D(t)` the rigid leaf
motion used only by the evaluator, `Z*(t) = D(t) Z0`, and `Zhat(t)` the saved
publication. The point error is `||p_hat - p_*||`. The oriented normal error is
between their normalized +X axes; full rotation error is scored separately.
Normal/tangential point components use the expected normal. This measures drift
of the original observed zone under a rigid-leaf assumption, excluding seed
alignment error; it does not establish initial metrology, leaf ownership, contact
selection, slip or finite-footprint support.

For an exact saved teacher world command `G(t)`, conditional target propagation is
`Ghat(t) = Zhat(t) inverse(Z*(t)) G(t)`. Its intended local target is held fixed
only in the evaluator. This quantifies the world-position/orientation effect of
the measured rigid transport at the recorded target, using its actual lever arm.
It is not an actual A4 prediction/segment rollout or a truth-derived action fed
to the system. The terminal observation has no next command and stays missing.
No command is shifted to a neighboring timestamp or chosen for a better result.

A3 rotates both free vectors through a **static hinge-frame rotation**. Its
origin does not affect this primitive transform. Point2Pose's moving-zone
orientation cannot be substituted for the static A3 reference. The saved runs
contain no supported closed hinge frame or A3 actions; actual A3 frame/action
errors remain unavailable. The existing observed `fit_motion_axis` is reconstructed
causally from accepted saved poses/bounds, preserving its 32-sample window,
displacement gate and known floor. It yields only a diagnostic origin and relative
angle. Root ownership checks and operational publication are not reconstructed;
closed yaw remains unknown. No privileged hinge is inserted into a provider.

A4 additionally needs an admitted hinge frame, angle and full panel-local segment
with explicit stages/ticks. No such predictions/admissions are saved. The
operational adapter retains its admitted reference and latches on incompatible
updates; this analysis does not claim the graph spike reached a controller.
See [[../topics/action-representations-and-adapters|action consumers]].

## Full saved-window impact

Errors exclude the constructed seed zero. Coverage includes every scheduled row;
finite lost/rejected poses remain scored separately, with their original flags.
Each system/condition has 2,857 finite non-seed zone measurements. Both bounded
and baseline runs retain all 2,858 original rows. There are 62,876 all-candidate
row records across the four saved runs; none is interpolated or reselected.

| Primary condition/system | Accepted rows | Zone point p95 / max, mm | Normal p95 / max, degrees | Conditional teacher target p95 / max, mm |
|---|---:|---:|---:|---:|
| light baseline | 2,311 | 7.79 / 22.37 | 1.18 / 6.22 | 15.38 / 48.28 |
| light bounded | 2,791 | 15.43 / 45.17 | 1.20 / 5.17 | 16.66 / 23.18 |
| nominal baseline | 1,582 | 10.20 / 36.07 | 1.18 / 9.06 | 11.91 / 69.01 |
| nominal bounded | 2,282 | 9.28 / 178.61 | 0.40 / 29.81 | 7.28 / 116.23 |

Bounded light's target sample count is 2,790 because its last accepted observation
has no command. All-finite point/normal/target p95 values are 8.24 mm/1.18 degrees/
15.55 mm for baseline light, 15.71/1.20/16.71 for bounded light, 34.39/2.65/16.94
for baseline nominal and 19.02/2.07/11.96 for bounded nominal. Target errors use
2,856 non-seed samples per complete run. `impact-summary.json` separates accepted,
all-finite, lost-finite and accepted-push results for every candidate; target
rotation errors retain full-pose semantics. Improved percentile or smaller target
maximum does not erase availability differences or accepted outliers.

| Primary condition/system | Diagnostic axis-fit rows | Origin error p95 / max, mm | Relative-angle error p95, degrees |
|---|---:|---:|---:|
| light baseline | 1,820 | 23.28 / 26.69 | 1.20 |
| light bounded | 2,282 | 20.02 / 30.16 | 1.09 |
| nominal baseline | 912 | 27.94 / 32.04 | 1.27 |
| nominal bounded | 1,717 | 28.83 / 29.59 | 0.53 |

These are conditional diagnostic fits, not full A3 reference scores. Their
position-bound p95 is 1.17–1.50 m, not hardware calibration or useful admission.
The zero axis-direction error is imposed by the vertical-axis model and the
recording's vertical hinge. Motion-sample timestamps remain explicit; the nominal
spike is not appended because its inherited correspondence bound exceeds its
displacement, leaving the last sample 16.67 ms old. Full closed-frame accuracy
and actual A3/A4 action errors remain missing.

## Nominal: graph publication at 45.6667 s

The fixed window is 45.4–46.1 s, with later saved probes through the terminal
loss. The event is preceded by another accepted graph outlier at 45.6333 s:
28.10 mm position error and 47 active landmarks changed, maximum 87.33 mm.
At 45.6667 s:

| Stage | Zone point error, mm | Full rotation error, degrees |
|---|---:|---:|
| Registration seed before SDF | 1.17 | 0.20 |
| SDF-refined registration/frontend | 2.92 | 0.58 |
| Actual post-graph publication | 178.61 | 32.73 |

SDF cost falls from 0.001254 to 0.000723 while its small pose error increases;
it does not generate the large publication jump. Post-graph normal error is
29.81 degrees. Point displacement has 147.86 mm normal and 100.19 mm tangential
components. The conditional teacher target shifts 116.23 mm, with 32.73-degree
orientation error and a 610.16 mm zone-to-target lever arm.

The second graph input is a delayed keyframe: its input pose matches the saved
frontend at 45.6167 s, three source frames earlier. It uses that anchor's nine
inliers, not the current frontend's 22. Only one is an original seed reference.
Its inlier cloud has RMS principal extents 54.90/24.84/4.08 mm. The 50 ms source
age is distinct from computation latency and cannot explain 32.73 degrees by
itself. The graph returns all six historical keyframe poses; 50 shared active
landmarks change, maximum 131.01 mm, while 30 pending points become valid.
No reference is retired in either nominal event.

The native LM pose-chain gate uses **the mean over every registration residual**,
including outliers. At the two anchors this is 12.54/14.41 mm, versus inlier means
2.45/2.60 mm. Both exceed its unchanged 10 mm gate. The saved live graph counters
increase by nine factors each, matching nine landmark factors with no between-pose
factor; landmark variables grow 37→44→47. This is code/record evidence of missing
pose-chain constraints, not proof that one constraint would fix the result.
Sparse old-reference anchoring and accumulated landmark inconsistency remain
plausible contributors; factor Jacobians and an ablation are unavailable.

The publication audit holds map and pose independently fixed:

| Same current pairs | Inliers within 4 mm | Of the 22 declared inliers retained |
|---|---:|---:|
| Frontend pose + frontend map | 22 | 22 |
| Published pose + frontend map | 2 | 0 |
| Frontend pose + published map | 22 | 21 |
| Published pose + published map | 10 | 1 |

These are algebraic residual decompositions on saved arrays, **not stateful
ablations or replacement poses**. The worst declared published residual is
433.79 mm. Self-consistency of ten current pairs does not validate the pose.

The TSDF also changes: primary voxels grow 8,660,412→11,449,438→21,086,856; saved
all-candidate rebuild counts grow 8→9→10 over the two updates. On extent growth,
the existing adapter rebuilds with retained keyframes using their updated poses
before integrating the new keyframe. Voxel values and precise SDF-surface changes
were not saved and cannot be inferred from voxel counts alone.

At 45.6833 s the actual publication returns to 1.95 mm/0.24 degrees and the
conditional target to 0.87 mm. All active map coordinates are unchanged from the
bad frame: publication recovery does not restore the map. Of 52 inliers, 46 come
from the two newly promoted cohorts born at 45.5833/45.6167 s. The next graph at
46.5 s moves landmarks again, maximum 30.39 mm. Final nominal loss still spans
60 original frames, 77.6333–78.6167 s; its finite terminal pose error remains
22.06 mm/3.14 degrees. No repaired-state continuation is claimed.

## Light: first accepted deterioration at 52.1333 s

The fixed window is 51.7–52.6 s. Published point errors on the three consecutive
frames at 52.1167/52.1333/52.15 s are 3.69/12.37/5.54 mm. At the middle frame,
error is already 13.30 mm before SDF and becomes 12.37 mm after it. SDF cost
falls 0.000438→0.000162. There is no graph update, landmark change, new point,
promotion or retirement at this frame. Full rotation/normal errors are
1.145/1.144 degrees; conditional target error is 13.11 mm at a 516.74 mm lever.
The zone error is chiefly tangential: 11.91 mm, versus 3.34 mm normal.

All four pose/map residual combinations agree on 13 current inliers; inherited
and published support masks agree. Only four satisfy independent historical
material anchors within 4 mm in the evaluator, and none is an original seed
inlier. Eight of the 13 are from the cohort born at 51.7 s; the remaining five
come from 46.4333, 47.4333, 48.3667 and 50.8833 s. For that recent cohort,
map-to-first-observed-anchor discrepancies at the event reach roughly 8–12 mm
among its supporting points. Such anchors assume rigid membership and carry
first-observation bias; they do not prove physical identity.

The recent cohort was born under an 8.30 mm estimate, then promoted at 51.75 s;
that graph update changes 52 shared landmarks, maximum 16.84 mm. Its nine graph
inliers belong to the earlier 51.7 s anchor, contain no seed reference, and only
two satisfy the independent material audit. Its pose-chain factor is present:
8.96 mm mean residual and ten added factors. This differs from nominal. Sixteen
primary retirements occur between 51.8 and 52.1 s; none occurs at 52.1333 s.
They change the surviving evidence, but their independent causal effect is not
isolated by comparing baseline with a variant whose RNG/history already diverged.

At 52.2167 s another graph moves 55 shared landmarks, maximum 17.25 mm; its
52.1667 s anchor omits the pose-chain factor (10.80 mm mean; nine added factors).
Light TSDF extent remains 12,558,924 primary voxels across these nearby graph
updates. It still integrates keyframes, so unchanged size is not unchanged map.
Without an extent rebuild, previously fused observations are not reintegrated
when historical graph poses change. The effect of this mixed fusion history is
an open SDF-consistency hypothesis; saved voxels are unavailable.

Deterioration is intermittent and persists later: 14/53 accepted frames exceed
1 cm between 52.1333 and 53 s. From the onset to the recording end, 541 accepted
point errors exceed 1 cm among 1,524 accepted/1,590 scheduled frames; 65 are native
lost. At 61.2667 s a graph yields 17.23 mm with nine published inliers, only two
from the inherited set; at the end, 20.83 mm remains accepted with six internally
consistent pairs and zero independent material-compatible inliers. The evidence
supports reference/map bias and changing correspondence subsets before SDF, not
a unique attribution to SDF, retirement or one pixel tracker failure.

## Decision, limitations and next comparison

Saved stages, reference histories and live graph/TSDF counters distinguish the
two mechanisms sufficiently for this diagnosis. No CUDA replay is indispensable.
A fresh short replay at either event would reset the very map/history under
investigation; reconstructing the original causal prefix would be required for
an intervention. No ablation was performed. Any subsequent controlled comparison
must retain the same prefix/controls and measure landmark/TSDF changes plus later
frames, including final loss; suppressing only a displayed pose is insufficient.

Before adopting the renewal policy, the next focused comparisons should test
publication/support consistency across the graph/map update and preservation of
material evidence during replacement. They must distinguish absent pose-chain
constraints, delayed keyframe publication and map/SDF history from observed
reference bias. Closed-reference/contact admission and hardware calibration remain
separate requirements. Neither the percentile gains nor restored next-frame pose
qualifies A3/A4 use, loaded contact or the unchanged 150 ms operational limit.

Only two image sheets are produced: `nominal-three-frames.png` and
`light-three-frames.png`, each showing the original pre/event/post frames with
baseline and bounded actual publications. RGB/depth/frame alignment is checked,
and all values correspond to the displayed original timestamp. There is no
choice of a better stage pose. The archived failed unbounded trials still retain
1,506/2,239 missing rows; no derived figure replaces their evidence.
