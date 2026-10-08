# Selected SAM3 OFF: Memory and Quality Diagnosis

Baseline: `main` at `7b60c17`, with no pre-existing tracked changes. The selected
[[p2p-sam3-unified-frontend|SAM3 OFF recipe]] is preserved: one SAM and one original
leaf, `door surface`, seed0, TAPIR Large/480/four iterations/all-query, SuperPoint,
graph off, partial reference batch on, rollback off, five inliers/4 mm, vectorized
SDF Jacobian and 120 active references. Operational defaults, A3/A4 and policies
are unchanged. No model comparison, new acquisition or live resampling is added.

The selected settings are maintained in `configs/point2pose_selected.json`;
`models/perception/README.md` documents the explicit `offline_episode` invocation.
The original ignored `selected-recipe.json` remains historical evidence.

Original light/nominal recordings, all-query baseline, unbounded SAM3 OFF and
all previous attempts remain intact. New evidence uses the separate ignored root
`outputs/b1/perception/p2p-sam3-selected-01/`. Both conditions use original rows
1860–4717 (2,858 captures), 31–78.6166667 s, original timestamps and initialization.
Ground truth is restricted to evaluators. Steps are sequential: memory equivalence,
saved-data diagnosis, then one justified quality change or a discriminating check.

## SAM3 payload lifetimes

The pinned source is `2345a4ad109ac29c569da749c91d84f10dc08c40`. Audit the actual
SAM3 consumers rather than copying SAM3.1's age cutoff. `sam3_tracker_base.py`
`frame_filter` selects up to 15 qualifying nonconditioning outputs using
`eff_iou_score > mf_threshold` (0.01), then includes the immediately previous frame
regardless of score. The recipe uses stride 1, seven spatial memories and 16 object
pointers. A useful high-score frame can remain selectable across an arbitrarily
long low-score interval: its age is not an expiry criterion.

After encoding/output, `release_sam3_forward_history` retains the latest 15
qualifying outputs plus current output, preserving their original tensors and
ordering. Displaced older qualifying outputs cannot re-enter any future forward
selection; old nonqualifying outputs never become qualifying. The original
conditioning output, constants, identity/confirmation/occlusion state, text and
native feature caches stay intact. Reconditioning must be OFF; stride/selection
settings outside the supported recipe are refused rather than guessed.

Expired nonconditioning payloads are removed from the aggregate output dictionary,
per-object views, temporary views, consolidation indices and tracked-frame records.
Past output masks and per-frame exported scores/suppression records are released
only after the current output has been consumed. The input image/stage/prompt
containers retain seed and current payload with original global indices; released
reads fail explicitly. There is no GPU-to-RAM offload. Native backbone caching
already evicts previous features and remains unchanged.

A 301-frame native audit attributes RAM image growth to 6,096,384 bytes per capture
and GPU growth to retained tracker outputs and cached output masks. At index 300,
input storage is 1,835,011,584 CPU bytes, tracker state 1,153,459,912 GPU bytes and
cached masks 173,376,000 GPU bytes; these are named overlapping state measurements,
not independent allocations to sum. `state-audit.jsonl` preserves the containers.
Two failed audit launches before model initialization are preserved in logs; the
supported Isaac Lab launcher then completed the audit on RTX 4090.

`selector-equivalence.json` exercises the exact pinned native selector for 10,000
synthetic arrivals including long low-score gaps. Its selected indices and original
payload identity match at every frame; at most 16 nonconditioning payloads and two
input images remain. A retained useful memory reaches age 1,549 frames. This
mechanical check complements, and does not replace, complete recorded CUDA replay.

Both complete CUDA replays pass: 2,858/2,858 frames per condition, no inference
failure/retry, zero differing masks, poses, tracking/acceptance decisions, reference
history or complete native registration traces. Full P2P runtime/configuration is
identical; SAM runtime differs only in `bounded_forward_memory`. Every original
capture and failed tracking row remains. All three original physical-target scores
exactly reproduce saved SAM3 OFF at 10/15/20 mm and 5 degrees. Seed is excluded
from correctness but retained in the scheduled denominator. `memory-comparison.json`
and `complete-equivalence.jsonl` preserve the complete checks.

| Condition | System | SAM allocated first/final (GiB) | SAM allocator peak (GiB) | SAM CPU high-water (GiB) | Joint sampled GPU peak (GiB) | Complete p50/p95/max (ms) |
|---|---|---:|---:|---:|---:|---:|
| light | Preserved SAM3 OFF | 3.65 / 7.84 | 8.39 | 20.21 | 14.41 | 232 / 264 / 1003 |
| light | Bounded SAM3 OFF | 3.65 / 3.66 | 4.21 | 7.46 | 9.79 | 226 / 253 / 951 |
| nominal | Preserved SAM3 OFF | 3.65 / 7.84 | 8.39 | 18.51 | 12.92 | 233 / 281 / 978 |
| nominal | Bounded SAM3 OFF | 3.65 / 3.66 | 4.21 | 7.46 | 8.29 | 230 / 270 / 975 |

CPU high-water includes model loading and is not current RSS. New simultaneous
100 ms samples measure per-worker live RSS: SAM medians in the first/final five
seconds stay near 2.22/2.27 GiB; P2P map/history costs remain separately reported.
Saved OFF has no per-worker live-RSS series; its high-water must not be relabeled
as live growth. The SAM allocator reaches its bounded plateau after warmup, without
offloading expired tensors to RAM. Startup/seed are separate in the JSON report.
The sampler closes without errors; inference and every original frame resource
record are complete. No nonseed request reaches 150 ms.

**Adopt bounded history for the selected experimental SAM3 OFF recipe**: exact
full-window behavior with substantially lower CPU/GPU resources and a small observed
latency reduction. This is one replay per condition, not a repeated timing study or
an operational freshness claim. Enable `bounded_memory=True` with `version="sam3"`,
`detection_reconditioning=False`. The preserved unbounded recipe and all older
results remain available; operational defaults are unchanged.

## Saved-data residual diagnosis

Memory-only implementation/complete validation: `b911887`. Diagnosis uses saved
OFF output, six short windows and five inspected overlays, without new model
inference. `saved-diagnosis.json`, `local-fit-evaluator.json` and
`diagnosis-{light,nominal}-*.png` retain the full chain. Material labels are
approximate exclusive prepared collision hull membership at 4 mm margin, including
Panel/Handle versus Frame; overlap/unclassified remain separate. Per-point motion
checks assume rigid transport from measured birth depth and evaluator motion;
they are explanatory audits, not a strict material-ground-truth dataset.

**Mask → reference selection.** SAM retains ID0 and a nonempty mask on all 2,858
captures in each condition. Masks visibly include parts of the fixed frame. The
seed already has three/four fixed-only reference births light/nominal; the original
seed is preserved. First contaminated renewals are 35.4167 s light (29 moving-only,
one fixed-only) and 36.2333 s nominal (22 moving-only, seven fixed-only, one
unclassified). They are sampled after that frame's frontend registration: new IDs
cannot explain its current output, and become later references. At these frames
the published primary errors are 1.97/1.27 mm with nine moving inliers and zero fixed
inliers. Presence of contaminating references is demonstrated; downstream causal
attribution is not established by their count alone.

**Selection → TAPIR → registration/SDF → publication.** Light's accepted primary
peak in the final five seconds is 21.26 mm at 75.05 s, with eight published inliers,
all currently moving-only and no current fixed-only inlier. Cluster/SDF seed error
is 21.17 mm; SDF changes it to 21.26 mm and that pose is published. The seven
moving-at-birth inliers have median pixel displacement 2.91 px, measured 3D material
transport discrepancy 11.92 mm and reference-map discrepancy 13.67 mm. A local
unweighted SVD fit retaining only these seven still gives 20.80 mm. These show
correspondence/reference bias on the leaf and do not isolate one unique source of
its accumulation. A mask-only frame exclusion is not a demonstrated cure.

Nominal's exact 70.6 s peak starts from an 8.53 mm previous pose; registration/SDF
seed is 45.27 mm and SDF reduces it to 43.10 mm. Seven inliers satisfy the unchanged
4 mm gate: five currently moving-only and two fixed-only. Publication equals the
frontend/registration output; graph is off and no graph discontinuity occurs.
Re-fitting the saved seven published pairs gives 44.12 mm; restricting to the five
moving-at-birth pairs gives 45.43 mm. Thus removing the two fixed pairs alone does
not repair this frozen local fit. The five-pair cross-covariance singular values
are 1.0701, 0.001997, 0.0000755: the second direction is weak relative to the first.
Median moving-birth pixel/3D/reference discrepancies are 0.81 px/8.01 mm/8.34 mm.
Sparse, poorly spread support and biased references are observed; their separate
causal contributions over the complete history remain hypotheses. Six of the
seven peak inliers were born at a depth span above the configured 10 mm limit,
including four of the five moving-born inliers.

At nominal 78.6 s, SAM's mask remains present and the frontend has 77 selected pairs
(53 currently moving-only, 20 fixed-only; remaining unclassified). There is no
accepted registration cluster and no SDF refinement; the previous pose is retained
and flagged lost. Only four current moving pairs lie within 4 mm of that retained
pose. The final six lost-window budget deferrals are on supported frames; the lost
frames do not renew. Lack of an accepted five-inlier cluster, rather than absence
of the SAM mask, is the immediate refusal. At 78.6167 s native tracking returns with
five moving/two fixed published inliers; that is not strict recovery qualification.

## Configured versus applied depth-jump gate

The actual class is `SuperPointBalancedSampler`. Configuration enables
`sample_filter_enable` and `sample_reject_depth_edge`, radius 2 and span 0.01 m.
Its override of `sample` calls `_valid_depth_mask` for validity/range, but never
calls `_candidate_safety_keep_mask` or `_depth_edge_keep_mask`; inherited attributes
are present but unused by selection. This is a demonstrated configuration/runtime
discrepancy. The inherited helper's radius 2 means a 5×5 patch, despite the YAML's
3×3 comment. Apply the helper's executable meaning, not the comment.

Saved birth-pixel evaluation with the exact native helper rejects the first
renewal's one/seven fixed-only references while retaining 11/12 moving references
light/nominal. It also rejects 18/10 moving references and nominal's one ambiguous
reference. Across the old full reference history, it would reject 468/610 moving
births light and 295/402 nominal; 96/186 nominal fixed births would still pass. It
is a local depth-discontinuity gate, not a general frame/material classifier.
No quality gain can be inferred from these counterfactual counts alone.

## Single quality intervention to evaluate

Repair only the configured depth-jump gate for newly selected renewal references.
Use the inherited `_depth_edge_keep_mask`, radius 2, 10 mm, after the existing balanced
selection and before bounded admission. Preserve original seed references, masks,
SuperPoint ranking/order/crop, TAPIR, registration/SDF, graph off, partial batch on,
rollback off, five inliers/4 mm and every other control. Do not replenish rejected
points, activate other ignored spacing controls, add geometric ownership filters,
change prompts or use truth in workers. Keep this opt-in and separate from memory.

This is justified by the inactive configured gate and measured contaminated first
renewals. It does not claim that depth edges explain every later error. The
required discriminating comparison is one full original CUDA replay per condition,
with identical initialization, original timestamps/failures and the same three
physical targets at 10/15/20 mm and 5 degrees. Adoption must weigh lost/correct
availability, precision, peaks, gaps/tails and resources; a lower local peak cannot
outweigh a worse full sequence. Complete results and adoption follow below.


## Complete quality comparison and adoption

Diagnosis: `bc6fca5`. The sole quality change is opt-in
`reference_depth_edge_filter=True` in the offline coordinator/Point2Pose worker.
`point2pose_sampling.py` applies the inherited gate to native selected renewal
points before bounded admission; initialization is bypassed exactly. The native
configuration, source models, masks, number/order/ranking of candidate selection,
registration thresholds and all other P2P controls remain unchanged.

Each condition completes all 2,858 original CUDA captures without failure or retry.
Seed masks, poses, references and native initialization trace are identical; every
SAM mask over both complete openings is identical to bounded/unbounded OFF.
First pose/object differences occur 35.4833 s light and 36.3 s nominal, after the
first changed renewals. All 354/516 admitted new reference pixels light/nominal pass
the exact native configured gate in saved-data evaluation. Nominal directly records
169 gate calls and 4,554 rejected selected candidates (this includes repeats across
sampling calls, not unique historical IDs). Light's per-call gate trace is missing
because its observer initially compared the acquisition ID with the internal index.
That observer is corrected for nominal; all 354 light admitted pixels, initialization,
masks/poses and chronological receipts remain independently auditable. Missing
light rejection totals are unavailable, not zero; no replay is repeated to improve
instrumentation. Two initial unit failures from optional trace attributes were
corrected; focused regressions and the real CUDA storage check pass.

`quality-comparison.json`, `quality-birth-gate-audit.json`, `report.md`,
`selected-recipe.json`, `full-comparison.png` and `original-target-error-cdf.png`
retain every target distribution, accepted/all-finite error, precision, peak time,
correctness gap, tail, startup/seed and named CPU/GPU/latency measure. Saved baseline
and OFF target metrics exactly reproduce. Original scheduled denominators include
initialization; it is excluded from correctness/errors. Correctness requires both
the position bound and 5 degrees, separately from native presence.

| Condition | Original target | SAM3 OFF availability % 10/15/20 mm | New gate availability % 10/15/20 mm | New gate accepted precision % 10/15/20 mm |
|---|---:|---:|---:|---:|
| light | 0 | 70.33/96.19/99.86 | 85.83/99.79/99.93 | 85.86/99.82/99.96 |
| light | 1 | 69.03/94.05/99.76 | 87.12/99.90/99.93 | 87.15/99.93/99.96 |
| light | 2 | 71.66/97.13/99.90 | 83.80/99.58/99.93 | 83.83/99.61/99.96 |
| nominal | 0 | 87.40/96.54/97.73 | 99.13/99.72/99.86 | 99.19/99.79/99.93 |
| nominal | 1 | 86.21/96.50/97.76 | 99.13/99.69/99.86 | 99.19/99.75/99.93 |
| nominal | 2 | 87.82/96.68/97.76 | 98.74/99.65/99.83 | 98.81/99.72/99.89 |

| Condition/system | Primary position p50/p95/p99/peak mm | Primary rotation p95/peak degrees | Accepted nonseed /2,857 | Native lost /2,857 |
|---|---:|---:|---:|---:|
| light / memory | 4.37/14.62/17.74/21.26 | 0.69/1.04 | 2857 | 0 |
| light / quality | 4.95/12.16/13.62/23.82 | 1.10/1.66 | 2857 | 0 |
| nominal / memory | 4.45/11.71/15.50/43.10 | 0.47/3.85 | 2800 | 57 |
| nominal / quality | 2.23/7.43/9.39/31.48 | 0.68/4.43 | 2856 | 0 |

| Condition/target | OFF maximum correct gaps s 10/15/20 mm | New maximum correct gaps s 10/15/20 mm | OFF last 5 s correct at 10/15/20 mm (/301) | New last 5 s correct at 10/15/20 mm (/301) |
|---|---:|---:|---:|---:|
| light/0 | 3.100/0.100/0.017 | 2.617/0.033/0.017 | 2/212/298 | 8/296/300 |
| light/1 | 5.033/0.333/0.033 | 0.967/0.033/0.017 | 1/170/295 | 19/299/300 |
| light/2 | 3.050/0.100/0.017 | 3.100/0.050/0.017 | 5/235/299 | 7/290/300 |
| nominal/0 | 0.150/0.117/0.117 | 0.033/0.017/0.017 | 153/232/241 | 286/297/300 |
| nominal/1 | 0.200/0.117/0.117 | 0.033/0.017/0.017 | 132/230/241 | 286/296/300 |
| nominal/2 | 0.133/0.117/0.117 | 0.033/0.017/0.017 | 164/233/242 | 283/297/300 |

| Condition/system | Complete p50/p95/max ms | SAM allocator peak GiB | P2P allocator peak GiB | Joint sampled GPU peak GiB | SAM/P2P CPU high-water GiB |
|---|---:|---:|---:|---:|---:|
| light/memory | 226/253/951 | 4.21 | 1.17 | 9.79 | 7.46/2.67 |
| light/quality | 210/235/966 | 4.21 | 0.95 | 12.76 | 7.46/2.40 |
| nominal/memory | 230/270/975 | 4.21 | 1.17 | 8.29 | 7.46/2.91 |
| nominal/quality | 223/260/971 | 4.21 | 0.92 | 7.99 | 7.46/2.63 |

**Adopt bounded SAM3 OFF plus the renewal-only depth gate as the selected experimental
recipe.** All six physical targets improve correct availability at 10/15/20 mm versus
preserved OFF, with much better nominal tails and no native losses. Nominal still
has one integration rejection (`degenerate_registration`, 66.7167 s), retained in
the denominator. At the old 70.6 s primary spike, error is 6.67 mm versus 43.10 mm;
the new nominal maximum is 31.48 mm at 39.7 s. It is relocated/reduced, not eliminated.
Nominal accepted p95 improves 11.71→7.43 mm; all-finite p95 is 7.45 mm and includes
the rejected pose. Complete p95 improves 253→235 ms light and 270→260 ms nominal.

The tradeoffs are explicit. Light's primary median worsens 4.37→4.95 mm, rotation
p95 worsens 0.69→1.10 degrees and its peak increases 21.26→23.82 mm at 75.9333 s.
Nominal rotation p95 also worsens 0.47→0.68 degrees, with maximum 4.43 degrees.
Light target 2's longest 10 mm gap increases 3.050→3.100 s; the final primary 10 mm
score remains only 8/301, versus 100/301 in the old multi-object baseline. This
retains a substantial unresolved light tail despite improved 15/20 mm tails. The
older baseline is stronger on some fine-accuracy/continuity measures and is kept
as scientific evidence, not rerun or replaced on disk.

Quality's joint sampled GPU peak rises 9.79→12.76 GiB light (late, 76.4167 s), while
nominal falls 8.29→7.99 GiB. This is not SAM history growth: SAM remains 3.65→3.66 GiB
and peak 4.21 GiB. Light P2P's final active allocation decreases 0.52→0.43 GiB, but its
Torch allocator reserve increases 3.69→6.72 GiB. That named reserve measurement
accounts for the observed residency tradeoff; fragmentation or a precise allocation
cause is not established. No allocator optimization is introduced. SAM live RSS
stays around 2.20–2.28 GiB and P2P CPU high-water decreases in both conditions.
The measured 24 GiB workstation completes both windows, but this is finite replay
evidence rather than a general bounded-residency guarantee for P2P's allocator.

No nonseed request reaches 150 ms; cold startup, seed, sampled residency and allocator
peaks are distinct. The serial evaluator has one request in flight and does not
measure live acquisition backlog. Material recovery, larger opening angles, hardware
tracking and contact/provider/A3/A4 qualification are not established. Models,
operational defaults, baseline, original OFF configuration and all prior evidence
remain preserved. Those experiments did not include general cleanup, live
resampling, additional optimization or policy changes.

The [[p2p-selected-residuals|new-trajectory same-ID residual diagnosis]] now completes
that saved-window pairing, independently of this preceding unfiltered diagnosis.
Light combines bias already stored at birth, promotion changes and current
correspondence/rotation effects; its peak has nine moving-only pairs. Nominal's
39.7 s peak instead selects seven fixed/one moving pair, while its distinct angular
maximum includes a five-inlier-gate fallback. Keep the selected recipe: there is
no justified single observable common correction. Independent material-pixel
chains and, if needed, unchanged-run filter/SDF state are the next discriminating
evidence; missing historical light rejection totals remain unavailable.
