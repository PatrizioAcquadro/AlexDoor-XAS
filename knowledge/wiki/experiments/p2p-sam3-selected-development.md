# Selected SAM3 OFF: Memory and Quality Diagnosis

Baseline: `main` at `7b60c17`, with no pre-existing tracked changes. The selected
[[p2p-sam3-unified-frontend|SAM3 OFF recipe]] is preserved: one SAM and one original
leaf, `door surface`, seed0, TAPIR Large/480/four iterations/all-query, SuperPoint,
graph off, partial reference batch on, rollback off, five inliers/4 mm, vectorized
SDF Jacobian and 120 active references. Operational defaults, A3/A4 and policies
are unchanged. No model comparison, new acquisition or live resampling is added.

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
regardless of score. The recipe uses stride1, seven spatial memories and 16 object
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
and GPU growth to retained tracker outputs and cached output masks. At index300,
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
`sample_filter_enable` and `sample_reject_depth_edge`, radius 2 and span0.01 m.
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
outweigh a worse full sequence. Results remain pending.
