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
The sampler's shutdown-only error, if any, stays in its original log; inference and
all original frame resource records are complete. No nonseed request reaches 150 ms.

**Adopt bounded history for the selected experimental SAM3 OFF recipe**: exact
full-window behavior with substantially lower CPU/GPU resources and a small observed
latency reduction. This is one replay per condition, not a repeated timing study or
an operational freshness claim. Enable `bounded_memory=True` with `version="sam3"`,
`detection_reconditioning=False`. The preserved unbounded recipe and all older
results remain available; operational defaults are unchanged.

## Residual diagnosis and quality decision

The complete memory-only equivalence gate passes. Next inspect saved masks, sampled
references, TAPIR correspondences, registration/SDF and actually published poses
at first contaminated renewal, final light drift, nominal 70.6 s peak and final
nominal losses. Verify configured versus applied depth-jump filtering before any
quality intervention. No speculative combination of geometric filters and spatial
prompts is authorized.
