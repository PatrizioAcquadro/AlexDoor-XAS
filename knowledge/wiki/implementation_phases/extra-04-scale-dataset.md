# Extra 04 — Scale Dataset

> Historical phase record. The retained data contract is documented in [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]].

## Objective

Produce one matched A2/A3 dataset supporting controlled N50-N500 comparisons.

## Subphase E4.1 — Master and Retained Views

#### Implementation

The completed `v3_scale_master` contained 550 matched episodes, 110 per D0-D4 pose, with nested N50, N100, N250, and N500 training memberships and fixed 25-episode validation and test sets.

B0 payloads and their orchestration were removed. Generic grouping, split and normalization algorithms remain; the historical contract is recorded at Git `9c16e3d`.

#### Key Decisions

- Content grouping prevents exact-trajectory leakage across splits.
- Each action-space/view pair uses train-only normalization.

#### Problems / Limitations

- The dataset covers one simulated door family.
- Pose-plan, candidate generation, merge, ledger, and publication tooling were removed.

## Artifacts

No local B0 dataset payload remains. The counts, memberships strategy and conclusions are retained as documentation, not as an active generation workflow.

## Files

- `datasets/README.md`
- `src/alexdoor_xas/dataset/`
