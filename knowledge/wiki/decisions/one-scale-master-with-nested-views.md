# Decision — One Scale Master with Nested Views

## Context

Independent N50, N100, N250, and N500 datasets would change episode composition and holdouts together with training size.

## Decision

For the completed scale study, use one 550-episode matched A2/A3 master with nested N50, N100, N250, and N500 training memberships and fixed 25-episode validation and test sets. Maintain train-only normalization for each action-space/view pair.

B0 payloads were removed during cleanup. Generic split/view and normalization algorithms remain, but this decision records a completed study, not a B1 dataset. Historical source state: Git `9c16e3d`.

## Consequences

- Every larger training view contains the smaller view.
- Validation and test membership remain fixed across size and representation.
- Direct split membership and normalization recomputation define current validity.
- The result remains limited to one simulated door family and one completed seed-0 study.

The scale-generation, merge, ledger, cluster-sweep, and publication workflows are historical and no longer executable repository features.

## Version Notes

- 2026-08-13 — Clarified that the nested-view rationale is retained after retirement of the B0 products.
