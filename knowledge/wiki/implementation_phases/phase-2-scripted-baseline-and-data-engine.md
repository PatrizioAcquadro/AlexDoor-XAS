# Phase 2 — Scripted Baseline and Data Engine

> Historical phase record. Current contracts are documented in [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Objective

Create a deterministic door-push generator and matched A1-A4 episode exports.

## Subphase 2.1 — Scripted Execution and Recording

#### Implementation

This phase established the approach/contact/push/release controller, pre-action recording alignment, terminal-response capture, and matched representation export from one physical episode.

The B0 generator, dataset identity and old-schema compatibility were retired. Numerical recording and matched-export utilities remain; B1 integration is deferred to Phase 6.

#### Key Decisions

- Door-relative frames are explicit and hinge anchored.
- Representation products share physical identity and factual outcome.
- Reusable exports remain in `datasets/`; scripted staging lives in the runtime cache.

#### Problems / Limitations

- A1 is not learned and A4 has no learned policy.
- Scale-candidate, paired-master publication, and legacy sidecar workflows were removed.

## Artifacts

B0 local datasets and generation workspaces were removed. Their documented results remain historical; code/contract history is available at Git `9c16e3d`.

## Files

- `src/alexdoor_xas/policies/scripted/`
- `src/alexdoor_xas/recording/`
- `src/alexdoor_xas/dataset/export.py`
