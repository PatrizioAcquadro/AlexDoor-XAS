# Preserved evidence

Keep selected results, baselines, distinct failures, diagnostics and their inputs
here. Payloads are local and ignored by Git; source history cannot restore them.
New runs start in [`../generated/`](../generated/README.md). Runtime/expert
verification normally belongs in `~/.cache/alexdoor-xas/verification/`.

## Interpretation and recovery

`b1/perception/` retains completed perception studies under original run names.
The [experiment pages](../../knowledge/wiki/index.md#experiments) own questions,
methods, decisions and scientific limits; per-run reports own detailed metrics,
protocols, source revisions, traces, masks and decisive images. Corrected aligned
component scores supersede initial offset scores without erasing them.

`b1/perception/cleanup.json` records relocation prefixes, canonical retained copies,
symlinks and historical removals. Interpret embedded old execution paths through
that inventory. Preserve restore/readback pointers; a removed ignored payload is
not recoverable merely because its launcher remains in Git.

Historical removals include superseded recordings/weights/caches on October 2,
and 4.60 GiB of SAM3.1/SAM2 Small weights and identical registration-trace copies
under `maintenance_20261008`. Distinct results/failures and relative links to the
retained trace survive. Superseded recording headers are in
`b1/perception/superseded-recording-headers-20261002.json`. These records do not
promise complete executable reproduction.

## Protected consumers and result boundaries

- Static human-reviewed regions: `b1/perception/operational-scan-diagnosis-01/final/<asset>/<condition>/final-report.json`.
- Local footprint/reachability evidence: `b1/perception/operational-contact-readiness-01/`; endpoint checks do not certify a continuous loaded path.
- Saved residual diagnosis depends on `b1/perception/p2p-sam3-followup-01/hull_evaluator.py`, prepared collision geometry and `b1/perception/p2p-performance-01/full-target-evaluation.json`; retain these inputs.
- Offline `frames.jsonl` keeps scheduled/missing/rejected/lost rows, native/integration denominators and capture-time errors. The interrupted `point2pose-offline-60hz-01/` retains 4,216 processed rows, 502 unprocessed rows and eleven unstarted attempts.
- Earlier bounded maintenance checks remain under `cleanup-validation-20261002/`, `p2p-cleanup-20261008/`, `consolidation-20261008/` and `model-cleanup-20261008/` below `b1/perception/`. Their prefixes do not qualify full openings or 150 ms freshness.

New cleanup preserves all existing evidence. Historical launchers importing
retired APIs use their recorded Git revision; maintained saved-data readers use
current imports. No preserved simulation/teacher/diagnostic result becomes a
qualified learned-policy or physical result through relocation or documentation.
