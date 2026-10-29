# TSDF Deferred Snapshots: CUDA Cost and Failure Audit

## Question and method

Can materializing CPU TSDF snapshots only on rebuild return remove repeated copies without changing
successful or failed-state queries?

Against `1f9cb20`, a local subclass defers snapshot construction while preserving native integration
order, argument handlers, weights, grid and filters. Captured light/nominal rebuilds contain 17/six
integrations. One warmup and three alternating CUDA pairs per condition compare full arrays,
metadata and signed queries; refusal/error cases examine actual worker reentry.

## Result and decision

Success-path outputs match exactly. Median rebuild time falls 650.2→281.7 ms light (56.7%) and
468.5→268.9 ms nominal (42.6%). Snapshot downloads/allocation/copies dominate the removable host
cost; these are local event measurements, not complete-provider gains.

A failed in-place integration changes the last retained CPU snapshot and the next actual worker
queries. Final-only materialization therefore fails the error-state contract despite the
successful-path speedup.

**Reject adoption.** Production stays at the restored baseline. Complete fixed-input and
timed-ingress replays remain closed; adopted savings are zero. A successor must preserve the last
completed state across a partial write.

## Limits and evidence

The 34 completed local cells include timed pairs, warmups, profiles and 16 refusal/error cells;
failures remain preserved. The externally removed original worktree is not a source of recoverable
ignored payloads. Current preserved evidence contains the reproduced local checks, not a new full
quality/freshness qualification.

Evidence: `outputs/evidence/b1/perception/tsdf-snapshot-20261014/`, including `local_rebuild.py`,
`local-summary.json`, `worker-reentry-summary.json`, `completion-audit.json` and captured volumes.
Git: audit `f960c2f`, completed decision `99fa48e`. See [[tsdf-readonly-launch-inputs]] and
[[tsdf-rebuild-double-buffer]].
