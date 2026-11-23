# Preserved evidence

Preserve selected results, baselines, distinct failures, diagnostics and their
interpretation here. Payloads remain local and ignored by Git. Relative paths
below start at `outputs/evidence/`.

`b1/perception/` consolidates completed perception experiments under their
original run names, including `geometric-*`, `operational-tracking-*` and the SAM3
video comparison. Retained records include protocols, corrected per-door scores,
initial/interrupted failures, source revisions and decisive images. Corrected aligned
component scores supersede the original offset scores. Reports retain their original
content. Interpret embedded execution paths using the prefix relocations in
`b1/perception/cleanup.json`; maintained local readers use the current layout.
This inventory also records payload removals and preserved recording headers.

The earlier October 2 cleanup removed superseded recordings, discarded weights,
temporary scripts/caches and repeated payloads as recorded in that inventory.
The initial October 8 Point2Pose cleanup preserved ignored payloads. Subsequent
repository consolidation removed SAM3.1/SAM2 Small weights and eight identical
registration-trace copies, reclaiming 4.60 GiB. Relative symlinks preserve saved
reader/report paths to the retained trace: selected-recipe copy when available,
otherwise original producer. Digests, canonical paths and removals are recorded
in `cleanup.json` under `maintenance_20261008`. Distinct results and failures remain.
Ignored payloads are not Git-recoverable; the retained results do not promise
complete executable reproduction. See
[Perception Findings](../../knowledge/wiki/experiments/b1-perception-findings.md).

`b1/perception/operational-scan-diagnosis-01/` preserves the corrected 6.0B static
reconstructions and human-reviewed region evidence. Final per-case records use
`final/<asset>/<condition>/final-report.json`; duplicate reports were removed.
`operational-contact-readiness-01/` preserves finite mesh-band support and endpoint
IK comparisons. Human local-role confirmation, geometry and reachability remain
distinct from motion identity, continuous collision and loaded-control qualification.

The cleanup's fresh frozen CUDA smoke and four 0–25 s scans use
`b1/perception/cleanup-validation-20261002/`. Earlier cleanup validation
remains historical in `cleanup-validation/`. No offline/dynamic/release flag is set.
The active CLI also supports Point2Pose smoke, operational replay/observer checks
and serial offline evaluation. Retired tracker traces are historical results.

Point2Pose offline attempts retain an incremental `frames.jsonl` and separate
native/integration coverage, capture-time errors, losses/recoveries and
initialization outcomes. Their 60 Hz schedule never depends on compute latency;
the operational 150 ms deadline is not an offline scoring criterion. Existing
operational attempts remain distinct. All outputs remain local and ignored.

`b1/perception/point2pose-offline-60hz-01/` is stopped on user request after 4,216
rows in its first attempt. Its original frame records, 502 unprocessed rows,
11 unstarted attempts and full expected-frame counts are preserved. Its read-only
4–10 s audit and plot localize drift to native frontend output and distinguish
verified frame/composition contracts from open registration hypotheses.

Prepared assets remain under `datasets/doors/b1/`. Qualification and Purdue runtime
verification normally remain under `~/.cache/alexdoor-xas/`. No B1 policy result or
qualified perception release exists.

The selected recipe's original and failed runs remain under `p2p-sam3-selected-01/`
and the earlier experiment roots. `p2p-sam3-residual-02/` retains same-ID image
pairs and trajectory-stage figures for future material correspondence checks.
Do not delete `p2p-sam3-followup-01/hull_evaluator.py` or
`p2p-performance-01/full-target-evaluation.json`: later diagnosis depends on them
and the prepared collision geometry. Historical source alone cannot restore
these ignored inputs, traces or models.

`b1/perception/p2p-cleanup-20261008/` contains fresh 33-frame selected smokes per
condition, the existing 12-sample operational smoke, original-source reconstruction
and exact-prefix verification. It is bounded cleanup evidence, not a new complete
replay, latency comparison or provider qualification.

`b1/perception/consolidation-20261008/` verifies the package reorganization with
exact pose/mask/decision comparisons on the same selected 33-frame windows and
12 operational samples, plus one CUDA synthetic probe per handedness. The import
migration record is `b1/perception/consolidation-imports-20261008.json`; maintained
saved-data readers use current imports, while retired launchers require their
original source revision. These checks do not extend scientific qualification.

`b1/perception/model-cleanup-20261008/` retains four before/after CUDA image
comparisons for removal of unused DINOv3 extraction. Boxes, masks and all RGB-D
surface geometry match exactly; measured availability is reported separately.
The original comparison including wall-clock availability remains as a failed
check beside the corrected geometry comparison. Local inventory records the
DINOv3 and SAM3.1 metadata-directory removals and preserves their provenance.

## Linked checkout relocation

The November 23 integration preserved the policy-pilot and D3 evidence under their corresponding `outputs/` paths in the Desktop checkout. Machine-local files with conflicting names, the original policy-pilot patch and the exact source-to-destination map are in `worktree-local-20261123/`. Use `relocation.json` to resolve an old checkout path. For frozen protocols that embed absolute paths, make a working copy for a new run and replace only the relocated prefixes. Keep the original receipts with their acquired paths.

## ACT evidence on workstation DATA

Mount `/media/pacquadr/DATA` to use the completed ACT results. The checkout's
`act-a1-complete-cycle-20261026/remote-results/` link points to
`/media/pacquadr/DATA/AlexDoor-XAS/evidence/act-a1-complete-cycle-20261026/remote-results/`.
The original host and run names are preserved. The
[storage report](../generated/storage-reclaim-20261123/REPORT.md) lists the
host-local files still required for replay or evidence custody.

The `act-a1-diverse-20261019/cold-hdf5/` link mirrors each archived file's
original campaign-relative path with a `.zst` suffix. Restore an individual
file in a working directory with `zstd -d ARCHIVE.zst -o episode.hdf5`.
The adjacent `ARCHIVE.zst.json` gives its original name, SHA-256, size, mode
and modification time. Training and development inputs in the retained ACT index
remain directly readable in the checkout.
