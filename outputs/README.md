# Outputs

Generated results remain local and ignored by Git. Selected pretrained resources
live in `models/perception/`; reusable recordings/calibration live in `datasets/`.

`b1/perception/evidence/` consolidates completed perception experiments under their
original run names, including `geometric-*`, `operational-tracking-*` and the SAM3
video comparison. Retained records include protocols, corrected per-door scores,
initial/interrupted failures, source revisions and decisive images. Corrected aligned
component scores supersede the original offset scores. Reports retain their original
content; embedded execution paths resolve through `cleanup.json` relocations.
That inventory also records payload removals and preserved recording headers.

Superseded recordings, discarded model weights, temporary scripts, feature/mask
caches, repeated dumps/media and logs without residual value were removed.
Ignored payloads are not Git-recoverable; the retained results do not promise
complete executable reproduction. See
[Perception Findings](../knowledge/wiki/experiments/b1-perception-findings.md).

`b1/perception/operational-scan-diagnosis-01/` preserves the corrected 6.0B static
reconstructions and human-reviewed region evidence. Final per-case records use
`final/<asset>/<condition>/final-report.json`; duplicate reports were removed.
`operational-contact-readiness-01/` preserves finite mesh-band support and endpoint
IK comparisons. Human local-role confirmation, geometry and reachability remain
distinct from motion identity, continuous collision and loaded-control qualification.

The cleanup's fresh frozen CUDA smoke and four 0–25 s scans use
`b1/perception/evidence/cleanup-validation-20261002/`. Earlier cleanup validation
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

Prepared assets remain under `assets/doors/b1/`. Qualification and Purdue runtime
verification normally remain under `~/.cache/alexdoor-xas/`. No policy result or
qualified perception release exists.
