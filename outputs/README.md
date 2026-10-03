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
The active CLI supports only image smoke and static scan diagnosis; retired tracker
traces are results, not executable entry points.

Prepared assets remain under `assets/doors/b1/`. Qualification and Purdue runtime
verification normally remain under `~/.cache/alexdoor-xas/`. No policy result or
qualified perception release exists.
