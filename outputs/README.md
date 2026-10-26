# Outputs

Run payloads are ignored by Git. Active inputs stay local; verified inactive
archives may reside in persistent project storage:

- [`generated/`](generated/README.md) — new runs, previews and intermediate output.
- [`evidence/`](evidence/README.md) — selected results, baselines, failures and
  diagnostics retained for interpretation or future work.

Start new commands in a fresh `generated/` directory. Retain useful results in
`evidence/` together with their configuration, source revision and required inputs.
Moving a run requires updating its readers; do not rewrite historical measurements.
A failed run can be useful evidence. Neither directory implies qualification.

Reusable data belong in [`datasets/`](../datasets/README.md); model setup is in the
[wiki](../knowledge/wiki/topics/perception-model-setup.md). Isaac qualification
and runtime verification normally stay under `~/.cache/alexdoor-xas/`.

Some inactive historical HDF5 payloads now have adjacent `.archive.json` pointers.
Restore the original path with the pointer's overwrite-protected command before
using its historical reader. The [complete-cycle campaign record](../knowledge/wiki/experiments/act-a1-complete-cycle-collection.md)
documents the verified persistent archives and protected active dependencies.
Ignored payloads are not recoverable from Git.
