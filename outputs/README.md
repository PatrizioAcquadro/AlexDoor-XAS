# Outputs

Run payloads stay local and ignored by Git:

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
