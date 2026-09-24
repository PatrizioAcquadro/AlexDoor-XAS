# Episode and Dataset Contracts

These are reusable numerical recording/export utilities. The B0 dataset and its
robot-specific provenance compatibility were removed. The B1 observation, truth,
action and demonstration integration is specified in
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]].

## Recording and Export

`EpisodeBuffer` stores metadata, timestamped actions, proprioception, object state,
contact and safety fields. `EpisodeOutcome` records factual termination, success,
final angle and simulator termination/truncation flags. Serialization supports
`phase2.v2` HDF5 only; old schemas are rejected rather than upgraded implicitly.

`dataset.export.export_datasets` exports one explicitly recorded episode set:
A2 HDF5, A3 using its recorded frame actions, A4 JSONL chunks and A1 when joint
targets plus the terminal target are present. IDs and outcomes remain matched.
Callers own recording alignment and must provide those fields; this function
does not generate demonstrations or infer missing B1 supervision.

Export validates every representation in staging before publication and refuses
existing version directories. A failed publication removes only its new outputs;
previous datasets remain untouched. Episode filenames use full IDs and recording
refuses an existing filename, so distinct IDs sharing a prefix cannot collide.

Robot identity is an asset ID plus source fingerprint. Export rejects mixed
tasks, robots or identities; no Alex V2 manifest or URDF is required.

## Splits, Normalization and Batches

Datasets use `datasets/<task>/<action_space>/<version>/`. Split and optional view
files assign disjoint episode IDs. Content grouping prevents duplicate numerical
trajectories leaking across splits. Normalization uses only the selected training
IDs and is validated by recomputation.

`EpisodeDataset`, `A4ChunkDataset`, `ChunkSampler` and `BatchIterator` provide
records, padded windows and seeded batches. Callers must supply ordered `obs_keys`,
for example `("joint_pos", "joint_vel")`, to observation assembly, normalization
and sampling. Only recorded proprioception is selectable. Numeric object-state
and contact fields live in `record.diagnostics`; the original tables remain in
`record.buffer`. Matching and content grouping include diagnostics independently
of policy observations. The caller remains responsible for recording provenance.

`DatasetNormStats` owns serialization shared by files and checkpoints. Loading
requires the same observation keys in the same order, selected training IDs and
optional view; no preset conversion or normalization fallback occurs. Policy data
loading requires an explicit `datasets_root`, independent of the package location.

Numerical validation checks shapes, finite values, timing, factual termination,
declared contact source and any recorded joint-target position limits. It imposes
no historical force threshold: physical admission belongs to B1 qualification.
These utilities do not implement the Phase 6 RGB-D recording or learning path.

Tests use explicit small arrays and synthetic identities. No local B0 recordings
or external robot asset are needed to verify these contracts.
