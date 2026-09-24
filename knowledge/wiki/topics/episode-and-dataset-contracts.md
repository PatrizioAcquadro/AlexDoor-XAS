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

Robot identity is an asset ID plus source fingerprint. Export rejects mixed
tasks, robots or identities; no Alex V2 manifest or URDF is required.

## Splits, Normalization and Batches

Datasets use `datasets/<task>/<action_space>/<version>/`. Split and optional view
files assign disjoint episode IDs. Content grouping prevents duplicate numerical
trajectories leaking across splits. Normalization uses only the selected training
IDs and is validated by recomputation.

`EpisodeDataset`, `A4ChunkDataset`, `ChunkSampler` and `BatchIterator` provide
validated records, padded windows and seeded batches. The retained numerical
presets `core`, `core_contact`, `core_door_pose` encode state-vector fields; they
are **not the B1 observed-only policy interface**. B1 must keep simulated hinge,
contact and door truth out of model observations, as defined in Phase 6.

Tests use explicit small arrays and synthetic identities. No local B0 recordings
or external robot asset are needed to verify these contracts.
