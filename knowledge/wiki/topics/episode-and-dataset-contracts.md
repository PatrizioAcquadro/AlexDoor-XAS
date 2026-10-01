# Episode and Dataset Contracts

These are reusable numerical recording/export utilities. The B0 dataset and its
robot-specific provenance compatibility were removed. The B1 observation, truth,
action and demonstration integration is specified in
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]].

The separate B1 RGB-D engineering format and causal recording interface are in
[[shared-door-perception|Shared Door Perception]]. The numerical contracts below
remain unchanged and do not imply B1 policy-dataset compatibility.
6.0A preserves `b1.rgbd.v1` and its sensor keys: old recordings have no measured
torque, even if commands include effort values. Replay reports feedback absent.
No recording migration or torque feature is added. Policy compilation explicitly
checks the observation geometry profile against its binding; release v2 remains
legacy and cannot silently accept operational observations.

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

## Separate B1 Policy Data

`dataset/b1.py` adds `b1.policy-dataset.v1`: one physical episode with N+1 common
encoded observations (including terminal), N A1/A2/A3 commands, and A4 segment
rows with explicit start indices. The four views retain the same physical identity
and clock. Existing samplers operate at control ticks for A1-A3 and segment
boundaries for A4. Teacher commands/stages supply labels only.

Raw preparation reuses the live observation builder and robot FK, with the
recorded calibration and canonical arm/neck order. Only an initial inspection
prefix is excluded from learned arm actions; missing/invalid manipulation inputs
are rejected, not silently dropped. Raw recordings must explicitly have purpose
`b1_matched_policy_demonstration`. The preserved 6.0 engineering recordings are not
promoted into this dataset, and no policy dataset has been generated.

Export requires a frozen train/development identity assignment, all four action
paths, complete A4 stages, aligned boundaries and a qualified/frozen perception
binding. Publication refuses existing destinations and stages new output before
renaming. The new path does not reinterpret numerical `phase2.v2` artifacts.

`load_b1_data` builds train-only normalization and the two family batch factories;
see [[learned-policy-stack|Learned Policy Stack]] for fixed A4 category scaling
and the distinct B1 checkpoint formats.

Tests use explicit small arrays and synthetic identities. No local B0 recordings
or external robot asset are needed to verify these contracts.
