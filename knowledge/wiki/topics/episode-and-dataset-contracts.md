# Episode and Dataset Contracts

B1 maintains causal `b1.rgbd.v1` recordings and matched `b1.policy-dataset.v1`
policy data. The previous `phase2.v2` reader/export/split/view pipeline is retired;
historical source remains at `097d578`. It is not a second supported data path.

## Recording

The RGB-D engineering interface is documented in
[[shared-door-perception|Shared Door Perception]]. Sensor observations, applied
commands and evaluator annotations remain separate. Existing recordings have no
measured torque; commands do not substitute for feedback. No recording migration
or policy-dataset production is performed by repository consolidation.

## Matched Policy Data

`dataset/b1.py` provides `b1.policy-dataset.v1`: one physical episode with N+1 common
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

## Normalization and Sampling

`ChunkSampler` and `BatchIterator` consume B1 records, preserving padded action
windows, explicit ordered observation keys and seeded batch ordering. Statistics
use training identities only. `DatasetNormStats` retains the B1 checkpoint wire
format; its reserved `view_id` is null. Separate historical view files and loaders
are retired. Reordered observations, invalid statistics and stale membership fail
before training or policy loading. Asset identity remains mandatory.
