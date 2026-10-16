# Episode and Dataset Contracts

B1 maintains causal `b1.rgbd.v1` recordings and matched `b1.policy-dataset.v1`
policy data. The previous `phase2.v2` reader/export/split/view pipeline is retired;
historical source remains at `097d578`. It is not a second supported data path.

The explicitly unqualified [[b1-policy-pilot|policy pilot]] additionally supports
`b1.policy-dataset.pilot.v1` with A1/A2 only. Its binding, purpose and policy contract
are distinct; partial data cannot enter the final matched format below.

## Recording

The RGB-D engineering interface is documented in
[[shared-door-perception|Shared Door Perception]]. Sensor observations, applied
commands and evaluator annotations remain separate. Preserved engineering recordings have no
measured torque; commands do not substitute for feedback. New recordings retain a
separate 60 Hz measured feedback channel and 120 Hz robot feedback/dynamics stream,
including source, joint-output semantics, support times and canonical joint order.
Partial pilot exports preserve complete raw sensor/calibration/command/phase groups. No recording migration
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

## Learned pilot input and reuse boundary

The `rgbd-conv-v1` pilot stores four causal 5x60x96 image tensors and current
18-value proprioception; the convolution is trained and serialized inside each
family model, never fitted using development images/statistics. Static/recent
channel support remains independently timed. Preparation and inference share
`RGBDEncoder`; `make_model` binds the learned projection to the recorded recipe.
See [[b1-policy-pilot|the shared implementation and cost]].

The [[../experiments/b1-shared-learned-pilot|three-episode A3/A4 reuse audit]] verifies
retained raw observations/calibration/commands/feedback and supervisor-only label
reconstruction. It leaves observed initial geometry, A4 bootstrap, sweep/response
support and the operational release contract explicitly unresolved. Annotation
truth is supervision, not an observed estimate or admission. Partial historical
recordings are not a final matched dataset and are not relabeled for new control.
Dense concatenated image vectors also require a lazy/deduplicated image-loading
path before production scale; the current in-memory numerical loader remains
suitable only for a bounded pilot. No such storage redesign is silently claimed.
