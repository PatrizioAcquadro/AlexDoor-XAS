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
Prepared observations support contiguous arrays, compressed/VDS arrays and the
`frame-indexed-v1` HDF5 group layout. Raw preparation streams through the unchanged
observer into a temporary frame store: exact image blocks are stored once while
per-control image indices and 18-value proprioception reconstruct the same vector.
Learned RGB/depth frames are uncompressed; coverage uses lossless LZF. Image values
use native float32 only when the round trip is exact, widening to float64 otherwise;
reconstructed vectors preserve their original dtype. Exact matching preserves
signed zero too. No overlapping history tensors or trainable encoder outputs are stored. The binding
still fixes preprocessing, support and existing history; storage selects no new
observation or temporal policy. Raw recordings retain all original information.

`B1Dataset` checks identity, actions, boundaries, first/terminal inputs and clocks
at open; sampling and normalization check selected values before use. Set
`check_observations=True` for a complete bounded-memory scan, including unused
rows. Finite-value failure can otherwise occur on first access instead of open.
Compressed/VDS observations stay selective; contiguous historical files retain
read-only memory maps. A shared pool limits HDF5 readers to four files with 1 MiB
chunk caches per open dataset. `close()` drains handles; later reads reopen them. Batch reads
group by episode and restore the same seeded order, duplicates and padding.

`index_dataset(source, destination)` is a non-destructive streaming compatibility
path for historical prepared files. It preserves values, dtype, metadata, splits
and normalization membership. Embedded raw data stays linked to its retained
source unless copying is explicitly requested. Direct random reads from historical
VDS layouts remain usable but can be slower; conversion avoids their repeated
virtual-map traversal. `materialize_train_data` still provides exact-float32 dense
train-only files when required by an older consumer; it now reads indexed data in
blocks too. A1 datasets are released before A2; no dataset redesign or physical
qualification follows from these infrastructure changes.


## Pilot rollout retention

`recording.rollout.RolloutWriter` and `pilot_physics.py rollout` retain evaluation
artifacts separately from imitation data. `b1.policy_rollout.v1` stores the raw
images actually consumed at 10 Hz, all 60 Hz proprioception/full robot/tool
states, submitted commands and applied targets, and the native 120 Hz robot
feedback. Local rollout images are lossless, uncompressed single-frame chunks;
retained captures are stored once and per-control static/three-history indices
join them to acquisition/availability clocks. Invalid feature support and sampled
failed inputs remain explicit. RGB/depth/mask bytes, NaN payloads and raw SDK
intrinsics are preserved. Numeric streams use bounded batches of at most 128 rows,
drained before the unchanged 60-control-tick flush and on finish/close. Pending
rows participate in logical clock and column validation. Storage errors leave the
file incomplete and cannot retry a partly written batch. Historical compressed
recordings remain readable. See [[../experiments/b1-runtime-speed|the adopted
runtime layout and measured limits]].

Pose-error source coordinates are emitted as native Python numbers. A live
recording exposed NumPy float32 camera coordinates nested in contact/admission
diagnostics, which strict JSON could not serialize. Converting those coordinate
scalars preserves their exact numeric values and retains strict non-finite
rejection; geometry, ownership and contact decisions are unchanged.

The IO recording hook snapshots applied targets after execution and before a
response failure replaces them with a stop anchor. Cartesian submissions retain
the actual submitted world goal; A1 retains its submitted joint increment and
bounded joint targets. Inspection holds and learned submissions have distinct
codes. Independent contact records, original categories, feedback semantics and
door state go to evaluator sidecars/streams, never back to the observer. A closed
failed episode is complete evidence; interrupted files remain incomplete. This
format is not admitted by the demonstration exporter and supplies no success or
contact qualification. Numerical round trips and callback ordering pass; live
Isaac recording and full retained-input audit await the GPU extension.
