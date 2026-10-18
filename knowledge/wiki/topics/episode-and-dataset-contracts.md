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
Prepared views now use exact float32 file-backed observations and streaming
batches, preserving original train-only normalization membership. A1 datasets are
released before loading A2; host/process RAM guards remain. This bounds the pilot
path without establishing production-scale or final matched-data qualification.


## Pilot rollout retention

`recording.rollout.RolloutWriter` and `pilot_physics.py rollout` retain evaluation
artifacts separately from imitation data. `b1.policy_rollout.v1` stores the raw
images actually consumed at 10 Hz, all 60 Hz proprioception/full robot/tool
states, submitted commands and applied targets, and the native 120 Hz robot
feedback. Lossless gzip/shuffle image tiles avoid repeating the four-image input
at each control tick. Per-control static/three-history image indices and feature
acquisition/availability clocks preserve causal reconstruction without saving
repeated prepared tensors. Invalid feature support remains explicit; sampled
failed inputs are retained. Raw validity, NaN bit patterns and SDK intrinsics
are preserved. Dataset handles remain open for the writer lifetime so bounded
HDF5 chunk caches survive between rows; partial image tiles are compressed at
flush/close instead of on every reopen. The 60-control-tick flush cadence,
column validation and completion marker are unchanged. Recorded-input checks
preserve all numeric/image bytes across chunk boundaries, including NaN payloads.
The robot occlusion filter also reuses its selected depth array within one
capture instead of copying it twice per mesh. It recomputes the array on every
call; no observation, pose, freshness or mask is reused across captures. Sampled
recorded head/arm configurations retain identical complete masks. These component
improvements do not establish an end-to-end runtime target.

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
