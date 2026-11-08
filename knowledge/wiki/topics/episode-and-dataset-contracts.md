# Episode and Dataset Contracts

B1 maintains causal `b1.rgbd.v1` recordings and matched `b1.policy-dataset.v1`
policy data. The previous `phase2.v2` reader/export/split/view pipeline is retired;
historical source remains at `097d578`. It is not a second supported data path.

The explicitly unqualified [[b1-policy-pilot|policy pilot]] additionally supports
`b1.policy-dataset.pilot.v1` with A1/A2 only. Its binding, purpose and policy contract
are distinct; partial data cannot enter the final matched format below.

The separate opt-in `b1.temporal-recording.v1` and `b1.temporal-frames.v1` path
supports independent native sensor/control/feedback streams and finite-history
ACT. Preparation defaults to diagnostic-only labels; explicit admission and
shared readiness determine training rows without image reads. History sampling,
action interval and prediction horizon are independent views. Raw NaNs, failures
and interruptions remain available. See
[[act-a1-observations-and-time|temporal APIs, semantics and measured limits]].
The existing matched/pilot formats and prior efficiency results are unchanged.

## Recording

The RGB-D engineering interface is documented in
[[shared-door-perception|Shared Door Perception]]. Sensor observations, applied
commands and evaluator annotations remain separate. A teacher's applied target
profile must match the declared action interval and execution path. A controller
that changes motor targets between control boundaries needs corresponding
recording and executor support; selecting its first, final or mean target does
not reproduce a nonconstant profile. The isolated native-rate geometry component
exposes this distinction and remains outside qualified A1 sources. The isolated
causal held-target successor preserves this interface in software and conditional
models; native custody must still establish that all four applied targets match
the recorded command and its ordinary A1 joint-delta reconstruction. The first
native prefix exposed one target differing by one float32 ULP after delta
encoding. The proposal now uses the existing executor mapping before dispatch;
the exact replay check remains required, alongside all original physical gates.
The corrected native run passes exact replay on its available targets but remains
incomplete after a loaded physical acceleration refusal. Its final interval has
one delivered native step, with three not advanced. It must not be described as
an undispatched command or a complete four-step profile. The final measured
feedback and declared goal position are unavailable; commands never fill those
gaps. Derived full-motion qualification is explicitly scoped to the completed
0 N case. A passing process exit or prefix reader cannot qualify the whole run.
The later minimum motor norm components complete all eight N/H+/H−/P 0/10 N
cases: all 37,272 control intervals retain four identical native targets and exact
A1 replay, and all 28,632 geometric proposals join their current declared goals
and delivered payments. This qualifies the components' action profiles; observed
recordings, actual contact and full teacher-source admission remain separate
requirements.

The actual-contact Temporal format records absolute world-pose requests as kind
2, together with current state acquisition, dispatch/application times and the
applied target. These known command inputs must be joined to the preceding
completed native state. Post-command force, pose and contact-loss fields are
outcomes, never predictor inputs. A missing preceding mass sample remains
unavailable even when a later control-state row is present. A later contact
intervention has 1,135 exact held A1 profiles and 4,540 matched native feedback
packets, despite its negative task outcome. Its independent geometric proposal
reconstruction matches only 368/374 targets; six one-step float32 disagreements
remain distinct from the exact recorded action profile. Quaternion normalization
alone does not resolve them. The next diagnostic journal retains the actual
pre-physics root quaternion and position as current model inputs; configured
calibration must not be fitted to later achieved tool poses. For predicted
material references, the causal receipt must join original program input,
current measured motion and the actual Temporal dispatched request. Predicted
future pose never becomes measured state. The completed timing experiment
joins all 97 current material inputs to their actual predicted requests and
reconstructs all 126 geometric targets and paid goals exactly using native root
inputs. Its 888 action profiles pass despite the negative task. This resolves
the input ambiguity for that recording only; the six previous disagreements
are not retroactively relabelled.

Preserved engineering recordings have no measured torque; commands do not substitute for feedback. New recordings retain a
separate 60 Hz measured feedback channel and 120 Hz robot feedback/dynamics stream,
including source, joint-output semantics, support times and canonical joint order.
The B1 writer retains lossless single-frame LZF RGB/depth/mask chunks and buffers
small numeric/text columns by at most 128 rows. It drains before each unchanged
60-control-tick flush and on finish/close. Pending clocks participate in ordering
checks. Orderly interruption drains buffered values while retaining complete=False;
a write failure cannot retry a partial drain or publish completion. Raw values,
including NaN payloads, unavailable feedback and failed outcomes, remain unchanged.
Native teacher quality review also retains initialization failures with zero
commands and zero or one actual initial state. Missing command/evaluator streams
then yield zero eligible labels/windows, unavailable duration when no state exists,
and the unchanged original outcome. This diagnostic result never qualifies a
source; an empty claimed completion fails integrity. Review must not overwrite
the initial failure with a missing-command-field exception.
Pilot preparation references these retained originals by default; explicit
`--copy-raw` creates a self-contained export for transfer.

Temporal teacher recordings also retain `teacher-trace.jsonl` beside the raw
episode. Each completed transition flushes its actual clock/generation and
separate privileged diagnostics, including native force mean, each physics-substep normal force and contact
separation, and pressure-control feedback. A subsequent refusal or snapshot error leaves those completed records
readable; orderly recording close flushes/closes the sidecar before model teardown.
Nonfinite diagnostic values serialize as null while raw arrays and masks retain
their original values. This file is outside the observation/target reader and
confers no completion or source eligibility. Historical missing traces remain
unavailable. CPU lifecycle checks cover normal and exceptional transitions;
the resumed D2-RP diagnostic exercises the mean/feedback journal; the newly
added per-substep force/separation fields still require native validation.

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
Large contiguous, compressed and VDS observations stay selective; contiguous
historical arrays up to 8 MiB retain read-only memory maps. This prevents a full
normalization scan from making a multi-GiB visual mapping resident. A shared pool limits HDF5 readers to four files with 1 MiB
chunk caches per open dataset. `close()` drains handles; later reads reopen them. Batch reads
group by episode and restore the same seeded order, duplicates and padding.
Family batch factories prefetch one next normalized batch in one reader thread to
overlap CPU reads with GPU computation. The queue contains no dataset copy or
workers per episode. `prefetch=False` keeps synchronous iteration. Early close
joins the current read; input exceptions propagate before the affected batch is used.
Close an active prefetched iterator before explicitly closing its dataset.

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
states, submitted commands and applied targets, and native120/240Hz robot
feedback. The default remains uncompressed single-frame RGB-D chunks.
`--rollout-compression lzf` or `--rollout-compression gzip` opts into built-in
lossless compression of those same
frames when retained-evidence capacity matters; it costs additional CPU write time.
The file records the mode. Raw RGB/depth/validity bytes, including NaN payloads,
remain exact and existing readers require no conversion. Retained captures are
stored once; per-control legacy visual-support indices join them to acquisition
and availability clocks, while temporal history is reconstructed from the raw
independent streams. Invalid feature support and sampled
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

The [[../experiments/b1-data-pipeline|bounded pipeline study]] reports comparable
saved-data preparation, loading, ACT input cost, writer tradeoffs and scaling.


The gzip option uses level1 and retains one frame per chunk, preserving bounded
selective reads. Its RGB/depth/validity bytes, including NaN payloads, match the
original input exactly. Numeric streams, clocks and joins are unchanged. The
uncompressed default and existing recordings remain intact; compression choices
control storage only. Current measured space/write tradeoffs are recorded in
[[../experiments/act-a1-first-completion]].
