# Learned Policy Stack

ACT and Diffusion remain reusable numerical models and tensor trainers. The B0
run orchestration is retired. B1 RGB-D/action integration belongs to
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]],
and training/evaluation orchestration to
[[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7]].

The shared B1 perception boundary is documented separately in
[[shared-door-perception|Shared Door Perception]]. The B1 software path below
connects its observed encoding to both families; qualification and physical
validation of the eight learning paths remain pending.

The [[b1-policy-pilot|common policy pilot foundation]] adds a separate, explicitly
unqualified binding for independent encoder/policy and geometry development.
Partial A1/A2 data and inference use the same family APIs. The explicit
`rgbd-conv-v1` recipe adds jointly trained RGB-D convolution inside both models,
with the common causal history and checkpoint-bound preprocessing. The grid
recipes remain compatible historical evidence. A local action admission route needs bounded observed material
and sweep evidence; qualified-release and A3/A4 contracts remain strict. Measured
solver feedback and robot-only residual load/stop checks are wired but unqualified
for contact. See the pilot page for measured cost and outstanding prerequisites.

The [[experiments/b1-act-diffusion-a1a2-pilot|bounded A1/A2 experiment]] exercises
fresh teacher recordings, observed-state label reconstruction, unloaded physical
replay, fixed-minibatch learning, four short CUDA runs and checkpoint round trips.
It retains ACT A2 development regression and does not qualify learned contact or
the eight-path final benchmark.

The explicit `rgbd-finite-history-v1` ACT recipe additionally accepts structured
RGB-D and measured-state history, with a shared CNN and one finite attention
block. Its current-state readout retains q/dq explicitly. Temporal checkpoints
and inference are separate from the legacy flat recipe; both remain maintained.
The v2 temporal checkpoint removes static-reference elapsed time. Its explicit
`temporal_a1_runner` bridge executes six of twelve rows through the existing
per-row admission/controller/stop path and validates freshness inside a chunk.
See [[act-a1-observations-and-time|the causal contract, API, measured costs and collection boundary]]. No new policy has been trained or physically qualified.

## Maintained Components

- `ActModelCfg` / `DiffusionModelCfg` and their training dataclasses configure the
  models directly, without run-specific YAML or a second configuration loader.
- ACT uses a conditional variational transformer and z-score action scaling.
  Diffusion uses a denoising transformer, min/max action scaling, DDPM/DDIM and
  optional EMA. Both use normalized numerical observation tensors.
- `train_act` and `train_diffusion` accept batch factories, support validation
  and return epoch history. Checkpoint callbacks expose optimizer, random-state,
  epoch and policy-specific scheduler/EMA state for exact training continuation.
- `act_chunk_source(policy, observe, ...)` provides optional temporal ensembling;
  `diffusion_chunk_source(policy, observe, ...)` provides receding-horizon execution.
  The caller supplies `observe(context)` and owns observation provenance.

## Data and Checkpoints

B1 data loading checks frozen identity membership, train-only normalization and
ordered observation keys. Checkpoints contain weights, model configuration,
normalization, robot identity and the perception/action contract. Invalid shapes,
non-finite weights and incompatible identities are rejected; saving is atomic.

The standalone numerical data pipeline and `v3` checkpoint constructors are
retired. Historical source remains at `097d578`. Models, tensor trainers,
normalizers and chunk execution remain shared by the B1 integration below.
Numerical model checks do not establish manipulation quality.

## B1 Family Integration

`policies/b1.py` provides `load_b1_data`, `batch_factories`, `make_model`,
`save_policy` and `B1Policy.from_checkpoint`, with the family tensor trainers. The existing ACT and
Diffusion models/trainers consume the same observed tensor and action widths
7/6/6/17. No model training or policy-dataset production is launched by these APIs.

Statistics are computed on train identities only; batch/checkpoint preparation
rejects changed membership or stale statistics. ACT keeps z-score scaling and
Diffusion keeps min/max scaling. A4 stage/termination columns instead use fixed
mean 0, std 1, min 0 and max 1: ACT sees 0/1 and Diffusion sees -1/+1. Continuous
columns retain train-derived scaling. Padded rows remain masked by existing losses.

B1 checkpoint formats are `alexdoor_xas.act.b1.v1` and
`alexdoor_xas.diffusion.b1.v1`. They embed the exact observation/action contract,
canonical arm/neck order, robot identity, normalization and qualified/frozen
perception binding. A v3 numerical checkpoint cannot enter this path through
matching dimensions. B1 inference defaults to CUDA and validates artifact
compatibility before model construction.

ACT temporal ensembling is available for A1-A3 only. A4 retains discrete segment
boundaries and rejects ensembling. Both families predict segment chunks;
`B1Runner` executes the first segment, then queries again at its boundary.

CPU checks cover data, scaling, serialization envelopes and contract rejection.
CUDA forward/loss/gradient and prediction round-trip checks cover the eight B1
paths and retained families. No optimizer update was performed during cleanup;
training regressions remain maintained but were excluded. These checks do not
qualify perception or establish physical policy performance.

## Observed Execution and Replay

`B1Observer(provider, binding)` requires a provider bound to the exact qualified
release. The future provider must verify its named artifact bytes before loading,
convert sensor arrays to its device and provide geometry plus static/recent visual
features. No qualified provider or model-specific loader is supplied. See
[[shared-door-perception|the model-independent contract]].
`PurdueIO` accepts an already-created prepared-door environment with cameras and
the common setup. It reconstructs camera/tool poses from calibration and observed
joints, reuses the common inspection/parked-tool hold, and leaves constant fingers
outside learned actions. It does not launch or modify another simulator.

Once the required 6.0 release and B1 policy artifacts exist, the programmatic path
is `B1Observer(provider, binding)` → `B1Policy.from_checkpoint(...,
binding=observer.binding, runtime_asset=robot_asset)` →
`PurdueIO(env, binding=observer.binding, robot_asset=robot_asset, setup=setup)` →
`B1Runner(policy, observer, io).run()`. The shared setup limits the manipulation
budget; the inspection and causal warmup occur first. A new run resets perception,
policy sampling, queued actions and A4 stage state.

`MatchedReplay(dataset, episode_id)` can replace the policy for explicitly labeled
recorded-action diagnostics. It enforces recorded timing and uses live estimated
geometry for A3/A4; it does not inject recorded door transforms. A replay completion
or predicted A4 termination reports execution only, never task success.

`PurdueSafety` consumes per-substep normal-force/contact diagnostics and simulator
geometry exclusively for stop decisions. It checks distal surfaces, forbidden
contacts, joint/finger limits and controlled contact at the policy's actual panel
point; it never requires the teacher's exact point, adds corrective motion, or
uses an expert stopping angle. Physical verification remains pending.

The historical `rgbd-grid-fixed-units-v2` reader uses fixed observation units
in shared B1 batches and checkpoint inference, with unchanged train-only action
normalization. V1 retains its original behavior. See
[[experiments/b1-pilot-preprocessing|the isolated preprocessing comparison]].


The active B1 path uses the family tensor trainers directly; the unused
`train_model` dispatch wrapper is retired. Training and checkpoint contracts are unchanged.
