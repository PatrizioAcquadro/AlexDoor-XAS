# Learned Policy Stack

ACT and Diffusion remain reusable numerical models and tensor trainers. The B0
run orchestration is retired. B1 RGB-D/action integration belongs to
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]],
and training/evaluation orchestration to
[[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7]].

The shared B1 perception estimator is documented separately in
[[shared-door-perception|Shared Door Perception]]. The B1 software path below
connects its observed encoding to both families; qualification and physical
validation of the eight learning paths remain pending.

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

The shared data loader checks split membership, train-only normalization,
ordered observation keys and per-episode robot identity against dataset metadata.
A dataset/view cannot silently reuse stale statistics.

Inference checkpoints contain weights, dimensions, model configuration, dataset
descriptor, normalization and robot identity. Formats are
`alexdoor_xas.act.v3` and `alexdoor_xas.diffusion.v3`, with explicit `obs_keys`.
Earlier checkpoint formats are rejected without migration. Invalid dimensions,
non-finite weights, incompatible normalization, missing identity and mismatched
runtime identity are rejected. Saving is atomic. Normalization serialization is
shared with dataset statistics; matching dimensions alone cannot hide reordered
observation fields.

There is no maintained B0 training CLI, W&B wrapper, run-directory protocol or
closed-loop evaluator. No old checkpoint is claimed to be a B1 policy. Numerical
model correctness and small overfit tests do not establish manipulation quality.

## B1 Family Integration

`policies/b1.py` provides `load_b1_data`, `batch_factories`, `make_model`,
`train_model`, `save_policy` and `B1Policy.from_checkpoint`. The existing ACT and
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
`tests/test_b1_policy_models.py` defines all eight CUDA forward/loss/gradient and
prediction round-trip checks; they remain unexecuted while collection owns the
GPU. The estimator encoding parity test is likewise pending. These software
interfaces do not qualify 6.1 or establish policy performance.

## Observed Execution and Replay

`load_frozen_observer` checks the release gates and estimator/backbone artifact
bytes, loads the existing estimator, rejects unqualified confidence, and freezes
all perception parameters. It never turns a failed checkpoint into a valid one.
`PurdueIO` accepts an already-created prepared-door environment with cameras and
the common setup. It reconstructs camera/tool poses from calibration and observed
joints, reuses the common inspection/parked-tool hold, and leaves constant fingers
outside learned actions. It does not launch or modify another simulator.

Once the required 6.0 release and B1 policy artifacts exist, the programmatic path
is `load_frozen_observer(...)` → `B1Policy.from_checkpoint(...,
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
