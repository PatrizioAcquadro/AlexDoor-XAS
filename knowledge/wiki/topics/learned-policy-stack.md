# Learned Policy Stack

ACT and Diffusion remain reusable numerical models and tensor trainers. The B0
run orchestration is retired. B1 RGB-D/action integration belongs to
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]],
and training/evaluation orchestration to
[[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7]].

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
