# Learned Policy Stack

AlexDoor-XAS maintains state-only ACT and Diffusion policies for A2 and A3. They retain historical B0 dataset, training and checkpoint contracts. Purdue
learned execution and RGB-D model integration remain deferred to Phase 6.

## Configuration and Data

`configs/act.yaml` and `configs/diffusion.yaml` are loaded through strict OmegaConf-backed dataclasses. Shared dataset, run, and rollout types live under `policies/common/`; policy-specific model and training fields remain with ACT or Diffusion.

The loader validates the selected dataset, shared split membership, train-only normalization, observation preset, retained view when present, and Alex V2 robot identity. ACT uses z-score normalization; Diffusion uses its configured action scaling and scheduler path.

## Models and Training

ACT predicts action chunks with `ACTModel`. Diffusion predicts denoising trajectories with `DiffusionTransformer` and may use EMA. Both trainers expose deterministic seeded construction, validation, epoch history, and resumable state.

`scripts/train_policy.py --policy {act,diffusion}` allocates an exclusive UTC run under `outputs/door_push_alex_v2/{act,diffusion}/`. Configuration overrides apply only to a new run. `--resume` loads the run's frozen configuration and rejects additional overrides.

## Checkpoints and Resume

`best.pt` is a compact self-contained inference checkpoint. It contains the policy format, model weights and dimensions, model configuration, dataset descriptor, normalization statistics, robot identity, and selection metadata. ACT and Diffusion use `alexdoor_xas.act.v2` and `alexdoor_xas.diffusion.v2`.

`last.pt` is the incomplete-run resume checkpoint. It contains current model weights plus optimizer, epoch/global-step, history, random-state, and policy-specific training state such as Diffusion scheduler/EMA data. It is written atomically before training and after completed epochs, then removed only after successful publication. Errors retain resumable state and write `error.log`.

Older checkpoint formats, missing required Alex V2 identity, incompatible normalization, and cross-model loading fail closed.

## Run Outputs

Each training run owns immutable `resolved_config.json`, one narrative `report.md`, compact training history/plot, compact open-loop metrics/plot, checkpoints, and optional error state. Empty optional media or trace directories are not created.

Optional W&B tracking is disabled by default. When enabled through `WANDB_MODE`, the scripts call the SDK directly, write standard state under `outputs/wandb/`, and log compact scalar metrics without automatic artifact publication.

## Closed-Loop Evaluation

B0 execution has been retired. `scripts/eval_policy.py` and
`scripts/verify_policy_rollout.py` now terminate with a migration explanation
before simulator startup or output allocation. Existing checkpoints still load
for offline use, and historical closed-loop results remain untouched. They
cannot be interpreted as Purdue-compatible checkpoints.

## Limits

- Learned policies support only A2 and A3.
- Observations are state-only.
- Training/open-loop metrics do not replace closed-loop evaluation.
- Historical Phase 3 checkpoints remain evidence of that completed study; they are not automatically regenerated or migrated.

## Version Notes

- 2026-08-13 — Reconciled configuration, training, checkpoint/resume, run-output, and immutable evaluation behavior with the simplified policy implementation.
