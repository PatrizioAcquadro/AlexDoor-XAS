# Phase 3 — Non-VLA Learned Baselines

> Historical phase record. Current training and evaluation behavior is documented in [[topics/learned-policy-stack|Learned Policy Stack]].

## Objective

Train and evaluate state-only ACT and Diffusion policies through shared data, adapter, and closed-loop contracts.

## Subphase 3.1 — Data, Policies, and Evaluation

#### Implementation

This phase introduced validated A2/A3 model data, ACT, Diffusion, adapters, checkpoint loading, and matched closed-loop evaluation. The scientific study later expanded to a sixteen-cell policy/representation/data-size matrix.

Tensor training, model checkpoints, resume state and generic adapters remain. B0 run creation, evaluation, reports and CLI configuration were retired; Git `9c16e3d` preserves that source state.

#### Key Decisions

- The completed study learned A2 and A3 only.
- Training/open-loop metrics do not replace simulator evaluation.
- Runtime execution requires a matching robot identity.

#### Problems / Limitations

- The completed benchmark was success-saturated and selected no winner.
- Results are state-only, simulation-only, and limited to seed-0 training.

## Artifacts

Historical conclusions are retained in [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]] and [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]]. Removed runner and evidence files remain available through Git history.

## Files

- `src/alexdoor_xas/policies/`
- `src/alexdoor_xas/adapters/`
