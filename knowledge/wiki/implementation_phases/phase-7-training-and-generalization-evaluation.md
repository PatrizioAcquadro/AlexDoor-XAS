# Phase 7 — Training and Generalization Evaluation

> Planned. Requires the Phase 6 dataset and protocol. No training or evaluation
> was performed by the 2026-09-22 documentation revision.

## Objective

Compare A1-A4 x ACT/Diffusion on qualified held-out door identities, using
expert-normalized controlled opening and explicit validity. The authoritative
metric and stress-reference rules are in
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 7.0 — Forty-Checkpoint Training Matrix

#### Implementation

Train two model families x four action representations x five independent seeds
for 40 selected checkpoints. Hold physical episode membership, observations,
training budget, and checkpoint selection fixed. Select on development doors
only, using the same validity-adjusted progress objective for all cells.

#### Key Decisions

- Use actual compatible GPUs. The workstation owns Isaac workloads; Gilbreth
  A100s or the available Gautschi H100 allocation can run non-Isaac training.
- Report failed training cells and seed variation; do not silently replace
  failures with extra favorable seeds or report only the best checkpoint.
- Keep training independent of simulator imports and test-door data.

#### Problems / Limitations

Complete when all 40 planned runs have explicit outcomes and selected loadable
checkpoints where successful. Resolve infrastructure failures transparently;
do not call an incomplete matrix a completed comparison.

## Subphase 7.1 — Paired Nominal and Stress Evaluation

#### Implementation

Evaluate the selected checkpoints over 20 paired conditions per evaluated door
in ID, GEO, POSE, LIGHT, DYN, and COMPOUND. Match conditions/seeds across cells and
use only observed RGB-D/proprioception and frozen perception/gaze in inference.
Expert and policy use the same physical scenario, validity rules, sustain
duration, and episode time budget. Neither stops at a shared opening target.

For POSE, DYN, COMPOUND, and any reset perturbation that changes physical
reachability, compute the frozen probe reference for that exact physical
condition before examining policy outcomes. Cache it across checkpoints.
LIGHT-only changes may reuse the corresponding physically identical reference.
Test expert traces remain isolated evaluation data and never tune the probe.

#### Key Decisions

- A zero, invalid, or unresolved expert reference is an unqualified scenario.
  Report reference coverage and the reason for all models; do not silently omit
  a difficult policy result or substitute a nominal denominator.
- Do not revise test assets, randomization ranges, or common setup after seeing
  test-policy performance. A systemic evaluation defect must be corrected and
  the affected paired comparisons rerun transparently.
- Preserve raw angles, whole-episode safety/physics validity, stop reasons, and
  release status. Unsafe or otherwise invalid policy episodes receive zero valid
  progress while staying in the aggregate denominator. A safe timeout or incomplete
  release alone does not erase earlier controlled sustained progress; use the
  decision's distinction between partial task performance and invalid execution.
- Simulator truth is permitted for scoring and a common safety monitor, never
  to produce helpful policy/gaze/adapter commands or angle-based early success.

#### Problems / Limitations

Complete only with interpretable paired results and declared expert-reference
coverage. Report software/infrastructure interruptions separately from policy
failures; they cannot be silently removed to improve a model's score.

## Subphase 7.2 — Analysis and Reproducible Release

#### Implementation

Rank by GEO validity-adjusted expert-normalized progress, aggregated within each
door and then equally across doors. Report raw sustained angles, per-door ratios,
per-split/tier results, ID-minus-GEO generalization gap, and training-seed variation.
Keep ratios above one visible. Report validity and force diagnostics alongside
progress; they are not alternate optimization targets.

Publish the licensed manifest/attribution, split, common setup/probe, expert
references, observation/action/perception contracts, dataset recipe, selected
checkpoint information, protocol, and aggregate/per-door results. Separate
oracle diagnostics from main rankings.

#### Key Decisions

- Interpret the main result as progress relative to a frozen practical expert,
  not percentage of an absolute robot optimum.
- Describe the qualified 24-door population and excluded-domain coverage.
  More rollout repetitions do not increase the number of independent door identities.
- The real stereo error model, soft-finger mechanics, hardware force safety,
  and sim-to-real transfer remain unvalidated unless separately demonstrated.

#### Problems / Limitations

Conclusions cover unseen push-door instances under the declared conditions.
They do not establish handle operation, other object families, mobile-base
behavior, hardware deployment, or general physical safety.

## Artifacts

Future outputs: training outcomes/checkpoints, paired evaluation records,
expert-condition references, per-door/split report, and reproducible release.

## Files

Expected surfaces: `src/alexdoor_xas/policies/`, `src/alexdoor_xas/eval/`,
shared rollout reporting, `scripts/train_policy.py`, `scripts/eval_policy.py`,
`configs/`, and publication documentation.
