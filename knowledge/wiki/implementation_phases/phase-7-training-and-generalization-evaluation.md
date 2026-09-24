# Phase 7 — Training and Generalization Evaluation

> Planned. Two subphases train the models, then evaluate and analyze them.
> Requires Phase 6; no training or evaluation is performed by this revision.

## Objective

Compare A1-A4 x ACT/Diffusion on qualified held-out door identities using
expert-normalized controlled opening and explicit validity. The authoritative
metrics and matched stress references are in
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 7.0 — Forty-Checkpoint Training Matrix

#### Implementation

After the small Phase 6 pilot passes and its dataset/protocol are frozen, train
two models x four representations x five independent seeds. Hold physical episode
membership, observations, training budget, and checkpoint selection fixed. Select
on development doors only with the common validity-adjusted progress objective.

#### Key Decisions

- Use actual compatible GPUs. The workstation owns Isaac workloads; Gilbreth
  A100s or the available Gautschi H100 allocation can run non-Isaac training.
- Report failed cells and seed variation; do not replace failures with favorable
  extra seeds or report only the best run.
- Keep training independent of simulator imports and test-door data.

#### Problems / Limitations

Complete when all 40 planned runs have explicit outcomes and selected loadable
checkpoints where successful. Resolve infrastructure failures transparently;
never label an incomplete matrix a completed comparison.

## Subphase 7.1 — Evaluation, Analysis, and Reproducible Results

#### Implementation

Run and analyze **ID and GEO first** to answer the central seen-versus-unseen-door
question. Use 20 paired conditions per evaluated door/tier, matching conditions
and seeds across cells. Report valid expert-normalized progress, raw sustained
angles, per-door/split scores, generalization gap, and training-seed variation.
This is an explicit core-result milestone, not a claim that stress evaluation
is already complete.

Then execute the planned POSE, LIGHT, DYN, and COMPOUND tiers using the ranges
and eligibility rules fixed before sealed evaluation. Do not expand accessory
experiments or retune the protocol based on ID/GEO test results. Main inference
uses only observed RGB-D/proprioception and frozen perception/gaze.

For every physically changed condition, including reset perturbations, compute
its matched frozen-probe reference before examining the corresponding policy
outcomes. Reuse it across checkpoints; purely visual changes may reuse the
physically identical reference. Expert and policy share physical conditions,
validity rules, sustain duration, and episode time budget, without a shared
angular stopping target.

Average rollout scores within each door, then weight doors equally. Rank by GEO
validity-adjusted expert-normalized progress. Preserve ratios above one and
report raw angles, invalid counts, force diagnostics, stop reasons, and release
status. Analyze results as they are produced rather than opening another subphase
for the same records.

Keep the configurations, dataset recipe, licensed manifest/attribution, split,
expert references, selected checkpoint information, and per-door/aggregate
results needed to reproduce the comparison. Prepare a public-release checklist
at the end; public packaging/publication is not a separate research subphase or
a prerequisite to reading and delivering the scientific results.

#### Key Decisions

- A zero, invalid, or unresolved expert reference means an unqualified scenario.
  Report coverage/reason for every model; do not silently drop difficult cases
  or substitute nominal denominators.
- Do not replace test doors or tune ranges/common setup after policy results.
  Correct systemic evaluation defects transparently and rerun affected comparisons.
- Unsafe or otherwise invalid policy episodes receive zero valid progress and
  stay in the denominator. A safe timeout or incomplete release alone does not
  erase earlier sustained controlled progress; an unsafe release invalidates it.
- Simulator truth may score and trigger common safety stops, never supply helpful
  policy/gaze/adapter commands or angle-based early success.
- Keep oracle diagnostics separate and run them only to investigate a concrete
  failure, not as an obligatory extra matrix. Report infrastructure interruptions
  separately from policy failures.

#### Problems / Limitations

Deliver ID/GEO conclusions before spending on the stress extension, while keeping
that extension visible as pending until executed. Full planned evaluation closes
only after the retained stress tiers and analysis are complete. Interpret progress
relative to a practical expert, not an absolute robot optimum. Conclusions concern
the qualified push-door domain; stereo errors, soft fingers, physical safety,
sim-to-real, handle operation, and other object families remain outside the claim.

## Artifacts

Future outputs: training outcomes/checkpoints, core ID/GEO report followed by
stress results, matched expert references, per-door analysis, and reproducibility
materials. No result or public release was produced by this revision.

## Files

Expected surfaces: `src/alexdoor_xas/policies/`, future B1 rollout/evaluation
and reporting modules, training/evaluation entry points, `configs/`, and result
documentation. The retired B0 evaluation package is not a maintained dependency.
