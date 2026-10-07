# AlexDoor-XAS Technical Wiki

Topic pages distinguish the operational Purdue runtime from historical B0 results. Implementation-phase pages distinguish planned work from concise historical records; experiment pages retain durable completed results.

## Project Status

- [[status|Project Status]] — Current capabilities, entry points, storage boundaries, historical conclusions, and limits.

## Implementation Phases

Planned and historical development records:

- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]] — Initial package, dependency, and simulator foundation.
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]] — Scripted execution, recording, and matched export foundation.
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]] — State-only policy, adapter, and evaluation foundation.
- [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4 — Robot and Task Configuration]] — GPU-verified Alex003/control/RGB-D, historical synthetic setup and the shared zero-yaw revision assessed on collected doors.
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5 — Door Corpus and Qualification]] — Complete: 32 qualified doors, reviewed geometry families and frozen 19/6/7 identity split.
- [[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6 — Perception, Actions, and Demonstrations]] — Open 6.0 perception/contact qualification, partially validated 6.1 action software and unstarted 6.2 policy data.
- [[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0 — Operational Perception and Observed Contact]] — 6.0A/B maintained; previous trackers retired; unqualified CAD-free Point2Pose 6.0C prototype and contact/qualification requirements.
- [[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7 — Training and Generalization Evaluation]] — Two planned subphases: training, then evaluation/analysis with ID/GEO before stress tests.
- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]] — Migration from provisional assumptions to fixed-base Alex V2.
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]] — Closed-loop and force-semantics stabilization.
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]] — Completed two-cell A100 compatibility check.
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]] — Completed 550-episode master and nested-view construction.
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]] — Completed sixteen-cell training matrix.
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]] — Completed 576-rollout matched evaluation.

## Topics

Current technical behavior and explicitly labeled planned contracts:

- [[topics/system-architecture|System Architecture]] — Operational runtime/preparation, reusable algorithms and storage boundaries.
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]] — Retired B0 protocol, results and scientific limits.
- [[topics/purdue-b1-robot-and-contact|Purdue B1 Robot and Contact Contract]] — Operational Purdue/WSG/ZED configuration, derived push frame, measured pedestal and validation limits.
- [[topics/action-representations-and-adapters|Action Representations and Adapters]] — A1-A4 meanings and maintained execution boundaries.
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]] — `phase2.v2`, matched exports, splits, views, normalization, and model data.
- [[topics/learned-policy-stack|Learned Policy Stack]] — Reusable ACT/Diffusion models, tensor training, checkpoints and explicit observation boundary.

- [[topics/shared-door-perception|Shared Door Perception]] — Static scan, official Point2Pose panel/zone prototype, recording and estimator-independent contracts; qualification pending.

- [[topics/pre-a4-initialization-and-hinge-budget|Shared Pre-A4 Initialization and Simulated Hinge Budget]] — Policy-independent initialization routes, insufficiency, uncertainty sources and unchanged A4 command sensitivity.

## Key Decisions

Current architectural and scientific contracts:

- [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]] — Hold physical experience and task geometry aligned across A1-A4.
- [[decisions/visuoproprioceptive-generalization-benchmark|Visuoproprioceptive Generalization Benchmark]] — Approved question, held-out door benchmark, matched data strategy, and articulated-object progression.
- [[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]] — Historical B0 rationale for translation-only IK and exact-panel contact.
- [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]] — Historical rationale for fixed holdouts and nested memberships.

## Experiments

Current engineering and historical scientific records:

- [[experiments/b1-perception-findings|B1 Perception Findings and Direction]] — Corrected model screening, static/local findings, failed Point2Pose qualification diagnostics and preserved evidence.

- [[experiments/point2pose-consumer-impact-and-failure-windows|Point2Pose Consumer Impact and Two Failure Windows]] — Saved zone/normal and conditional target errors, missing A3/A4 inputs, nominal graph/map jump and light reference deterioration.

- [[experiments/point2pose-bounded-global-graph-ablation|Point2Pose Bounded Global Graph Ablation]] — One full CUDA graph-off attempt per original recording; removed event spikes, improved primary light, worse nominal terminal loss and conditional target tail; no common replacement adopted.

- [[experiments/point2pose-isolated-improvements|Isolated Point2Pose Improvements]] — Saved geometric/RANSAC, promotion, pixel and initialization diagnostics; three independent complete CUDA comparisons and retained/rejected controls.

- [[experiments/point2pose-refit-hypotheses|Point2Pose Refit Hypothesis Preservation]] — Complete five-inlier/4 mm own-seed rollback: nominal gains, light regression, rejected combination, jump refusals and unsupported tail.

- [[experiments/p2p-a3-a4-integration-diagnosis|P2P to A3/A4 Integration Diagnosis]] — Verified closed local seeds/static signs, delayed and unqualified hinge bootstrap, and targeted CUDA cost on the preserved partial-only graph-off reference.

- [[experiments/p2p-observed-geometry-and-sdf-jacobian|Observed Geometry and SDF Jacobian]] — Separately attributed observed-reference/hinge/angle changes and CUDA Jacobian evaluation.

- [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]] — Completed sixteen-cell training result and limits.
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] — Saturated matched evaluation with no selected winner.
- [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]] — Reproducible force event and bounded perturbation result.

## Sources

No user-owned raw source has been ingested.
