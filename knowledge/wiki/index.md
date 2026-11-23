# AlexDoor-XAS Technical Wiki

Start with [[status|Project status]], [[topics/system-architecture|maintained workflows and consumers]]
and [[log|milestones]]. Current runtime contracts, future requirements and
historical scientific results have separate canonical pages.

## Use and develop

- [[topics/system-architecture|System Architecture]]
- [[topics/purdue-b1-robot-and-contact|Purdue B1 Robot and Contact Contract]]
- [[topics/perception-model-setup|Selected perception resources]]
- [[topics/shared-door-perception|Shared Door Perception]]
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]]
- [[topics/action-representations-and-adapters|Action Representations and Adapters]]
- [[topics/learned-policy-stack|Learned Policy Stack]]
- [[topics/cluster-training|Official NVIDIA Cluster Training]]
- [[topics/pre-a4-initialization-and-hinge-budget|Shared Pre-A4 Initialization and the Simulated Hinge Budget]]
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]]

- [[topics/b1-policy-pilot|Pilot commands and temporal observations]]
- [[topics/paired-physics-benchmarks|Paired backend benchmarks]]

## Implementation phases

Phases retain their identities and distinguish commissioned software from planned
qualification and data/evaluation work. B0 and completed extras are historical.

- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]]
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]]
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]]
- [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4 — Robot and Task Configuration]]
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5 — Door Corpus and Qualification]]
- [[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6 — Perception, Actions, and Demonstrations]]
- [[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0 — Operational Perception and Observed Contact]]
- [[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7 — Training and Generalization Evaluation]]

Completed historical extras:

- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]]
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]]
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]]
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]]
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]]
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]]

## Decisions

- [[decisions/calibrated-position-only-alex-v2-execution|Decision — Calibrated Position-Only Alex V2 Execution]]
- [[decisions/door-relative-task-and-matched-representations|Decision — Door-Relative Task and Matched Representations]]
- [[decisions/one-scale-master-with-nested-views|Decision — One Scale Master with Nested Views]]
- [[decisions/visuoproprioceptive-generalization-benchmark|Visuoproprioceptive Generalization Benchmark]]

## Experiments

Each page owns its question, method, decisive results, limits and evidence/Git
references. Numerical success, native acceptance, qualified perception and learned
manipulation success remain distinct.

- [[experiments/observed-cycle-collection|Observed teacher collection and contact studies]]
- [[experiments/policy-pilot-results|Policy selection, execution and demonstration audit]]

### Perception foundations, selection and diagnosis

- [[experiments/b1-perception-findings|B1 Perception Findings and Direction]]
- [[experiments/point2pose-consumer-impact-and-failure-windows|Point2Pose Consumer Impact and Two Failure Windows]]
- [[experiments/p2p-a3-a4-integration-diagnosis|P2P to A3/A4: Initialization, Hinge and Runtime Cost]]
- [[experiments/p2p-observed-geometry-and-sdf-jacobian|Observed Geometry and SDF Jacobian: Separate Interventions]]
- [[experiments/p2p-evaluation-and-performance|Point2Pose Evaluation and Isolated Performance]]
- [[experiments/p2p-sam3-unified-frontend|Unified SAM3 Video Frontend for Point2Pose]]
- [[experiments/p2p-sam3-selected-development|Selected SAM3 OFF: Memory and Quality]]
- [[experiments/p2p-selected-residuals|Residuals of the Selected Renewal-Filtered Trajectory]]
- [[experiments/p2p-persistent-session-integration|Persistent SAM3 Inspection and Tracking]]
- [[experiments/p2p-operational-selection-and-light-peaks|SAM3 Selection, Operational Timing and Integrated Light Peaks]]
- [[experiments/p2p-sam3-cost-diagnosis|Selected SAM3 + Point2Pose Cost Diagnosis]]

### Controlled alternatives and adoption decisions

- [[experiments/point2pose-bounded-global-graph-ablation|Point2Pose Bounded Global Graph Ablation]]
- [[experiments/point2pose-isolated-improvements|Isolated Point2Pose Improvements After Graph-Off Diagnosis]]
- [[experiments/point2pose-refit-hypotheses|Point2Pose Refit Hypothesis Preservation]]
- [[experiments/p2p-selected-registration|Selected-Candidate Registration with Preserved Alternatives]]
- [[experiments/p2p-temporal-renewal-admission|Temporal Evidence Before P2P Reference Promotion]]
- [[experiments/tsdf-readonly-launch-inputs|TSDF Read-Only Launch Inputs]]
- [[experiments/tsdf-deferred-snapshot-audit|TSDF Deferred Snapshots: CUDA Cost and Failure Audit]]
- [[experiments/tsdf-rebuild-double-buffer|TSDF Rebuild Double Buffer]]
- [[experiments/sam3-official-compilation|Official SAM3 Compilation]]
- [[experiments/p2p-ransac-point-buffers|Point2Pose RANSAC Point Buffers]]

### Historical B0 study

- [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]]
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]]
- [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]

## Sources

No user-owned raw source has been ingested. `knowledge/raw` is read-only to agents.
