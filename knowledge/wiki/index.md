# AlexDoor-XAS Technical Wiki

Start with current state, then the canonical technical contract for the workflow.
Experiments preserve questions, results and limits; phase pages distinguish
implemented boundaries from future requirements. Git retains ordinary edit history.

## Start here

- [[status|Project Status]] — Current execution, capabilities, measured outcomes and open questions.
- [[topics/system-architecture|System Architecture]] — Runtime ownership, maintained workflows and repository boundaries.
- [[log|Project Milestones]] — Concise durable changes.

## Technical contracts and use

- [[topics/act-a1-observations-and-time|ACT A1 Observations and Time]]
- [[topics/action-representations-and-adapters|Action Representations and Adapters]]
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]]
- [[topics/b1-policy-pilot|B1 Policy Pilot]]
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]]
- [[topics/learned-policy-stack|Learned Policy Stack]]
- [[topics/perception-model-setup|Selected perception resources]]
- [[topics/pre-a4-initialization-and-hinge-budget|Shared Pre-A4 Initialization and the Simulated Hinge Budget]]
- [[topics/purdue-b1-robot-and-contact|Purdue B1 Robot and Contact Contract]]
- [[topics/shared-door-perception|Shared Door Perception]]

## Implementation roadmap and historical phases

- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]]
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]]
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]]
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]]
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]]
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]]
- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]]
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]]
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]]
- [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4 — Robot and Task Configuration]]
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5 — Door Corpus and Qualification]]
- [[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0 — Operational Perception and Observed Contact]]
- [[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6 — Perception, Actions, and Demonstrations]]
- [[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7 — Training and Generalization Evaluation]]

## Decisions

- [[decisions/calibrated-position-only-alex-v2-execution|Decision — Calibrated Position-Only Alex V2 Execution]]
- [[decisions/door-relative-task-and-matched-representations|Decision — Door-Relative Task and Matched Representations]]
- [[decisions/one-scale-master-with-nested-views|Decision — One Scale Master with Nested Views]]
- [[decisions/visuoproprioceptive-generalization-benchmark|Visuoproprioceptive Generalization Benchmark]]

## Experiments and retained scientific results

- [[experiments/d3-observed-cycles|D3 observed-cycle diagnosis]]
- [[experiments/act-a1-complete-cycle-collection|ACT A1 Complete-Cycle Collection]]
- [[experiments/act-a1-complete-cycle-pause-diagnosis|ACT A1 pause diagnosis — existing evidence only]]
- [[experiments/act-a1-complete-demonstration-audit|ACT A1 complete-demonstration audit]]
- [[experiments/act-a1-diverse-collection|Diverse bounded ACT A1 collection campaign]]
- [[experiments/act-a1-failure-diagnosis|ACT A1 failure diagnosis]]
- [[experiments/act-a1-first-completion|ACT A1 First-Completion Experiment]]
- [[experiments/act-a1-four-door-expansion|ACT A1 Four-Door Expansion]]
- [[experiments/act-a3-n50-seed-112-force-diagnostic|Experiment — ACT-A3-N50 Seed-112 Force Diagnostic]]
- [[experiments/approach-contact-corrections|Approach-to-contact corrections and guards]]
- [[experiments/b1-act-a1-calibration-stopping|ACT A1 calibration and stopping correction]]
- [[experiments/b1-act-a1-two-door-diagnostic|ACT A1 Two-Door Diagnostic]]
- [[experiments/b1-act-diffusion-a1a2-pilot|Bounded ACT/Diffusion A1/A2 Physical-Data Pilot]]
- [[experiments/b1-approach-controller-compatibility|Historical Approach / Current Controller Compatibility]]
- [[experiments/b1-data-pipeline|Bounded B1 Data Pipeline]]
- [[experiments/b1-end-to-end-pilot-resume|Resumed End-to-End A1/A2 Pilot]]
- [[experiments/b1-initial-placement-pilot|Bounded Initial Placement Study]]
- [[experiments/b1-joint-limit-aware-ik|Joint-Limit-Aware Pose IK]]
- [[experiments/b1-loaded-hold-and-stop|B1 Loaded Hold and Measured-Anchor Stop]]
- [[experiments/b1-observed-contact-certificate|Observed Contact Certificate Study]]
- [[experiments/b1-operational-tolerances|ACT A1 operational tolerance investigation]]
- [[experiments/b1-perception-findings|B1 Perception Findings and Direction]]
- [[experiments/b1-pilot-approach-release|ACT A2 Approach and Release Residuals]]
- [[experiments/b1-pilot-contact-prerequisites|Pilot Contact Prerequisites]]
- [[experiments/b1-pilot-offline-and-robot-dynamics|Pilot Checkpoint Diagnosis and Robot-Only Dynamics]]
- [[experiments/b1-pilot-preprocessing|B1 Pilot Fixed-Unit Preprocessing]]
- [[experiments/b1-runtime-speed|ACT A1 Runtime Speed]]
- [[experiments/b1-shared-learned-pilot|Shared Learned A1/A2 Door-Pushing Pilot]]
- [[experiments/gilbreth-nested-scale-sweep|Experiment — Nested Scale Sweep]]
- [[experiments/p2p-a3-a4-integration-diagnosis|P2P to A3/A4: Closed Initialization, Hinge Bootstrap and Runtime Cost]]
- [[experiments/p2p-evaluation-and-performance|Point2Pose Evaluation and Isolated Performance]]
- [[experiments/p2p-observed-geometry-and-sdf-jacobian|Observed Geometry and SDF Jacobian: Separate Interventions]]
- [[experiments/p2p-operational-selection-and-light-peaks|SAM3 Selection, Operational Timing and Integrated Light Peaks]]
- [[experiments/p2p-persistent-session-integration|Persistent SAM3 Inspection and Tracking]]
- [[experiments/p2p-sam3-cost-diagnosis|Selected SAM3 + Point2Pose Cost Diagnosis]]
- [[experiments/p2p-sam3-selected-development|Selected SAM3 OFF: Memory and Quality Diagnosis]]
- [[experiments/p2p-sam3-unified-frontend|Unified SAM3 Video Frontend for Point2Pose]]
- [[experiments/p2p-selected-registration|Selected-Candidate Registration with Preserved Alternatives]]
- [[experiments/p2p-selected-residuals|Residuals of the Selected Renewal-Filtered Trajectory]]
- [[experiments/p2p-temporal-renewal-admission|Temporal Evidence Before P2P Reference Promotion]]
- [[experiments/phase-3-unified-evaluation|Experiment — Phase 3 Unified Evaluation]]
- [[experiments/point2pose-bounded-global-graph-ablation|Point2Pose Bounded Renewal Without the Global Graph]]
- [[experiments/point2pose-consumer-impact-and-failure-windows|Point2Pose Consumer Impact and Two Failure Windows]]
- [[experiments/point2pose-isolated-improvements|Isolated Point2Pose Improvements After Graph-Off Diagnosis]]
- [[experiments/point2pose-refit-hypotheses|Point2Pose Refit Hypothesis Preservation]]
