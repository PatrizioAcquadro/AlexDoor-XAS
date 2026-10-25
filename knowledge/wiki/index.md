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
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]] — B1 RGB-D, observed-input policy data, atomic exports and train-only normalization.
- [[topics/learned-policy-stack|Learned Policy Stack]] — Reusable ACT/Diffusion models, tensor training, checkpoints and explicit observation boundary.
- [[topics/b1-policy-pilot|Common B1 Policy Pilot Foundation]] — Causal RGB-D/data, measured robot feedback, bounded local A1/A2 admission and separate offline/contact readiness.
- [[topics/act-a1-observations-and-time|ACT A1 Observations and Time]] — Maintained causal streams, finite-history ACT, guarded execution and future complete-collection requirements; physical qualification remains open.
- [[topics/act-a1-diverse-collection|Diverse ACT A1 Collection]] — Declared 112-episode campaign, qualification gates, resources and outcomes.
- [[topics/act-a1-four-door-expansion|ACT A1 Four-Door Expansion]] — Qualified four-door train recordings, hold continuations, one posture pilot, observed convergence, source-row views, costs and next physical coverage.

- [[topics/perception-model-setup|Perception Model Setup]] — Local models, isolated dependencies and supported commands.

- [[topics/shared-door-perception|Shared Door Perception]] — Static scan, official Point2Pose panel/zone prototype, recording and estimator-independent contracts; qualification pending.

- [[topics/pre-a4-initialization-and-hinge-budget|Shared Pre-A4 Initialization and Simulated Hinge Budget]] — Policy-independent initialization routes, insufficiency, uncertainty sources and unchanged A4 command sensitivity.

## Key Decisions

Current architectural and scientific contracts:

- [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]] — Hold physical experience and task geometry aligned across A1-A4.
- [[decisions/visuoproprioceptive-generalization-benchmark|Visuoproprioceptive Generalization Benchmark]] — Approved question, held-out door benchmark, matched data strategy, and articulated-object progression.
- [[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]] — Historical B0 rationale for translation-only IK and exact-panel contact.
- [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]] — Historical rationale for fixed holdouts and nested memberships.

## Experiments

- [[experiments/act-a1-first-completion|ACT A1 First Learned Completion]] — Authorized temporal training, guarded execution and fixed two-direction completion experiment.
- [[experiments/act-a1-failure-diagnosis|ACT A1 Failure Diagnosis]] — Saved-trajectory semantics, phase-specific 61k/75k errors and two genuine-history native diagnostics; no training or threshold changes.

- [[experiments/b1-data-pipeline|Bounded B1 Data Pipeline]] — Saved-data streaming preparation, lossless frame reuse, bounded loading and unchanged ACT consumer throughput.

- [[experiments/b1-act-a1-calibration-stopping|ACT A1 Calibration and Stopping]] — Side/load support, corrected loaded stops, independent verification and preserved failures.
- [[experiments/b1-act-a1-two-door-diagnostic|ACT A1 Two-Door Diagnostic]] — Two completed nominal failures, independent physical results, operational censoring and proposed data/history work.
- [[experiments/b1-runtime-speed|ACT A1 Runtime Speed]] — Frozen 1,631/7,217-command timing, lossless recording improvements and both measured targets met; learned task remains incomplete.

- [[experiments/b1-operational-tolerances|ACT A1 Operational Tolerances]] — Bounded empirical-position and directional material-margin diagnosis; fixed checkpoint and separately scored reference execution.

- [[experiments/b1-end-to-end-pilot-resume|Resumed End-to-End Pilot]] — Paused, incomplete learned simulation pilot; frozen outcomes, safe stop and reconciled resource accounting.

- [[experiments/b1-observed-contact-certificate|Observed Contact Certificate Study]] — Three fixed current-placement camera diagnostics; signed distance component remains unqualified, tool-center/contact semantics separated and six prerequisite cases unspent.

- [[experiments/b1-pilot-contact-prerequisites|Pilot Contact Prerequisites]] — All three complete teacher diagnostics fail; valid partial progress retained, nine cases unspent and observed contact-certificate gap demonstrated.

- [[experiments/b1-initial-placement-pilot|Bounded Initial Placement Study]] — One train/dev-derived common placement passes all four unloaded approach/stop cases; load/contact and pose variation remain unqualified.

- [[experiments/b1-joint-limit-aware-ik|Joint-Limit-Aware Pose IK]] — One authorized IK cycle: numerical checks pass, four physical cases reject the left path; original solver restored.

- [[experiments/b1-shared-learned-pilot|Shared Learned A1/A2 Pushing Pilot]] — Bounded incomplete pilot: learned RGB-D and Gilbreth verification pass, execution/admission and genuine learned episodes remain blocked.

- [[experiments/b1-approach-controller-compatibility|Historical Approach / Current Controller Compatibility]] — Saved-reference audit and frozen-control CUDA comparison: right tracking passes, left timed path fails; local stops pass, contact remains unqualified.

- [[experiments/b1-pilot-approach-release|ACT A2 Approach and Release Residuals]] — Saved alignment, actual phase exposure, input ambiguity, historical tracking residuals and one fixed-budget sampling test.

- [[experiments/b1-pilot-preprocessing|B1 Pilot Fixed-Unit Preprocessing]] — Fixed-unit inputs and paired ACT A2 mean gains, with release/tail regression and unqualified generalization.

- [[experiments/b1-loaded-hold-and-stop|B1 Loaded Hold and Measured-Anchor Stop]] — Paired CUDA diagnosis, benchmark-local common impedance, continuing-physics latch/reset and unchanged-calibration limits.

- [[experiments/b1-pilot-offline-and-robot-dynamics|Pilot Diagnosis and Robot-Only Dynamics]] — Physical-unit checkpoint/phase/chunk errors, clipping attribution and separate bounded 6.0D evidence.

- [[experiments/b1-act-diffusion-a1a2-pilot|Bounded ACT/Diffusion A1/A2 Pilot]] — Fresh physical teacher data, observed-state labels, bounded CUDA learning and operational refusal evidence.

- [[experiments/p2p-sam3-cost-diagnosis|Selected SAM3 + Point2Pose Cost Diagnosis]] — Current stage costs, mature rebuild peaks, operational age/replacement accounting and comparable producer contention; diagnosis only.

- [[experiments/p2p-temporal-renewal-admission|Temporal P2P Renewal Admission]] — One causal saved-data criterion rejected before CUDA activation: insufficient motion within the existing wait budget and severe support depletion in both conditions.

- [[experiments/p2p-operational-selection-and-light-peaks|SAM3 Selection, Operational Timing and Integrated Light Peaks]] — Explicit ingress, recorded/live accounting, saved peak diagnosis and one controlled saved-mask light CUDA bridge.

Current engineering and historical scientific records:

- [[experiments/p2p-persistent-session-integration|Persistent SAM3 Inspection and Tracking]] — Shared frontend, strict initialization, bounded historical CUDA equivalence and separate inspection/temporal integration checks.
- [[experiments/b1-perception-findings|B1 Perception Findings and Direction]] — Corrected model screening, static/local findings, failed Point2Pose qualification diagnostics and preserved evidence.

- [[experiments/point2pose-consumer-impact-and-failure-windows|Point2Pose Consumer Impact and Two Failure Windows]] — Saved zone/normal and conditional target errors, missing A3/A4 inputs, nominal graph/map jump and light reference deterioration.

- [[experiments/point2pose-bounded-global-graph-ablation|Point2Pose Bounded Global Graph Ablation]] — One full CUDA graph-off attempt per original recording; removed event spikes, improved primary light, worse nominal terminal loss and conditional target tail; no common replacement adopted.

- [[experiments/point2pose-isolated-improvements|Isolated Point2Pose Improvements]] — Saved geometric/RANSAC, promotion, pixel and initialization diagnostics; three independent complete CUDA comparisons and retained/rejected controls.

- [[experiments/point2pose-refit-hypotheses|Point2Pose Refit Hypothesis Preservation]] — Historical own-seed rollback: nominal gains, light regression and rejected combination; implementation retired, evidence retained.

- [[experiments/p2p-a3-a4-integration-diagnosis|P2P to A3/A4 Integration Diagnosis]] — Verified closed local seeds/static signs, delayed and unqualified hinge bootstrap, and targeted CUDA cost on the preserved partial-only graph-off reference.

- [[experiments/p2p-observed-geometry-and-sdf-jacobian|Observed Geometry and SDF Jacobian]] — Separately attributed observed-reference/hinge/angle changes and CUDA Jacobian evaluation.

- [[experiments/p2p-evaluation-and-performance|Point2Pose Evaluation and Isolated Performance]] — Saved tolerance/target evaluation and compute comparisons; all-query retained, rejected model/tracker/SVD variants retired.

- [[experiments/p2p-sam3-unified-frontend|Unified SAM3 Video Frontend]] — Historical four-way comparison selecting SAM3 OFF; SAM3.1 retired, original targets, failures, material audits and resource tradeoffs retained.

- [[experiments/p2p-sam3-selected-development|Selected SAM3 OFF Memory and Quality]] — Selected opt-in recipe, exact bounded-history equivalence, renewal-only depth gate and retained accuracy/memory tradeoffs.

- [[experiments/p2p-selected-residuals|Selected Renewal-Filtered Trajectory Residuals]] — Same-ID birth/map/TAPIR diagnosis, separate new peaks, angular and registration/SDF contributions; trace limitations retained.

- [[experiments/p2p-selected-registration|Selected-Candidate Registration]] — Rejected periodic-registration comparison and nominal tail regression; code historical, traces and evaluators preserved.

- [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]] — Completed sixteen-cell training result and limits.
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] — Saturated matched evaluation with no selected winner.
- [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]] — Reproducible force event and bounded perturbation result.

## Sources

No user-owned raw source has been ingested.
