# System Architecture

AlexDoor-XAS maintains the B1 Purdue manipulation software and the reusable
components needed for A1–A4 × ACT/Diffusion. The commissioned simulator is
`AlexDoor-DoorPush-Purdue-v0`: fixed-base Alex003, WSG32/UMI v1 and head ZED RGB-D.
The external Alex package owns robot/camera assets and factories. No repository
command controls physical hardware. [[../status|Status]] owns qualification state.

## Maintained capabilities and consumers

| Capability | Current or planned consumer | Canonical entry points |
|---|---|---|
| Runtime, control and sensing | Expert qualification, recording and future observed execution | `DoorPushPurdueEnv`; `scripts/verify_purdue_runtime.py`; [[purdue-b1-robot-and-contact]] |
| Door preparation and qualification | Frozen corpus and future admitted assets | `scripts/prepare_doors.py`, `scripts/qualify_door.py`, `scripts/verify_door_preparation.py`; [[../implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] |
| Synthetic setup and corpus verification | Runtime regressions and source/record admission | `scripts/verify_synthetic_setup.py`, `scripts/verify_door_corpus.py`; [[../implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] |
| Environment preflight | Workstation setup and diagnostics | `scripts/check_env.py`; project README |
| Persistent observed perception | Static inspection, serial replay, continuous publication and future provider qualification | `PerceptionSession`, `SessionProvider`, `session_provider`; [[shared-door-perception]] |
| Explicit model/scan diagnostics | Development of perception and reading existing engineering recordings | `scripts/perception.py`; [[perception-model-setup]] |
| Engineering recording | Reusable train/development RGB-D and future qualified demonstration acquisition | `scripts/collect_perception.py`, `B1Writer`, `ExpertRecorder`; [[episode-and-dataset-contracts]] |
| Matched B1 data and normalization | Planned Phase 6.2 and training | `prepare_recording`, `export_dataset`, `B1Dataset`, `load_b1_data`; [[episode-and-dataset-contracts]] |
| ACT/Diffusion and checkpoint continuation | Planned Phase 7 training and inference | Family tensor trainers, `B1Policy`; [[learned-policy-stack]], [[cluster-training]] |
| A1–A4 execution and labeled replay | Future qualified provider integration and numerical regressions | `ActionAdapter`, `B1Observer`, `PurdueIO`, `B1Runner`, `MatchedReplay`; [[action-representations-and-adapters]] |
| Guarded observed teacher collection | D3 material-motion research, D4 hold/release studies and native Steel component diagnostics | `scripts/pilot_physics.py collect`, `scripts/collect_d4_observed.py`. Full D4 attempts use the beta2 RTX runtime. [[../experiments/observed-cycle-collection]] |
| Offline pilot, checkpoint evaluation and closed-loop replay | A1/A2 pilot, temporal ACT and readers for saved training runs | `scripts/pilot_offline.py`, `scripts/train_temporal_act.py`, `scripts/pilot_physics.py`; [[b1-policy-pilot]] |
| Calibration and performance | Working-envelope and stop verification, backend selection and data-loading costs | `scripts/pilot_workspace_survey.py`, `scripts/verify_loaded_stop.py`, `scripts/verify_pilot_dynamics.py`, `scripts/analyze_loaded_stop.py`, `scripts/benchmark_temporal_pipeline.py`; [[paired-physics-benchmarks]] |
| Independent metrics and evidence readers | Saved-result interpretation and future qualification | `perception.evaluation`, diagnostic metrics/target transport; [[../experiments/b1-perception-findings|experiment findings]] |

These interfaces remain maintained even when their final scientific consumer is
planned. A test-only duplicate provider is not a second runtime: static tests
inject the external image worker into the real session, and geometry tests use
`ScanMemory`. The former `GeometryProvider`, `CueEngine` and `tracking_provider`
wrapper are retired. `session_provider` is the single provider factory.

## Observation, control and evaluation boundaries

Captured observations contain synchronized RGB, metric optical-axis depth,
validity, seven-arm/two-neck joint positions/velocities, camera calibration/FK,
frame IDs and acquisition times. The environment's Gym tensor is the 18 joint
values; B1 image/history encoding belongs to the provider/observer boundary.
Runtime metadata, asset identity, teacher commands and evaluator door truth never
become policy features or geometry observations.

The policy chooses an action; adapters transform it; the common controller
executes it; the monitor can stop. Truth-based expert/evaluator and legacy stop
diagnostics are separate. RGB-D cannot certify load, tangential force or actuator
feedback. Observed admission requires its explicit geometry, uncertainty,
continuous-sweep and stop/load contracts; missing bounds remain unavailable.
See [[shared-door-perception]] and [[purdue-b1-robot-and-contact]].

Prepared-door geometry supplies the scripted expert's material trajectory and
physical bounds, never learned observations. Preparation includes the task floor;
qualification needs valid sustained hold/release independently of maximum angle.
Promotion stages dependencies and publishes `prepared.json` last, preserving the
candidate on failure. The frozen corpus selects recording identities but does not
replace episode admission or train-only normalization. Detailed rules belong to
[[../implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]].

## Storage and verification

Tracked door records and ignored prepared payloads share `datasets/doors/b1/`.
Engineering recordings live in `datasets/recordings/b1/`; all 50 engineering-v2
episodes are retained. Local model weights/isolated runtimes live in ignored
`models/`; setup belongs to [[perception-model-setup]]. New work uses
`outputs/generated/`, durable evidence `outputs/evidence/`, and runtime/expert
verification `~/.cache/alexdoor-xas/`. The repository storage guides own relocation,
removal and recovery pointers. Git cannot recover removed ignored payloads.

B0 orchestration/data/scenes, `phase2.v2` data APIs, standalone `v3` loaders and
rejected perception variants are retired. Historical source is available at
`097d578` or the revision named by each experiment; obsolete launchers use that
revision. Maintain readers needed by retained evidence, not new writers for
retired formats. `knowledge/raw` remains user-owned.

CI checks Ruff, portable unit tests on Python 3.11/3.12, wheel/sdist construction
and installed-wheel imports outside the checkout. Local GPU/integration checks
cover the external runtime boundaries. The README owns commands; tests establish
software behavior, not perception quality, hardware safety or learned success.
