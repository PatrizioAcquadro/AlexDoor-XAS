# Pilot commands and temporal observations

## Choose a workflow

Run from the source checkout. The pilot uses training and development identities selected by an explicit protocol. Keep raw/source files and use a fresh writable output for new evaluations.

| Workflow | Entry point and inputs |
|---|---|
| Audit, prepare, index, materialize, train or evaluate matched pilot data | `scripts/pilot_offline.py MODE --protocol configs/b1_learned_pilot.json --root PILOT_ROOT`. Use `evaluate --selection-protocol configs/b1_pilot_selection.json` for the frozen development selection metrics. |
| Collect, qualify, replay or roll out in the calibrated simulator | `scripts/isaac-run.sh scripts/pilot_physics.py MODE --help`. Supply the protocol, prepared asset root and mode-specific inputs. The default is Isaac Sim PhysX for these calibrations. |
| Train/resume temporal ACT-A1 | `scripts/train_temporal_act.py --manifest MANIFEST --output NEW_RUN`. The manifest selects admitted sources. Use `--resume` for a compatible checkpoint. |
| Review temporal recordings or compare prepared loading and inference | `scripts/verify_temporal_collection.py --help` and `scripts/benchmark_temporal_pipeline.py --source RAW --prepared PREPARED --output NEW_RUN`. The latter compares the same captured inputs. |
| Check operational coverage and stopping | `scripts/pilot_workspace_survey.py` surveys the working envelope. `scripts/verify_pilot_dynamics.py` and `scripts/verify_loaded_stop.py` measure robot response. `scripts/analyze_loaded_stop.py` reads the saved stop traces. |

## What a checkpoint can consume

The temporal A1 actor consumes a static inspection view and causal RGB-D and robot histories. Joint position/velocity, verified measured torque and previously applied controller setpoints each carry their own validity. Acquisition time, availability and reset generation determine eligibility at a decision. Missing feedback stays masked. Teacher phase, task success, door identity and simulator truth belong to labels and evaluation.

The default four-second history has 22 recent RGB-D slots plus a static reference, 240 × 384 RGB-D and 241 robot tokens. The standard temporal ACT recipe predicts 12 rows and executes six before replanning.

At 60 Hz, an action specifies a seven-joint target increment. Observation history, prediction horizon and executed chunk length are independently configured. The reader validates the robot identity, preprocessing and timing contract against the saved model. Train-only statistics and source membership travel with the checkpoint. Existing recordings and selected checkpoints remain readable. [[episode-and-dataset-contracts]] · [[learned-policy-stack]]

The general matched A1–A4 benchmark also requires qualified action-specific geometry. The pilot supplies a narrower executable path. Historical selection results and source limitations are in [[../experiments/policy-pilot-results]].
