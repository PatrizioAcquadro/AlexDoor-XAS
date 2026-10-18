# B1 Policy Pilot

The maintained pilot uses ACT/Diffusion × A1/A2 with `rgbd-conv-v1`, observed RGB-D
and proprioception. It remains paused and incomplete. No final matched A1–A4 dataset,
perception release or full learned opening/hold/release result is delivered.
[[../status|Project status]] owns current panels and resource accounting;
[[../experiments/b1-end-to-end-pilot-resume|the end-to-end experiment]] preserves all
attempts, interruptions, negative outcomes, corrections and frozen recipes.

## Interfaces and ownership

| Component | Input/output and responsibility |
|---|---|
| `RGBDEncoder` | Sensor acquisition to `VisualEncoding`; weight-free preparation and causal history. |
| `B1Observer` | Encoding plus current proprioception to `PolicyObservation`; geometry support stays independent. |
| `PilotBinding` | Explicit encoder, dimensions, timing and operational geometry profile; always unqualified. |
| `ActionAdapter` / `B1Runner` | Interpret the prediction, obtain admission for the actual command, dispatch through IO and stop on refusal. |
| `PilotMotionEvidence` | Proposed motion plus current observed material/workspace, robot geometry and calibrated load/stop support to admission. |
| `ExpertRecorder` / `RolloutWriter` | Complete teacher episodes or actual learned inputs/commands, plus separate evaluation/feedback clocks. |

A1 joint deltas and A2 world-pose deltas need no door hinge for interpretation.
Interpretation is separate from admission. A3 requires a supplied hinge frame;
A4 requires supported segment/hinge/panel/angle state and retains its admitted
frame through the segment. Those physical paths remain unqualified.

The optional geometry provider keeps `binding`, `generation`, `reset()` and
`update(sensor) -> DoorEstimate`. Its old visual encoding is unused by the pilot.
Missing geometry does not prevent feature assembly and cannot permit motion.
`between_inference_ticks` retains the previous geometry/support timestamps;
episode changes invalidate all cached geometry and features. Public local patch
state is `tracking`; stale patches are `unavailable`. Ownership, uncertainty,
continuous sweep/response and load evidence remain necessary.

## Observations and model

`B1Observer(provider_or_none, binding, encoder=encoder)` requires matching encoder
identity, `reset()` and `update(sensor, *, generation) -> VisualEncoding | None`.
Each `FeatureBlock` contains vector, acquisition/availability times, source frame
and episode generation. A missing result retains its original timestamps. Recent
features expire at `max_gap_s`; static inspection support persists only in its
own episode. Repeated or backward acquisition times/frames are refused.

The current recipe prepares five 60×96 channels: RGB/255, valid metric depth
saturated at 5 m and divided by 5 m, and valid-depth coverage. Recorded near/far
still define sensor validity. Missing depth contributes zero with explicit
coverage; entirely missing RGB-D cannot become fresh input. Static inspection
freezes its last sampled view. Three recent frames at 10 Hz are oldest-first,
with reset-only left padding. Nine joint positions and nine velocities are appended
at the current acquisition. Four images plus proprioception give 115,218 values.

`RGBDProjection` lives inside both action models: shared-per-frame Conv2d
5→16→32→64, stride 2, GroupNorm/SiLU, a 3×4 spatial pool and 64 features per image,
then the existing 128-wide action conditioning. Visual and action weights train
jointly from the declared seed. No pretrained descriptor, crop, evaluator label,
asset identity or per-door lookup enters the model.

Observation units are fixed, with the existing ±10 clipping; actions retain
family-specific train-only normalization. Model configuration, visual weights,
recipe, history, sensor interval and robot/action ordering travel with the existing
checkpoint binding. `make_model`, save/load and inference reject mismatches.
Historical grid recipes retain their original arithmetic and normalization for
reading/inference; they are not defaults for new production. See
[[../experiments/b1-pilot-preprocessing|the completed preprocessing comparison]].

## Timing and control

Physics remains 120 Hz and control 60 Hz. Primitive chunks contain 12 actions;
execute six ticks, then replan. Both families support `n_action_steps=6`; ACT
ensembling is disabled in the pilot. A4 replans only at segment boundaries.
Encoder inference, policy inference and geometry publication are separate events:
none refreshes another channel's support. Synchronous simulation completion uses
its simulation clock; wall-time cost does not establish 60 Hz hardware execution.

An operational `admission_source(observation, tool, action, remaining_ticks)` must
support exactly the predicted command. The adapter checks source, action, initial
tool pose, schedule, current support and remaining margin. A1/A2 local admission
can be hingeless but still needs finite observed geometry, response, background
clearance and stopping support. Updated support spends the original margin;
missing bounds and exhausted margin still refuse. IO independently checks before
physics, then validates actual response. Refusal clears queued actions and stops
without an added release motion or control tick.

The inference-only runner mode interprets one prediction and stops with
`unqualified_pilot_inference_only`, zero executed policy commands and no success
claim. It is distinct from genuine learned execution in the retained panels.

## Contact, load, workspace and stopping

`ObservedContactMonitor` combines observed finite material/normal-depth support
with separately calibrated robot-load intervals. Transported footprint support
and direct current contact-depth compatibility are separate. Self-occlusion,
unmeasured motion growth or missing original coverage returns unavailable;
current support cannot fabricate future zero-motion certainty.

`PurdueLoadSource` retains both 120 Hz native substeps, measured torque, calibration
domain and full-wrench peaks. Independently supported normal bounds can refine
the normal interval; full pose bounds remain necessary for material/sweep checks.
Weak-load acceleration/position extensions require their verified domains and
complete future stopping support. A command or RGB-D image never substitutes for
measured feedback. Source calibration is not a consequence of tracking acceptance.

Observed workspace/background rays, finite robot collision-cell covers and
self-clearance support the actual proposed path. Coupled revolute/prismatic travel
and the stopping allowance belong in the robot cover. It proves geometric
coverage, not observed free space, material ownership or contact by itself.
Open-pose head-ZED surveys are separate pre-task calibration, never evaluator
geometry injected into observations. Setup/controller/force/physical criteria stay
frozen; all rejected component variants remain in the experiment records.

`verify_stop_continuation` independently checks two continuing simulated seconds
after a stop. Cancellation raises outside Kit callbacks, closes writers and
preserves exit130 and partial recordings. Historical missing post-stop evidence
stays explicitly unavailable. No runtime cleanup changes these guarantees.
See [[../experiments/b1-loaded-hold-and-stop]],
[[../experiments/b1-observed-contact-certificate]] and
[[../experiments/b1-end-to-end-pilot-resume]] for qualification domains and failures.

## Data and checkpoint contracts

`PilotBinding` uses `b1.policy-pilot.v1` and `operational-v1`; `offline_passed`,
`dynamic_passed` and `frozen` stay false. Partial policy contracts use
`b1.policy-contract.pilot.v1`; datasets use `b1.policy-dataset.pilot.v1`, purpose
`b1_partial_policy_pilot`, A1/A2 only. A pilot cannot load as a qualified release.
The final `b1.policy-dataset.v1` still requires matched A1–A4, complete A4 stages,
physical identity and the original qualified release binding.

Compilation/export requires N commands, N+1 causal observations, robot FK,
identity, declared train/development membership and 60 Hz control. Engineering
recordings are not demonstrations. Invalid/excluded supervision stays excluded
with explicit reasons; failed raw episodes remain preserved. Teacher annotations,
privileged physical scoring and feedback stay outside policy inputs.

Raw episodes preserve RGB-D/calibration, applied targets, phases, terminal
observations and separate feedback/robot-state streams. Learned rollout recording
retains the consumed input, actual commands, native dynamics and independent
clocks; missing historical streams stay missing. `--reference-raw` avoids copying
retained RGB-D while checking identity; default export remains self-contained.
Neither partial labels nor schema edits promote data to final matched evidence.

Materialization accepts only declared train data and exact float32 file-backed
observations. All five eligible train records retain normalization fitted on the
original two. Development never fits statistics. Streaming batches and releasing
A1 before A2 bound RAM; the current 40 GiB orderly guard, 42 GiB hard limit and
12 GiB host reserve remain. Diffusion retains its frozen action range and clipping;
new out-of-range labels remain visible, without a normalization refit.

## Maintained commands and resumption

`pilot_offline.py` supports audit, preparation, materialization, training and
checkpoint evaluation; `pilot_physics.py` supports collection, replay and rollout.
Both require `--protocol`. Its `recipe` selects the encoder and must agree with
prepared data and checkpoint contracts. New preparation/materialization/training
and collection use `rgbd-conv-v1`. Historical audits/evaluation/replay/inference can
supply the exact `--recipe` when an old protocol has no recipe field; an explicit
argument cannot override a conflicting protocol. No command resumes work by itself.

The initial numerical protocol, `pilot_diagnosis`, `compare_pilot_preprocessing`
and unused `PilotRecorder` remain recoverable at `8bbeefe`. Their evidence and
historical checkpoints are preserved. Reusable benchmarks, workspace surveys and
load/control/dynamics/stop verification remain maintained.

Training states are atomic and retain raw model, optimizer, RNG, EMA/scheduler,
history and protocol. Resume preserves completed cells and refuses changed
experiments. Phase-balanced continuation uses only declared train identity/time/
phase joins; development and policy inputs never receive supervisor phases.
The approved continuation retains complete four-cell states and frozen selection
rules. Resource limits and previous failures do not reset on resumption.

The pilot remains paused. Further training, collection, simulation or physical
panels require new instructions after review. Software regressions, numerical
math and checkpoint compatibility do not establish physical success, perception
qualification, real-time operation or generalization.
