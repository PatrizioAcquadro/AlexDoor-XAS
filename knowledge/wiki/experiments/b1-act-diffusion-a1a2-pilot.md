# Bounded ACT/Diffusion A1/A2 Physical-Data Pilot

This supervised imitation pilot uses the common foundation at `182a4bc` on
`PatrizioAcquadro/b1-policy-pilot`. It is separate from Phase 6.2 matched-data
production and the final training matrix. The executable protocol is
`configs/b1_policy_pilot.json` at historical revision `8bbeefe`. That initial
protocol and its fixed comparison launchers are retired from the active checkout.

## Declared protocol

Three fresh nominal episodes: train `animated-door-1-88abf40` (right) and
`door-2738468b94d74c5f` (left); development `psx-industrial-001-4f5561b` (left).
Frozen corpus membership is unchanged. Collection seeds are 6100/6200, training
seed 17 and policy inference seed 23. No sealed-test payload or evaluation is used.

Each combination first receives 64 updates on one fixed 32-window train minibatch.
A fresh model then receives three epochs of at most 32 minibatches of 32 windows,
learning rate 1e-4. Architecture width remains 128. Both families use 12-action
chunks and six-tick replanning; Diffusion uses EMA, constant LR with eight warmup
updates, and ten-step DDIM. Fixed development sampling uses 256 windows, with
Diffusion loss sampling seed 12345. Only the final checkpoint is used; there is
no development-based checkpoint selection or favorable behavioral retry.
ACT z-score L1 and Diffusion min-max sampled L1 have different units; raw action
errors are reported separately. Validation uses the maintained policy observation
clipping at ten normalized units. Loss alone cannot establish task success.

Full observed-state label reconstruction is audited: A1 is the applied joint
target minus measured joints; A2 is the submitted world goal relative to observed
robot FK, including left-multiplied axis-angle rotation. Bounded physical replay
includes the common 25 s inspection and 180 approach ticks in each representation.
Thresholds remain 1 cm and 5 degrees; unexpected contact stops this unloaded
diagnostic. Loaded replay is not established by this prefix.

The frozen teacher, robot, fingers, camera, controller, force rules and complete
inspection/approach/contact/push/hold/release sequence are reused. Privileged
teacher labels stay in raw annotations and evaluator reports; model inputs are
only the existing 1,298 RGB-D/proprioceptive features. N+1 observations, applied
joint targets, full world goals, neck targets, calibration, timestamps, phases,
terminal observations and separate 120 Hz feedback are retained. Both action views
derive from exactly the same physical episodes. Old engineering recordings cannot
be relabeled as pilot demonstrations.

At most one operational runner attempt per combination uses the same development
identity, nominal setup and seed, with at most 120 policy ticks after inspection.
No admission source or calibrated load source is invented. A refused attempt with
zero policy commands cannot demonstrate closed-loop manipulation.

## Results

Raw data, failures, checkpoints and reports are local under
`outputs/generated/b1-policy-pilot/act-a1a2-pilot/`; `pilot-report.json` consolidates
the results. Entry points are `scripts/pilot_physics.py` (collect/replay/rollout)
and `scripts/pilot_offline.py` (audit/prepare/train).

### Physical teacher data and labels

| Identity | Split | Sustained opening | Peak normal force | Stop | Release |
|---|---|---:|---:|---|---|
| animated-door-1-88abf40 | train | 61.32 degrees | 7.43 N | tracking margin | passed |
| door-2738468b94d74c5f | train | 57.99 degrees | 5.95 N | tracking margin | passed |
| psx-industrial-001-4f5561b | development | 46.97 degrees | 7.27 N | tracking margin | passed |

All three complete simulated teacher episodes passed the maintained rules;
push/hold contact-valid fractions were 100%. These are teacher results, not
learned-policy behavior. The left train and development episodes reached zero
normalized joint margin; their AABB clearance lower bound was zero. Neither
quantity supplies operational swept-space admission. Forces are evaluator normal
contact forces, not measured torque or a verified robot wrench estimate.

There are 14,658 raw observations and 10,155 learned command rows after the shared
inspection prefixes: 7,052 train and 3,103 development. The export contains only
the explicit partial A1/A2 schema. Every original sensor, annotation, command and
feedback group remains in the raw files and exported raw groups.

Full-episode reconstruction maximum errors are 7.46e-9 rad for A1, zero translation
and 1.09e-14 rad for A2. A2 rotation labels are nonzero, reaching 0.24453 rad in the
left train episode. Causal 1,298-feature arrays are finite. Representative reset,
middle and terminal RGB images were inspected for all doors; the dark left train
material was retained unchanged. This is input validity, not perception qualification.

Both physical replay prefixes passed: measured joints and tool positions matched
exactly, rotation differed only by roundoff, and no contact occurred. A1/A2 replay
cycle p95 was 57.99/59.47 ms, with retained maxima 144.62/222.67 ms. This includes
rendering, physics and replay/FK work, without model inference. It does not establish
loaded replay or wall-clock 60 Hz execution.

### Learning and checkpoint results

| Path | Fixed-minibatch L1, before → after | Development L1, before → final | Inference p95 |
|---|---:|---:|---:|
| ACT A1 | 0.6931 → 0.2184 | 0.8119 → 0.5962 | 0.75 ms |
| Diffusion A1 | 0.7238 → 0.2643 | 0.7261 → 0.3360 | 10.59 ms |
| ACT A2 | 0.8286 → 0.2230 | 0.9378 → 1.0069 | 0.77 ms |
| Diffusion A2 | 0.6426 → 0.2487 | 0.6480 → 0.3289 | 10.69 ms |

All four fixed-minibatch reductions exceed the declared 20% criterion. Each fresh
short model completed exactly 96 optimizer updates; no budget was extended. Eight
overfit/final policy checkpoints saved and loaded on RTX 4090 reproduce seeded
predictions exactly. ACT's total L1+10 KL objective fell 7.8435→1.4189 (A1) and
7.8382→1.4027 (A2); Diffusion noise MSE fell 0.7936→0.1804 and 0.8002→0.1734.
Full epoch histories, normalization and model contracts are retained.

ACT A2 development regression is retained with the prescribed final checkpoint.
ACT z-score and Diffusion min-max L1 must not be ranked against each other. Both
representations/families use the same sampled physical windows. Development clips
3.01% of normalized observation entries at the existing ten-unit policy limit;
development never contributes to normalization.

For A2, mean/p95 delta-translation errors are ACT 12.12/26.32 mm and Diffusion
16.80/42.09 mm. Delta-rotation p95 is 1.91/8.66 degrees. These are offline action
prediction errors, not physical tool tracking or contact validity. Per-joint A1
errors and all inference samples/maxima are retained in the individual reports.
Inference timings cover ready features through normalization, model and output;
they exclude camera, encoder, physics, FK, geometry and admission. One seed and
one development identity cannot establish population generalization.

### Operational runner

All four operational starts completed the same 1,512 inspection ticks (25.2 s),
then returned `invalid_perception` at the first post-inspection monitor check.
Each executed zero policy commands, had zero evaluator normal contact force,
froze targets at measured joints without an extra tick, and attempted no release.
Maximum door drift during inspection was 0.00515 degrees; there is no learned
task-progress result. Solver feedback was present, with approximately 15.96 N m
peak absolute joint torque; this is not contact force.

No qualified observed geometry producer was supplied: the support handle held no
estimate. Load and action-admission sources were also absent. The first refusal
does not physically test the later load/stop/admission gates and does not evaluate
SAM3/Point2Pose quality. No qualification flag, privileged estimate or hidden
corrective controller was substituted. These are operational startup/refusal
checks, not successful closed-loop manipulation trials.

Full inspection-cycle p95 was 39.23/39.12/39.42/39.80 ms for ACT A1, Diffusion A1,
ACT A2 and Diffusion A2, with a retained maximum 221.96 ms. This includes sensors,
physics, robot FK, observed checks/encoding and passive telemetry, without model
queries or geometry/admission production. It exceeds the 16.67 ms wall-clock
budget of 60 Hz; simulation-clock cadence is not real-time qualification.

The demonstrated chain is complete physical teacher recording → causal observed
features/labels → train-only normalization → bounded CUDA optimization → saved and
loaded prototype → operational startup and refusal. Learned door opening, loaded
control, moving/contact stopping, A3/A4 execution and generalization remain unqualified.

## Infrastructure interruptions

The first launch completed inspection but failed before approach while serializing
a `numpy.float32` tolerance. Commit `ece1aeb` changes its Python type without
changing its value or guards. The repaired launch used identical inputs and seed.

The app then externally archived worktree `445b` during that launch. Code was
recovered from snapshot `78cb1b5` into the calling `23f1` worktree on the same branch.
The active incomplete 9.5 GB HDF5 and log were recovered from open descriptors
before terminating that process; last logged progress was 38.61 degrees and 0.620 N.
The first incomplete inspection recording and old ignored probes were removed by
the external archive and were not recoverable. The restart retains identical
inputs/seed and the interrupted file is excluded from learning. Neither restart
is a selected successful behavioral retry.

## Follow-up diagnosis

The [[b1-pilot-offline-and-robot-dynamics|offline and robot-only follow-up]] preserves these checkpoints and operational refusals, and separates physical-unit prediction errors from dynamic load/stop verification. It does not repeat these unconfigured operational starts.

## Control boundary and minimum next experiment

Contact rollout requires verified dynamic/peak residual bounds, a moving/contact
stop envelope, and observed local ownership, both finger supports, continuous
sweep/clearance and material-response evidence. Measured joint torque remains
distinct from evaluator normal contact force. A stationary gravity fit and a
trained checkpoint cannot supply these prerequisites.

The [[b1-pilot-offline-and-robot-dynamics|robot-only follow-up]] now tests unloaded
acceleration/deceleration, known external perturbation and stopping without door
contact. It retains substantial loaded stop travel and estimator-envelope exits.
The next control step remains robot-only: verify sustained-load tracking/stopping
and justified dynamic residual bounds without widening thresholds from failures.
Supported local geometry and independent 6.0E, together with the required 6.0D
checks, must pass before diagnostic door interaction 6.0F. Any later supported
result covers only its tested local state/dynamic/stop envelope, not arbitrary
learned commands.

## Reuse and retraining

Raw sensors, full tool goals/rotations, canonical measured joints, phases,
calibration, timing and separate annotations/feedback support later A3/A4
preparation. A3 also needs supported observed hinge/frame estimates; A4 needs
supported angle/panel state and complete segment fitting. Privileged teacher
geometry is retained for supervision/audit, not substituted for observed inputs.
Encoders, normalization, trainers, checkpoint loading, family numerics and the
bounded runner remain reusable.

A3/A4 need newly derived labels and new output heads/checkpoints. A different
encoder/history requires re-encoding the same raw episodes and training again.
Changing the physical robot/tool, command meaning, controller or control cadence
requires new relevant physical demonstrations and training. A verified exact unit
or order conversion alone does not imply another optimization run.
Improving only independent stop-only evidence/calibration does not automatically
require retraining when observations, action semantics, controller and task
distribution remain fixed; physical rollout evaluation must still be repeated.
