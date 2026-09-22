# Phase 6 — Perception, Actions, and Demonstrations

> Planned. Requires Phases 4 and 5. No implementation, pilot, dataset generation,
> or training was performed by the 2026-09-22 documentation revision.

## Objective

Make A1-A4 genuinely trainable and executable through the same observed RGB-D
and proprioceptive interface, then produce matched training demonstrations.
Follow [[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 6.0 — Multi-Door Observations and Recording

#### Implementation

Complete the asset-indexed environment and synchronized RGB-D/proprioceptive
recording using the Phase 4 sensor integration and Phase 5 corpus. Use Isaac Sim
Replicator for shared visual conditions and annotations. Keep annotations and
expert simulator state separate from policy observations and inference state.
Verify every admitted door's reset, capture, and expert execution interfaces
without changing the common robot/contact setup.

#### Key Decisions

- Asset identity, split, scene condition, and randomization seed are metadata,
  not hidden policy inputs.
- Synthetic configuration probes and held-out qualification traces do not enter
  B1 learned training data. Fit observation/action normalization on training data.

#### Problems / Limitations

Complete only when timestamps, depth units/mask, action timing, reset, and
terminal-state recording are consistent. Ideal rendered depth retains the
approximation stated in the robot contract.

## Subphase 6.1 — Frozen Perception and Optional Gaze

#### Implementation

Using training doors only, train the door-frame and articulation estimator
needed by object-relative execution. Select preprocessing/backbone and depth
fusion on development doors, then freeze them for every comparison cell.
The existing DINOv3 ViT-B/16 versus SigLIP 2 Base Patch16 pilot remains a bounded
candidate comparison; RGB-only encoders do not by themselves consume metric
depth, so freeze an explicit shared depth/mask processing path as well.

If Phase 4 found a static-view deficit, implement bounded deterministic gaze
using RGB-D/estimated geometry and neck proprioception. Train any supporting
perception on training doors and select it on development doors. Validate
occlusion, loss/reacquisition, and safe behavior when estimates are unavailable.
Otherwise retain the frozen fixed neck pose.

#### Key Decisions

- Define estimator accuracy and closed-loop usability criteria before training.
- A3/A4 and gaze receive estimated geometry only in the main learned evaluation.
  Ground-truth state is restricted to teacher, labels, evaluator, or a separately
  labeled oracle diagnostic that cannot select models.
- Freeze the same perception/gaze components for all representations. Never
  retune them per test door or use simulator door pose to cache a perfect reset.
- If perception fails, expose that failure. No silent oracle substitution.

#### Problems / Limitations

Complete after development-set perception and, if needed, gaze pass their
predeclared gates. Visibility diagnosed in Phase 4 is not proof that learned
perception works. Test assets cannot set tolerances or select this stack.

## Subphase 6.2 — Complete and Verify All Four Action Paths

#### Implementation

Implement the full dataset/model/normalization/adapter/rollout path for both ACT
and Diffusion under every action representation:

| Representation | Required executable behavior |
|---|---|
| A1 | Seven joint-target deltas in the frozen arm order; direct joint execution without IK. |
| A2 | World-frame 6D tool-pose deltas; both translation and rotation executed through seven-joint IK. |
| A3 | Hinge-frame-relative 6D tool-pose deltas; estimated frame transform followed by the A2 executor. |
| A4 | Complete object-centric approach/contact/push/hold/release sequence; explicit tensor encoding and stage boundaries, using estimated geometry. |

Phase 4 supplies the shared low-level pose controller; this subphase completes
its learned interfaces and the A1/A4 paths. Do not treat A2/A3 as already correct
merely because the old translation-only learned pipeline runs. Verify action
semantics, rotational execution, tool offset, joint order, limits, and causal
observations across all eight paths.

#### Key Decisions

- Keep neck and finger commands outside the learned arm action arrays.
- Adapters may validate, transform, and apply shared safety limits, but must not
  invent missing A4 stages or append an expert push to an incomplete prediction.
- No representation receives the expert angle, true hinge/frame state, or a
  hidden controller that finishes the task. Predicted motion goals are allowed;
  common benchmark stopping angles are not.
- Learned trajectories need not reproduce the teacher's exact point or path.
  They must obey the same push-only, authorized-surface, collision, force, and
  controlled-contact validity rules. This permits legitimate progress above
  the fixed expert reference without giving an adapter privileged assistance.

#### Problems / Limitations

Complete only when all eight paths can replay matched actions and perform
end-to-end smoke rollouts with observed geometry. Legacy six-joint/state-only
datasets and checkpoints must not be relabeled as compatible B1 products.

## Subphase 6.3 — Data and Evaluation-Condition Pilot

#### Implementation

Run a small train/development-only pilot through recording, matched A1-A4 export,
loading, training smoke, and closed-loop execution. The frozen scripted expert
is the primary teacher. Use Replicator for visual variation; admit a bounded
Isaac Lab Mimic trial or targeted teleoperation only for an identified coverage
gap and after validity/pairing checks. Defer RL teachers.

Determine dataset size, history, A4 sequence length, randomization ranges,
training budget, and selection rules. Define 20 meaningfully varied paired
evaluation conditions per door/tier; repeating the identical deterministic
episode does not create additional geometric evidence. Define the stress-test
expert procedure before sealed test evaluation, as specified in the decision.

#### Key Decisions

- Keep the accepted physical episodes and observations identical across A1-A4.
- Keep teacher identity explicit; extra teachers do not redefine the frozen
  qualification reference or use test-door results to improve the probe.
- Fit choices on train/development only. Do not adapt base pose, contact rule,
  or the admitted corpus in response to learned failures.

#### Problems / Limitations

Pilot data are engineering evidence, not the final benchmark dataset. Complete
only after validity, coverage, pairing, synchronization, and learnability pass.

## Subphase 6.4 — Protocol Freeze and Final Demonstrations

#### Implementation

Freeze the shared perception/gaze, observation history, accepted teacher mix,
dataset membership/size, action encodings, normalization, randomization ranges,
training budget, five-seed list, checkpoint-selection rule, and evaluation
conditions. Generate the matched demonstration dataset from training doors.
Development doors support selection/evaluation; sealed test qualification and
evaluation evidence never become demonstration training data.

#### Key Decisions

- Use one accepted episode set for every matrix cell, with representation-specific
  action arrays and train-only normalization.
- Verify pairing, schema, replay, and split separation. Later protocol changes
  require an explicit new benchmark version, not silent reranking.

#### Problems / Limitations

Complete only when the final dataset and frozen protocol are usable by all eight
paths. Phase 7 owns full training and sealed evaluation.

## Artifacts

Future outputs: synchronized observed-input pipeline, frozen perception/gaze,
eight executable learning paths, pilot evidence, protocol, and matched dataset.

## Files

Expected surfaces: `src/alexdoor_xas/envs/`, `src/alexdoor_xas/action/`,
`src/alexdoor_xas/adapters/`, `src/alexdoor_xas/recording/`,
`src/alexdoor_xas/data_engine/`, `src/alexdoor_xas/dataset/`,
`src/alexdoor_xas/policies/`, `configs/`, and supported `scripts/` entry points.
