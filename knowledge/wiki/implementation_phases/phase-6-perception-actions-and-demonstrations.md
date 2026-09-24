# Phase 6 — Perception, Actions, and Demonstrations

> Planned. Three subphases cover observed inputs, all action paths, and a gated
> pilot-to-dataset workflow. Requires Phases 4 and 5; nothing is executed by this revision.

## Objective

Make A1-A4 trainable and executable with the same observed RGB-D/proprioceptive
interface, then produce matched training demonstrations. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 6.0 — Multi-Door Observations and Perception

#### Implementation

Extend the Phase 4 capture path to the Phase 5 asset-indexed environment and
synchronized recording. Do not rebuild camera integration. Verify reset,
timestamps, depth/mask units, action timing, terminal state, and expert execution
without changing the common robot/contact setup. Keep annotations and privileged
expert state separate from observed inputs and inference state.

Use one suitable shared visual backbone and explicit depth/mask processing path.
Do not run a mandatory backbone comparison: consider alternatives only if a
concrete train/development failure makes that work necessary. Train the door-frame
and articulation estimator using training-door data, select it on development
doors, then freeze preprocessing, history, features, and outputs for every cell.
Early training-door recordings may support this engineering work; they do not
constitute the final matched policy dataset.

If Phase 4 demonstrated a fixed-view deficit, implement bounded deterministic
gaze using observed RGB-D/estimated geometry and neck proprioception. Verify
occlusion, loss/reacquisition, and safe behavior when estimates are missing.
Otherwise retain the fixed neck pose and omit gaze implementation.

#### Key Decisions

- Define estimator accuracy/closed-loop usability gates before training. Freeze
  one shared perception/gaze stack, never tuned per representation or test door.
- A3/A4 and gaze use estimated geometry in main evaluation. Simulator state is
  restricted to teacher, training labels, evaluator, or a separately labeled
  oracle diagnostic that cannot select models.
- Asset identity, split, and randomization seed are metadata, not hidden inputs.
  No oracle reset cache, segmentation-derived sensor mask, or silent fallback.
- Synthetic setup probes and held-out qualification traces are not learned
  training data. Fit normalization on training data only.
- Reuse existing Replicator APIs and the Phase 4 camera capture for visual
  variation/annotations, following the Subphase 6.2 dataset recipe. Keep RGB,
  depth, valid-depth mask, proprioception, and actions synchronized; training
  annotations remain separate from policy inputs.

#### Problems / Limitations

Complete after synchronized recording and development-set perception pass,
including gaze only when needed. Ideal depth remains an explicit approximation;
visual randomization does not reproduce real ZED stereo errors or missing depth.
Synthetic visibility alone does not prove learned perception works; no separate
stereo-error modeling project is required.

## Subphase 6.1 — Complete All A1-A4 Learning and Execution Paths

#### Implementation

Complete dataset, model-output, normalization, adapter, and rollout support under
both ACT and Diffusion:

| Representation | Required behavior |
|---|---|
| A1 | Seven joint-target deltas in the frozen arm order, executed directly without IK. |
| A2 | World-frame 6D tool-pose deltas, with translation and rotation executed through seven-joint IK. |
| A3 | Hinge-frame-relative 6D tool-pose deltas, transformed through estimated geometry and the A2 executor. |
| A4 | Complete object-centric approach/contact/push/hold/release sequence, with explicit tensor encoding and stage boundaries. |

Reuse Phase 4's low-level pose controller. Verify joint order, tool offset,
rotational execution, limits, causal observations, and action semantics through
all eight paths. The old translation-only A2/A3 pipeline is not sufficient B1
support. Verify matched-action replay and observed-geometry smoke rollouts before
the data pilot.

#### Key Decisions

- Keep neck and constant finger commands outside learned arm action arrays.
- Adapters may validate, transform, and apply shared safety limits. They cannot
  invent missing A4 stages, append expert motion, or read true door state.
- Predicted motion goals are allowed; shared benchmark stopping angles and
  expert-reference inputs are not.
- Policies may differ from the teacher's exact point/path while obeying common
  push-only, authorized-surface, collision, force, and controlled-contact rules.
- Do not relabel legacy six-joint/state-only data or checkpoints as compatible B1.

#### Problems / Limitations

Complete only when all eight ACT/Diffusion x A1-A4 paths work. A partial A1/A4
implementation cannot support the intended representation comparison.

## Subphase 6.2 — Small End-to-End Pilot and Final Dataset

#### Implementation

First run a small train/development-only pilot through recording, matched export,
loading, brief training, and closed-loop execution. Diagnose validity, timing,
coverage, pairing, and learnability before large-scale generation. The frozen
scripted expert is the default teacher. Generate physical demonstrations directly
on the 12 training doors with balanced coverage of doors and permitted conditions.
Each accepted demonstration includes approach, contact, opening, hold, and release
under the common contact and safety rules, without a shared angular stopping target.

Use Replicator for plausible variation in lighting position/intensity/color,
surface appearance, and background elements that do not obstruct manipulation.
Keep these settings constant within an episode by default. Visual material changes
must not silently change friction or other physical properties. Any variation in
initial states or dynamics uses the environment and declared training ranges,
preserving the frozen common robot/contact setup. Keep held-out stress ranges
distinct from training ranges and select neither from sealed test outcomes.

Assess motion coverage separately from visual variety: repeated execution with
different images does not add new corrective behavior. If recovery examples are
needed, record safe, successful expert corrections from permitted perturbed states;
adding random action noise alone is not a recovery demonstration. Check physical
validity, contact, force, timing, and matched replay before accepting episodes.

Use a bounded Isaac Lab Mimic trial or targeted teleoperation only when the pilot
identifies a motion-coverage gap that direct expert generation cannot address
economically. Mimic requires source demonstrations, subtask annotations, and
environment/controller integration. Re-execute and validate every adapted candidate;
do not assume trajectory transformations preserve a door's contact arc when width
or handedness changes. Keep only candidates meeting the same demonstration rules.
If scripted data suffice, omit Mimic and teleoperation.

Use the pilot to fix dataset size, observation history, depth processing, A4
encoding/length, teacher mix, randomization, normalization, training budget,
five-seed list, checkpoint selection, and paired evaluation conditions. Define
20 meaningfully varied conditions per evaluated door/tier; identical deterministic
repeats are not extra geometric evidence. Freeze eligibility and stress ranges
on train/development data before any sealed test-policy results, even though
Phase 7 reports ID/GEO first and stress results afterward.

Only after the pilot passes, freeze the protocol and generate the final matched
demonstrations from training doors. All representations use the same accepted
physical episodes and observations, with representation-specific action arrays.
Verify schema, pairing, split separation, and replay. Development data support
selection/evaluation; test evidence never becomes training demonstrations.

#### Key Decisions

- Combining pilot and production in one subphase does not remove the pilot gate.
  Do not generate the full dataset or launch the 40-run matrix before it passes.
- Teacher source remains explicit. Added teachers do not redefine expert
  qualification or tune the frozen probe from test results.
- Learned failures cannot retune base pose/contact rules or replace admitted doors.
- Defer RL teachers and VLA work. Alternative backbones, gaze, Mimic, and
  teleoperation are conditional remedies, not mandatory benchmark activities.
- Changes after protocol freeze require an explicit new benchmark version.

#### Problems / Limitations

Pilot recordings are engineering evidence. Complete only with a passed pilot,
frozen protocol, and final dataset usable by all eight paths. Phase 7 owns full
training and sealed evaluation.

## Artifacts

Future outputs: synchronized multi-door recordings, one frozen perception stack
and optional gaze, eight executable learning paths, pilot evidence, protocol,
and matched dataset. None was produced by the documentation revision.

## Files

Expected surfaces: `src/alexdoor_xas/envs/`, `src/alexdoor_xas/action/`,
`src/alexdoor_xas/recording/`, `src/alexdoor_xas/dataset/`,
`src/alexdoor_xas/policies/`, `configs/`, and supported `scripts/` entry points.
Learned-action integration will be implemented against the B1 runtime; no B0
adapter package remains to be extended.
