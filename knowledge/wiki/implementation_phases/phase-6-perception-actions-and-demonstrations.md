# Phase 6 — Perception, Actions, and Demonstrations

> Subphase 6.0 remains unqualified. 6.0A contracts and 6.0B static evidence/queries
> are maintained. Previous dynamic trackers are retired; 6.0C integrates
> official CAD-free Point2Pose as an unqualified prototype. 6.0D-H remain planned.
> Subphase 6.1 software is partially validated; final integration is pending.
> Subphase 6.2 is not started. Phases 4 and 5 are complete.

## Objective

Make A1-A4 trainable and executable with the same observed RGB-D/proprioceptive
interface, then produce matched training demonstrations. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 6.0 — Operational Perception and Observed Contact

#### Current implementation boundary

Frozen GroundingDINO/native SAM3/DINOv3 and calibrated RGB-D/multiview geometry
support 6.0B static object memory, ownership alternatives, hinge hypotheses and
finite contact/relevant-space queries. The static provider exposes scan evidence
but no qualified state, policy encoding or admission to move. Human-reviewed local
regions and corrected contact covers remain; whole-object identity and axes are
unresolved. Existing recordings lack torque and the observed monitor blocks loaded
execution.

Legacy geometric tracking/evaluation, custom material tracking and video comparison
are retired after preserving their failures. Independent full-state/local contracts,
explicit selection, timing/reset, admission and feedback interfaces remain. Future
The Point2Pose prototype tracks candidate panel pose from distributed visual references and RGB-D,
compensating camera kinematics and preserving local contact checks. No integration,
training, collection or release migration is implemented. All 50 engineering-v2
episodes remain for future common qualification. See
[[experiments/b1-perception-findings|findings]] and
[[topics/shared-door-perception|maintained implementation]].

#### Approved implementation sequence

The canonical implementation specification is
[[phase-6-0-operational-perception-and-contact|Phase 6.0 — Operational Perception and Observed Contact]].
It defines inputs, uncertainty, contact selection, provisional action admission,
force feedback, scoring, failure handling and staged qualification. Keep this parent
page as the phase overview; use the detailed plan for requirements and exit checks.

| Package | Outcome | Prerequisites |
|---|---|---|
| 6.0A — complete | Explicit operational/provisional/load contracts, measurable admission rules and numerical consumer compatibility | Approved protocol and current consumer audit |
| 6.0B — complete | Static object fusion, fixed/unresolved support, hinge alternatives and geometric queries; bounded scans and targeted evidence/contact corrections; identity/interaction remain unqualified | 6.0A |
| 6.0C — prototype, unqualified | Official CAD-free Point2Pose, distributed RGB-D references, camera-motion compensation, panel pose and panel-fixed push zone; local contact checks retained | 6.0B |
| 6.0D | Documented robot feedback, common compliant execution and physical stop | 6.0A; independent of B/C with numerical fixtures |
| 6.0E | Independent dual-profile evaluator and both pilot replays | 6.0A-C |
| 6.0F | Bounded diagnostic manipulation on the two train pilots | 6.0D/E gates |
| 6.0G | One common recipe on all existing train/development episodes | 6.0F |
| 6.0H | Dynamic qualification, evidence-backed freeze and 6.1 handoff | 6.0G and 6.0D |

#### Key decisions

Preserve A4, calibrated FK/IK, causal observation/availability semantics and frozen local
visual models. Move from a single selected plane to observed leaf surfaces and
motion. Total dimensions are optional for operational use unless an action needs
them. The diagnostic controller chooses a supported contact patch; the teacher's
0.295/1.09 m rule and recorded labels remain unchanged. Policies choose their own
actions; adapters and monitors never supply corrective task motion.

Use documented joint-torque feedback initially in the common monitor, not as a
new policy feature. Its semantics, uncertainty and timing require verification;
no wrist force sensor or perfect simulated contact measurement is assumed.
Separate geometric qualification, bounded provisional action admission and loaded
control. Safety/loss stops latch and clear pending actions; reacquisition cannot
restart motion. Simulator truth remains evaluator-only for this path.

The new operational-v1 profile retains 1 cm / 5 degree limits for required hinge,
angle and contact state, plus 95% coverage and joint accepted-state precision.
Provisional and missing estimates remain in the denominator. Legacy full-state
results retain their original dimensions/prescribed-contact gates and failures.
The detailed plan explicitly permits bounded pilot dynamics after pilot offline
and feedback/stop gates; qualification dynamics still require all train/development
offline gates. Neither exception nor an incomplete prototype is a release.

#### Completion and limits

Complete only after 6.0H and both operational gate groups. Keep ideal depth and
unverified hardware feedback/mounting limitations explicit. This is operational
manipulation qualification, not a claim of exact full-door reconstruction or
hardware safety. No training, new demonstration collection or sealed-test access
belongs to this revision. Phase 6.2 retains future collection/training ownership.

## Subphase 6.1 — Complete All A1-A4 Learning and Execution Paths

#### Current implementation boundary

The model-independent B1 observation builder, matched dataset interface, full action
contracts, ACT/Diffusion bindings and execution/replay runner are implemented.
Numerical checks cover all eight data/scaling/dispatch paths, rotations, A4 boundaries,
reset, stale/lost observations, replay timing and stop-only safety. CUDA forward,
loss/gradients, inference and checkpoint prediction checks pass for the maintained
families without optimizer updates during cleanup. The four optimizer-based training
regressions were deliberately not run and remain maintained.

No qualified/frozen perception provider or B1 policy dataset exists. Final raw/live
encoding equivalence, matched physical replay and observed-geometry rollout across
contact/hold/release and loss/reacquisition remain blocked by Phase 6.0. Numerical
fixtures are not physical validation. Subphase 6.1 remains open and 6.2 not started.
See [[topics/action-representations-and-adapters|Action Representations]],
[[topics/episode-and-dataset-contracts|Data Contracts]] and
[[topics/shared-door-perception|Perception]].

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
on all training doors assigned by the Phase 5 frozen split, with balanced coverage
of those doors and permitted conditions. Using the full qualified corpus across
the three partitions does not make development/test doors training data.
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
and matched dataset. Current engineering recordings and the unqualified estimator
are preparation artifacts, not the final matched dataset or a qualified stack.

## Files

Expected surfaces: `src/alexdoor_xas/envs/`, `src/alexdoor_xas/action/`,
`src/alexdoor_xas/recording/`, `src/alexdoor_xas/dataset/`,
`src/alexdoor_xas/policies/`, `configs/`, and supported `scripts/` entry points.
Learned-action integration will be implemented against the B1 runtime; no B0
adapter package remains to be extended.
