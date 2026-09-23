# Visuoproprioceptive Generalization Benchmark

## Context and Status

The completed state-only Phase 3 study covered A2/A3, ACT/Diffusion, one door
family, and one training seed. Its 576 successful rollouts validated the pipeline
but did not distinguish representations. That historical Alex V2 path is
**B0**; its six-joint, translation-only behavior is not the desired B1 contract.

This approved 2026-09-22 decision replaces the former all-in-one Phase 4 plan.
It defines **B1** across Phases 4–7 without claiming implementation. The separate
preparatory code audit is recorded in [[status|Project Status]]; it does not
implement or execute any revised phase.

## Scientific Question

Holding the Purdue robot, observations, physical demonstrations, dataset size,
training budget, and evaluation protocol fixed, how do A1-A4 and ACT/Diffusion
change safe opening progress on geometrically unseen push-door assets?

Zero-shot means acting on held-out door identities without retraining or
adaptation. The comparison covers four representations x two models, with five
independent training seeds each. Handles and other articulated objects are later
studies; this benchmark performs panel pushing without grasping or latch operation.

## Robot and Task Boundary

Use fixed-base Purdue Alex with WSG32/UMI v1, its seven right-arm joints, and the
measured Purdue pedestal. Reuse the existing external Alex package and ZED X Mini
Wide integration. The canonical selection, closed fingers, authorized tip
surfaces, operational frame, physical source facts, and simulation limitations
are in [[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]].

Phase 4 selects one robot-plus-pedestal floor pose relative to the center of the
closed opening, holding measured height fixed. Optimize the minimum sustained
expert angle over four synthetic cases: widths 1.20 and 0.65 m x left/right
hinges. Neither the common robot pose nor the contact rule changes per door.

The expert follows one panel material location, specified by a common fraction
of width from the hinge and an absolute floor height. Start synthetic exploration
at fraction 0.90; freeze the final fraction/height, tool orientation, controller,
limits, and horizon before collected assets. The tool points into the panel with
its vertical axis upward during contact/push/hold. Handles remain collidable
obstacles rigidly attached to the panel, not targets for grasping.

The resulting maximum is a practical expert reference under these constraints,
not a proof of global robot reachability. Mechanical limits, collision/force
safety, and a common finite horizon remain necessary. No shared task-angle
threshold ends expert or policy motion. A timeout or solver stall is not evidence
of a kinematic limit; record the limiting cause explicitly.

## Observations, Perception, and Privilege

All eight cells receive the same synchronized head left RGB, metric left-aligned
depth, valid-depth mask, and robot proprioceptive history, including arm and neck
state. A wrist camera is not required. RGB-D is chosen because the intended ZED
provides metric depth useful for manipulation; the initial rendered depth remains
an explicitly ideal geometric approximation, not a reproduced stereo-error model.

Start with one suitable shared backbone and depth-processing path; no mandatory
backbone comparison is required. Evaluate alternatives only for a concrete
train/development failure. Use one frozen perception stack. An estimator supplies
the door frame/articulation required by A3/A4. First test a fixed neck pose; add
a common deterministic observed-input gaze controller only for a demonstrated
visibility deficit. Never give gaze a perfect simulator door position.

| Consumer | Allowed information |
|---|---|
| Qualification expert and synthetic setup search | Simulator geometry/state and measured simulation contacts. |
| Training supervision | Labels from training-door episodes, kept distinct from model observations. |
| Learned policy, A3/A4 adapters, and optional gaze | Sensor observations, robot proprioception/forward kinematics, and frozen perception estimates. |
| Evaluator and common safety monitor | Simulator truth for scoring/validity and safety stops; no helpful motion commands or angle-based early success. |
| Oracle diagnostic | Explicitly separate results, never main ranking or test-driven model selection. |

No true door pose may enter through reset initialization, cached transforms,
segmentation-derived validity masks, action adapters, or hidden completion logic.
The expert reference angle and asset identity are evaluation metadata, not policy
inputs. If perception loses the door, handle that observed failure explicitly.

## Action Representations

| Representation | B1 execution contract |
|---|---|
| A1 | Seven right-arm joint-target deltas, executed directly without IK. |
| A2 | World-frame 6D tool-pose deltas, including actuated rotation, using seven-joint pose IK. |
| A3 | Hinge-anchored door-frame 6D deltas transformed using estimated geometry, then executed through A2. |
| A4 | Complete object-centric approach/contact/push/hold/release sequence with explicit encoding and boundaries. |

A2/A3 remain six-dimensional Cartesian representations; seven is the arm's
actuated joint count. Fix their current ignored-rotation behavior before measuring
reachability. Complete all dataset, model-output, normalization, adapter, and
rollout paths under both ACT and Diffusion before final data production.

Finger opening is constant and neck control is common/separate. Adapters validate,
transform, and enforce shared safety rules, but never invent missing sequence
stages or complete a failed prediction with the expert. Learned trajectories may
differ from the expert's exact point/path while obeying the same push-only,
authorized-surface, controlled-contact, force, and collision validity rules.

## Asset Corpus and Demonstrations

Target 24 unique qualified identities, 12 left- and 12 right-hinged. Subphase 5.0
first delivers validated reusable normalization and static/GPU-physics checking
tools, before candidate intake begins. Then the user provides one URL at a time:
the model reviews it, the user manually downloads an acceptable candidate, and
the model prepares/checks the local payload using those tools. Each prepared door
can proceed directly to the expert check in 5.1; no initial batch of 24 downloads
is required. Apply the legal, normalization, static, and physics criteria in
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]].
Only CC0 or CC BY 4.0 assets and redistributable dependencies are admitted.

Apply the frozen setup/probe to every asset. A contact fraction naturally produces
a different metric distance on a different width; this is the same rule. Moving
that fraction or height to avoid a particular handle is per-asset tuning and is
not permitted. Replace candidates that are legally/technically invalid or excluded
by the predefined reachable-domain gate. Diagnose probe bugs or unresolved limits
before deciding that a door is unsuitable.

Freeze 12 training, 4 development, and 8 test identities, balanced left/right as
6/6, 2/2, and 4/4. Keep related mesh families together. Mirrors/recolors are not
new identities. Sealed test qualification proves feasibility only; its traces
and images never enter learned training, normalization, tuning, or selection.

The **asset corpus** is normalized doors, provenance, qualification references,
and splits. The **demonstration dataset** is synchronized RGB-D/proprioception
and actions from accepted physical episodes on training doors. It is generated
later, then exported to matched A1-A4 representations. These are distinct products.
Synthetic doors configure the common robot setup only; they are not B1 data.

## Expert Reference and Minimum Admission

For an episode, define the maximum sustained angle as the largest angle maintained
for a contiguous 0.5-second interval of valid controlled panel contact. In sampled
form, this is the maximum of window minima over eligible intervals. Use measured
seconds, not a hard-coded number of ticks when cadence changes. An uncontrolled
impact/coasting peak does not establish controlled opening. Expert qualification
requires a valid hold and release; the door need not remain open forever after
release. Policy release completion is reported separately from partial progress.

Phase 5 runs the frozen probe twice from an identical reset. Valid outcomes and
limiting causes must agree, and sustained maxima must differ by at most 2 degrees.
Diagnose failures without best-of-many selection. Freeze `theta_expert_d` as the
lower of the two valid maxima. This is a conservative deterministic reference,
not a statistical reliability percentile or global optimum.

Require `theta_expert_d >= 45 deg` solely for nominal asset admission. Exclude
lower-angle candidates as outside the chosen reachable domain, even if their
assets are valid. Do not stop the probe at 45 degrees. An unresolved timeout/stall
cannot establish the reference. Do not derive a shared primary angle from the
worst asset. The old `theta_primary`, `success@45deg`, `success@60deg`,
`success@75deg`, adaptive `n_qual`, bootstrap selection, and identical-rollout
percentile protocol are superseded and are not part of B1.

## Metrics and Validity

For a nominal door `d` and policy rollout `r`:

`expert_normalized_progress_d,r = policy_max_sustained_angle_d,r / theta_expert_d`

Use the same sustain definition, physical scenario, control/validity rules, and
episode time budget for expert and policy. A valid episode with no qualifying
sustained-opening interval has zero progress. Preserve ratios above one: a policy
can outperform the frozen probe. Never change the denominator after seeing that
result, and never use it as an inference stopping target.

For scientific aggregation, define `valid_progress_d,r` as this ratio when the
whole policy episode passes the shared safety/physics/control validity gates,
and zero when it violates them (including force or forbidden contact). Keep
invalid episodes in the trial denominator. A safe stall, horizon exhaustion,
or incomplete task sequence alone does not erase previously sustained valid
progress; report its stop reason and release status. This avoids reintroducing
binary task completion as a hidden primary gate. Uncontrolled intervals cannot
satisfy the sustain definition, and any unsafe release invalidates the episode.
Retain their raw angles, ratios where defined, stop reasons, and violation counts
as diagnostics; never report a valid-only average as the main ranking.

Average paired rollout scores within each door, then weight doors equally within
a split/tier; report variation across training seeds. **Primary ranking is GEO
validity-adjusted expert-normalized progress.** Report raw maximum sustained angle,
per-door normalized progress, per-split/tier results, ID-minus-GEO generalization
gap, validity rate, and force diagnostics. Force is a validity gate, not an
alternate main optimization target. Report interruptions due to infrastructure
separately and resolve affected paired comparisons rather than hiding failures.

## Evaluation Conditions and Matched Expert References

| Tier | Evaluation domain |
|---|---|
| ID | Training identities under nominal physical and visual conditions. |
| GEO | Sealed identities under nominal conditions; primary generalization result. |
| POSE | Seen identities at held-out door poses within the declared domain. |
| LIGHT | Seen identities under held-out illumination. |
| DYN | Seen identities under held-out physical parameters. |
| COMPOUND | Sealed identities with held-out pose, illumination, and dynamics. |

Run and report ID/GEO first, then the planned POSE/LIGHT/DYN/COMPOUND stress
extension. A core result may be delivered while stress work is explicitly pending;
it is not a completed full evaluation. Define all ranges and eligibility rules
using train/development data before any sealed policy evaluation, so core test
results cannot tune the later stress protocol. Use 20 paired conditions per
evaluated door/tier and the same conditions/seeds for every model/representation.
Distinct conditions and training
seeds provide variation; identical deterministic repeats do not create new doors.

For condition `c` that changes physical reachability, replace the nominal
reference with `theta_expert_d,c` from the same frozen probe and physical reset,
using the same sustain/validity protocol. Compute it before policy outcomes and
reuse it across all checkpoints. A purely visual change can reuse the physically
identical reference. This includes physical reset perturbations in nominal tiers,
not just POSE/DYN/COMPOUND. Report nominal and stress scores separately.

The 45-degree rule applies to nominal asset admission; it is not reapplied to
remove difficult stress cases. A positive valid stress reference supports the
ratio even below 45 degrees. A zero, invalid, or unresolved reference makes that
scenario unqualified: report its coverage/reason for every model and do not claim
a complete comparison or silently switch denominators. Never tune the expert,
resample test conditions, or replace a test door to improve learned results.

## Data Strategy and Remaining Freeze

The frozen scripted expert is the primary teacher; Replicator supplies visual
variation and training annotations through the existing capture path. Assess
visual variety and motion coverage separately. A bounded Mimic trial or targeted
teleoperation is conditional on a pilot-demonstrated motion gap that direct expert
generation cannot address economically; adapted trajectories must satisfy the same
physical validity and demonstration rules. The generation recipe belongs to
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6, Subphase 6.2]],
without a new subphase. Gaze is likewise implemented only for a
demonstrated fixed-view deficit. Defer RL teachers, VLA extensions, and unrelated
experiments.

The small end-to-end pilot remains mandatory: pass it before final dataset
production or full training. Combining pilot and production into Subphase 6.2
does not remove this gate. Phase 6 selects/fixes perception and optional gaze,
history, depth processing, A4 encoding/length, accepted teacher mix,
dataset membership/size, randomization
ranges, normalization, training budget, five seeds, checkpoint selection, and
paired evaluation conditions using train/development doors only. Use the same
accepted physical episodes for all representations. Full training produces
2 x 4 x 5 = 40 selected checkpoints, without favorable replacement seeds.

## Implementation Order and Essential Deliverables

The plan has nine subphases, grouped by concrete outputs:

| Phase | Subphases | Output |
|---|---|---|
| [[implementation_phases/phase-4-robot-and-task-configuration|4 — Robot and Task Configuration]] | 4.0 Operational Alex003/control/RGB-D; 4.1 common pose/reachability/visibility. | Usable robot and frozen synthetic-qualified setup. |
| [[implementation_phases/phase-5-door-corpus-and-qualification|5 — Door Corpus and Qualification]] | 5.0 tools first, then one-URL-at-a-time preparation; 5.1 expert qualification and final split. | Qualified doors, references, and 12/4/8 split. |
| [[implementation_phases/phase-6-perception-actions-and-demonstrations|6 — Perception, Actions, and Demonstrations]] | 6.0 observations/perception; 6.1 complete A1-A4; 6.2 passed pilot followed by final dataset. | Shared observed-input stack, eight usable learning paths, and matched demonstrations. |
| [[implementation_phases/phase-7-training-and-generalization-evaluation|7 — Training and Generalization Evaluation]] | 7.0 training; 7.1 evaluation and analysis, ID/GEO before stress tiers. | Core generalization result, subsequent stress analysis, and reproducible records. |

Retain licensed asset provenance, split, setup/probe and expert references,
observation/action contracts, dataset recipe, frozen protocol, and per-door and
aggregate results. Public-release packaging is a final checklist, not a separate
subphase or a blocker to delivering the research result. Claims cover this
qualified push-door domain, not handle manipulation, other object families,
physical safety, or sim-to-real.

## Version Notes

- 2026-09-23 — Specified Replicator dataset variation, separate motion-coverage
  checks, and the pilot-dependent criterion for using Mimic within Subphase 6.2.
- 2026-09-22 — Consolidated to nine subphases; tools precede sequential URL intake,
  optional work is conditional, and ID/GEO reporting precedes stress analysis.
- 2026-09-22 — Replaced the shared-angle design with Purdue/WSG32/head RGB-D,
  synthetic minimax setup, per-door/per-condition expert progress, and Phases 4–7.
