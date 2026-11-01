# B1 Policy Pilot

The maintained pilot uses ACT/Diffusion × A1/A2 with `rgbd-conv-v1`, observed RGB-D
and proprioception. The full panels remain deferred and incomplete. Current
execution authority belongs in [[../status|Project status]]. No final matched A1–A4 dataset,
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

The original pilot uses 120 Hz physics and 60 Hz control; the current scoped
ACT acquisition/stop qualification uses 240 Hz physics with four native substeps
per 60 Hz control interval. Primitive chunks contain 12 actions;
execute six ticks, then replan. Both families support `n_action_steps=6`; ACT
ensembling is disabled in the pilot. A4 replans only at segment boundaries.
Encoder inference, policy inference and geometry publication are separate events:
none refreshes another channel's support. Synchronous simulation completion uses
its simulation clock; wall-time cost does not establish 60 Hz hardware execution.

The observed runtime accepts `geometry_inference_period_s` in its operational
recipe, retaining the 0.1 s default. The shared session validates the period
against its existing maximum gap. A shorter period processes actual new RGB-D
captures with the same clocks, robot filter, ownership and uncertainty rules;
it never refreshes support by relabeling an old result. Synchronous inference
can increase wall-time cost and does not qualify real-time hardware throughput.

An optional `inspection_acquisition` in the operational recipe declares actual
view waypoints and sample times for both native collection and the observed
provider. It must retain mounting, neck limits and every other inspection field;
the normal trajectory validator still applies. The observer/IO binding uses this
same schedule, while independent workspace recordings retain their original
recipe and timestamps. Loaded-start pre-roll schedules remain explicit. Extra
views establish support only after actual native acquisition and independent
review; they do not qualify a full cycle or relax identity/material gates.

Local normal conditioning checks the original finite triangle plus geometric
alternatives seeded at the observed coordinate extrema. Every certificate pays
the same maximum point/model residual over the entire measured planar core;
only a complete independently valid bound can be selected. Collinear or otherwise
unconditioned support remains unavailable. This does not shrink metric error,
correspondence residuals, material coverage, motion bounds or stopping criteria.

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

An optional `pause_aware_seek` clock distinguishes an unchanged submitted target
from active searching. Target change uses independently verified robot FK
resolution; actual settling must stay inside the existing command-response
position/rotation envelope. These serve different purposes. Neither substitutes
material uncertainty for robot support. Cumulative target creep cannot renew a
small movement allowance. Before/after checks at one clock cannot erase the
submitted interval's search intent. Pause diagnostics report that effective
interval too: a same-clock after-check cannot label an already submitted active
command as a supported pause. Provisional unresolved target motion is reported
separately until it resolves into bounded settling or chargeable creep. The existing 3 s/30 mm budgets apply to active
search in this optional mode; supported held-target settling uses the already
qualified response envelope. Total elapsed time and measured travel remain
reported, and force/confirmation, identity, sample continuity and independent
action/stop checks remain unchanged. Existing
recipes retain the prior clock. The new clock passes the declared two-door
reference contact/stop checks. Saved changing ACT requests also retain bounded
active search, and genuine-history late-left braking/realignment and the widest
alignment/retreat/reapproach pass scoped contact/stop qualification. See [[../experiments/approach-contact-corrections]]
for that separate operational boundary and the independently qualified position
support. Neither establishes learned behavior.

`PurdueLoadSource` retains every configured native substep, measured torque, calibration
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
clocks; missing historical streams stay missing. Raw references are now the default;
`--copy-raw` explicitly creates a self-contained transfer, and the historical
`--reference-raw` spelling remains accepted. This avoids copying
retained RGB-D while checking identity. The library `export_dataset` default
remains self-contained; the maintained preparation CLI defaults to references.
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
load/dynamics/stop verification remain maintained. The old pre-latch control
smoke is retired; current runtime and stop checks cover maintained execution.

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


## Operational tolerance diagnostics

Empirical load position extrema are separate from physical joint limits.
`LoadCalibration.position_tolerance_rad` supplies independently validated lower/
upper allowances by joint, default zero, with an explicit force domain. Actual
robot q and Jacobian are used unchanged. Weak support cannot qualify future loaded
motion or stopping outside the main domain without observed free-space evidence.
Refusals carry numeric measurement/limit/allowance details. New position support
cannot combine with the independent extended-acceleration interval without
additional joint evidence.

Local interval response bounds describe the complete material motion during the
proposal, including response to actuation. The pilot supplies its existing linear
and angular rate envelopes times the interval duration. Temporal terms separately
cover acquisition age, crediting growth already in the tracker. The robot's
13 mm / 0.04 rad tracking response remains a distinct source.

Pose-error primitives retain their measured world rotation origins. Admission
propagates each primitive directly to both proposed finite footprints; it does
not add correlated angular travel via an arbitrary material anchor. Missing
primitive detail uses the conservative anchor/offset bound, and inconsistent
supplied primitives refuse. Finite material queries already erode the declared
geometry tolerance: `LocalActionSupport.geometry_position_prepaid_m` credits only
that amount, requires an independent positive robot sweep proof, and cannot
exceed the published geometry error. Default credit remains zero.

Contact material margins measure lateral footprint support. With an independent
robot collision/stop receipt, only the tangential projection of the extra normal
model discrepancy spends that margin; the full normal uncertainty remains in the
separation interval. Logs retain material margin and uncertainty terms, separately
from the whole-robot sweep residual. Missing support remains a refusal.

An explicit `pilot_physics.py rollout --reference-recording PATH` uses the same
operational runner and checks for a recorded A1 diagnostic. The fixed checkpoint
still establishes compatibility; replay commands are non-learned. Independent
scoring reports `reference_cycle_success` and never counts it as learned success.
See [[../experiments/b1-operational-tolerances|measurements, failures and limits]].

## Bounded data infrastructure

Preparation now yields one frame-indexed episode at a time through the unchanged
observer/FK and label audit. No corpus of overlapping visual tensors is retained
in RAM. `pilot_offline.py index --root DEST --protocol PROTOCOL --prepared-data SOURCE`
converts an existing prepared dataset without CUDA, preprocessing or source edits.
`DEST/data` must not exist. Metadata, train/development membership and original
normalization fit remain unchanged; embedded raw data is linked unless `--copy-raw`
is explicit. Separately referenced raw paths must be available or relocated for
raw inspection; policy loading is independent of those privileged records.

Loading guards budget active batches/normalization and small eager metadata;
logical image bytes remain a separate diagnostic, including indexed and VDS data.
The input factories keep one bounded prefetched batch, with a synchronous library
option. See [[episode-and-dataset-contracts|storage, validation and compatibility]]
for close/reopen and value-check timing. These changes select no new encoder,
history, training strategy, checkpoint or task outcome.

## Identified normal-bound aging

For an unloaded whole-robot proof, the interval erosion already covers relative
material motion. The tracker publishes the same identified linear-aging primitive
inside pose and normal-position bounds. Receipt renewal credits its positive
normal-bound growth once, capped at the explicit interval motion term. Other
uncertainty growth still spends the original remaining margin; absent attribution,
missing prepaid erosion, missing support or true margin exhaustion still refuses.
This changes accounting, not physical limits. The finite native prefix validation
and preserved pilot refusals are in [[act-a1-first-completion]].

The producer also retains each complete swept sphere's own eroded separation
margin. Renewal spends that sphere's measured normal-error growth against its
corresponding margin, with strict positive residual required for every sphere.
The global minimum remains the admission summary; it cannot be combined with
unrelated distant-body growth. Missing per-sphere margins retain the older,
more conservative global-minimum rule. Incomplete, nonpositive or inconsistent
margin vectors refuse. No point/model error, sphere radius, temporal erosion,
collision proof, physical limit or stopping criterion is reduced. Exact saved
D3/D4 receipts expose the spatial mixing; this numeric correction still requires
guarded native full-cycle qualification.


## Temporal loaded-phase diagnostics

`pilot_physics.py rollout --temporal-act --intermediate-state PATH` separately
assesses a learned continuation from an existing qualified selected training state.
It reuses loaded-state reconstruction and independent start-stop qualification,
then acquires fresh native observation history with unavailable prefix masked.
After handoff, the unchanged temporal ACT runner owns manipulation; observed
admission and physical stops remain active. Initialized opening cannot establish
full task completion: this mode always sets full-task `passed` false, and reports
only a separate intermediate-phase outcome. See [[act-a1-first-completion]] for
retained tests, native applicability and the fixed normal-start completion panel.


Loaded diagnostic pre-roll reproduces the maintained source controller before ACT
handoff, recording its commands as initialization. The six-step loaded-start check
uses the existing physical force window and current-gap gate, retaining native
samples; it does not require a single arbitrarily chosen tick to have positive
force. Empty startup failures remain incomplete recordings with explicit outcomes.
The experiment page preserves original harness failures and revised native cells.


Loaded startup acquires the original number of genuine geometry views during its
pre-roll, with fresh tracking afterward. This live schedule is separate from the
original workspace-recording capture times and does not change calibrated bounds.
Observed selection then permits the normal load estimator/contact window to warm
from actual native feedback before policy handoff. Ordinary full-start inspection
keeps the common schedule.
