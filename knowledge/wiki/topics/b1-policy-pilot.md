# Common B1 Policy Pilot Foundation

This foundation starts at `9b94978` on `PatrizioAcquadro/b1-policy-pilot`.
It enables independent policy and geometry development. It is explicitly
unqualified: no trained encoder/policy, matched dataset, physical interaction or
perception release is delivered. SAM3 and Point2Pose are unchanged.

## Ownership and Exchange

| Owner | Maintained surface | Exchange |
|---|---|---|
| Policy branch | Encoder, observation assembly, data, ACT/Diffusion wiring, runner, common control/IO | `PilotBinding`, `VisualEncoding`, `PolicyObservation`, `Command` |
| Companion geometry branch | Providers and A3/A4-specific geometry integration | Existing `DoorEstimate`, `OperationalState`/`LocalMaterialState`, `FieldSupport`, `ActionAdmission` |

The geometry branch should consume these interfaces before changing shared policy,
dataset or runner code. Action schema, tensor widths, joint order and coordinate
conventions remain unchanged. Any new admission route for A1/A2 without a hinge
must be agreed at this boundary; the foundation does not add that route.

## Four Independent Questions

1. **Can the policy observe?** `require_features(observation, binding)` checks the
   numerical feature width and the pilot's visual support. The vector remains
   `rgbd_static`, `rgbd_recent`, nine arm/neck positions, nine velocities.
2. **What geometry is observed?** `observation.estimate` retains the existing
   geometry, support bounds, acquisition/availability/support times and episode
   generation. `geometry_available` applies existing geometry checks independently
   of features. Missing hinge/angle/contact/dimensions stay absent. Bounds of
   `None` stay unmeasured. `valid` retains its existing combined meaning.
3. **What does a representation require?** `ActionAdapter.interpret` converts A1
   joint deltas and A2 world-frame pose deltas without door geometry. A3 conversion
   requires a supplied hinge frame. A4 still requires segment state, hinge,
   coherent panel/angle and its predicted local target. Conversion is not admission.
4. **May this command execute?** `ActionAdapter.command`, operational admission,
   `PurdueIO` and the stop monitor retain current geometry, contact, freshness,
   clearance, force/load and stop requirements. Feedback remains a separate
   `RobotFeedback` channel. Missing torque and unverified load control still stop
   loaded interaction; neither RGB-D nor actuator commands replace measured feedback.

## Encoder and Provider Protocols

`B1Observer(provider_or_none, pilot_binding, encoder=encoder)` requires an encoder
with the same binding, `reset()` and
`update(sensor, *, generation) -> VisualEncoding | None`.
The encoder receives only acquisition time/frame, left RGB, aligned metric depth,
boolean valid-depth mask and arm/neck proprioception. Camera extrinsics/intrinsics
remain available to the geometry provider. Teacher/evaluator labels and feedback
enter neither path.

`VisualEncoding(static, recent)` contains two `FeatureBlock` values. Each block
carries its vector, `acquired_s`, `available_s`, source `frame` and `generation`.
Only completed, causal results from this episode are usable. Returning `None`
retains the previous encoding with its original timestamps. Recent features expire
at `max_gap_s`; static inspection features persist within the episode. Current
proprioception is appended at each acquisition. Encoder recipe identity and block
widths belong to the pilot binding; an actual encoder remains future work.

The optional geometry provider keeps `binding`, `generation`, `reset()` and
`update(sensor) -> DoorEstimate`. Its `encoding` is unused by the pilot. A result
marked `between_inference_ticks` retains the previous geometry snapshot without
advancing any support timestamp. Generation changes invalidate cached features
and geometry. A missing/unavailable geometry provider does not prevent policy
feature assembly. The qualified-release observer path remains available unchanged.

## Timing, Runner and Stop

Physics remains **120 Hz**, control **60 Hz**. `PurdueIO` rejects a different pilot
cadence. Acquisition supplies distinct synchronized camera frames; the observer
rejects repeated or backward frames/times. Encoder inference, policy chunk
inference and geometry publication are separate events: none renews another
channel's support. This foundation does not introduce a new scheduler or promise
60 Hz model inference. Pilot data retain one observation per 60 Hz control tick.

`B1Runner(..., pilot_inference_only=True)` explicitly requires a pilot binding.
After the common inspection/hold prefix, it checks features, queries and interprets
one prediction, then stops with `unqualified_pilot_inference_only`, zero executed
policy commands and no task-success claim. Inspection safety still applies.

For future operational dispatch, the optional
`admission_source(observation, tool, action, remaining_ticks) -> ActionAdmission`
callback supplies evidence for exactly the predicted command. No callback means
no operational command. The adapter validates source, action, initial tool pose,
trajectory schedule and current support; A4 retains its admission through the
segment. IO performs its independent checks before physics. Refusal clears queued
commands and stops without adding release motion or an extra control tick.

## Partial Data and Release Compatibility

`PilotBinding` uses `b1.policy-pilot.v1`, requires an explicit encoder identity and
`operational-v1` geometry checks, and keeps `offline_passed`, `dynamic_passed` and
`frozen` false. Pilot policy contracts use `b1.policy-contract.pilot.v1`.
`PerceptionBinding` continues to require the qualified/frozen legacy release v2;
a pilot cannot be loaded as that release or silently substituted in its checkpoints.

With a pilot binding, `compile_episode`, `export_dataset` and `B1Dataset` use
`b1.policy-dataset.pilot.v1`, purpose `b1_partial_policy_pilot`, and A1/A2 only.
No hinge, A4 stages or fabricated A3/A4 arrays are needed. N commands, N+1 causal
observations, robot FK, physical identity, train/development membership and 60 Hz
control remain required. Raw pilot recordings must declare that purpose explicitly;
engineering recordings are still rejected. Existing ACT/Diffusion normalization,
batching and checkpoint APIs accept this explicit pilot contract.

The final matched `b1.policy-dataset.v1` still requires A1–A4, complete A4 stages,
matched physical identity/boundaries and the original release binding. Producing
A3/A4 labels later requires supported geometry; changing a pilot's schema cannot
promote partial data into the matched dataset.

## Deferred Decisions

Choose the actual encoder recipe/widths and bounded inference schedule on the
policy branch. The companion branch must supply observed A3/A4 geometry and
trajectory evidence with honest support times/bounds. A hinge-free A1/A2 physical
admission route, measured feedback/load/stop validation, matched data collection
and training remain separate future work. This milestone stops before those steps.
