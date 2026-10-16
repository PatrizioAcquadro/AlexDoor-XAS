# Common B1 Policy Pilot Foundation

This foundation starts at `9b94978` on `PatrizioAcquadro/b1-policy-pilot`.
It enables independent policy and geometry development. The separate
[[experiments/b1-act-diffusion-a1a2-pilot|bounded A1/A2 pilot]] now supplies three
physical teacher episodes and four CUDA-trained prototype checkpoints. The new
[[../experiments/b1-shared-learned-pilot|shared learned pilot]] adds the explicit
`rgbd-conv-v1` input/model recipe below; its pilot remains incomplete. These remain
unqualified; no final matched A1-A4 dataset, policy contact execution or perception
release is delivered. SAM3 and Point2Pose are unchanged.

## Ownership and Exchange

| Owner | Maintained surface | Exchange |
|---|---|---|
| Policy branch | Encoder, observation assembly, data, ACT/Diffusion wiring, runner, common control/IO | `PilotBinding`, `VisualEncoding`, `PolicyObservation`, `Command` |
| Companion geometry branch | Providers and A3/A4-specific geometry integration | Existing `DoorEstimate`, `OperationalState`/`LocalMaterialState`, `FieldSupport`, `ActionAdmission` |

The geometry branch should consume these interfaces before changing shared policy,
dataset or runner code. Action schema, tensor widths, joint order and coordinate
conventions remain unchanged. `LocalActionSupport` now supplies a bounded A1/A2
admission route without a hinge. The geometry branch owns production of its
observed swept-space, material-response and uncertainty evidence.

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
widths belong to the pilot binding. `RGBDEncoder` implements the recipe below.

The optional geometry provider keeps `binding`, `generation`, `reset()` and
`update(sensor) -> DoorEstimate`. Its `encoding` is unused by the pilot. A result
marked `between_inference_ticks` retains the previous geometry snapshot without
advancing any support timestamp. Generation changes invalidate cached features
and geometry. A missing/unavailable geometry provider does not prevent policy
feature assembly. Fresh local material patches export the canonical public
`tracking` state; the tracker retains its internal `tracked` lifecycle. Stale
patches export `unavailable`. This interface correction does not fill missing
geometry/FK bounds, resolve ownership or produce sweep/response evidence, so
admission still refuses unsupported patches. The qualified-release observer path
remains available unchanged.

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

For operational dispatch, the optional
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

The companion branch must supply observed geometry and continuous trajectory
support with honest times/bounds, including the local A1/A2 route below. Dynamic
load calibration and contact/moving-stop qualification remain open. The bounded
partial offline pilot does not start final matched-data production, the final
training matrix or a contact rollout campaign.


## Executable observation and data recipe

`policies/encoding.py` supplies `pilot_binding(inspection)` and `RGBDEncoder`.
The fixed encoder has no downloaded or trainable weights. Each image is spatially
pooled to 8×8 cells: RGB / 255, mean valid metric depth / far range, and valid-depth
coverage. Invalid depth contributes zero and never contaminates the depth average.
The depth interval, grid, history and period are checkpoint-bound. Historical
pilot recordings actually bind approximately 0.01–100,000 m. The explicit opt-in
`rgbd-grid-fixed-units-v2` recipe instead uses depth/5 m with fixed observation
units (no empirical observation z-score); sensor validity keeps its recorded
interval. Training and inference share the existing +/-10 clip, while action
normalization remains train-only. Legacy v1/default checkpoints are unchanged.
See [[experiments/b1-pilot-preprocessing|the isolated preprocessing comparison]]
for the saved-data diagnosis, paired protocol and current evidence.
No crop, augmentation, DINOv3, calibration or evaluator label enters the encoder.
ACT/Diffusion's existing observation projection and all policy weights are trainable.
This deliberately coarse baseline establishes infrastructure, not visual sufficiency.

Static support freezes the last sampled inspection view. Recent support concatenates
three frames at 10 Hz, oldest first, spanning 0.2 s, with reset-only left padding.
Between encoder ticks, support times remain unchanged while current nine joint
positions and velocities are appended. The vector has 1,298 values (320 + 960 + 18).
The same stateful observer is replayed from reset offline, including inspection.
Synchronous simulation blocks during inference; `available_s` uses that simulation
clock. This is not an asynchronous/hardware latency guarantee.

Initial primitive schedule: 120 Hz physics, 60 Hz control, 12-action (0.2 s) chunks,
execute six ticks (0.1 s), then replan. Both families accept `n_action_steps=6`;
ACT ensembling is disabled. A4 still replans at segment boundaries. Geometry keeps
its independent provider/publication cadence and original support timestamps.

`PilotRecorder` requires explicit `b1_partial_policy_pilot` purpose. It preserves
RGB-D, calibration, timestamps, canonical proprioception, post-controller applied
joint/neck targets, submitted world goals, applied effort commands, phases,
terminal observations and the
separate optional measured-feedback stream. Engineering recordings remain rejected.
Pilot validity requires a passed physical diagnostic and explicit termination,
without demanding an evaluator door angle. Legacy engineering validation is unchanged.
`prepare_recording` checks calibration and uses live encoding/FK; `export_dataset`
retains the complete raw groups alongside partial A1/A2 tensors, enabling later
supported A3/A4 derivation. No partial schema promotes itself to a matched dataset.
Train statistics reject duplicate/development membership even at the low-level API.
Raw preparation also rejects a split that differs from the frozen assignment;
export refuses raw provenance from a different physical episode or identity.

The CUDA round-trip check compares offline/live feature arrays exactly, preserves
raw sensor/command rows, tests development outliers against train-only statistics,
and loads/predicts both families for A1/A2. Existing eight-path loss/gradient checks
remain separate; that original numerical smoke performed no optimizer update or
physical demonstration collection. The subsequent bounded pilot is reported separately.

`benchmark_policy_pilot.py` measures 600×960 CUDA sensor arrays through CPU support
validation, RGB-D encoding, normalization, family inference and output transfer.
On RTX 4090 (30 warm samples/path, random weights, default width 128, chunk 12,
Diffusion DDIM 10 steps), A1/A2 total p95 was ACT 3.21/2.43 ms and Diffusion
14.90/15.08 ms. Across all eight numerical paths the largest maximum was 15.76 ms.
These measurements exclude rendering, physics, geometry, FK and admission; they
cannot establish the full rollout rate. The historical report path was
`outputs/generated/b1-policy-pilot/model-cost.json`, removed by the later external
worktree archive described below.

Observation/data infrastructure is ready for a bounded offline training invocation
once explicitly recorded pilot demonstrations and identity splits exist. No real
pilot dataset or trained checkpoint is delivered by the numerical smoke.


## Observed control and local admission

`PurdueIO.robot_feedback()` reads `root_view.get_dof_projected_joint_forces()`:
PhysX projects incoming joint spatial forces onto each canonical joint axis.
The declared signal is simulated joint-output torque in N m. Applied effort
commands are retained separately in `commands/effort_target_nm`, in canonical
arm/neck order; they never populate measured feedback. Neither
`joint_effort_target` nor `applied_torque` is read as a measurement. These values
are robot feedback, not contact-force ground truth. Missing backend support is
reported unavailable. Both 120 Hz substeps capture measured torque, robot state,
Jacobian and inverse-dynamics expectation; raw recordings retain this separate
`physics_feedback` stream as well as the 60 Hz feedback channel.

`ResidualLoadModel` subtracts robot inverse dynamics (mass × acceleration,
Coriolis and gravity), applies the independently verified sign, and bounds a
single distal world wrench through J.T. Rank loss, inconsistent redundant-joint
residuals and calibration-envelope exits reject estimation. Joint residual bounds
propagate through the pseudoinverse into a force-norm upper bound and a normal
force lower bound. `PurdueLoadSource` uses both physics samples, preserving peak
upper load and conservative lower contact load. Normal-direction uncertainty is
also deducted. Unmodelled friction, state/model error and acceleration estimation
must be covered by the calibration; one stationary fit does not establish this.

`ObservedPurdueSafety` accepts that source through `PurdueIO(..., load_source=...)`.
The monitor checks feedback identity/freshness/health, calibrated dynamics/stopping,
force upper limits and observed orientation. `ObservedContactMonitor` separately
combines signed separation intervals at the two calibrated distal finger geometries,
qualified local identity/support and submitted swept-space admission, bounded robot
load and causal temporal consistency. It reports indirect contact evidence, seeking,
separation, unavailable support or latched loss. An occluded interface is never
reported as directly measured. Positive load away from the intended material fails.

`ContactMonitorConfig` requires an explicit measured operational recipe, including
visible-surface/model disagreement, force hysteresis, confirmation and bounded
acquisition. Patch, robot and rotational errors stay separate; a fit residual alone
is insufficient. Expected unloading is recognized from the submitted withdrawal,
not generated by the monitor. Unexpected loss stops and requires episode reset.
Explicit push/hold requires confirmed contact; axis-free primitives may acquire
contact only within the declared bounded search. No source/calibration/monitor
means loaded execution stops explicitly. The monitor reads no prepared door,
articulation, contact categories or evaluator forces and cannot correct task motion.

The evaluator's 0.1 mm gap and 0.02 N contact score remain unchanged and do not
specify operational sensor resolution. The former tool-center/gap-upper comparison
was an integration defect, now removed. Software tests are not physical monitor
qualification; see [[../experiments/b1-end-to-end-pilot-resume|resumed pilot]].

`admit_local_action` accepts A1/A2 only. A selected local material needs explicit
policy/diagnostic source, resolved rigid-material ownership (`observed_rigid_material`
for observed static material or the existing `observed_moving_rigid_candidate`),
tracking, current identity/pose support, the existing 1 cm / 5 degree bounds,
and two supported finger footprints. The producer must bound the actual continuous
sweep plus stopping envelope, relevant collision space, material response,
robot tracking error, relative motion and stop travel. Unknowns and None bounds
refuse admission. Unloaded proposals additionally prove contact separation.
Freshness, episode generation, selection changes, uncertainty increases and
response-envelope exits invalidate a receipt. This is action admission from local
evidence, not a qualified full-door estimate; `PolicyObservation.valid` is unchanged.
A3/A4 still require their hinge/angle/segment contracts.

`env.preview_targets` is the same pure bounded A1/IK function used by dispatch.
`PurdueIO.preview(Command)` exposes the world FK endpoint to evidence producers.
Before execution IO independently checks the control interval and that this actual
controller endpoint fits the receipt's robot-error bounds. Receipts cannot describe
a safe endpoint while sending a different bounded joint command. On stop, pending
commands clear and all position targets freeze once at measured joints, with no
extra physics tick or release trajectory. The environment also latches command
refusal, clears velocity targets and retains gravity compensation; repeated stops
do not move the anchor. Zero Gym actions can continue stopped physics. Reset is
required before motion resumes. Common benchmark impedance and effort semantics
are defined in [[purdue-b1-robot-and-contact|the robot/control contract]].

## Bounded physical evidence and readiness

`verify_pilot_control.py --headless --cameras --output ...` runs one synthetic,
obstacle-cleared robot probe: hold, random-weight inference without actuation,
a known 4 N distal perturbation for evaluation, tiny A1/A2 preview checks and
stop. The test input force is never passed to the residual estimator. It does
not collect demonstrations or run a learned contact policy.

On RTX 4090 the unloaded stationary torque differed from robot gravity prediction
by at most 4.7e-5 N m. The separate 4 N probe supported the negative residual sign
and estimated force components [-4.122, 0.036, 0.097] N; the diagnostic upper bound
was 4.512 N. Its 0.02 N m residual assumption remains unverified and is never
marked calibrated. The probe after perturbation retained up to 0.302 N m of
residual during recovery, showing why the stationary fit is insufficient.
A1/A2 previews matched applied targets within float32 roundoff (32 epsilon), and
post-probe frozen-target travel was 1.30 mm over 0.2 s. These are individual
observations, not a worst-case stopping envelope or contact-stop qualification.

The combined probe includes physics, camera rendering/acquisition, calibration FK,
robot tool FK/Jacobian, both substep feedback/dynamics captures, RGB-D observation
assembly and policy normalization/inference/output. With inference every six ticks,
cycle p50/p95 was ACT 44.09/51.75 ms and Diffusion 44.95/58.07 ms. Query maxima
were 112.91/62.02 ms; the ACT outlier is retained. The short timing sample is not a
real-time guarantee. The 60 Hz controller and 10 Hz policy are simulation-clock
cadences; this camera-enabled configuration runs slower than wall-clock real time.
No qualified geometry/sweep producer or admitted contact cycle was available;
its additional cost is unmeasured. It must be included before any rollout-rate claim.
Those historical probe paths were under `outputs/generated/b1-policy-pilot/`
(`control-substeps` for the final probe). The app later archived their owning
worktree and removed the ignored payloads. Current pilot data and the recovered
interrupted recording are under `act-a1a2-pilot/`; see the experiment's
infrastructure-interruption record. The historical probe values are not new measurements.

| Readiness | State | Remaining prerequisite |
|---|---|---|
| Offline training infrastructure | Exercised on real simulated teacher recordings | Keep explicit partial purpose, causal inputs and frozen train/development identities |
| Offline training on physical pilot data | Demonstrated for ACT/Diffusion × A1/A2: three episodes, fixed-minibatch learning, 96-update short runs and checkpoint round trips | One development identity/seed is not general qualification; ACT A2 development regression is retained |
| Robot-only 6.0D diagnostic | Local 0/4 N hold/stop and command latch verified on CUDA with original calibration; historical failures retained | [[../experiments/b1-loaded-hold-and-stop|Verified regime and remaining uncertainty/fault/contact checks]] |
| A1/A2 rollout with contact | Blocked | Verified dynamic/peak residual calibration and moving/contact stopping envelope; observed local sweep/response evidence and supported contact |
| ACT/Diffusion × A1–A4 reusable numerics | Maintained; eight-path GPU forward/loss/gradient/checkpoint checks pass | A3/A4 supported geometry and complete matched demonstrations |

The original foundation's broader unit run found four pre-existing failures in
`test_perception_cli.py`: the mock argument namespace lacks `selection`.
An isolated `cd5398f` baseline reproduces all four; policy/control changes do not
touch that CLI or fixture. Targeted policy, data, admission and control checks pass.
That foundation run performed no optimizer update or extended collection/training;
the subsequent bounded partial pilot now has separate measured results.

Share the observation/data commit with the geometry branch for the common encoder,
recording/export and timing conventions. Share the control commit for
`LocalActionSupport`, the extended `ActionAdmission`, robot feedback/load channels,
pure controller preview and stop semantics. Providers remain geometry-owned; these
commits neither qualify them nor supply missing bounds on their behalf.


A concrete initial setup uses `ActModelCfg(chunk_size=12)` or
`DiffusionModelCfg(horizon=12)` and `B1Runner(..., n_action_steps=6)`. Twelve ticks
bound the predicted horizon to 0.2 s while six executed ticks limit stale open-loop
control to 0.1 s. The three visual samples cover that horizon's recent past;
current 60 Hz proprioception still changes between image encodings. These are
initial pilot choices, not tuned performance claims. The renderer bottleneck and
retained query outlier must be resolved/measured before wall-clock deadlines matter.

The [[experiments/b1-pilot-approach-release|approach/release follow-up]] preserves
v2 with uniform sampling. One phase-balanced 96-update cell improves release
but regresses other phases; no observer, controller or geometry interface changes.
The [[../experiments/b1-approach-controller-compatibility|current-controller comparison]]
now passes right unloaded approach tracking and two-point local stopping, while
left timed-path p95 remains 28.78 mm / 7.87 deg with active IK limits despite static
support. Native feedback is complete, but old load calibration covers only 108 of
3,384 samples. Validate a feasible approach reference under the frozen controller
before policy changes; old observations/actions are not relabeled, and matched
new-controller/contact demonstrations and required 6.0D/E checks remain pending.

## Learned RGB-D foundation

New pilot work explicitly selects `rgbd-conv-v1` with `--recipe` in both
`pilot_offline.py` and `pilot_physics.py rollout`; both retain the historical grid
default for existing commands. Runtime derives sensor bounds from the current
camera and still rejects a checkpoint contract mismatch. The grid recipes remain
historical compatible paths, with no new comparison. `RGBDEncoder` prepares five
60x96 image channels: RGB/255, mean valid metric depth saturated at 5 m and divided
by 5 m, and valid-depth coverage. Invalid depth contributes zero and the coverage
channel retains its absence; recorded near/far still define sensor validity. Entirely
missing RGB-D is rejected by `B1Observer`; unsupported input is never imputed as
fresh. Static inspection plus three oldest-first frames at 10 Hz use the existing
reset, left-padding and acquisition/publication semantics. Current q/dq remain
nine canonical joints each. Input width is 115,218: four images plus 18 joint values.

The prepared data contain images rather than pretrained/frozen descriptors.
`common/visual.py:RGBDProjection` sits **inside both action models**: shared-across-
frames Conv2d 5->16->32->64 with stride 2, GroupNorm/SiLU, a 3x4 spatial pool,
64 values per image, then four concatenated image features plus proprioception
projected to the existing 128-wide action conditioning. PyTorch initializes all
weights from the declared training seed; the encoder and action model optimize
jointly. No downloaded weights, extra dependencies, annotation input or hidden
per-door lookup is added. The 3x4 spatial pool preserves coarse layout; visual
sufficiency must still be established by genuine learned rollouts.

`make_model` derives `visual_recipe` from the data binding. Both family model
configurations save it and all convolutional weights in the normal checkpoint;
saving/loading rejects a mismatch with the input recipe. `None` keeps the legacy
linear observation projection and historical weights. Fixed observation units
avoid empirical pixel z-scores; action normalization remains train-only. The
network recipe, depth scale, image size, history/timing and robot/action contract
travel with the existing checkpoint binding.

CUDA verification covers offline/live equality, causal/reset/static/missing-depth
behavior, image-sensitive nonzero visual gradients, train-only statistics,
ACT/Diffusion x A1/A2 checkpoint and seeded prediction parity. The combined targeted
suite passes 128 checks. This establishes software paths; it is not training on
new-controller demonstrations or learned physical task success. Sensor-array to
prediction p95 is 2.24/2.31 ms ACT and 14.46/15.11 ms Diffusion A1/A2 on RTX 4090
with random weights and DDIM 10 steps. Camera/rendering, physics, FK, geometry,
admission and training memory are excluded; no wall-clock deadline is qualified.

The recipe uses maintained PyTorch [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
[GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)
and [adaptive pooling](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html).
No architecture search is performed. Data and model cost/limits are in the shared
pilot experiment; learned checkpoints and admitted rollouts remain pending.


New raw pilot recordings additionally retain `robot_state`: actual calibrated tool-link pose and
all robot joint positions/velocities at each RGB-D timestamp. This stream stays
outside model inputs and supports later supervision/physical audits. Its clock
must match observations exactly; old recordings remain readable without it.
See [[../experiments/b1-end-to-end-pilot-resume|the resumed pilot]] for the deferred
A3/A4 execution boundary and outstanding physical validation.


For tracked material, transported finite footprint support and current direct
depth compatibility are separate diagnostics. Self-occlusion can leave the first
available while the interface is unobserved. This supplies a bounded geometric
model only: the contact monitor still needs verified robot load, identity,
uncertainty and independently produced relevant sweep/stop evidence. Missing
original coverage or pose bounds remains unavailable.


`pilot_offline.py prepare --reference-raw` avoids duplicating retained RGB-D in
learned-view exports. It verifies raw episode identity and records the original
path; preserve that separate raw archive for later A3/A4 work. Default exports
remain self-contained and copy all raw groups. Both forms train/load the same
observed feature/action views without reading supervisor annotations.
