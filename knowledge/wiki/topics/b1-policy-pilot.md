# Common B1 Policy Pilot Foundation

This foundation starts at `9b94978` on `PatrizioAcquadro/b1-policy-pilot`.
It enables independent policy and geometry development. The separate
[[experiments/b1-act-diffusion-a1a2-pilot|bounded A1/A2 pilot]] now supplies three
physical teacher episodes and four CUDA-trained prototype checkpoints. The new
[[../experiments/b1-shared-learned-pilot|shared learned pilot]] adds the explicit
`rgbd-conv-v1` input/model recipe below; its pilot remains incomplete. These remain
unqualified; no final matched A1-A4 dataset, policy contact execution or perception
release is delivered. Native SAM3/Point2Pose models and selected controls remain unchanged.

## Ownership and Exchange

| Owner | Maintained surface | Exchange |
|---|---|---|
| Policy branch | Encoder, observation assembly, data, ACT/Diffusion wiring, runner, common control/IO | `PilotBinding`, `VisualEncoding`, `PolicyObservation`, `Command` |
| Companion geometry branch | Providers and A3/A4-specific geometry integration | Existing `DoorEstimate`, `OperationalState`/`LocalMaterialState`, `FieldSupport`, `ActionAdmission` |

The geometry branch should consume these interfaces before changing shared policy,
dataset or runner code. Action schema, tensor widths, joint order and coordinate
conventions remain unchanged. `LocalActionSupport` now supplies a bounded A1/A2
admission route without a hinge. This bounded pilot must produce its own observed sweep, response and uncertainty
evidence; companion A3/A4 qualification is deferred.

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
upper load and explicit native intervals of conservative lower normal load.
Contact confirmation uses a configured causal mean over completely covered
intervals; duplicates add no evidence and missing intervals remain unavailable.
This does not smooth the instantaneous force safety limit. Optional nonnegative
velocity/acceleration residual terms use whole-arm motion norms to cover coupling;
they require independent calibration and leave the legacy constant-bound model
unchanged when absent. Normal-direction uncertainty is
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
visible-surface/model disagreement, force hysteresis, covered force window,
confirmation and bounded acquisition. Total contact uncertainty must fit inside
the independently admitted finite-space margins; inflating uncertainty cannot
turn a remote surface into admissible contact. Patch, robot and rotational errors stay separate; a fit residual alone
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
tracking, finite current identity/pose bounds and two supported finger footprints. The producer must bound the actual continuous
sweep plus stopping envelope, relevant collision space, material response,
robot tracking error, relative motion and stop travel. Unknowns and None bounds
refuse admission. Unloaded proposals additionally prove contact separation.
Freshness, episode generation, selection changes and response-envelope exits
invalidate a receipt. Revalidation uses the original finite-error formula and
remaining control interval: any increased uncertainty and lost local clearance
spend the original remaining margin. Exhausted margins and missing bounds refuse;
a stronger rigid-to-moving identity witness for the same selected patch is allowed. This is action admission from local
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


The selected material-zone cover stays distinct from the actual tool-relative
query, so parked approach does not imply contact. Actual action admission still
requires its own finite footprints and relevant swept-space margins. Initial
health checking precedes receipt construction; actual dispatch remains gated.
Measured arm-speed checks include both native substeps, and robot stopping
precedes perception reset/worker cleanup.


The shared inspection parks the arm at its configured joint anchor while moving
only the neck. Cartesian parking had introduced nullspace self-motion and changed
the subsequent physical condition. The teacher scan and `PurdueIO.hold_step` now
use the same joint-anchor convention; old recordings remain historical and new
matched references require physical verification.


The physical pilot CLI accepts explicitly declared train/development rollout
conditions through `--condition`, using `evaluation_panel` when present. Sealed
identities remain excluded by both protocol selection and corpus split checks.
Preparation can read each declared episode's `directory`, allowing two physical
conditions per identity without overwriting recordings or duplicating identity
selection. Defaults retain the historical one-condition layout.


A declared `lighting_scale` multiplies the maintained scene's dome-light intensity
before reset, and is retained in runtime and raw metadata. It changes rendered
observations while preserving rigid-body geometry/dynamics. The default is 1.0;
this hook does not qualify a visual range or camera noise model. A future pilot
panel must freeze its common scales using train/development evidence before
learned evaluation, and preserve every outcome.


Pilot observations may use the maintained geometry provider's independent recipe
with the separately bound visual encoder. Both must declare the same inspection
and the provider's freshness limit cannot exceed the policy contract. Episode
generation remains mandatory; release-mode binding identity stays strict. Purdue
safety uses the IO's operational profile, rather than a prototype recipe's absent
profile default. This interface compatibility neither resolves material identity
nor supplies a load calibration, sweep receipt or motion permission.


The session provider now exposes an explicit synchronous-simulation mode with a
measured simulation-clock callback and a declared inference period no greater than
the existing freshness limit. It waits for actual compute completion while the
simulation step is blocked, stamps tracking/publication with that unchanged clock,
and retains wall compute/publication costs separately. A clock that advances during
the blocked call refuses; asynchronous and recorded-latency replay defaults remain
unchanged. This supports simulation slower than wall time without backdating an
asynchronous result or claiming hardware/150 ms wall-time readiness. The frontend
and selected model settings are unchanged; exact chosen-cadence physical quality
and local admission still require validation before a learned command can run.

The session factory also forwards its supplied robot/camera calibration into local
tracking, enabling actual-tool footprint queries when FK/geometry are supplied.
Calibration error and relative-motion bounds remain explicit separate inputs;
forwarding the calibration does not invent either bound.


Learned image views contain 115,218 scalars per tick. The B1 loader now maps
contiguous HDF5 observation storage read-only and exposes ordered column/matrix
views, so sampling does not duplicate every episode in RAM. Normalization merges
finite train-only row blocks instead of concatenating all episodes; values, counts,
identity checks and fixed encoder units remain unchanged. Chunked legacy files
retain their existing read path. No cache copy or new dataset format is introduced.
This removes a demonstrated allocation problem in the code path; full real-data
preparation/training memory and throughput still need measurement.


The explicit pilot geometry recipe selects a tracker seed from observed candidates
when the semantic door mask contains several planes: require a unique maximum
visible surface area and an exclusive interior pixel. Ties and fully overlapping
alternatives remain unresolved. The choice is latched through the existing explicit
selection API, uses no prepared geometry or evaluator annotation, and generates
no robot motion. The maintained default recipe is unchanged. This seed choice
does not certify rigidity, intended leaf identity, local bounds or contact; native
tracking and independent physical validation remain necessary.

For a real short training run, `overfit_updates: 0` skips the already completed
infrastructure minibatch probe. It records zero probe updates and retains the
same four independent training paths and checkpoint round trips. Positive values
retain the previous probe. Count all optimizer updates against the common pilot
ceiling, including infrastructure updates; the remaining allocation permits
12,000 initial real updates and at most 5,836 additional updates in total.

Local tracking publishes initial material geometry and dynamic frame-origin
uncertainty separately. The complete fingertip budget remains in finite-cover
queries; it is not republished as frame translation and charged a second time
with the same rotations during admission. The contact monitor explicitly includes
both geometry and dynamic pose terms. Missing calibration or motion bounds remain
missing; this correction changes neither operational field limits nor physical
evaluator tolerances and does not establish qualification.

Purdue IO and the observer share the same pilot provider-binding check: independent
visual and geometry recipes must use identical inspection and compatible causal
age limits. The encoder/checkpoint binding stays exact. This avoids rejecting
the explicitly separate geometry recipe at execution after accepting it during
observation assembly; release bindings remain exact.

The runtime attaches calibrated finite finger covers to each observed native
reference at acquisition. It chooses a measured material point maximizing the
minimum boundary clearance of both covers over a fixed deterministic sample.
The native masks and object tracking are unchanged; this changes material
coordinates, not a policy goal or robot trajectory. Missing finite support and
ambiguous ownership still refuse. Physical extrema can be lines and remain
separate from the existing 3 mm distal-band projection used for support checks.

A1/A2 admission requires finite measured local bounds and spends their full
position/rotation effects against the actual finite-footprint, continuous sweep,
response and stopping margins. It does not additionally apply the global 10 mm
A3/A4 frame-qualification cap to that marginal proof. Large uncertainty reduces
the remaining margin and can still refuse; missing bounds always refuse. The
physical 10 mm tracking score, 0.1 mm contact gap, force, collision, hold and release
criteria are unchanged. The contact monitor spends only its additional
visible/collision-model discrepancy against the receipt's remaining margin, since
that receipt already charges geometry, pose and robot errors. It still reports
the complete uncertainty in the fingertip separation intervals.

For bounded A1/A2 bootstrap, local material identity can be witnessed before
opening: the explicitly selected semantic region must support both finite finger
covers in at least three calibrated views whose camera origins are separated
beyond twice the supplied geometry/calibration error. Witness frames, timestamps,
origins and clearances are retained. Current native registration and measured
root geometry must remain consistent before publishing `observed_rigid_material`.
Missing calibration, repeated stationary views, semantic mismatch and insufficient
finite support cannot establish this identity. This witnesses the selected local
material, not a hinge, movable-leaf ownership for the whole mask or A3/A4 operational
initialization. Independent intended-door/negative-case validation remains required
before a pilot recipe is qualified.

`configs/b1_learned_pilot.json` prospectively declares the real learned-encoder
pilot: at most six fresh physical episodes, four cells with 3,000 updates each,
and all 24 learned outcomes on two doors × nominal/dim/bright simulated dome
lighting × four cells. Train doors each have nominal and one cross-selected
lighting condition; development dim/bright episodes must qualify the moderate
0.9–1.1 range before learned evaluation. The unobserved train-door lighting
condition stays in the panel. No dynamics or placement variation is claimed.
Invalid episodes remain evidence and never become imitation targets automatically.
The first fresh nominal episode and the independently preserved valid diagnostic
form the prospective matched-condition expert pair under unchanged pair criteria;
historical denominators/scores remain untouched. The current global resource
ceilings still apply; this protocol is not permission to exceed them.

Physical pilot entry points write a case-start marker only after creating the
actual Isaac environment. Launcher/preflight errors remain separate from physical
cases; all created collection/replay/rollout environments remain accountable,
including later refusals or failures.

The physical entry point now latches the controller and continues physics for
two seconds after collection/replay/rollout, preserving native speed, pose and
contact diagnostics separately from imitation commands. A collection's final
recording outcome includes this stop result; a failed tail cannot remain a valid
imitation episode merely because the teacher's earlier release passed. Interrupted
starts retain a separate stop prefix.

`scripts/pilot_workspace_survey.py` acquires separate simulated pre-task calibration
with the same head ZED. Two common fixed observer placements view the nominal
robot work volume before manipulation. The accepted teacher/policy placement and
initial condition are unchanged; the robot base stays fixed within each survey.
These are observations, not favorable task starts, extra policy inputs, additional
sensors, TSDF work or a hardware-readiness claim. Retained RGB-D/calibration and
clock provenance must establish usable common free space; no map, current
observation or admission certificate is inferred from the camera visibility model.
Unknown support and dynamic-door exclusions remain necessary checks.

Local patch publication now keeps measured finite material clearances separate
from the uncertainty-eroded slip diagnostic. The geometric query still preserves
mask holes, finite finger covers, image/pixel support and its local depth tolerance.
Pose, calibration and temporal errors remain explicit in `FieldSupport` and are
charged by the action proof against its actual footprints/sweep. A negative slip
margin does not erase the measured material model or itself admit a command.
Missing measured support, missing error bounds and insufficient action margins
still refuse. Current pixel visibility remains a separate occlusion diagnostic.

The saved valid teacher cycles contain a brief push-to-hold unloading interval.
The first hold endpoint reused the just-completed push reference, pausing it for
one control tick. The common handoff now carries the final commanded angular
increment into the first hold endpoint, capped by the existing lead limit.
Normal force targets, compression update, IK/controller and physical scoring are
unchanged. Numerical endpoint continuity passes; physical verification of this
specific handoff correction remains pending. Earlier scores remain original.

`LoadCalibration` can optionally supply a three-coefficient empirical error bound
for the robot tool-normal wrench projection, with an explicit maximum total-force
upper bound for that weak-load domain. Native measured joints and calibrated FK
supply the tool normal; its difference from the observed material normal carries
the existing full wrench-error radius. The original total force upper bound,
strong-load fallback, joint/speed/acceleration envelope and missing-data stops are
unchanged. A scalar recipe must be independently validated before use; the optional
fields do not change model or stop qualification flags. The current candidate
remains unqualified and no live contact execution is enabled by this interface.

Isaac-free offline preparation accepts `--urdf` for a relocated robot asset on
Gilbreth. Its bytes must match the existing recorded `RobotAssetRef.sha256` before
FK or preprocessing; raw recordings/runtime metadata remain unchanged. The actual
preparation path is retained in the label audit. `--reference-raw` avoids another
large raw copy. Relocation and CLI checks pass; real remote preparation/training
remain pending.

`CommonWorkspace` reads only observed depth, depth validity, intrinsics, calibrated
camera poses and their original capture clocks from complete pre-task recordings.
It checks recorded camera/FK agreement and propagates the supplied camera bounds
into covering-ball queries. Every supplied calibration condition must certify
free space; one unknown/occupied condition refuses. This is static workcell evidence,
not a new dynamic timestamp, current target identity, dynamic-door exclusion,
self-collision proof, stopping bound or command receipt. No asset identity or
privileged annotation selects geometry at runtime. The three common front surveys
are retained; coarse single-sphere body coverage still has unknown shoulder regions
and must be refined before it can support an actual action.

The handoff correction independently passes the full right teacher cycle: all
180 hold ticks satisfy the unchanged physical criteria, and the minimum native
0.1 s mean is 0.0822 N. Fourteen isolated zero-force substeps remain reported;
the maximum consecutive zero interval is 8.33 ms, replacing the earlier 83 ms
handoff unloading. No requirement of positive force at every native substep is
invented. Opening is 51.1111 degrees; the frozen 51.2603-degree nominal reference
remains more demanding and unchanged.

The two pre-correction fresh nominal recordings keep their original valid physical
scores, raw data and paired-reference role. Their known handoff command pause is
excluded from imitation through explicit protocol eligibility/reasons. The four
remaining declared conditions use the same corrected teacher: right dim/left bright
for training and both development lights for validation. The raw collection ceiling
stays six, and the fixed 24-outcome denominator is unchanged. This is a partial
pilot data selection, not the final matched A1–A4 dataset.

Static workspace queries may cache a certified free ball. Reuse spends center
motion, radius growth and the increased camera-calibration lever against the
original radial margin, without changing the observation clocks or certifying
new dynamic geometry. Input ray banks are copied; unknown volumes never enter
the cache. A numerical adaptive cover of the first right command certifies all
14 moving convex shapes with 108 balls across the three observed conditions.
Initialization takes 0.70 s; 200 small artificial cache displacements take a
median 0.194 ms/query. This is CPU geometry feasibility/timing, not a physical
trajectory, whole-task clearance or hardware real-time qualification.

`RobotGeometry` supplies calibrated full-joint URDF transforms, conservative convex
collision cells and visual/collision-volume exclusion of robot depth returns.
Missing joints or sweep bounds refuse. The cells cover shape interiors and add
serial-chain joint-travel bounds; they do not command motion or import door assets.
Pixel queries can restrict their domain to the region the maintained geometry fit
may inspect, including enclosed mask omissions. Removed pixels remain unobserved.
Numerical tests compare the pixel prefilter with complete volume membership and
cover intermediate rotating volumes. Actual full Purdue meshes load successfully.

The observed Purdue monitor now checks all four WSG32 jaw positions from complete
robot proprioception, preserving the existing 1 mm closed-gripper criterion.
Missing jaw channels, nonfinite states and opening stop before load/admission.
The existing measured native arm-speed guard still runs on both substeps.

The maintained mask/depth surface extractor exposes an optional dominant-plane
local query. It returns the largest connected measured part of the first robust
plane and preserves invalid depth and true holes. Existing complete extraction
remains the default. This supports local A1/A2 contact-geometry investigation with
the same SAM3/RANSAC components; it neither supplies identity/ownership nor labels
an occluded interface directly observed. Correspondence-based rigid pose bounds,
current surface normal and finite material support remain distinct evidence.

Operational Purdue dispatch now compares the actual post-step tool FK with the
admitted endpoint and its declared robot position/rotation bounds, after the
existing pre-dispatch controller preview. An excess or invalid measured FK
latches the physical stop immediately; the runner still counts the command that
actually executed and reports its fault. Subsequent commands remain refused
until reset. This verifies measured response rather than treating a commanded
limit as compliance. Numerical fault cases pass; loaded policy qualification and
independent physical scoring remain required.

The pilot RTX camera publishes SDK intrinsics with half-integer sample centers.
Geometry now converts those intrinsics once at session ingress to integer pixel
indices; new calibration metadata declares `[0.5, 0.5]`, and the explicit pilot
recipe provides the convention for retained legacy recordings. Conflicting
metadata refuses. Raw SDK intrinsics, RGB-D, clocks and learned image preprocessing
remain unchanged. Static workspace maps are rebuilt from normalized rays; prior
native 3D tracking outputs are not relabeled as corrected.

Independent saved-data checks explain the systematic bias: four visible-floor
ROIs fall from up to 6.160 mm error to below 1.57 micrometers. Five other fixed
ROIs include occluders and do not establish floor accuracy. On the fixed sparse
train contact-plane samples, normalized geometry gives maximum plane errors
0.06844/0.00199 mm and normal errors 0.000268/0.000266 rad (right/left).
These are diagnostic numerical results, not a complete sensor uncertainty bound,
hardware qualification or observed-contact admission.

Pilot preparation records explicit exclusions and whether their unused raw files
are present on the current host. It does not require transferring excluded nominal
labels to Gilbreth and does not claim that remote preparation verified workstation
retention. Eligible train/development inputs remain mandatory.

An opt-in local surface constraint now combines the existing SAM3 depth plane,
a conditioned observed vertical edge and every measured native registration
inlier. It recomputes the translation residual and propagates finite plane/edge
baseline errors from an explicit metric point bound. Missing/ambiguous/degenerate
geometry or failed material consistency remains unavailable. Native poses and
A3/A4 diagnostics stay unchanged; no operational qualification is inferred.

The local contact contract can carry a separately supported normal-direction
error. Loaded alignment uses that direction bound while full pose uncertainty
continues to erode finite footprints and sweeps. Rotation about a surface normal
is not uncertainty in the normal direction. Without separate evidence the prior
full rotation bound remains the fallback. Geometry ingress can exclude calibrated
robot returns from geometric support while preserving raw policy RGB-D/validity;
excluded pixels remain unobserved.

The explicit pilot protocol can select the unique initialized observed material
zone before native support consumption, with a declared policy/diagnostic source.
This binds the intended target without qualifying ownership, filling missing
support or changing the selected semantic identity. Episode reset removes it.

The training runner also supports a host containing train records only. It retains
train-only normalization/optimizer membership and reports development metrics as
not evaluated; it never substitutes training examples for missing development.
The four fixed final checkpoints still require separate development and physical
checks. This permits training on the already transferred train data while the
separately requested development transfer remains blocked by automatic review.

Session construction can explicitly defer model startup to the first capture.
Unused constructor/runner resets then avoid repeatedly loading CUDA workers; a
used episode reset still destroys/recreates the full temporal/model state. Startup
metadata is written only after actual worker creation. This changes initialization
cost, not clocks, observations, readiness or qualification.

Calibrated robot-return exclusion runs at the declared geometric capture cadence,
before submission to both native tracking and geometry. Intermediate control ticks
consume bounded existing material support and cannot publish unfiltered robot
returns as fresh direct-depth evidence. Raw policy inputs remain unchanged.

The constrained material anchor's normal coordinate is measured from the current
observed plane; native pairs retain the tangential translation. The anchor must
belong to the initial observed plane, and the complete pair residual is recomputed
after the constraint. Inconsistent native translation therefore increases the
published error rather than disappearing. This estimates material geometry and
never changes a submitted policy command.

The local contact patch can now publish independently propagated plane-normal
position and direction errors at its material anchor. They include the explicit
metric point/model terms, observed conditioning, angular lever, robot/camera
calibration and temporal growth; a fit residual alone never supplies them. Full
material-pose uncertainty still protects finite footprints and sweeps. The contact
monitor uses these current normal bounds, when present, for fingertip separation
intervals and separately calibrated actual-tool FK errors rather than commanded
tracking error. Missing normal evidence keeps the conservative full-pose fallback.

Causal three-dimensional consistency retains the complete pose bound. Acquisition
travel now accumulates measured world tool path length, so material-coordinate
jitter cannot masquerade as robot travel and an oscillation cannot evade the
existing travel ceiling by returning to its anchor. Identity, sweep receipts,
verified load, confirmation/loss, stop behavior and independent physical thresholds
remain unchanged. Numerical contact/no-contact/loss/ambiguity cases pass; complete
physical false-admission/miss/delay/coverage measurements remain pending.

A saved-input wiring audit identifies a separate initialization defect in the old
ad-hoc component replay: it passed rank-one distal extrema to a finite-footprint
query. The existing maintained `PushGeometry.contact_covers()` supplies rank-two
covers; without changing masks, zones or observations, the two saved train seeds
then each provide a valid finite zone and 13 static material witnesses. This is
an input-geometry correction, not qualified ownership or a new native result.
Corrected-convention component replay is prepared with finite covers, full observed
robot-return exclusion and explicit diagnostic selection; it awaits local budget.

Current-capture normal bounds need no assumed future speed. Without a measured
speed model they are available at the exact source timestamp and become unavailable
between captures; no zero-motion bound is fabricated. The preserved whole-finite
model discrepancy reaches 1.973 mm, so the existing 3 mm reserve is retained.
A temporary unqualified 1 mm reserve proposal is preserved as incomplete and was
never used for physical admission. Complete calibration/contact/sweep validation
remains required.


## Continuous robot cell sweeps

`RobotGeometry.world_ball()` covers each collision cell with its intrinsic sphere
and joint travel, including the measured stopping allowance supplied by the
caller. Revolute displacement now uses the cell center distance to each actual
joint axis plus its cell radius. Proximal-to-distal telescoping preserves these
lever bounds under simultaneous serial-chain motion; prismatic travel is added
directly. This avoids charging the full downstream chain length to every small
cell. Coupled bent-chain and prismatic numerical coverage passes. It supplies a
continuous geometric cover, not observed free space, self-clearance, contact or
a stopping qualification. Unsupported observed queries still refuse.
