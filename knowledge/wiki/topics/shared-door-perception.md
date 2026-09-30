# Shared Door Perception

Current integration contract: `b1.perception.release.v2` binds named artifact
SHA256 identities, explicit static/recent `visual_dims`, `warmup_s`, inspection
and freshness. `B1Observer` accepts a provider with matching `binding`,
`reset()`, `update(sensor)` and `encoding`; the provider owns device conversion
and must verify artifacts before loading. `PerceptionBinding.verify_artifacts`
checks exact bytes. The model-specific loader is retired. No qualified provider
is available; old release v1 is rejected. The estimator discussion below is
historical and will be consolidated during the approved cleanup.

Subphase 6.0 prepares one observed-only stack for every B1 model/representation.
The initial trained estimator is **not qualified**: run-01 fails the development
gates. The [[experiments/b1-perception-run-01|run-01 diagnosis]] separates measured
failures from the subsequent loss correction. The
[[experiments/b1-perception-fit-check-01|two-door fitting check]] passes contact
gates on train but exposes inaccurate articulated components. The subsequent
[[experiments/b1-perception-fit-check-02|complete-state fit check]] passes all
component and contact checks on the same two train doors. The subsequent
[[experiments/b1-perception-run-02|full-corpus run-02]] fails all six development
doors, with worsening geometry and excessive confidence. Development accuracy
is insufficient and closed-loop usability remains unverified.
The [[experiments/b1-perception-input-diagnosis|input diagnosis]] motivated the
opt-in common inspection and camera-mount study, causal static memory, a separate
metric encoder, bounded contact refinement and frozen-geometry confidence fitting.
The implementation and scoped validation are tracked in
[[experiments/b1-perception-corrections|Perception Corrections]]. The refreshed
50-episode campaign and run-03 have since completed. Its last
checkpoint passes complete geometry on all 19 train doors but none of the six
development doors. The [[experiments/b1-perception-model-comparison|model comparison]]
records this generalization gap and public pretrained/geometric component results.
The [[experiments/b1-perception-approved-models|approved DINOv3/SAM 3 trials]]
find no qualified replacement: the backbone swap retains development failure.
Boxed SAM 3 improves partial plane success on the aligned sample but retains
severe outliers; it is a candidate visual aid for the next prototype, which must
explicitly retain metric geometry across views. No estimator is qualified for
dynamic use.
The implementation contract is in
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]].

## Recording boundary

`scripts/collect_perception.py` reads the frozen asset corpus, accepts only train
or development identities, and launches a fresh Isaac process per door/condition.
The default is one worker; `--jobs 2` permits two independent processes on the
verified 24-GiB workstation. Each worker has its own log under `worker-logs/`.
A missing/incomplete artifact is failure even when Isaac exits with code zero.
The legacy mode reuses the frozen scripted expert, camera and fixed neck.
The optional inspection mode retains the arm/contact setup and safety rules. It does not publish new qualification references. Nominal and moderate
Replicator light conditions are constant within each episode; physics is unchanged.
The initial campaign contains two episodes per identity: 38 train and 12 development.
These are perception engineering recordings, not the final matched policy dataset.

`b1.rgbd.v1` HDF5 is independent of numerical `phase2.v2`. With N commands it stores
N+1 observations and annotations in separate groups, with episode settings,
calibration, outcome and collection diagnostics in a dedicated `metadata` group.
This includes reset at episode time zero and the
terminal observation. Missing reset observations are rejected.
Command t acts between observations t and t+1. The command includes the compensated
world tool goal actually passed to `command_pose`, seven arm targets and, in new
recordings, two neck targets;
the placeholder zero action passed to `env.step` is not the expert command.
Writing is incremental and refuses existing files. Image checks reject empty black
frames; mean brightness is diagnostic, not an arbitrary exclusion of dark materials.
Interrupted files remain incomplete; physically invalid episodes remain preserved but are rejected by loaders.

Observed inputs are RGB, left-aligned image-plane depth in meters, its finite/range
validity mask, seven arm and two neck positions/velocities, timestamps, intrinsics
and calibrated camera pose. Camera pose is reconstructed from the URDF neck chain,
fixed mount and fixed robot base calibration. Simulator camera pose is compared
only as a collection diagnostic. Door transforms never initialize inference.
Identity, split, handedness, phase, expert state and geometric targets are metadata
or annotations. The estimator input loader cannot select them as observations.

## Frozen backbone and estimator

The initial backbone is `facebook/dinov2-small` (DINOv2 ViT-S/14), loaded locally
through the workstation's existing Transformers runtime
([official DINOv2 reference](https://huggingface.co/docs/transformers/model_doc/dinov2)). Its parameters have no
gradients and it stays in evaluation mode. RGB uses ImageNet normalization and
224-pixel letterboxing; it never crops away the image edges. Depth/mask use aligned
nearest resampling, adjusted intrinsics and masked XYZ patch features. Cached
RGB features and live inference both use float16; metric XYZ stays float32.
Rendered depth is ideal geometry, not a ZED stereo-error simulation.

`configs/perception_metric.json` is the new CLI default. RGB and metric XYZ
have separate equal-width encoders and normalization before fusion. Metric data
uses a 32×32 grid, excludes depths beyond 3 m before pooling, and transforms
observed points through the calibrated camera pose. Hinge position is an observed
3-D centroid plus a learned residual. Centered geometric features make a common
world translation shift the predicted hinge/contact by the same metric amount.
This structural property does not prove correct semantic localization.

Seven timestamp-selected inspection observations remain available throughout an
episode. Their pooled encoding predicts static hinge, dimensions and local contact;
a GRU over the four recent 10-Hz observations, conditioned on inspection memory,
predicts articulation. Asset identity, phase and truth never enter either encoder.
Train-only normalization and equal-door/phase sampling remain in use. The legacy
`configs/perception.json` and old head remain explicitly loadable for diagnostics.

`ObservedEstimator` clears both memories on reset, missing/insufficient depth,
nonmonotonic input or a sampling gap. Missing inspection samples cannot be filled
from future frames. Warmup, low confidence and unqualified confidence produce
invalid estimates. Consumers must check freshness before commanding motion; there
is no oracle fallback. Signed angle is right-handed about the estimated hinge Z.

## B1 Policy Encoding Interface

The metric estimator optionally returns its static inspection encoding and recent
GRU state alongside the unchanged geometry prediction. `B1Observer` concatenates
these two frozen encodings with the current nine joint positions and velocities,
in the order `rgbd_static`, `rgbd_recent`, `joint_position`, `joint_velocity`.
`DoorEstimate` remains a separate adapter input. Offline preparation and live
rollout use this same builder; annotations and metadata are not selectable inputs.

Between inference ticks only a still-fresh successful encoding/estimate is held;
current proprioception is refreshed. Loss, unqualified/low confidence, invalid
proprioception and reset invalidate the held output. The existing 6.0 inference
behavior remains the default; policy encoding is opt-in. CUDA prediction parity
and offline/live encoding equivalence remain unvalidated during collection.

`b1.perception.release.v1` is the downstream release declaration required by 6.1:
the exact perception config, estimator and backbone SHA256 references, and true
`offline_passed`, `dynamic_passed`, `frozen` gates. It does not perform or replace
qualification. No such evidence-backed release is available yet; a confidence-
qualified checkpoint alone is insufficient. Producing and validating the release
belongs to the pending 6.0 integration, not to these numerical fixtures.

`load_frozen_observer` verifies those artifact references before loading weights,
requires qualified complete-state confidence, and disables perception gradients.
The collection/estimator-training entry points and their recipes are unchanged.

## Common inspection and mount study

`--inspection configs/perception_inspection.json` enables one common 25-second
neck trajectory for all doors. The ZED rigid mount pitches upward by 10 degrees;
the final neck pitch compensates this for manipulation. The trajectory respects
URDF neck limits and a 0.4 rad/s command-speed bound. The parked arm holds its
initial tool pose. Forbidden contact, excessive tool drift, door motion or neck
tracking error aborts recording. Camera FK is checked throughout the scan.
The mount change is a simulation study requiring a corresponding hardware bracket
and calibration before real deployment; the external Alex package is unchanged.

Inspection is appended as phase code 5, preserving existing phase numbers and one
continuous observation/command clock from reset. It is scoring metadata only.
New metric caches require the exact inspection config in recorded metadata; old
fixed-view recordings cannot silently satisfy the new contract. Top/bottom coverage
is an offline truth/depth audit, not a privileged inference input. Seeing portions
of the boundary is distinct from seeing the entire silhouette.

`--inspection-only` records a bounded camera diagnostic without the expert. Such
files explicitly lack hold/release and are rejected by training episode validation.
They must be stored outside an engineering campaign.

## Complete-state objective and train-only fitting

The `articulated-state-v3` recipe corrects the compensating-state failure measured
in [[experiments/b1-perception-fit-check-01|fit-check-01]]. That historical run
passes contact gates while its hinge and local contact geometry remain wrong;
its old `fit_check_passed` result is not a complete-state qualification.

Every geometric component now has weight 1 and a dimensionless Smooth L1 loss.
Euclidean hinge-origin, dimension-vector, local-contact and world-contact errors
are divided by 0.01 m. Hinge, local-contact, panel and world-contact rotations
and the wrapped signed articulation error are divided by 5 degrees. These are
engineering component budgets matched to the existing operational length/angle
resolution; they prevent individual components hiding errors larger than the
contact budget. They do not guarantee compound accuracy, so the original contact
gates remain independently required. No component thresholds are inferred from
the new fitting outcome.

Rotational distance uses `atan2` of the relative rotation's skew norm and trace;
signed articulation uses wrapped `atan2(sin(delta), cos(delta))`. Unlike the
previous chord objective, the angular loss retains sensitivity near 180 degrees
while keeping finite gradients at zero. Exact antipodal rotations still have
an ambiguous shortest direction. Fresh output weights are small, rotational
biases form identity bases, and the angle starts near `(sin, cos)=(0, 1)`.
These are generic trainable initial values, shared across doors, not oracle
initialization. Initial confidence is below the unchanged acceptance threshold.

Geometry optimization now freezes the confidence output, including weight decay.
Selection and stagnation use only the equal-door worst normalized geometry p95;
geometry-passing checkpoints rank before failing checkpoints. The standard
`ReduceLROnPlateau` scheduler multiplies the step by 0.3 after four evaluations
without a 0.5% improvement, down to 3e-6, before the existing stagnation stop.
Scheduler state is checkpointed and resume requires the same schedule. `loss`/`train_loss` report the optimized geometry sum; `loss_terms` separately
reports all components, including the diagnostic confidence loss. `refine` uses a
fresh optimizer at 3e-5 from a compatible checkpoint and evaluates only train data;
it stops when every train door passes geometry or after at most ten minutes.

`calibrate` freezes geometry, fits only the confidence row on train observations
for at most four passes/ten minutes, then evaluates unchanged development doors.
Labels require all nine physical errors within budget, with missing inputs negative.
The implementation verifies that geometric weights and normalization are unchanged.
Qualification requires per-door p95 errors within budget, at least 95% joint geometric
coverage, at least 95% confidence coverage and at least 95% precision among accepted
states. The fixed 0.5 confidence threshold is not lowered. Failed qualification
still saves the classifier for diagnostics, with `qualified=false`; online inference
rejects it. Evaluation scores the candidate confidence; online validity additionally requires
qualification metadata. Geometry fitting success and confidence qualification
are separate results.

`fit-check --train-doors LEFT_ID RIGHT_ID` fits a fresh estimator on exactly one
train identity per handedness, using all selected episodes and causal windows. A bounded pilot may use
`--partial`; full training and calibration always require the complete campaign. Normalization, balanced sampling, selection and evaluation use
only that subset. It refuses development/test identities and stops at complete
geometry success, train stagnation, 200 epochs or at most ten minutes of loop time.
Final evaluation/checkpoint writing can slightly exceed that time budget. A
passing fit is not evidence of held-out accuracy, dynamic usability or completed
Subphase 6.0.

```bash
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py fit-check --train-doors door-2738468b94d74c5f animated-door-1-88abf40 --output outputs/b1/perception/NEW_FIT_DIRECTORY
```

New checkpoints use `b1.perception.checkpoint.v3` and retain the articulated-state
confidence contract. The config selects the correct model architecture. Legacy
v1/v2 weights remain diagnostic-loadable; absent qualification metadata means
unqualified. A v1 contact-only confidence cannot construct `ObservedEstimator`.
Resume requires the same recipe, scope, data and stopping settings; the new scope
also records separate confidence fitting. Calibrated artifacts omit the geometry
optimizer and are not resumable training checkpoints.

Train-only summaries report `development_evaluated=false`; full training evaluates
development. A successful geometry fit is not development, confidence or dynamic
qualification. Preserve all attempts and use new output directories.

## Pretraining gates and future validation

The unchanged engineering gates in `configs/perception_metric.json` apply per development door
during contact/push/hold: valid coverage at least 95%, operational-point positional
error p95 at most 0.01 m and orientation error p95 at most 5 degrees. Report both
handednesses separately, including the worst per-door positional/orientation p95
for each side. Confidence is trained against geometric usability on train
samples for the complete articulated state, including missing-input examples;
it is not a simulator visibility flag.
Development checkpoint selection weights doors equally and cannot use test evidence.

These are offline gates. Subphase completion additionally requires demonstrated
loss/reacquisition and dynamic use of estimates under the unchanged control/safety
rules. The offline evaluator explicitly reports dynamic validation pending and
cannot mark Subphase 6.0 complete. The initial scan addresses the diagnosed view deficit. Alternate backbones and
closed-loop gaze remain unvalidated future changes.

## Operator commands

Run commands from the repository root with the supported Isaac Lab Python launcher.
On the authoritative workstation, set `ISAAC_LAB_DIR=/home/pacquadr/IsaacLab`.
CUDA execution requires access to the real GPU and simulator caches. Never
substitute CPU for a sandbox denial.

```bash
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/collect_perception.py --output datasets/b1/perception/engineering-v2 --inspection configs/perception_inspection.json --jobs 1
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py prepare --output outputs/b1/perception/preparation-v2.json
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py check --output outputs/b1/perception/readiness-v2.json
```

These commands produced the completed refreshed campaign and prepared caches.
Their existing destinations must not be overwritten. The new defaults require
`engineering-v2`/`features-v2` and
fail against old caches. A pilot supplies explicit `--recordings`, `--features`,
`--partial` and, for `fit-check`, exactly two train door IDs. Preparation refuses
existing destinations. Check verifies unchanged weights without optimizer updates;
partial checks cannot claim full-campaign readiness. Collection resume skips only
validated complete files; retain interrupted attempts in separate directories.

The historical run-03 launch is shown below. It has finished and its geometry
fails development; do not repeat this command or calibrate it as a path to release.
A future run requires a fresh destination and a separate decision:

```bash
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py launch --output outputs/b1/perception/run-03 --hours 1
```

`launch` validates the corpus/cache/CUDA and submits a persistent `systemd --user`
service with console and launch records. It never restarts automatically. `train`
runs in the foreground. `configs/perception_training.json` retains the geometric
recipe and stops after ten completed evaluations without a cumulative 0.5%
improvement, never before 15 epochs. This stopping rule does not relax physical
gates. The loop budget excludes final evaluation/checkpoint writes. Status and
summary report the stop reason, per-door metrics and saved checkpoints.

Legacy diagnostics must explicitly pass `--config configs/perception.json`,
`--recordings datasets/b1/perception/engineering-v1` and
`--features datasets/b1/perception/features-v1`. For example, `refine --checkpoint
outputs/b1/perception/run-02/last.pt --output NEW_DIRECTORY` adds those three flags.
No sealed test door, policy training, new full training or overnight extension is
implied by a successful pilot.
