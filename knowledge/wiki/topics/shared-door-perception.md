# Shared Door Perception

Subphase 6.0 prepares one observed-only stack for every B1 model/representation.
The initial trained estimator is **not qualified**: run-01 fails the development
gates. The [[experiments/b1-perception-run-01|run-01 diagnosis]] separates measured
failures from proposed corrections. Passing recording checks or a CUDA forward/loss
does not establish development accuracy or closed-loop usability.
The implementation contract is in
[[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]].

## Recording boundary

`scripts/collect_perception.py` reads the frozen asset corpus, accepts only train
or development identities, and launches a fresh Isaac process per door/condition.
The default is one worker; `--jobs 2` permits two independent processes on the
verified 24-GiB workstation. Each worker has its own log under `worker-logs/`.
A missing/incomplete artifact is failure even when Isaac exits with code zero.
It reuses the frozen scripted expert, camera, fixed neck, physical setup and safety
rules. It does not publish new qualification references. Nominal and moderate
Replicator light conditions are constant within each episode; physics is unchanged.
The initial campaign contains two episodes per identity: 38 train and 12 development.
These are perception engineering recordings, not the final matched policy dataset.

`b1.rgbd.v1` HDF5 is independent of numerical `phase2.v2`. With N commands it stores
N+1 observations and annotations in separate groups, with episode settings,
calibration, outcome and collection diagnostics in a dedicated `metadata` group.
This includes reset at episode time zero and the
terminal observation. Missing reset observations are rejected.
Command t acts between observations t and t+1. The command includes the compensated
world tool goal actually passed to `command_pose` and resulting seven joint targets;
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

A small spatial fusion module and GRU use four samples at 10 Hz. Each history is
causal and remains inside one episode. Train-only proprioceptive normalization
and equal-door/phase sampling prevent development leakage and domination by long
opening traces. The learned output includes the hinge frame, signed articulation,
panel orientation, dimensions, contact geometry and confidence. `ObjectFrame`
compatibility makes future A3 integration explicit; it does not implement Phase 6.1.

`ObservedEstimator` clears history on reset, missing depth, nonmonotonic input or
a sampling gap. Warmup and low confidence produce invalid estimates. Consumers must
check freshness before commanding motion; there is no oracle fallback. The signed
angle follows right-handed rotation about the estimated frame's Z axis; it is
negative for the canonical right-hinged opening direction.

## Pretraining gates and future validation

The fixed engineering gates in `configs/perception.json` apply per development door
during contact/push/hold: valid coverage at least 95%, operational-point positional
error p95 at most 0.01 m and orientation error p95 at most 5 degrees. Report both
handednesses separately, including the worst per-door positional/orientation p95
for each side. Confidence is trained against geometric usability on train
samples, including missing-input examples; it is not a simulator visibility flag.
Development checkpoint selection weights doors equally and cannot use test evidence.

These are offline gates. Subphase completion additionally requires demonstrated
loss/reacquisition and dynamic use of estimates under the unchanged control/safety
rules. The offline evaluator explicitly reports dynamic validation pending and
cannot mark Subphase 6.0 complete. Gaze and alternate backbones remain conditional
on a diagnosed train/development failure.

## Operator commands

Run commands from the repository root with the supported Isaac Lab Python launcher.
On the authoritative workstation, set `ISAAC_LAB_DIR=/home/pacquadr/IsaacLab`.
CUDA execution requires access to the real GPU and simulator caches. Never
substitute CPU for a sandbox denial.

```bash
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/collect_perception.py --output datasets/b1/perception/engineering-v1 --smoke --condition nominal
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/collect_perception.py --output datasets/b1/perception/engineering-v1 --resume --jobs 2
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py prepare --output outputs/b1/perception/preparation.json
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py check --output outputs/b1/perception/readiness.json
```

`--resume` on collection skips only validated complete recordings; incomplete
attempts must be preserved outside the active campaign before retrying. Feature
preparation refuses an existing destination. Check uses no optimizer or backward
pass and verifies unchanged weights. `check --partial` is a diagnostic only and
cannot report campaign readiness.

The initial engineering campaign has 50 validated recordings and 50 complete
feature caches. `outputs/b1/perception/preparation.json` records the cache build;
`outputs/b1/perception/readiness.json` records the full-corpus CUDA check. This
establishes readiness to start estimator training, not development accuracy or
Subphase 6.0 completion.

**User-requested handoff:** preparation does not launch training. The commands below
describe the first run; `run-01` now exists. Use a new directory for a revised recipe. The smaller model
should submit one autonomous one-hour run with `launch`, confirm startup once,
then end its turn. No active assistant, polling loop or scheduled task is needed.
The user requests the analysis after the run ends; no overnight extension starts
automatically.

```bash
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py launch --output outputs/b1/perception/run-01 --hours 1
"$ISAAC_LAB_DIR/isaaclab.sh" -p scripts/perception.py evaluate --checkpoint outputs/b1/perception/run-01/best.pt --output outputs/b1/perception/development-01.json
```

`launch` validates the corpus, feature cache and CUDA before submitting a
`systemd --user` service, then returns immediately. The service runs independently
of the Codex task; keep the workstation awake and the desktop user session running.
It never restarts automatically. `train` remains available for foreground execution.
The service is `alexdoor-perception-run-01.service`; submission records and the
persistent console are the sibling files `run-01.launch.jsonl` and
`run-01.console.log`. An existing run directory requires explicit `--resume`.

`configs/perception_training.json` controls early stopping independently of the
frozen feature configuration, so stopping settings do not invalidate the caches.
The initial rule stops after **10 consecutive completed development evaluations**
without a cumulative improvement of at least **0.5%** in the equal-door selection
score, and never before **15 completed epochs**. Small gains accumulate against
the last significant improvement. This is an engineering stopping rule, not a
change to the per-door accuracy gates. Every absolute best score still saves
`best.pt`, even when its improvement is below the stopping threshold.

The run stops at the first applicable time, stagnation or epoch limit; nonfinite
loss, gradients or development score fail explicitly. The one-hour budget covers
the training loop; finishing the current step, final development evaluation and
checkpoint writing can add time. Final `status.json` and `summary.json` record
`time_budget`, `development_stagnation`, `epoch_limit` or `error`. The summary
includes the best development results and checkpoint paths. Startup failures before
the training loop are visible in the console/service result.

After completion, start the analysis from these files:

```bash
cat outputs/b1/perception/run-01/summary.json
cat outputs/b1/perception/run-01/status.json
tail -n 3 outputs/b1/perception/run-01/metrics.jsonl
```

Check finite loss, CUDA memory, per-door errors, confidence coverage and the
train/development gap. `last.pt` stores model, optimizer, random states and the
early-stopping counter; `best.pt` is selected on development only. Resume requires
the same data/config and stopping settings;
an interrupted partial epoch starts a fresh weighted sample on resume. Preserve
failed runs and diagnose missing/invalid data, NaNs or GPU failures before retrying.

A later `launch --resume outputs/b1/perception/run-01/last.pt --output
outputs/b1/perception/run-01 --hours 10` is supported but **not authorized to run
by preparation alone**. Decide after the first cycle whether more optimization
addresses the measured failure. A resumed invocation has its own time budget and
retains the stagnation counter;
a run stopped for stagnation needs diagnosis before an extension. Do not lower gates,
open test doors, change the backbone or launch policy training to make a run pass.
