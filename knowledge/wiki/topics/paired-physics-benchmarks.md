# Paired Physics Backend Benchmarks

The common runner measures native PhysX GPU and Newton Steel with CUDA Graph on
one RTX GPU, one process at a time. The fixture is a four-prismatic-joint gantry
with two opposing fingers. Both engines load the same authored USD colliders,
material, mass, offsets, initial conditions, commands, and explicit PD effort
controller. The default experiment exercises tracking at one and eight
independent environments, a lift and release of a free block, and the same
block motion with Isaac RTX RGB-D, color/depth perception, and recording.

## Run and read the results

From the repository root, set `ISAAC_SIM_DIR` to the installed Isaac Sim,
`NEWTON_SITE` to an isolated Python site with Newton 1.8.0rc2, Warp 1.18.0,
OpenUSD, the Newton USD schemas, and the OVPhysX schema package, and
`BENCH_GPU_UUID` to the RTX UUID from
`nvidia-smi -L`. Run:

```bash
bash scripts/benchmark_physics.sh configs/physics_benchmark/common.json outputs/generated/paired-physics
PYTHONPATH=src python3 -m alexdoor_xas.benchmarks.summarize outputs/generated/paired-physics
```

Append `tracking-single`, `tracking-batch`, `block`, or `rgbd` to select cases.
The configuration's `backends` list selects the engines to run.
Use a fresh output directory for each rerun. The default configuration pairs the
same three seeds after one warm-up. Each result includes process and scene
startup, separate warm-up measurements, timed resets, steady episode wall time,
synchronized physics steps, controls, evaluation, and where applicable rendering
with readback, perception, and compressed recording. `environment_steps_per_s`
counts each environment's physics steps. VRAM is peak device memory sampled by
`nvidia-smi` every 100 ms during the measurement windows, including resident
system graphics allocations. The process logs record the native PhysX GPU flag
and the captured Steel graph. The summary validates both backend receipts.

The task measures bilateral native contact at 60 Hz, object displacement relative
to the hand during lift/carry, release after supported contact, final support
height, and post-release drift and velocity. The RGB-D fixture uses a fixed
320 × 240 Isaac RTX camera at 30 Hz, with positive finite metric depth, advancing
frame IDs, non-flat RGB, and observed red-object detection. Its detector logs
the centroid and median depth without changing the common controller. Recorded
NPZ windows include RGB, depth, simulation timestamps, and the detection outputs.

## Project adapters

Set `factory` globally or in a named case to a `module:callable` that receives
`(simulation_cfg, config, case, backend, env_roots)` and returns an experiment
fixture. It owns its simulation context and exposes `sim`, `start`, `reset`,
`control`, `dispatch`, `advance`, `observe`, and `evaluate`. RGB-D workloads
also provide `capture`, `perceive`, `record`, and `finish_recording`. The shared
runner owns the warm-up, seed pairing, timers, device reservation, and receipts.
A workload with an existing high-level step can supply
`control_tick(time_s, substeps, physics_dt, measure)` in place of the three
control and dispatch methods. It can time the existing step as one operation or
instrument exclusive component boundaries through `measure`. The runner checks
the actual number of physics steps and reports missing physics attribution as
unavailable.

| Project | Existing interfaces to bind | Acceptance to verify on the project scene |
| --- | --- | --- |
| AlexDoor | `qualification.expert.probe_config` supplies the prepared door and `DoorPushPurdueEnvCfg`. `DoorPushPurdueEnv` exposes `step_a1`, `command_pose`, and measured contact and RGB-D capture. | Projected joint feedback, material-point support, load and stop bounds, and observed push/hold/release/stabilization with mounted ZED. |
| VLA-LEGO | `vla_lego.isaac.environment.AssemblyEnvironment` owns `step`, `capture`, `observe`, `task_truth`, and episode recording. `PickPlaceEvaluator.update` and `result` own the task verdict. | Inject the physics config at context construction. Verify the actual WSG32 mimic, contact and friction, observed head frames, and existing pick/place criteria on the LEGO block. |
| SquadBot | `squadbot_av.simulation.rig.AlexSensorRig` owns `step(head_command=...)`, `state`, and `capture_camera`. `pilot_evaluation.evaluate_pilot` owns episode scoring. | Inject the physics config at context construction. Verify neck response, ZED-to-head extrinsics, audiovisual time alignment, room and acoustic integration, and existing orientation metrics. |

The common fixture uses one static camera and no acoustic scene. Application
constructors that create their own PhysX config need a physics-config input.
The common benchmarks provide a runtime comparison and component evidence.
Project-specific task or autonomy qualification requires each project's normal
controller, observations, recorder, and evaluation.
