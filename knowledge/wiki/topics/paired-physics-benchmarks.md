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

The reusable Steel backend lives in `isaac_physics_profile`. On the workstation's
Isaac Lab beta2 stack, launch with the selective Newton/Warp environment from
`src/isaac_physics_profile/launch.sh` and inject `steel_config(...)` into
`SimulationCfg.physics` after AppLauncher. The profile keeps the Kit OpenUSD
runtime in-process and imports colliders through the separate wheel's OpenUSD.
Its solved normal-force and mounted-camera adapters are used by VLA-LEGO's
existing scene, controller, recorder, and evaluator. Project launchers can use
`ISAAC_PHYSICS_PROFILE=newton_steel` or `isaacsim_physx`. Solver options and
device graph capture should be recorded alongside task results.

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
| VLA-LEGO | `vla_lego.isaac.environment.AssemblyEnvironment` injects the profile at context construction and owns `step`, `capture`, `observe`, `task_truth`, and episode recording. `PickPlaceEvaluator.update` and `result` own the task verdict. | Verify physical pick/place completion and timing under the existing evaluator before choosing a default. |
| SquadBot | `squadbot_av.simulation.rig.AlexSensorRig` owns `step(head_command=...)`, `state`, and `capture_camera`. `pilot_evaluation.evaluate_pilot` owns episode scoring. | Inject the physics config at context construction. Verify neck response, ZED-to-head extrinsics, audiovisual time alignment, room and acoustic integration, and existing orientation metrics. |

The common fixture uses one static camera and no acoustic scene. Application
constructors that create their own PhysX config need a physics-config input.
The common benchmarks provide a runtime comparison and component evidence.
Project-specific task or autonomy qualification requires each project's normal
controller, observations, recorder, and evaluation.

## AlexDoor D4-N contact readback

The D4-N comparison adapter uses `SteelContactSnapshot` to batch the native
contact transfers into pinned host buffers with one stream synchronization.
The raw observer and the filtered finger/panel normal and friction views share
one snapshot per physics step. The snapshot uses the solver's pre-step body
poses for contact geometry and preserves pair orientation, signed forces,
sensor ordering, and overflow checks.

In the short D4-N replay with Alex003, WSG32 and ZED, the optimization reduced
Steel's contact and state bucket from 26.90 s (previous two-run mean) to 23.06 s.
The fresh pair completed 5,556 steps and 1,389 valid RGB-D frames each. Steel
with CUDA Graph took 69.53 s for replay through stabilization versus 70.48 s
for native PhysX GPU. Including startup and warm-up, Steel took 79.90 s versus
77.10 s. Both reached roughly 3° after retreat; peak load during the push
itself differed (0.27 N with Steel, 2.83 N with PhysX).

The bounded loaded-stop comparison uses the first 9 s of recorded targets and
then runs the existing measured-anchor stop evaluator for 2 s plus its 4 s
continuation. Both backends passed from roughly 0.2 N of preload, with 900 valid
ZED frames each. The complete short test took 51.45 s with PhysX and 56.25 s
with Steel, including startup. The stop evaluator measures continuing physics
at a fixed target, independently of the replay controller.

The live observed Steel attempt reached confirmed contact and 0.84° of
opening before the user ended the long test during push. Its independent stop
passed and the partial recording passed integrity review. The task threshold,
hold, release and stabilization remain unmeasured in that observed attempt.
Local commands, timing, and receipts are under
`outputs/generated/alexdoor-steel-contact-20261112/`.

## AlexDoor default after the bounded D4-N test

`scripts/isaac-run.sh` starts the Purdue environment with Newton Steel and
CUDA Graph. Set `ISAAC_PHYSICS_PROFILE=isaacsim_physx` to use native PhysX GPU.
The single-environment path imports the composed stage outside Kit and checks
the solver's capacity at the contact readout boundary.

Both backends passed a bounded D4-N cycle on the RTX 4090. Alex003 started
unloaded 3.9 mm from the panel. The controller used the mounted ZED's depth to
track opening and finger load to adjust normal motion. PhysX reached 2.86°
and Steel reached 2.86° before withdrawal. Both ended unloaded and stable with
about 27 mm of separation. The final pair measured 30.81 s with PhysX and
31.92 s with Steel in the loop, or 37.93 s and 43.25 s including startup and
warm-up. Steel passed without a backend or wheel-path override. The adoption
gate was functional. The paired probe covers a few degrees on one prepared
panel. Local logs, thresholds and repeat commands are under
`outputs/generated/alexdoor-closed-loop-20261112/`.

The same short trial was profiled before and after optimizing contact
readback. Initially Steel saved 8.55 s in physics but spent 10.73 s reading
and classifying contacts versus 2.68 s with PhysX. CPU classification
accounted for 8.08 s on Steel. The observer now reuses fingertip hulls, checks
the active contact values in batches and only reads poses for fingertip
contacts that can belong to the panel. Steel copies the sticky solver
status flags with the geometry and invokes Newton's full check for flagged
or optional statuses. The check still covers every observed native step.

The optimized pair passed on both backends. Steel contact time fell to
3.93 s, with 0.06 s spent interpreting the status flags on the host. Steel
took 26.55 s for the loop versus 30.70 s with PhysX and both took about 37.85 s
through the result including startup and warm-up. These timings cover one
short D4-N cycle per backend per configuration. The closed-loop runs reached
the opening criterion at different ticks. Timing breakdowns, step logs and
commands are under `outputs/generated/alexdoor-closed-loop-timing-20261112/`.

### Steel startup and warm-up

On the same short D4-N trial with warm runtime caches, Kit startup took about
4.0 s on both backends. Steel spent 1.65 s configuring the backend and 1.80 s
exporting and importing the Newton model. Reset with the ZED took 0.56 s and
the first control tick included 0.39 s of solver preparation and CUDA Graph
capture.

AlexDoor now starts the isolated importer before configuring Steel so its
library imports overlap with configuration and scene construction. Two
optimized runs reduced the model export/import span to 1.00 s and Steel
startup from 9.62 s to 8.88–8.89 s. Warm-up was 1.68 s. Steel's total through
the result was 37.32–37.52 s versus the Steel baseline of 37.87 s and paired
PhysX results of 37.15–37.50 s. Both backends passed each short trial. The
optimized Steel runs each captured CUDA Graph and checked capacity on all
2,620 native steps. Raw results and repeat commands are under
`outputs/generated/alexdoor-steel-startup-20261112/`.

Profiling the configuration boundary separately attributed most of its time
to Newton imports. AlexDoor now loads Newton's viewer, controller and inverse
kinematics APIs on first use. Isaac RTX and AlexDoor's own controller drive
the short D4-N trial. Configuration fell from 1.64 s to 1.40–1.41 s in the
two unprofiled pairs. Steel's totals were 37.01 and 37.53 s against 37.30
and 37.16 s for PhysX. The two pairs used opposite backend orders and
establish no repeatable total-time advantage for either engine. The
optimized Steel runs each passed with 661 valid ZED frames, CUDA Graph
active, and 2,644 capacity checks on 2,644 physics steps. Local timings,
profiles and repeat commands are under
`outputs/generated/alexdoor-steel-config-20261112/`.

On the same D4-N scene, Steel's warm-up baseline took 1.71 s. The ZED reset
took 0.42 s and graph preparation and capture took 0.57 s inside the first
control tick. CUDA's capture begin and end calls took roughly 36 microseconds
combined. The eager preparation pass specialized collision kernels and loaded
Warp modules before capture. Deferring Python cycle collection reduced that
span to 0.37 s, with the saved time appearing in the remaining warm-up ticks.

An adaptive ZED reset accepted three to five exposures instead of eight.
The full depth images and validity masks matched the reference. The mean RGB
difference reached 2.35 levels in an adaptive repetition and 2.87 in a second
fixed reset, indicating variation across runs at either exposure count.
Adaptive reset time remained near 0.42 s and warm-up did not improve
consistently. The fixed reset and native graph path were retained. The
experiment preserves the per-run timing and RGB-D evidence
under `outputs/generated/alexdoor-steel-warmup-20261112/`.
The final pair with the original reset took 37.44 s with Steel and
37.01 s with PhysX, including startup and warm-up. Steel's loop took
27.08 s and PhysX's took 29.88 s. Both passed; Steel ran all 2,644
capacity checks with CUDA Graph active.

### Reusing the graph across D4-N resets

The short near-panel trial ran three times in one Isaac session per backend
on the RTX 4090. Each episode restored the seeded Alex003 and hinge state,
acquired a new ZED RGB-D frame, ran 12 warm-up ticks, and used the same
closed-loop push, release, and stabilization criteria. The Steel trial runner
checked the preceding solver status and cleared the native solver history and
contact snapshot in place before each repeated reset. It verified that the same
graph object was launched on every native step after each reset.

Both backends passed all six episodes over two pairs with reversed execution
order. Steel took 92.26 s versus 103.03 s with PhysX in the first pair, and
91.95 s versus 99.90 s in the second pair. Each total includes one startup,
three resets and warm-ups, and three cycles. Steel spent about 18.43 s in
physics per session versus 44.51–44.63 s with PhysX. Its first control tick
took 0.77–0.79 s on initial capture and 0.038–0.043 s after reset. The log
shows one capture per Steel session and 8,076 launches and capacity checks
per session, including warm-up. The closed-loop completion tick can differ
by backend. All cycle frames passed the RGB-D checks and the settled contact
load was zero on both backends. The experiment covers repeated resets of the
same scene in one process. The runner, raw data, and repeat command are under
`outputs/generated/alexdoor-d4n-reused-graph-20261112/`.
