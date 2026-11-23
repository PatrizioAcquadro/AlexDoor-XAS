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

## Measured results

The verified November 12 run used one warm-up and three measured windows per combination on the RTX 4090. The table gives median elapsed seconds per window. All three measured block episodes passed on each backend, both with and without RGB-D.

| Fixture | PhysX | Steel |
|---|---:|---:|
| Single gantry tracking | 0.403 | 0.156 |
| Eight-environment tracking | 0.402 | 0.165 |
| Lift, carry and release | 1.151 | 0.469 |
| Same manipulation with RTX RGB-D and recording | 2.654 | 1.806 |

Steel’s finger-tracking RMSE rose from 2.6–3.1 mm to about 4.0 mm. The initial Steel run exhausted its 32 constraint rows. The preserved comparison includes the corrected capacity and native receipts. Six saved recordings supplied 291 frames per backend. Logs, seeds, failures, task tolerances and memory are in `outputs/generated/paired-physics-20261112/verified/`. These measurements describe the shared gantry. A project can reuse the timing protocol with its own scene, evaluator and acceptance criteria.

## AlexDoor measurements

| Diagnostic | Result and scope | Local record under `outputs/generated/` |
|---|---|---|
| Near-panel D4 cycle | Both engines reached 2.86° and ended unloaded and stable. After contact-readback optimization Steel took 26.55 s in-loop against 30.70 s for PhysX. Total time including startup was about 37.85 s for each. | `alexdoor-closed-loop-timing-20261112/` |
| Startup and warm-up | Two unprofiled pairs in reversed order gave no repeatable total-time advantage. Adaptive ZED reset did not consistently improve warm-up. Fixed reset was retained. | `alexdoor-steel-config-20261112/`, `alexdoor-steel-warmup-20261112/` |
| Three resets in one session | Both engines passed all six episodes across two pairs. Steel took 92.26/91.95 s per session, PhysX 103.03/99.90 s. A separate standard-environment check reused one graph for 8,052 native launches and retained the saturation stop. | `alexdoor-d4n-reused-graph-20261112/`, `alexdoor-d4n-standard-reset-20261113/` |
| Loaded contact and interrupted observed prefix | Both backends passed the bounded preload stop with 900 valid ZED frames each. A separate observed Steel prefix stopped at 0.84° after external interruption and passed its independent stop. | `alexdoor-steel-contact-20261112/` |

See [[purdue-b1-robot-and-contact]] for current launcher selection and reset semantics. [[../experiments/observed-cycle-collection]] covers whole-cycle task and safety outcomes. The complete benchmark development history is retained at `d22bea7`.
