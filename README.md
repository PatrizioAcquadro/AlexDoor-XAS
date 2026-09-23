# AlexDoor-XAS

AlexDoor-XAS studies how action representation affects learning and execution in contact-rich humanoid manipulation. 
Its current runtime uses fixed-base Purdue Alex003 in NVIDIA Isaac Sim and Isaac Lab.
The earlier Alex V2 door-pushing benchmark remains available as historical data and results.

The study compares matched episodes in four action representations (A1-A4), allowing the representation to change while the robot, task, and underlying experience remain fixed.

## Current scope

Subphase 4.0 replaces B0 execution with the Purdue Alex003 operational runtime:
WSG32/UMI v1, measured pedestal, seven-joint A1/full-pose A2/A3 control and head
ZED RGB-D/proprioception. Synthetic collidable fixtures commission the integration;
Subphase 4.1 freezes the common setup and probe on four synthetic doors. Subphase
5.0 adds verified door preparation, format conversion and static/GPU checks.
Real-door intake, expert qualification and learned-policy integration remain pending.

Historical datasets, ACT/Diffusion models, checkpoint loading and offline training
remain available. Full generation and learned evaluation currently stop with a
migration explanation before simulator startup. Old checkpoints cannot execute
as Purdue policies.

See [Project Status](knowledge/wiki/status.md) for evidence and remaining work,
and [Phase 4](knowledge/wiki/implementation_phases/phase-4-robot-and-task-configuration.md)
for the operational contract. No command controls physical hardware.

Use [Phase 5](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md)
for `scripts/prepare_doors.py` commands and the pre-download candidate checklist.

## Requirements

- Python 3.11 or newer through the supported Isaac Lab runtime.
- Isaac Sim 6.0.1 and Isaac Lab `release/3.0.0-beta2`.
- The external Alex package with Purdue/WSG, measured pedestal and pinned ZED Wide assets.

Do not use bare system `python3` for Isaac code.

## Installation and validation

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e /home/pacquadr/Desktop/Alex
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e ".[dev]"
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
```

Run the complete operational GPU gate:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/verify_purdue_runtime.py --viz none --device cuda:0
```

Reports, numeric traces and camera samples default to
`~/.cache/alexdoor-xas/verification/purdue/`; `--output` selects another location.
`--gate contacts` and `--gate rgbd` run focused diagnostic subsets; only `all`
can establish complete Subphase 4.0 evidence. `--no-cameras` is diagnostic only.

The public acquisition sample is `env.capture.sample`. It contains RGB, depth in
meters, a valid-depth mask, seven arm and two neck positions/velocities, simulation
time, episode and frame IDs. Model resizing, history and learning integration are
deferred to Phase 6.

## Repository layout

```text
src/alexdoor_xas/   package code for the benchmark, data, policies, and evaluation
scripts/            operational verification, B1 preparation and offline training
configs/            frozen synthetic setup, historical calibration and offline policies
tests/              deterministic regression and contract tests
knowledge/          user-owned raw research and the official technical wiki
datasets/           reusable local episodes, splits, and normalization artifacts
outputs/            canonical D0-D4 scenes and learned-policy runs
```

Machine-local assets, datasets, checkpoints, videos, logs, and runtime caches remain outside Git.

## Documentation

- [Technical Wiki](knowledge/wiki/index.md)
- [Project Status](knowledge/wiki/status.md)
- [System Architecture](knowledge/wiki/topics/system-architecture.md)
- [Action Representations and Adapters](knowledge/wiki/topics/action-representations-and-adapters.md)
- [Episode and Dataset Contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md)
- [Learned Policy Stack](knowledge/wiki/topics/learned-policy-stack.md)
- [Alex V2 Benchmark](knowledge/wiki/topics/alex-v2-benchmark.md)
- [Output Contract](outputs/README.md)

## License

This repository is proprietary. 
No license grant is provided unless a separate license file or written agreement states otherwise.
