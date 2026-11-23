# AlexDoor-XAS

AlexDoor-XAS studies action representation in contact-rich humanoid manipulation:
**A1–A4 × ACT/Diffusion** on held-out push doors, using fixed-base Purdue Alex003,
WSG32/UMI v1 and head ZED RGB-D.

The maintained workflows cover door qualification, observed teacher collection, recorded perception diagnostics and learned-policy training, replay and evaluation. The final matched A1–A4 comparison remains open. The [project status](knowledge/wiki/status.md) distinguishes teacher qualification, learned completions and safety-stop outcomes.

## Installation

Use Python 3.11+. The workstation supplies Isaac Sim 6.0.1, Isaac Lab
`release/3.0.0-beta2`, the external Alex package, PyTorch, Warp and CUDA.
Ordinary dependencies and optional development/diffusion tools are declared in
[pyproject.toml](pyproject.toml).

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e /home/pacquadr/Desktop/Alex
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e '.[dev,diffusion]'
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Robot/camera assets come from Alex; prepared doors, recordings and model weights
remain local. Perception uses isolated workers: follow
[model setup](knowledge/wiki/topics/perception-model-setup.md).
Repository scripts/configs run from the source checkout; the wheel supplies the
Python package. [Cluster training](knowledge/wiki/topics/cluster-training.md)
describes separate A100/H100 deployment boundaries.

## Verification and diagnostics

```bash
ruff check src scripts tests
ruff format --check src scripts tests
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/unit
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/gpu
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/integration
```

[CI](.github/workflows/ci.yml) checks lint/format, Python 3.11/3.12 unit tests,
wheel/sdist builds and installed-wheel imports. Unit tests need no Isaac launch,
model downloads or local payloads; numerical tensor contracts use CPU in CI.
GPU/integration checks require the corresponding local runtimes. A green CI
establishes software behavior, not simulator, perception or physical qualification.

## Run the simulator

Run from the checkout on the RTX workstation. The launcher selects the physics implementation by entry point.

| Entry point | Default runtime |
|---|---|
| `collect_perception.py`, `qualify_door.py` | Native Steel/Warp (`newton_steel_warp`) in the dedicated native environment |
| `collect_d4_observed.py` | Isaac Lab beta2 Newton Steel (`newton_steel`) with RTX RGB-D. Select `newton_steel_native` for native OVRTX or `newton_steel_warp` for a bounded Warp component. |
| `pilot_physics.py` and its dynamics, stop and workspace probes | Isaac Sim PhysX (`isaacsim_physx`) with the pilot calibration |

For standard collection and qualification, install the native runtime and converted assets as described in the [Purdue runtime guide](knowledge/wiki/topics/purdue-b1-robot-and-contact.md#install-the-native-b1-runtime). These bounded commands exercise the 4° fixture and write diagnostic results.

```bash
asset=psx-interior-wood-002-02e442c
fixture=configs/purdue_b1_bounded_diagnostic.json
scripts/isaac-run.sh scripts/collect_perception.py --asset-id "$asset" --condition light \
  --bounded-diagnostic "$fixture" --output outputs/generated/native-collection-check
scripts/isaac-run.sh scripts/qualify_door.py --asset-id "$asset" --condition nominal \
  --bounded-diagnostic "$fixture" --diagnostic --output-root outputs/generated/native-evaluation-check
```

For a guarded D4 component check, supply the protocol, setup and operational recipe from the same reviewed run. Resolve any historical absolute paths in a working copy of the recipe using the [evidence storage guide](outputs/evidence/README.md).

```bash
scripts/isaac-run.sh scripts/collect_d4_observed.py --protocol "$protocol" \
  --setup "$setup" --operational-recipe "$recipe" --urdf-cache-dir "$usd_cache" \
  --component-max-sim-s 0.1 --record-tracker-stream --output "$fresh_output"
```

The component time limit exercises interruption and continuing-physics stopping. To check the same protocol with native Steel and the Warp camera, use the native installer's converted asset layout and a separate output directory.

```bash
ISAAC_PHYSICS_PROFILE=newton_steel_warp scripts/isaac-run.sh scripts/collect_d4_observed.py \
  --protocol "$protocol" --setup "$setup" --operational-recipe "$recipe" \
  --component-max-sim-s 0.1 --record-tracker-stream --output "$fresh_native_output"
```

For a full native D4-N attempt, use the same declared protocol and recipe with a fresh output. The installer enables NVIDIA's OVRTX camera. On a cold shader cache, follow the compilation progress in the run's `ovrtx.log` and allow several minutes for the first frame.

```bash
ISAAC_PHYSICS_PROFILE=newton_steel_native scripts/isaac-run.sh scripts/collect_d4_observed.py \
  --protocol "$protocol" --setup "$setup" --operational-recipe "$recipe" \
  --record-tracker-stream --output "$fresh_native_output"
```

The [D4 experiment record](knowledge/wiki/experiments/observed-cycle-collection.md#native-steel-component-comparison) explains the measured differences. Run the corpus and synthetic runtime checks separately.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py
scripts/isaac-run.sh scripts/verify_purdue_runtime.py --viz none --device cuda:0
```

Every recorded perception diagnostic uses one explicit pilot episode and fresh output:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py smoke \
  --episode /absolute/path/to/pilot/nominal/episode.hdf5 \
  --output outputs/generated/b1/perception/NEW_SMOKE
```

[Supported diagnostics](knowledge/wiki/topics/perception-model-setup.md#supported-diagnostics)
cover static scan, P2P smoke, replay, offline evaluation and operational timing.
Live diagnostics require `--asset` and `--case`, with no recording dependency.
Each invocation starts one attempt; diagnostic completion never admits motion.

## Project navigation

Start at the [wiki index](knowledge/wiki/index.md) and
[architecture/workflow map](knowledge/wiki/topics/system-architecture.md).
[Data contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md) and
[policy stack](knowledge/wiki/topics/learned-policy-stack.md) describe the maintained
Phase 6–7 interfaces. [Datasets](datasets/README.md) and [outputs](outputs/README.md)
own storage and recovery guidance. B0 and rejected variants are historical;
each experiment names its Git revision and retained evidence.

## License

This repository is proprietary. No license grant is provided without a separate
license file or written agreement. Door assets retain their recorded third-party
terms: 29 redistributable, two local-only and one private/noncommercial.
