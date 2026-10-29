# AlexDoor-XAS

AlexDoor-XAS studies action representation in contact-rich humanoid manipulation:
**A1–A4 × ACT/Diffusion** on held-out push doors, using fixed-base Purdue Alex003,
WSG32/UMI v1 and head ZED RGB-D.

The B1 simulator, frozen door corpus, shared observed-perception pipeline and
learning software are maintained. Perception and physical learned execution
remain unqualified; the final matched B1 policy dataset is pending. The separate
A1/A2 pilot remains paused. No repository command controls physical hardware.
[Project status](knowledge/wiki/status.md) owns current results, limits and next work.

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

The full Purdue gate uses synthetic fixtures:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_purdue_runtime.py --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py
```

Runtime reports default to `~/.cache/alexdoor-xas/verification/`. Corpus verification
checks tracked records; `--evidence-root` also checks local expert reports.
[Phase 5](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md)
owns asset intake and qualification.

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
