# AlexDoor-XAS

AlexDoor-XAS studies how action representation affects contact-rich humanoid
manipulation: **A1–A4 × ACT/Diffusion** on held-out push doors, using fixed-base
Purdue Alex003, WSG32/UMI v1 and head ZED RGB-D.

## Current scope

- **Runtime and corpus:** Purdue control/sensing and Phase 5 are complete. The
  frozen corpus contains 32 qualified doors: 19 train, six development, seven test.
- **Perception:** maintained static GroundingDINO/SAM3 scans and an
  experimental CAD-free Point2Pose tracker. The selected SAM3 recipe is explicit;
  material identity, 150 ms freshness and loaded contact remain unqualified.
- **Learning:** B1 observed-input datasets, ACT/Diffusion models, normalization and
  A1–A4 execution adapters are maintained. Qualified perception integration and
  physical rollout validation remain pending. No matched B1 policy dataset or
  learned-policy result exists; the sealed test remains closed.

The [project status](knowledge/wiki/status.md) owns current results, limits and
next work. No repository command controls physical hardware.

## Installation

Use Python 3.11+. The supported workstation uses Isaac Sim 6.0.1, Isaac Lab
`release/3.0.0-beta2` and the external Alex package. Isaac, Alex, PyTorch, Warp and
CUDA are supplied by that runtime; ordinary Python dependencies are declared in
[pyproject.toml](pyproject.toml).

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e /home/pacquadr/Desktop/Alex
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e '.[dev,diffusion]'
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Robot/camera assets come from the external Alex package. Prepared door payloads,
recordings and model weights remain local and ignored by Git. Perception workers
use isolated dependencies; follow [model setup](models/perception/README.md).

## Verification

```bash
ruff check src scripts tests
ruff format --check src scripts tests
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/unit
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/gpu
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/integration
```

[GitHub Actions](.github/workflows/ci.yml) runs lint/format, unit tests on Python
3.11/3.12, wheel/sdist builds and an installed-wheel import check. Unit tests use
CPU tensors without Isaac, model downloads or local payloads. CUDA model/storage
regressions and integration with external Isaac/Alex/native sources run locally;
GPU tests explicitly skip when CUDA is unavailable. A green public CI does not
establish simulator, perception or physical qualification.

The complete Purdue runtime gate uses synthetic fixtures:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_purdue_runtime.py --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py
```

Runtime reports default to `~/.cache/alexdoor-xas/verification/`. Corpus verification
checks tracked membership and references; `--evidence-root` also checks local expert
reports. See [Phase 5](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md)
for prepared-door intake and qualification.

Run bounded frozen perception inference with a fresh output directory:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py smoke \
  --output outputs/b1/perception/NEW_SMOKE
```

The [model README](models/perception/README.md) documents scan, Point2Pose smoke,
operational replay, live and serial offline commands. Live diagnostics do not need
recordings. Successful execution and scientific quality are separate outcomes;
smokes do not admit motion or qualify a policy provider.

## Organization

| Path | Responsibility |
|---|---|
| `src/alexdoor_xas/` | Runtime, action math, preparation, recording, B1 data and policies. |
| `src/alexdoor_xas/perception/` | Shared contracts/geometry; `point2pose`, `sam3` and `diagnostics` subpackages. |
| `scripts/`, `configs/` | Supported entry points and common runtime/experimental settings. |
| `tests/unit`, `tests/gpu`, `tests/integration` | Checks separated by execution requirements. |
| `assets/doors/b1/` | Frozen corpus, canonical records and ignored prepared payloads. |
| `knowledge/` | User-owned raw research and canonical technical wiki. |
| `datasets/`, `outputs/`, `models/perception/` | Local data, results and pretrained resources. |

B0 workflows, numerical `phase2.v2` data APIs and standalone `v3` checkpoint loaders
are retired; historical source remains at `097d578`. B1 formats and observation/
action ordering are unchanged. New perception imports are documented in
[Shared Door Perception](knowledge/wiki/topics/shared-door-perception.md#module-layout).

Start at the [wiki index](knowledge/wiki/index.md),
[architecture](knowledge/wiki/topics/system-architecture.md),
[data contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md) and
[policy stack](knowledge/wiki/topics/learned-policy-stack.md).

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise. Door assets retain their
recorded third-party terms: 29 redistributable, two local-only and one
private/noncommercial.
