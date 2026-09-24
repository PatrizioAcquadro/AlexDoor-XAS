# AlexDoor-XAS

[GitHub repository](https://github.com/PatrizioAcquadro/AlexDoor-XAS).

The workstation checkout is `/home/pacquadr/Desktop/AlexDoor-XAS`. Python imports
remain `alexdoor_xas`; the distribution name remains `alexdoor-xas`.

AlexDoor-XAS studies how action representation affects learning and execution in
contact-rich humanoid manipulation. B1 compares A1–A4 × ACT/Diffusion on held-out
push doors with fixed-base Purdue Alex003, WSG32/UMI v1 and head ZED RGB-D.

Purdue control/sensing and the common four-door synthetic setup are implemented.
The prepared pool contains **32 doors**: 29 redistributable, two local-only and
one private/noncommercial. Real-door expert qualification, B1 demonstrations,
Replicator and learned-policy integration remain future work.

B0 workflows and local data have been retired. Their scientific conclusions and
limits remain in the wiki. Reusable action, recording, dataset and model
components remain; no repository command controls physical hardware.

## Setup and Verification

Use Python 3.11+ from the supported workstation stack: Isaac Sim 6.0.1 and
Isaac Lab `release/3.0.0-beta2`, plus the external Alex package with Purdue/WSG,
measured pedestal and ZED Wide assets. Isaac, PyTorch and CUDA are supplied by
that runtime, not installed as package dependencies.

From the checkout:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e /home/pacquadr/Desktop/Alex
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e '.[dev,diffusion]'
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
ruff check src scripts tests
```

Model tests use CUDA and explicitly skip if unavailable. Pure numerical tests do
not require a simulator. Run the Purdue integration gate on synthetic fixtures:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_purdue_runtime.py \
  --viz none --device cuda:0
```

Reports default to `~/.cache/alexdoor-xas/verification/purdue/`; use `--output`
for another location. `--gate contacts`, `--gate rgbd` and `--no-cameras` are
focused diagnostics; only the complete gate establishes full runtime evidence.

## Repository Layout

| Path | Purpose |
|---|---|
| `assets/doors/b1/` | Canonical door records and ignored source/final payloads. |
| `src/alexdoor_xas/` | Runtime, preparation and reusable learning components. |
| `scripts/` | Supported verification, synthetic setup and door intake. |
| `configs/` | Frozen common Purdue synthetic probe. |
| `tests/` | Behavioral regressions and GPU model checks. |
| `knowledge/` | User-owned raw research and canonical wiki. |
| `datasets/`, `outputs/` | Local future datasets/results; payloads ignored. |

## Documentation

- [Project Status](knowledge/wiki/status.md) — current state, limits and next action.
- [Technical Wiki](knowledge/wiki/index.md) — canonical navigation.
- [Phase 5](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md) — asset contract, corpus, intake commands and qualification protocol.
- [Architecture](knowledge/wiki/topics/system-architecture.md) — runtime and data boundaries.
- [Data Components](knowledge/wiki/topics/episode-and-dataset-contracts.md) and [Policy Components](knowledge/wiki/topics/learned-policy-stack.md) — maintained interfaces.

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise. Individual door assets retain
their recorded third-party terms and distribution scopes.
