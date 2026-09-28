# AlexDoor-XAS

[GitHub repository](https://github.com/PatrizioAcquadro/AlexDoor-XAS).

The workstation checkout is `/home/pacquadr/Desktop/AlexDoor-XAS`. Python imports
remain `alexdoor_xas`; the distribution name remains `alexdoor-xas`.

AlexDoor-XAS studies how action representation affects learning and execution in
contact-rich humanoid manipulation. B1 compares A1–A4 × ACT/Diffusion on held-out
push doors with fixed-base Purdue Alex003, WSG32/UMI v1 and head ZED RGB-D.

Purdue control/sensing and **Phase 5 are complete**. All 32 prepared doors have
qualified fresh-process expert pairs. The [frozen corpus](assets/doors/b1/corpus.json)
assigns 19 train, six development and seven test doors, preserving related
geometry families and both handednesses in every partition. Rights scopes remain
29 redistributable, two local-only and one private/noncommercial.

Subphase 6.0 preparation adds synchronized B1 recording and a shared frozen-backbone
perception estimator. Training has not started; final demonstrations and
ACT/Diffusion integration remain Phase 6 work. See the
[perception contract and handoff](knowledge/wiki/topics/shared-door-perception.md).
See the [corpus and qualification procedure](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md)
and current [project status](knowledge/wiki/status.md).

B0 workflows and local data have been retired. Their scientific conclusions and
limits remain in the wiki. Reusable action math, recording, dataset and model
components remain; no repository command controls physical hardware.

## Setup and Verification

Use Python 3.11+ from the supported workstation stack: Isaac Sim 6.0.1 and
Isaac Lab `release/3.0.0-beta2`, plus the external Alex package with Purdue/WSG,
measured pedestal and ZED Wide assets. Isaac, Alex, PyTorch, Warp and CUDA are
external runtime dependencies. B1 perception uses Transformers already supplied
by the workstation runtime. Ordinary Python dependencies are declared in
`pyproject.toml`; Diffusion and developer tooling use their respective extras.

From the checkout:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e /home/pacquadr/Desktop/Alex
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e '.[dev,diffusion]'
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
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

Qualify one prepared door with the frozen setup:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/qualify_door.py \
  --asset-id <id> --device cuda:0 --headless
```

The command stores a fresh report in the verification cache and updates only the
expert section of the door record. Read the [Phase 5 procedure](knowledge/wiki/implementation_phases/phase-5-door-corpus-and-qualification.md)
for outcomes, evidence review and the separate Phase 6 pilot boundary.

Verify frozen membership and references without starting Isaac Sim:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py
```

Add `--evidence-root ~/.cache/alexdoor-xas/verification/expert` to also check
local qualification reports, traces and capture inventories. Record-only checks
do not require ignored asset payloads or the cache.

Reusable numerical data APIs require an explicit dataset root and ordered
`obs_keys` drawn from recorded proprioception. Door state and contacts remain
diagnostics. Episodes retain `phase2.v2`; policy checkpoints use `v3` and reject
earlier formats. This is not yet the B1 RGB-D learning pipeline.

## Repository Layout

| Path | Purpose |
|---|---|
| `assets/doors/b1/` | Frozen corpus/split, canonical door records and ignored source/final payloads. |
| `src/alexdoor_xas/` | Runtime, preparation and reusable learning components. |
| `scripts/` | Supported verification, synthetic setup, intake and expert qualification. |
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
