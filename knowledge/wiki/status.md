# Project Status

Current as of 2026-09-24. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Frozen and verified on four synthetic doors. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | Not implemented/run on real doors; final corpus and 12/4/8 split pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

Implement [[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]]:
apply the frozen common probe, obtain two valid runs per candidate, derive
`theta_expert_d` and select the qualified domain. Preserve acquired preparation
results; do not rerun door preparation merely because metadata was simplified.

## Limits

Preparation does not establish robot reachability. The prison metal door retains
its 23.1° geometric stop; expert selection has not been performed. Rights scopes
constrain sharing independently of technical readiness.

RGB-D acquisition is operational; learned door perception is not. Simulator
contact/hinge truth remains diagnostic and cannot become policy input. B1 action,
perception and training contracts are defined in Phases 6–7, not by retained
state-vector model utilities. Simulation checks do not establish hardware safety.

## Maintained Surfaces

- [[topics/system-architecture|Architecture]] — runtime and storage boundaries.
- [[topics/purdue-b1-robot-and-contact|Purdue contract]] — control, sensing and physics.
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] — asset
  contract, 32-door table, intake and qualification protocol.
- [[topics/episode-and-dataset-contracts|Data components]] and
  [[topics/learned-policy-stack|Policy components]] — reusable numerical utilities.

Supported commands are `check_env.py`, `prepare_doors.py`,
`verify_door_preparation.py`, `verify_purdue_runtime.py`, `verify_synthetic_setup.py`
and `screen_synthetic_setup.py`. Verification reports stay in the runtime cache.

Historical B0 scientific conclusions and their limits remain in
[[topics/alex-v2-benchmark|B0 record]] and the experiment pages. The saturated
576-rollout study selected no winner; its 219.95 N event remains under review.
