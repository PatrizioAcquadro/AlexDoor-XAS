# Project Status

Current as of 2026-09-24. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | All 32 prepared doors have outcomes from the common RTX 4090 probe: **15 qualified, 15 out of domain, two unresolved, none unvisited**. Loader compatibility, leaf ownership, holding and the artificial prison-door stop were corrected before the routine campaign. Final corpus/split pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

Diagnose the two unresolved pairs without per-door retuning: actual forbidden
jaw/panel contact on `psx-front-001-ee7d5c6`, and early contact loss with no
valid final hold on `void-frame-studio-animated-classic-door-08bdf51b`. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]]
for their retained evidence and all published outcomes. At least seven additional
qualifiable right-door identities are still needed for the final 12/12 balance;
corpus selection and the split remain separate future work.

## Limits

Preparation does not establish robot reachability. The prison door's corrected
geometric stop is 47.0°, while its valid common-baseline reference is 44.91557°:
the measured tracking guard stops the push below the 45° admission threshold.
This is not proof that the robot cannot reach 47°. Six other valid paired
references also remain below 45°. PSX front-005 has an obstructed prescribed
footprint, and seven left-door frames physically intersect the fixed pedestal.
These 15 doors remain outside the current domain. Earlier outcomes and original
payloads remain archived. Rights scopes constrain sharing independently of
technical readiness.

RGB-D acquisition is operational, but sampled frame visibility fails in several
qualified real-door trials. Visibility remains a geometric diagnostic; learned
door perception is not established. Simulator
contact/hinge truth remains diagnostic and cannot become policy input. B1 action,
perception and training contracts are defined in Phases 6–7, not by retained
numerical model utilities. Simulation checks do not establish hardware safety.

## Maintained Surfaces

- [[topics/system-architecture|Architecture]] — runtime and storage boundaries.
- [[topics/purdue-b1-robot-and-contact|Purdue contract]] — control, sensing and physics.
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] — asset
  contract, 32-door table, intake and qualification protocol.
- [[topics/episode-and-dataset-contracts|Data components]] and
  [[topics/learned-policy-stack|Policy components]] — reusable numerical utilities.

Supported commands are `check_env.py`, `prepare_doors.py`,
`qualify_door.py`, `verify_door_preparation.py`, `verify_purdue_runtime.py`, `verify_synthetic_setup.py`
and `screen_synthetic_setup.py`. Verification reports stay in the runtime cache.

The maintained numerical data path requires explicit, ordered proprioceptive
`obs_keys` and a caller-supplied dataset root. Diagnostics are separate, episodes
retain `phase2.v2`, and checkpoints use `v3` with no legacy conversion. Dataset
export refuses existing destinations; preparation publication supports rollback.

Historical B0 scientific conclusions and their limits remain in
[[topics/alex-v2-benchmark|B0 record]] and the experiment pages. The saturated
576-rollout study selected no winner; its 219.95 N event remains under review.
