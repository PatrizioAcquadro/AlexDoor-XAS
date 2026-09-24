# Project Status

Current as of 2026-09-24. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | Pilot references remain right 59.33 degrees and left 46.37 degrees. Six subsequent doors were attempted; the campaign is paused pending loader compatibility and final-hold diagnosis. Prison-door collider correction passes static/GPU physics; its earlier expert reference is superseded pending a new pair. Across 32 doors: two qualified, two out of domain, five unresolved, one published `invalid_asset` under review, and 22 unvisited. Final corpus/split pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

Keep the remaining 22 doors unvisited until the loader compatibility problem
and repeated invalid final hold are diagnosed. The published `invalid_asset`
result must not be attributed to source geometry before the loader is checked.
See [[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]]
for per-door outcomes and retained evidence. Do not retune the pose, controller,
camera or assets to obtain a pass. Older results remain tied to their archived
setup. At least five additional qualifiable right identities are still needed
for the final 12/12 balance.

## Limits

Preparation does not establish robot reachability. The prison metal door retains
a corrected 47.0° geometric stop: the earlier 23.1° stop came from excess convex
volume at a frame corner. The original payload and reference remain archived. The previous four pedestal exclusions belong to the
superseded 45-degree pose; current results are tied to their recorded setup and
complete trial pair. Rights scopes constrain sharing independently of technical
readiness.

RGB-D acquisition is operational, but sampled frame visibility is insufficient in
the qualified real-door trials; learned door perception is not established. Simulator
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
