# Project Status

Current as of 2026-09-28. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | Twenty-two doors have qualified pairs: sixteen left and six right. The latest four pairs qualify at 45.09°, 54.30°, 54.41° and 62.66°, each with zero spread. Another 10 pairs and the final corpus/split remain pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

The focused `void-frame` repair is complete. Reusing the same physics scene
caused history-dependent contact impulses despite identical reset states.
Fresh Isaac processes now isolate both physics and renderer state between the
two trials; the final pair qualifies at 66.92° with valid holds/releases and
zero spread. Controller, geometry and admission thresholds are unchanged.
Continue the 10 pending pairs with `--rerun`, one door at a time. Their retained
qualified outcomes predate shared controller revisions. The first six newer
qualified pairs also predate the angular-guard revision; seven earlier newer
references predate process isolation. Published references remain tied to their
saved code/setup and evidence; all affected references need current validation
before final selection.
See [[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]].
The pool has only seven right identities, so even if all seven qualify, at least
five additional qualifying right identities are needed for the intended 12/12
balance. Corpus selection and the split remain future work.

## Limits

The 10 pending outcomes belong to the previous common placement/controller.
The recovery moves robot and pedestal together 5.7 cm back and 8 cm laterally,
changes the common fraction/height from 0.40/1.00 m to 0.295/1.09 m, and
corrects tracking/holding defects without lowering the 45-degree gate or physical
validity thresholds. All affected expert references must be remeasured before
final selection. Preparation alone does not establish robot reachability; rights
scopes constrain sharing independently of technical readiness.

RGB-D acquisition is operational. The visibility diagnostic now samples rendered
surfaces rather than convex collision proxies, and distributes frame points near
contact height. All nine targeted GPU snapshots on industrial-004, front-20d5505
and wooden-009 passed; their complete trajectories were not rerun. Visibility
warnings occurred on the new qualified pairs for `door-adf292f437f2` (556 of
3,949 checked frames in each cycle) and `door-door-metal-b21ec273` (all 3,453
checked frames in each cycle). Representative RGB frames for the latter still
show the door and robot hand; the geometric warning needs separate evaluation.
Visibility
remains separate from expert admission; learned door perception is not
established. Simulator contact/hinge truth remains diagnostic and cannot
become policy input. B1 action,
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
