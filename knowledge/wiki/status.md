# Project Status

Current as of 2026-09-28. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | All 32 prepared doors have published qualified pairs: 25 left and seven right. Twenty-five pairs used fresh-process trial isolation; the first seven precede that revision, and the first six also precede the angular-guard correction. Final same-implementation validation, corpus selection and split remain pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

The 24 pairs pending after the `void-frame` process-isolation repair are complete:
all qualified in one formal invocation each. Reports and published summaries
agree. Reusing the same physics scene had caused history-dependent contact
impulses despite identical reset states; fresh Isaac processes now isolate
physics and renderer state between trials. Controller, geometry and admission
thresholds were unchanged during the continuation. The first seven published
pairs predate process isolation, and the first six also predate the angular-guard
revision. Revalidate those seven under one implementation before final corpus
selection; keep each historical reference tied to its own saved code and evidence.
See [[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]].
The pool has only seven right identities, all qualified; at least five additional
qualifying right identities are needed for the intended 12/12 balance. Corpus
selection and the split remain future work.

## Limits

The recovery moved robot and pedestal together 5.7 cm back and 8 cm laterally,
changed the common fraction/height from 0.40/1.00 m to 0.295/1.09 m, and
corrected tracking/holding defects without lowering the 45-degree gate or physical
validity thresholds. The seven earlier published pairs need current remeasurement
before final selection. Preparation alone does not establish robot reachability; rights
scopes constrain sharing independently of technical readiness.

RGB-D acquisition is operational. The visibility diagnostic now samples rendered
surfaces rather than convex collision proxies, and distributes frame points near
contact height. Full-trajectory geometric visibility warnings remain on six
qualified pairs: the earlier `psx-industrial-004-4f5561b`,
`psx-front-20d5505` and `psx-wooden-009-20d5505`, plus the newer
`door-adf292f437f2`, `door-door-metal-b21ec273` and
`psx-front-008-ee7d5c6`. All nine targeted GPU snapshots on the first three
passed after the visibility correction; their full trajectories were not rerun.
Offline replay of all 750 saved RGB-D frames from the three newer pairs reproduces
the original counts. Their warnings reflect frame occlusion/out-of-view samples,
the 40-point sampling cap and panel self-occlusion in oblique views. No asset or
camera correction is justified by this diagnostic alone; retain the qualifications
and assess estimator usability in Phase 6. The scoped findings and evidence are
in [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]].
Visibility remains separate from expert admission; learned door perception is not
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
