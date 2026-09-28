# Project Status

Current as of 2026-09-28. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | All 32 prepared doors have published qualified fresh-process pairs under the current common implementation: 25 left and seven right. The authorized industrial-004 completion qualified at 47.92° with complete evidence. Final corpus review and split freeze remain pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

Complete the final corpus review and freeze the identity split. The last open
execution gate, `psx-industrial-004-4f5561b`, is closed: a separately authorized
single-door invocation on `cuda:0` published two valid fresh-process cycles at
47.92 degrees, with zero spread, complete RGB-D evidence and passing geometric
checks. The interrupted attempt remains in the cache; no asset, controller or
threshold change was needed. The other 31 door records are unchanged. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]].

The 32 published identities include 25 left and seven right doors from 15
source families. A concrete, non-frozen example keeps each family together and
covers both hands in train (15 left, four right), development (five left, one
right) and test (five left, two right). It uses every published qualified door;
there is no demonstrated Phase 5 need for additional right-hand intake or a
24-door cap. Development has sparse right-hand coverage, which later results
must disclose. No split manifest is frozen
and Phase 6 has not started.

## Limits

The recovery moved robot and pedestal together 5.7 cm back and 8 cm laterally,
changed the common fraction/height from 0.40/1.00 m to 0.295/1.09 m, and
corrected tracking/holding defects without lowering the 45-degree gate or physical
validity thresholds. All published pairs now use fresh processes and the current
common implementation. Preparation alone does
not establish robot reachability; rights scopes constrain sharing independently
of technical readiness.

RGB-D acquisition is operational. The visibility diagnostic now samples rendered
surfaces rather than convex collision proxies, and distributes frame points near
contact height. Full-trajectory geometric visibility warnings remain on three
current qualified pairs: `door-adf292f437f2`,
`door-door-metal-b21ec273` and `psx-front-008-ee7d5c6`. The older
industrial-004 report also has a warning; its complete new pair passes.
The fresh completed industrial-004, front-20d5505 and wooden-009 pairs pass the corrected
diagnostic in both full trajectories. Earlier targeted snapshots remain
separate from those complete checks.
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
