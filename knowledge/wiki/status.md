# Project Status

Current as of 2026-09-25. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification | Formal paired requalification under the corrected controller completed seven left-handed doors: six `qualified`, one `out_of_domain` at 17.92°. The campaign stopped at that first non-qualified result. The other 25 doors still require fresh pairs; final corpus/split pending. |
| 6–7 — Perception, data, learning | Approved specifications; no B1 demonstrations, Replicator dataset or learned-policy integration yet. |

## Next Action

Review the `psx-worn-20d5505` paired trace and the early orientation safety
guard before deciding how to resume formal qualification. Both valid cycles
stopped below the 45-degree gate; the result does not establish an asset defect
or global unreachability. Do not rerun this door automatically or launch another
door in the current campaign. The ignored `TODO.md` records seven completed
invocations and 25 remaining identities in order. Their retained 21 qualified
and four out-of-domain statuses predate the final controller, as do the six
recovery pairs. Four successful single-cycle diagnostics remain unpublished.
See [[implementation_phases/phase-5-door-corpus-and-qualification|Subphase 5.1]].
The pool has only seven right identities, so even if all seven qualify, at least
five additional qualifying right identities are needed for the intended 12/12
balance. Corpus selection and the split remain future work.

## Limits

The 25 unrerun outcomes belong to the previous common placement/controller.
The recovery moves robot and pedestal together 5.7 cm back and 8 cm laterally,
changes the common fraction/height from 0.40/1.00 m to 0.295/1.09 m, and
corrects tracking/holding defects without lowering the 45-degree gate or physical
validity thresholds. All affected expert references must be remeasured before
final selection. Preparation alone does not establish robot reachability; rights
scopes constrain sharing independently of technical readiness.

RGB-D acquisition is operational, but sampled geometric visibility failed in
both new cycles of industrial-004, front-20d5505 and wooden-009. Visibility
remains a diagnostic separate from expert admission; learned door perception is
not established. Simulator contact/hinge truth remains diagnostic and cannot
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
