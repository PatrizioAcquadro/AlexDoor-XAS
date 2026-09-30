# Project Status

Current as of 2026-09-29. The maintained project is **B1**; B0 execution,
compatibility, dataset payloads and run orchestration are retired.

| Area | Current state |
|---|---|
| 4.0 — Purdue runtime | Implemented and GPU-verified: seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head ZED RGB-D/proprioception and contact diagnostics. |
| 4.1 — Common setup | Common robot/pedestal placement at zero yaw and revised ready/parked posture validated on real left/right doors. Historical synthetic verification applies to the 45-degree setup only. |
| 5.0 — Prepared pool | **32 doors: 29 redistributable, two local-only, one private/noncommercial.** Acquired preparation results preserved; portable three-record layout. |
| 5.1 — Expert qualification and split | **Complete.** All 32 doors qualified under the current fresh-process protocol. `assets/doors/b1/corpus.json` freezes 12 reviewed families into train 19, development 6 and test 7, with both handednesses in each. |
| 6.0 — Perception | Refreshed 50-episode campaign and run-03 complete: 19/19 train geometry passes and 0/6 development. Approved DINOv3/SAM 3 GPU screening also finds no qualified replacement; the next step is an explicit observed metric-geometry prototype. No new full training. |
| 6.1 — Action paths | Independent software available for ACT/Diffusion × A1-A4: observed encoding, matched data, normalization/checkpoints, adapters and execution/replay runner. CPU contracts checked; CUDA checks, final integration and physical validation remain pending. **Not complete.** |
| 6.2–7 — Policy data and learning | Not started; no matched B1 policy dataset or learned-policy result. |

## Next Action

The [[experiments/b1-perception-model-comparison|current model comparison]] finds
accurate train fitting but severe development error after the refreshed collection
and run-03. Public geometric/pretrained components offer useful partial outputs;
none has established complete hinge/contact geometry or qualified confidence.
The [[experiments/b1-perception-approved-models|approved DINOv3/SAM 3 trials]] are
complete: both DINO heads fit all train doors in the small matched probe and fail
all six development doors. On the corrected component sample, boxed SAM 3 improves
development plane successes from 92/108 to 97/108, with rare severe errors still
present. Use it as a candidate visual aid with observed detector boxes, retain
SAM 2 as a reference, and implement multi-view metric geometry, border association,
tracking, local contact measurement and explicit ambiguity handling. This prototype remains
unimplemented. Validate full geometry per door before another full training run.
Existing recordings/checkpoints and failed attempts remain preserved; the sealed
test stays closed.

Subphase 6.1 can now bind the eventual frozen perception artifact without using
simulator geometry in the policy/adapters. Its deferred CUDA checks remain separate
from this perception comparison. Once 6.0 is qualified and frozen, validate final
integration, matched physical replay and observed-geometry execution across all
eight cells before closing 6.1. Do not start the 6.2 pilot yet.

The interfaces and autonomous launch commands remain in
[[topics/shared-door-perception|Shared Door Perception]].

Phase 5 is closed with all 32 qualified identities preserved. Cross-pack mesh
review merged the PSX essential/interior/front sources into one train family;
the previous source-only split example is superseded. The frozen counts are
train 15 left/four right, development five left/one right, test five left/two
right. Development's single right door and the 15-door residential train family
limit coverage. The freeze and evidence review are in
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]].

## Limits

The recovery moved robot and pedestal together 5.7 cm back and 8 cm laterally,
changed the common fraction/height from 0.40/1.00 m to 0.295/1.09 m, and
corrected tracking/holding defects without lowering the 45-degree gate or physical
validity thresholds. All published pairs now use fresh processes and the current
common implementation. Preparation alone does
not establish robot reachability; rights scopes constrain use and sharing
independently of technical readiness. Local-only assets require authorized local
access; the private/noncommercial door remains limited to private preparation/tests.

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
become policy input. B1 action and policy software is separate from the retained
numerical utilities; its available CPU checks do not establish perception or
physical rollout quality. Simulation checks do not establish hardware safety.

## Maintained Surfaces

- [[topics/system-architecture|Architecture]] — runtime and storage boundaries.
- [[topics/purdue-b1-robot-and-contact|Purdue contract]] — control, sensing and physics.
- [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] — asset
  contract, 32-door table, intake and qualification protocol.
- [[topics/episode-and-dataset-contracts|Data components]] and
  [[topics/learned-policy-stack|Policy components]] — reusable numerical utilities.

Supported commands are `check_env.py`, `prepare_doors.py`,
`qualify_door.py`, `verify_door_corpus.py`, `verify_door_preparation.py`, `verify_purdue_runtime.py`, `verify_synthetic_setup.py`
`screen_synthetic_setup.py`, `collect_perception.py` and `perception.py`.
Verification reports stay in the runtime cache; perception engineering data and
local preparation evidence use ignored datasets/outputs directories.

The maintained numerical data path requires explicit, ordered proprioceptive
`obs_keys` and a caller-supplied dataset root. Diagnostics are separate, episodes
retain `phase2.v2`, and checkpoints use `v3` with no legacy conversion. Dataset
export refuses existing destinations; preparation publication supports rollback.

Historical B0 scientific conclusions and their limits remain in
[[topics/alex-v2-benchmark|B0 record]] and the experiment pages. The saturated
576-rollout study selected no winner; its 219.95 N event remains under review.
