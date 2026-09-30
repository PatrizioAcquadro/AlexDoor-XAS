# Project Status

Current as of 2026-09-29. B1 is maintained; B0 execution and the failed custom
perception workflows are retired. No new training or collection occurred during
cleanup.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head RGB-D/proprioception and contact diagnostics implemented. Common zero-yaw setup supersedes the historical 45-degree synthetic setup. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights remain 29 redistributable, two local-only, one private/noncommercial. |
| 6.0 — Perception | Unqualified. Custom estimators retired after persistent development failures. GroundingDINO + SAM 3, explicit RGB-D/multiview geometry and optional DINOv3 features selected; new pipeline unimplemented. |
| 6.1 — Action paths | Model-independent observed-input contracts, matched data, ACT/Diffusion × A1–A4 and execution/replay software maintained. No qualified provider; final integration and physical validation remain pending. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Implement the shared observed metric-geometry provider against
[[topics/shared-door-perception|the retained interface and fixed gates]]. Address
border/plane association, multiview static geometry, articulation/contact geometry
and explicit ambiguity/loss. Then establish per-door development accuracy,
confidence and dynamic usability before releasing a frozen provider or opening
6.2. No further training or collection is implicitly authorized.

[[experiments/b1-perception-findings|Perception findings]] preserves the evidence:
run-03 fitted 19/19 train doors but passed 0/6 development; matched DINOv2/v3 probes
repeated that gap. Boxed SAM 3 improved aligned development plane successes from
92/108 to 97/108 with severe remaining failures. These are component results,
not full geometry or confidence qualification.

## Retained resources and limits

- Original engineering recordings: two 50-episode campaigns, two complete pilots
  and one inspection-only diagnostic. Calibration and split metadata remain intact;
  engineering data are not matched policy demonstrations.
- Selected model resources: `models/perception/`. Historical results and relocation
  records: `outputs/b1/perception/evidence/`. Five incomplete recordings, rejected
  weights and obsolete derived caches were removed explicitly; Git cannot restore
  those ignored payloads.
- Prepared assets, frozen corpus and qualification evidence remain unchanged.
  All 32 expert references use fresh processes. Visibility warnings on three
  qualified doors are sampling/observability diagnostics, not admission failures.
- Train has one 15-door residential family; development has only one right-hand
  door. Report per-door/handedness results without claiming population coverage.
- No policy/adapter input may contain privileged door geometry. Simulator truth
  remains teacher/evaluator information or stop-only safety input. The sealed test
  stays closed. Simulation validation does not establish hardware safety.

## Maintained surfaces

See [[topics/system-architecture|Architecture]],
[[topics/purdue-b1-robot-and-contact|Purdue contract]],
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]],
[[topics/episode-and-dataset-contracts|Data contracts]] and
[[topics/learned-policy-stack|Policy components]]. Supported scripts cover environment
checks, door intake/preparation/qualification, corpus verification, Purdue and
synthetic physics/probe verification, and RGB-D collection. Synthetic candidate
search and estimator orchestration are retired.

The numerical data API still requires explicit ordered proprioceptive `obs_keys`
and a dataset root. Episodes retain `phase2.v2`; numerical policy checkpoints retain
v3 and reject legacy formats. Export refuses existing destinations; asset publication
supports rollback. Historical B0 results remain in
[[topics/alex-v2-benchmark|the B0 record]]: the saturated 576-rollout study selected
no winner, and its 219.95 N event remains under review.
