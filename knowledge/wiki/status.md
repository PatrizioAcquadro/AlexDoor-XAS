# Project Status

Current as of 2026-09-30. B1 is maintained; B0 execution and the failed custom
perception workflows are retired. The geometric prototype uses existing recordings
and frozen local weights; no training, collection or test evaluation is authorized.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head RGB-D/proprioception and contact diagnostics implemented. Common zero-yaw setup supersedes the historical 45-degree synthetic setup. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights remain 29 redistributable, two local-only, one private/noncommercial. |
| 6.0 — Perception | Unqualified. Causal GroundingDINO/native SAM 3/DINOv3 provider, RGB-D geometry and full-state replay evaluator implemented. Both pilot doors failed full-state gates. Extended evaluation stopped by the user; retained pilot multiview/association diagnosis is the next action. Dynamic use remains unvalidated. |
| 6.1 — Action paths | Model-independent observed-input contracts, matched data, ACT/Diffusion × A1–A4 and execution/replay software maintained. No qualified provider; final integration and physical validation remain pending. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Diagnose boundary visibility, part identity, multiview fusion and hinge associations
on the two pilot doors. The user stopped the extended evaluation; completed and
partial evidence remains under `geometric-evaluation-02`, with an explicit stop
record. No further campaign is authorized until the main causes are clarified and
corrected, and common corrections are verified on both pilots. The
[[topics/shared-door-perception|full-state gates]] remain unchanged. Dynamic tests
still require every train/development door to pass offline; do not release a
provider or open 6.2 before both gate groups.

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
- No policy/adapter input may contain privileged door geometry. The prototype
  confines simulator truth to its evaluator, including future dynamic tests;
  its observed-only monitor cannot infer force from RGB-D. Other maintained
  teacher/legacy stop monitors have separate truth boundaries. The sealed test
  stays closed. Simulation validation does not establish hardware safety.

## Maintained surfaces

See [[topics/system-architecture|Architecture]],
[[topics/purdue-b1-robot-and-contact|Purdue contract]],
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]],
[[topics/episode-and-dataset-contracts|Data contracts]] and
[[topics/learned-policy-stack|Policy components]]. Supported scripts cover environment
checks, door intake/preparation/qualification, corpus verification, Purdue and
synthetic physics/probe verification, RGB-D collection and diagnostic perception
smoke/evaluation. Synthetic candidate search and estimator training orchestration
are retired.

The numerical data API still requires explicit ordered proprioceptive `obs_keys`
and a dataset root. Episodes retain `phase2.v2`; numerical policy checkpoints retain
v3 and reject legacy formats. Export refuses existing destinations; asset publication
supports rollback. Historical B0 results remain in
[[topics/alex-v2-benchmark|the B0 record]]: the saturated 576-rollout study selected
no winner, and its 219.95 N event remains under review.
