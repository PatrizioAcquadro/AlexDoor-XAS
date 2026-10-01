# Project Status

Current as of 2026-10-01. B1 is maintained; B0 execution and the failed custom
perception workflows are retired. The geometric prototype uses existing recordings
and frozen local weights. The operational successor protocol and implementation plan
are approved. 6.0A contracts, numerical admission rules and consumer compatibility
are implemented. 6.0B static object fusion and geometric queries are complete with
four scan-only CUDA diagnoses and a conditional video comparison. Static ownership
and physical hinge observability remain unresolved. No simulation, training,
collection, extended replay or sealed-test evaluation was started by these assignments.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, head RGB-D/proprioception and contact diagnostics implemented. Common zero-yaw setup supersedes the historical 45-degree synthetic setup. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights remain 29 redistributable, two local-only, one private/noncommercial. |
| 6.0 — Perception/contact | 6.0A/B complete: contracts/admission, static object memory, geometric queries and four bounded scan diagnoses. All scans retain ownership ambiguity and no supported physical hinge axis; the automatic-box video comparison retained no masks. Phase 6.0 is unqualified. 6.0C-H, feedback acquisition and dynamics remain pending; extended evaluation remains stopped. |
| 6.1 — Action paths | Model-independent observed-input contracts, matched data, ACT/Diffusion × A1–A4 and execution/replay software maintained. No qualified provider; final integration and physical validation remain pending. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Continue with 6.0C causal material tracking, articulation and field lifetimes, using the
[[implementation_phases/phase-6-0-operational-perception-and-contact|6.0 operational implementation plan]]
and the [[topics/shared-door-perception|implemented interfaces and static scan memory]].
The 6.0B handoff preserves competing objects, original material references and
unsupported edge-axis alternatives; static ambiguity must not become an accepted estimate.
The guide establishes a torque feedback interface; signal accuracy, simulated
sensor semantics, load inference and physical stopping remain unverified.

The approved operational-v1 profile retains required-state 1 cm / 5 degree and
95% coverage/precision limits, while separating optional full dimensions and
provisional action admission. Historical full-state failures remain unchanged.
Both pilot offline and feedback/stop gates precede bounded pilot dynamics; only
then may a common corrected recipe progress to extended replay. All train/development
offline gates precede qualification dynamics and both gate groups precede release.
6.0A/B supply interfaces, static numerical checks and bounded scan evidence,
not offline, dynamic or release qualification.
Keep 6.1 open and 6.2 unstarted.

The stopped `geometric-evaluation-02` and all pilot attempts remain preserved.
No broader campaign may bypass the main-cause correction and two-pilot checks.

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
  its current observed-only monitor lacks load feedback. The declared torque channel
  cannot expose privileged contacts or pretend to be a calibrated force sensor.
  Other maintained teacher/legacy stop monitors have separate truth boundaries. The sealed test
  stays closed. Simulation validation does not establish hardware safety.

## Maintained surfaces

See [[topics/system-architecture|Architecture]],
[[topics/purdue-b1-robot-and-contact|Purdue contract]],
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]],
[[topics/episode-and-dataset-contracts|Data contracts]] and
[[topics/learned-policy-stack|Policy components]]. Supported scripts cover environment
checks, door intake/preparation/qualification, corpus verification, Purdue and
synthetic physics/probe verification, RGB-D collection and diagnostic perception
smoke/evaluation and bounded scan diagnosis. Synthetic candidate search and estimator training orchestration
are retired.

The numerical data API still requires explicit ordered proprioceptive `obs_keys`
and a dataset root. Episodes retain `phase2.v2`; numerical policy checkpoints retain
v3 and reject legacy formats. Export refuses existing destinations; asset publication
supports rollback. Historical B0 results remain in
[[topics/alex-v2-benchmark|the B0 record]]: the saturated 576-rollout study selected
no winner, and its 219.95 N event remains under review.
