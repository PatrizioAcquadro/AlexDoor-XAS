# Project Status

Current as of 2026-10-08. **Static 6.0B is maintained; CAD-free Point2Pose is
implemented but unqualified. Phases 6.0 and 6.1 remain open.** The targeted cleanup
is complete; operational defaults, A3/A4 and policy contracts are unchanged.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and common zero-yaw setup maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts/static evidence | Operational/local support, explicit selection, admission and feedback contracts; object fusion, ownership alternatives, finite-cover and relevant-space queries. Scans/interfaces do not admit motion. |
| 6.0C — Tracking | Explicit selected SAM3 recipe with bounded history and renewal-only depth gate; SAM2 operational/baseline and live/serial evaluators retained. Useful accuracy improves offline, but material identity and 150 ms freshness remain unqualified. |
| 6.0D–H — Contact/qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Actions | ACT/Diffusion × A1–A4 observed-input/data/normalization/execution software maintained; qualified-provider integration and physical rollout validation pending. |
| 6.2–7 — Data/learning | Not started. No matched B1 policy dataset or learned-policy result; no sealed-test evaluation. |

## Next action and selected state

Keep [the selected configuration](../../configs/point2pose_selected.json) unchanged.
[[experiments/p2p-selected-residuals|Saved residual diagnosis]] found no justified
common correction. Next establish a few independent material-pixel chains in saved
RGB-D, separating image correspondence from reference/pose-feedback bias. The
historical rejected-light-candidate trace is missing; a future unchanged diagnostic
could record candidates/dense-field state, but cannot recover historical rejection
totals. This is not authorization to resume a campaign or change thresholds.

The selected [[experiments/p2p-sam3-selected-development|SAM3 development]] completed
both original 2,858-frame CUDA openings. Bounded history alone exactly preserved
masks/poses/decisions and reduced SAM peak 8.39→4.21 GiB without RAM offload. The
separate renewal-only gate improved all three targets; primary correct availability
at 10/15/20 mm is 85.83/99.79/99.93% light and 99.13/99.72/99.86% nominal, with zero
native losses and one nominal integration refusal. Complete p95 235/260 ms leaves
zero nonseed requests within 150 ms. Dense offline scoring does not measure backlog
or dropped-frame behavior.

Regressions remain explicit: light position peak 23.82 mm, rotation p95 1.10 degrees,
only 8/301 correct final 10 mm captures, and joint GPU peak 12.76 GiB through P2P
allocator reserve despite bounded SAM. Nominal's new position peak is 31.48 mm.
Detailed peak times, point IDs, coordinate conventions and SDF/rotation distinctions
belong in the residual diagnosis; neither native acceptance nor same-ID return
establishes material identity. Future latency work must preserve this recipe and
quality evidence and evaluate real queue/resampling separately.

## Cleanup and protected dependencies

The standalone numerical `phase2.v2` data and `v3` policy-loading pipeline is
retired. B1 retains its models, trainers, observed-input dataset, checkpoint
contracts and shared normalization/sampling; historical APIs remain at `097d578`.

`e494e8c` tracks the selected opt-in recipe; `c01c3a1` retires periodic selective
registration, SAM3.1/multiplex memory, refit rollback and rejected model/tracker/SVD
controls. Historical source remains at `57ad483`; all ignored payloads remain.
Static GroundingDINO/SAM3/DINOv3, SAM2 operational paths, baseline comparisons,
geometry/contact interfaces and evaluator helpers have active consumers.

Bounded CUDA validation in `outputs/b1/perception/p2p-cleanup-20261008/` matches
33 original frames per condition exactly against selected configuration, masks,
poses, decisions and registration traces. The existing 12-sample operational smoke
completes on CUDA TSDF with freshness still failed. Native source reconstructs
from the pinned archive, with migration/idempotence and unknown-source refusal.
These checks do not establish full-opening equivalence; no full replay, model
comparison or latency optimization was performed during cleanup.

Preserve recordings, original timestamps, baselines, failed attempts, images,
masks, traces and ignored diagnostic scripts. In particular the residual diagnosis
uses `p2p-sam3-followup-01/hull_evaluator.py`, prepared collision geometry and
`p2p-performance-01/full-target-evaluation.json`; all remain available. Original
31–78.6167 s recordings reach 63.7209 degrees, not the configured 90.7-degree limit.
Different automatic seeds prevent interpreting light/nominal as isolated lighting.
The earlier campaign stays stopped at 4,216 completed rows, 502 remaining and
11 unstarted attempts; [[experiments/b1-perception-findings|historical findings]]
preserve its failures and all intervening decisions.

## Action and qualification limits

The [[topics/pre-a4-initialization-and-hinge-budget|shared pre-A4 specification]]
applies before policy dispatch in every A1–A4 × ACT/Diffusion condition. Stationary
observations can support local surface/closed rotation, but these scans lack a
supported physical pin. Direct pin evidence or separately admitted leaf-motion
diagnosis must establish the reference before A4 approach; that pre-articulation
motion/admission path is absent. No controller is qualified by the specification.

Authored simulation floor/up and measured camera/FK agreement still need complete
interface inputs and uncertainty validation against proposed command/sweep/stop
margins. Correspondence envelopes and evaluator hinge p95 are different quantities;
neither substitutes for missing calibration bounds. A4 freezes each segment's
admitted frame, so a fixed local endpoint retains full origin error at approach
completion. Actual proposals, contact/load/stop and hardware calibration remain absent.

The unchanged gates require per-door 95% coverage/accepted precision and 1 cm/5
degrees; always unavailable fails. Pilot offline and feedback/stop checks precede
bounded interaction; all train/development offline gates precede qualification
dynamics. Both groups plus release compatibility precede 6.1 handoff. Tracking
alone supplies neither ownership, hinge nor safe continuous contact/sweeps.

All 50 engineering-v2 episodes, selected models, corpus/splits, prepared assets,
qualification evidence and `knowledge/raw` remain intact. Earlier September/October
cleanups removed explicitly inventoried payloads; `evidence/cleanup.json` records
those historical removals/relocations, which are not Git-recoverable. Current
recordings lack torque; the monitor blocks loaded execution. Depth is ideal,
camera mounting and hardware feedback/stopping remain unvalidated. Human local-role
review and endpoint IK do not qualify whole-object ownership or a continuous path.
Train contains one 15-door residential family, development one right-hand door;
report per-door/handedness results without population claims. The sealed test stays closed.

Historical learned estimators fitted train but qualified 0/6 development doors;
that does not justify another training run. B0 remains historical: the saturated
576-rollout study selected no winner and its 219.95 N event remains under review.
Supported commands/setup are in the root/model READMEs; runtime contracts and
future work are canonical in [[topics/shared-door-perception|Perception]] and
[[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0]].
