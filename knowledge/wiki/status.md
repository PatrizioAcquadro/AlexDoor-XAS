# Project Status

Current as of 2026-10-15. **Static 6.0B is maintained; CAD-free Point2Pose is
implemented but unqualified. Phases 6.0 and 6.1 remain open.** Repository consolidation
retains the full B1 core; the selected SAM3 recipe is now the common runtime
reference. A3/A4, policies and operational thresholds are unchanged.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and common zero-yaw setup maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts/static evidence | Operational/local support, explicit selection, admission and feedback contracts; object fusion, ownership alternatives, finite-cover and relevant-space queries. Scans/interfaces do not admit motion. |
| 6.0C — Tracking | One persistent SAM3 session shared by inspection/tracking, bounded history and renewal-only depth gate; live/replay and serial evaluation share the coordinator. Useful accuracy improves offline, but material identity and 150 ms freshness remain unqualified. |
| 6.0D–H — Contact/qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Actions | ACT/Diffusion × A1–A4 observed-input/data/normalization/execution software maintained; qualified-provider integration and physical rollout validation pending. |
| 6.2–7 — Data/learning | Not started. No matched B1 policy dataset or learned-policy result; no sealed-test evaluation. |

## Next action and selected state

The [[experiments/p2p-persistent-session-integration|persistent-session integration]]
uses the selected configuration throughout maintained scan/smoke/live/replay/offline
entry points. Historical 33-frame CUDA prefixes exactly preserve masks, poses,
decisions and traces with strict initialization and explicit BF16/TF32.
Two fresh selected sessions process all 4,718 original 0–78.6167 s captures per
condition, initializing P2P at 25.016667 s without restarting SAM3. Their short
mask/pose prefixes match exactly, but the full inspection histories are distinct
from historical 31 s starts. The unresolved geometric-selection attempt is preserved.

On the same 2,858 opening captures and three frozen physical targets, correct
availability at 10/15/20 mm and 5 degrees is 86.53–87.58/97.38–97.55/98.74–98.81%
light and 97.62–98.64/99.76–99.83/99.97–100% nominal. Neither has native/SAM3 loss;
light has eight nonseed degenerate-support refusals, nominal none. Light position
maxima 45.00–46.62 mm and rotation maximum 4.08 degrees regress; nominal position
maxima improve to 18.47–20.93 mm while median/p95 and 10 mm availability regress.
All three targets, precision, chronological gaps and tails are in the integration
record. Same effective configuration/BF16/TF32 and changed masks/seed history are
verified; no compensating quality change is introduced.

Full serial p95 260/246 ms still yields zero requests within 150 ms. Sampled joint
GPU peaks are 7.77/11.81 GiB, combined coordinator/worker RSS 6.03/6.10 GiB;
startup spikes remain unqualified. Final bounded wall-clock/replay ingress replaces
31 of 33 inputs and publishes two results about 3 s late. Its immutable seed and
scan publication times are checked independently. That 33-frame prefix did not
validate concurrent rendering or matched operational quality; the full ordered follow-up below supplies new, separate measurements.

The [[experiments/p2p-operational-selection-and-light-peaks|ordered operational follow-up]]
now implements explicit reference ingress and measures full continuous recorded
input followed by concurrent stationary rendering. Steady publication is 3.79/3.24 Hz,
with age p95 610/346 ms and zero within 150 ms. The recording retains 4,484 skips
and 27 native losses with zero measured pairs; dense quality does not transfer. The initial
live reference failure and subsequent post-run reporting failure are preserved.
The saved light diagnosis distinguishes an angular cluster/final-gate fallback,
a mixed frame/leaf position peak, and SDF-induced narrow supports for all eight
integration refusals. The single saved-mask CUDA bridge now completes all 2,858
light opening captures with an exactly matching historical 31 s seed and no SAM3
rerun or runtime change. Postseed masks alone change relative to history: primary
10/15/20 mm availability becomes 87.19/96.85/98.78%, with zero native losses,
12 geometric refusals and a new 74.55 mm maximum. The integrated 72.0667/78.15 s
peaks and eight refusal captures do not recur despite identical later masks;
combined P2P initialization/history matters, but seed time alone remains unisolated.
Reference audits retain 18/469/440 fixed-only births historical/bridge/integrated.
The [[experiments/p2p-temporal-renewal-admission|single temporal renewal prerequisite]]
is now implemented diagnostically and rejected before runtime activation: all
637/412 new light/nominal references are indeterminate within the 0.25 s waiting
budget, including 205/264 useful moving points. Projected opening support falls
below five pairs on 68.37/62.60% of captures and vanishes in both final tails.
No CUDA intervention or further variant ran; the selected pipeline stays unchanged.
The [[experiments/p2p-sam3-cost-diagnosis|selected-pipeline cost diagnosis]] now
separates native stages and mature TSDF rebuilds with unchanged dense/sparse
reference histories. SAM3 (about 107 ms), TAPIR (about 44 ms) and CPU registration
already exceed the 150 ms budget; tracing alone is insufficient. Continuous
nominal publishes at 4.28 Hz with age p95 327 ms, 4,479 replacements and zero
steady outputs within 150 ms. Comparable fixed-input producer probes increase
median model costs, but a paused nominal stall and full-opening contention remain
unresolved. The [[experiments/tsdf-readonly-launch-inputs|single TSDF readback candidate]]
exactly preserves both 3,217-frame CUDA histories and volume/query checks, but is
rejected for adoption: 2.60–3.74% fewer TSDF DtoH bytes yield no convincing
rebuild/service benefit. Four timed sessions retain mixed throughput/quality,
nominal loss/accuracy regressions and zero of 828 steady publications within
150 ms. Active code is restored to baseline; the reproducible candidate, failures
and residual profile remain. A follow-up [[experiments/tsdf-deferred-snapshot-audit|deferred-snapshot failure audit]]
now measures local CUDA rebuild medians of 650.2 -> 281.7 ms light and
468.5 -> 268.9 ms nominal, with exact successful volumes/queries. Final-only
snapshots are nevertheless rejected: after a failed in-place write, a real next
worker request consumes different SDF values. Point/image preparation remains
separate from measured snapshot copies. That final-only candidate stopped before
full fixed/timed replays because error equivalence failed. Dense replay does not
verify operational loss.

The [[experiments/tsdf-rebuild-double-buffer|two-buffer successor]] now has complete
recovered-source evidence: all local/error/worker gates and both 3,217-input fixed
pairs pass; repeated memory stabilizes with zero live pool arrays after retirement.
The four timed runs are complete, without new foreign compute. Local rebuilds
improve 35.2%/28.8%, but ordinary p95 service/age does not improve, nominal correct
20 mm counts fall from 194–195 to 162–163 across the targets, and native losses
rise from zero to 27. The nominal buffered history has no post-seed volume growth;
its shorter maximum stall is not a demonstrated buffered-rebuild benefit.
**Adoption is rejected; maintained and installed native code stay at baseline.**
Different timed seeds/histories limit causal quality attribution. The primary
remaining cost is ordinary sequential SAM3/P2P service (~108–113/~109–115 ms
medians), with sparse-input tracking still unqualified. All current evidence,
including the preserved light memory baseline and RustDesk interruption, remains
in the persistent primary-checkout recovery directory. Original worktree-loss
limitations remain historical; no completed valid gate was repeated.

The single [[experiments/sam3-official-compilation|official SAM3 compilation]]
variant now completes paired dense/sparse CUDA evaluation: all seven components
run, nine graphs stabilize before evaluation and sparse SAM3 medians improve
19.3%/16.7%. Native quality nevertheless regresses with exactly identical seed
masks and seed registration traces: correct 20 mm/5 degree counts become 148/180
for all light targets and 38/204 nominal, versus 153/180 and 153–154/204 eager.
Native losses rise 27→30 / 47→130. Adoption is rejected and maintained eager
source is restored; TSDF and every prior result remain unchanged. The pilot
quality gate closes full fixed/timed comparisons. Isolated-stage composition
suggests a residual near 199–204 ms before geometry/binding, not a measured
integrated benefit or operational age. All valid cells and two interference
prefixes remain; no other optimization or controller/training/contact work ran.

The single [[experiments/p2p-ransac-point-buffers|RANSAC point-buffer candidate]]
now passes isolated exact comparisons on 148 saved inputs and adversarial
support/threshold/degeneracy/failure cases, including all samples and final RNG.
The preset local benefit gate passes: median RANSAC falls 14.62%/15.18%, with
4.20/2.90 ms paired savings in the enriched saved-input corpus. Full fixed/timed
GPU comparisons await shared availability; integrated benefit is unvalidated.
SAM3 eager, TSDF baseline and all prior evidence are retained.

The earlier selected [[experiments/p2p-sam3-selected-development|SAM3 development]] completed
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
controls. Historical source remains at `57ad483`. Subsequent consolidation removed
only retired SAM3.1/SAM2 Small weights and eight identical trace copies, with
relative links preserving old trace paths. Local removal digests and canonical
copies are recorded in `evidence/cleanup.json` (`maintenance_20261008`).
GroundingDINO, the old static worker and separate SAM2 operation/setup are now
retired after consumer migration. Existing historical weights and all baseline
readers remain; SAM3 internal tracker components are preserved. Runtime CUDA math
no longer depends on the SAM2 import. Geometry/contact interfaces remain maintained.
A follow-up output-consumer audit retired DINOv3 extraction: its descriptors were
stored/fused but unused by maintained decisions. Model/config/license metadata and
SAM3.1 metadata remnants were consolidated into the local cleanup inventory.

Perception now separates model runtimes and diagnostics; tests separate portable,
GPU and local integration requirements. Public CI covers Python 3.11/3.12 unit
behavior, Ruff and packaging. See [[topics/system-architecture|Architecture]].

Bounded CUDA validation in `outputs/evidence/b1/perception/consolidation-20261008/` matches
33 original frames per condition exactly against selected configuration, masks,
poses, decisions and registration traces. The existing 12-sample operational smoke
matches saved object/mask outputs on CUDA TSDF with freshness still failed.
Synthetic probes pass for both handednesses. Native source reconstructs
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
