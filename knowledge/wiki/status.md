# Project Status

Current as of 2026-10-17. **Static 6.0B is maintained; CAD-free Point2Pose is
implemented but unqualified. Phases 6.0 and 6.1 remain open.** Repository consolidation
retains the full B1 core; the selected SAM3 recipe is now the common runtime
reference. A3/A4 semantics and operational thresholds are unchanged.

The active [[experiments/b1-end-to-end-pilot-resume|end-to-end continuation]] has
four real3000-update corrected-data checkpoints and frozen local development
inference. ACT A2 completes a120-command observed free-motion integration with
causal recording and a passing continuing-physics stop. There are no final-panel
outcomes or learned contact/opening/hold/release successes. Finite observed-material,
rigid-interface and target-envelope components pass bounded independent checks;
both complete retained observed paths now pass with100ms contact confirmation,
explicit detection errors and bounded-window limits. The first frozen full learned
execution is queued; no panel outcome or learned contact success is inferred.

The [[topics/b1-policy-pilot|policy pilot infrastructure]] now implements causal RGB-D
pooling/history, partial A1/A2 recording/export, train-only model input, measured
solver feedback and bounded local admission without a hinge. Offline infrastructure
and both family model paths are exercised by the
[[experiments/b1-act-diffusion-a1a2-pilot|bounded physical-data pilot]]: three complete
teacher episodes, ACT/Diffusion × A1/A2 short CUDA training and eight checkpoint
round trips. ACT A2 development regression is retained. The
[[experiments/b1-pilot-offline-and-robot-dynamics|offline follow-up]] localizes it to
action components and push/hold, and identifies static-feature clipping; the train
median remains stronger on aggregate development action error. The isolated
[[experiments/b1-pilot-preprocessing|fixed-unit preprocessing pair]] removes saved
input clipping and reduces ACT A2 consumed-prefix means from 11.93 to 6.44 mm
and 1.590 to 0.900 deg at identical initialization/data/budget. Release and p95
errors worsen, and train median remains better in aggregate. V2 is opt-in; this
one-identity offline result does not qualify generalization or historical data
under the newer controller. The
[[experiments/b1-pilot-approach-release|targeted residual diagnosis]] verifies
causal joins and limited release exposure, without demonstrating broad input
aliasing. One 96-update phase-balanced v2 cell improves development release
median/p95 to 8.10/29.22 mm, but regresses hold and train push and fails the declared
guards. Retain v2 with the original uniform sampler; no further variant or larger
budget is launched. The [[experiments/b1-approach-controller-compatibility|frozen-controller approach comparison]]
verifies saved frames/timing and four robot-only CUDA cases. Right approach p95
improves 15.24→0.67 mm; left improves 66.13→28.78 mm but still fails 10 mm/5 deg
tracking, with 100 rate-scaled/129 position-clipped IK steps and actual speed
0.527 rad/s. All poses have static IK support; torque clipping is absent. Moving
stops at 2.1 s travel at most 2.57/4.62 mm, but only 108/3,384 residual estimates
remain inside the old calibration domain despite complete native feedback.
The [[experiments/b1-shared-learned-pilot|shared learned pilot]] now preserves two
bounded reference candidates. A joint-feasible FK path improves left p95 to
11.94 mm / 3.22 degrees but still fails 10 mm and clips 184 targets; a 0.05 rad
interior endpoint variant refuses left local IK before motion. Both candidates
are rejected. One subsequently authorized [[experiments/b1-joint-limit-aware-ik|constrained-IK cycle]]
passes numerical checks but still fails left translation p95 (21.36 mm) and actual
speed (0.529 rad/s) in the four-case physical comparison. Right traces are
unchanged and all stops pass; the original solver is restored. Both bounded
stages remain rejected. A later authorized [[experiments/b1-initial-placement-pilot|common-placement study]]
passes all four unloaded cases with the original solver: approach p95 2.19/5.07 mm,
peak speed 0.319/0.277 rad/s. The prospective pilot floor Y is 0.482327 m; the
corpus setup is preserved. Earlier placement trials did not collect learned data.
Three historical episodes numerically reconstruct A3/A4 supervisor labels at existing tolerances,
but observed first-command geometry/bootstrap and continuous sweep/response/load
support remain essential missing production dependencies. The subsequent
[[experiments/b1-pilot-contact-prerequisites|three complete teacher diagnostics]]
all fail: 46.09/35.48/32.62 degrees valid sustain, invalid holding or lost contact,
no valid pilot cycle. Right full-cycle joint speed reaches 4.47 rad/s. The first
allocation stops with nine cases unspent; three later camera diagnostics preserve
those failures, leaving six of twelve prerequisite cases. The user subsequently
[[experiments/b1-end-to-end-pilot-resume|resumes the end-to-end pilot]], explicitly
correcting the sensor/evaluator distinction. The 0.1 mm /0.02 N physical score is
not operational sensor resolution. A fingertip/identity/load/temporal monitor is
implemented with explicit uncertainty and unchanged independent scoring; its
physical calibration remains pending. The resumed teacher has valid sustained
push after preload regulation, but fixed-pose and prescribed-decay holding lose contact. Pressure-consistent
holding preserves the final contact window after a physically verified joint-parked
inspection, then fails measured release speed. The common six-second release
correction completes both train cycles: 51.26/52.54 degrees controlled opening,
valid final hold/release and 0.320/0.284 rad/s actual joint peaks. Independent
contact/pose audits agree; post-release continuing-physics stops pass. The first
fresh right nominal episode independently passes physical, raw-clock and FK
audits, reconstructs A3/A4 supervision and establishes its matched expert pair
at 51.2603 degrees. The fresh left nominal episode also independently passes,
with its matched pair at 52.5360 degrees; both historical denominators are
unchanged. The one-tick push/hold reference pause is now physically corrected:
all 180 hold ticks pass. All six fixed fresh attempts complete; four corrected
teacher records (two train, two development) independently pass physical/raw/FK
and supervisor-reconstruction checks. Earlier nominal records/scores remain
preserved and their command labels are explicitly excluded from imitation.

Prerequisite count is unrestricted. Cases 14–17 preserve the original loaded-stop
and speed-coverage failures and the later common corrections. Actual right
first-contact stopping now passes at 9.70 N peak: 0.071 mm travel over two seconds,
with no violations. Left first-contact stopping also passes (7.68 N peak,
0.184 mm travel). Four independent moving-door stop holdouts pass against the
continuous whole-link stop calibration; complete operational contact/sweep
qualification remains open. Common observed workspace rays
cover all 14 moving robot shapes at the first approach condition; full moving
coverage is not inferred. The independently demonstrated RTX pixel-center bias
is corrected at geometry ingress. An opt-in observed plane/edge/native-pair local
constraint separates normal direction from full material-pose uncertainty;
corrected native tracking and full contact metrics remain unqualified.

The resumed cumulative ceilings are 15 local GPU hours, 3,600 remote allocated
GPU seconds, 18,000 optimizer updates and 300 GiB; six fresh demonstrations remain
consumed. The four real ACT/Diffusion × A1/A2 cells complete 3,000 updates each on
the same two eligible corrected train episodes in Gilbreth job11970038. Cumulative
remote charge is 2,743 GPU-seconds and optimizer count12,164 including infrastructure.
Returned checkpoint hashes and all four workstation CUDA loads pass. Development
stays local and uses unchanged checkpoint weights/train statistics; remote metrics
remain unavailable. Offline imitation error is not task success.

The maintained rollout CLI now wires real observed perception, native robot load,
contact monitoring and actual proposed-motion producers; the first two seconds of
both complete retained train paths pass numerical whole-robot/contact/admission
checks with calibrated native load and explicit detection limits. ACT A2 also completes120 actual learned commands in a two-second
free-motion prerequisite with causal recording and a passing continuing-physics
stop. Numerical finite-material margins now cover both retained train paths;
the complete retained operational-path check passes, while actual learned contact
execution and all24 panel outcomes remain pending. The recorder retains consumed images and separate full-rate commands,
robot/native feedback and independent evaluator streams. Current local counters
and all preserved failures are in [[experiments/b1-end-to-end-pilot-resume]].

Full A3/A4 observed initialization is deferred and does not block A1/A2. Retained
raw data and reconstructed supervisor labels do not qualify observed A3/A4
execution or constitute the final matched dataset. The learned `rgbd-conv-v1`
foundation and previous synthetic Gilbreth/RTX checkpoint round trips remain
preserved. Actual learned control, the unchanged 24-outcome panel and at least one
single checkpoint completing both train doors remain required. No production,
20-run campaign, sealed-test opening or hardware claim is started.

The earlier dynamic follow-up preserves the 106.84 mm loaded-stop failure. The separate
[[experiments/b1-loaded-hold-and-stop|load-bearing controller correction]] verifies
robot-only CUDA hold/stop with task-local common impedance and a measured-anchor
command latch: maximum hold/loaded-stop travel becomes 5.18/5.98 mm versus
56.62/111.47 mm in its paired baseline. All 2,880 corrected samples remain available
under the unchanged calibration; 72 stopped commands refuse and explicit reset
re-enables motion. This is local 0/4 N ready-pose evidence, with no contact or
calibration/qualification promotion. Loaded rollout remains blocked by the remaining
6.0D fault/envelope/contact checks and observed sweep/response evidence. Required
6.0D checks plus local geometric support and independent 6.0E must pass before diagnostic interaction 6.0F.
Camera-enabled cycles exceed real-time 60 Hz. No final matched dataset, policy
contact execution or sealed-test evaluation was produced.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and common zero-yaw setup maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families, frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts/static evidence | Operational/local support, explicit selection, admission and feedback contracts; object fusion, ownership alternatives, finite-cover and relevant-space queries. Scans/interfaces do not admit motion. |
| 6.0C — Tracking | One persistent SAM3 session shared by inspection/tracking, bounded history and renewal-only depth gate; live/replay and serial evaluation share the coordinator. Useful accuracy improves offline, but material identity and 150 ms freshness remain unqualified. |
| 6.0D–H — Contact/qualification | Common benchmark impedance and latched measured-anchor stop pass local robot-only 0/4 N CUDA checks with the original calibration; earlier failures retained. Full 6.0D uncertainty/fault/domain/contact checks and local geometry/6.0E remain open. |
| 6.1 — Actions | ACT/Diffusion × A1–A4 observed-input/data/normalization/execution software maintained; qualified-provider integration and physical rollout validation pending. |
| 6.2–7 — Data/learning | Partial A1/A2 offline pilot demonstrated; final matched A1-A4 data and training matrix not started. No learned contact task result or sealed-test evaluation. |

## Next action and selected state

For the resumed pilot, the user authorizes 15 cumulative local GPU hours and
repairs within 3,600 remote allocated GPU seconds. All four train-only Gilbreth cells complete 3,000 updates each; returned final
checkpoint bytes match the remote files and remote round trips pass. All four also load on workstation CUDA and complete local offline development
inference with frozen weights/statistics. Learned simulation results remain pending.
Finish observed geometry/contact/full-robot admission, verify integrated execution
and then retain the fixed 24 outcomes. Case count and deferred A3/A4 work are not
blockers. See [[experiments/b1-end-to-end-pilot-resume]].

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
unresolved. Discuss same-input-equivalent cost changes first; no optimization or
quality variant is implemented. Dense replay does not verify operational loss.
Material correspondence/recovery, hardware/FK/floor bounds and all
contact/action qualification remain open. Earlier selected quality is historical,
without automatic transfer to the persistent inspection session.

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
