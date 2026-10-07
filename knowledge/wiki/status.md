# Project Status

Current as of 2026-10-07. **6.0B is maintained; prior dynamic trackers are retired;
the CAD-free Point2Pose prototype is implemented but unqualified. Phase 6.0 and
6.1 remain open.** No training, new corpus or sealed-test evaluation was started.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and contact diagnostics. Common zero-yaw setup is maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts and static evidence | Operational/local support, explicit selection, admission and feedback interfaces retained. Static object fusion, ownership alternatives, hinge hypotheses, finite-cover and relevant-space queries maintained. Neither interfaces nor scans admit motion. |
| 6.0C — Panel tracking prototype | Retained diagnostic reference: graph off, partial batch on, rollback off, five inliers/4 mm, 120 active references per candidate; observed geometry and vectorized SDF Jacobian retained. Four full CUDA replays reject selected-only registration: accepted precision changes 85.75→96.32% light / 86.94→62.48% nominal, despite lower latency/memory. Both fresh all-registration references reproduce preserved poses exactly. The option remains disabled diagnostic code; provider remains local/invalid. Physical pre-A4 initialization, material/ownership, complete uncertainty, 150 ms freshness and loaded contact remain unqualified; full campaign stopped. |
| 6.0D–H — Contact and qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Action paths | ACT/Diffusion × A1–A4, observed-input/data contracts and execution/replay software maintained. Numerical/CUDA preparation is distinct from pending qualified-provider integration and physical validation. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Use the [[topics/pre-a4-initialization-and-hinge-budget|shared pre-A4 initialization specification]]
for every A1–A4 × ACT/Diffusion condition before policy dispatch. Stationary
observations support local surface/closed rotation but contain no supported
physical pin in these scans. Direct pin evidence or a separately admitted bounded
leaf-motion diagnostic must complete the physical reference before A4 approach;
the pre-articulation motion/admission path is absent. No movement/contact is
executed or admitted by this specification.

Close authored simulation floor/up and measured camera/FK input accounting, then
validate uncertainty against actual proposed command/sweep/stop margins. These
missing interface inputs are distinct from the conservative correspondence
envelope and evaluator-measured hinge errors; replacing a bound by p95 or pursuing
an arbitrary sub-centimeter bound is unjustified. Operational A4 freezes each
segment's admitted frame, so a fixed local endpoint retains the full origin
error at approach completion. Actual A4 proposals/contact inputs and hardware
calibration/load qualification remain absent.

The [[experiments/p2p-selected-registration|single selected-registration comparison]]
completes all four 2,858-frame CUDA attempts from original initialization without
inference retries/failures. It saves 31.43%/23.59% median request time, but loses
699 correct nominal poses and reduces its terminal 518-frame correct count from
325 to 30 without native loss triggering recovery. Retain all-registration as
the fixed reference and leave the new option disabled. Complete measured
observation p95 remains 578/619 ms even with the option; zero requests meet
150 ms. Queue estimates do not validate dropped-frame tracking or live admission.

## Preserved investigation evidence

Keep the full offline campaign stopped. Its original 4,216 completed rows,
502 unprocessed rows and 11 unstarted attempts remain preserved. Native loss/SDF
fixes and the failed moving-camera 4–10 s replay remain documented in
[[experiments/b1-perception-findings|B1 perception findings]].

The complete fixed-camera recordings retain all 2,858 original frames per condition,
31–78.6167 s, through 63.72 degrees; the configured 90.7-degree joint limit remains
unrecorded. Camera, motion and timestamps match across conditions, but automatic
light/nominal seeds differ. This is not causal lighting attribution.

The rejected unbounded final-inlier variant and its CUDA OOM/TSDF guard remain
preserved with all 1,506/2,239 unavailable process rows. Saved growth and a minimal
19-frame CUDA diagnostic measure 1.875 MiB of causal state per active reference.
The bounded renewal policy is now implemented at `2b127b4`, following telemetry
`94429d9`: 120 active references per candidate, confirmed gradual replacement,
independent compact CUDA storage and unreused historical IDs. Models, numeric
gates, candidate/point selection, graph, TSDF and native lost-state suppression
are unchanged. Every candidate remains audited; historical graph/keyframe/TSDF
growth is separate from the active-reference cap.

One fresh CUDA attempt per saved condition completes all 5,716 frames without
missing rows or process failure. Maximum all-candidate active references are
512/654, versus unbounded 3,025/2,160; historical references remain 2,094/1,830.
Compared at identical timestamps with baseline, correct supported poses gain/lose
417/443 light (net -26) and 788/99 nominal (net +689). Inaccurate accepted poses
increase from 35/88 to 541/99. All finite/lost/rejected errors remain scored;
counting every finite pose separately gives net -518/+59 correct poses. Nominal
still ends lost for 60 frames from 77.6333 s, with final finite error 22.06 mm /
3.14 degrees. Completion or a finite retained pose does not establish current support.

At 45.6667 s nominal, frontend error is 2.92 mm / 0.58 degrees, but the graph
publishes an accepted 178.61 mm / 32.73 degree pose. Only one of 22 declared inliers
remains inlier; ten recomputed pairs fit the published pose/current map. Therefore
metric self-consistency alone does not verify pose or material. Bounded flag
returns are 25/43, correct-pose returns 13/25, and strict historical-anchor returns
one/one; original seed-only returns are zero. Anchor ownership remains hypothetical.
The visibility guard blocks retirement on missing native visibility, depth or loss,
but independent audits flag 66/62 retired references as nearer-depth/self-occlusion
proxies despite native visibility; physical occlusion protection is not established.

The bounded policy is an unqualified prototype, not an adopted tracking solution.
Saved consumer diagnosis measures zone-point/normal drift and conditional effects
at recorded teacher targets, without creating missing A3/A4 inputs. Accepted target
p95 is 15.38→16.66 mm light and 11.91→7.28 mm nominal; the nominal spike reaches
116.23 mm/32.73 degrees. Actual static A3 frame errors and A4 predictions/admissions
are unavailable. Nominal's graph changes landmarks and rebuilds TSDF; its next
accurate pose does not restore the map. Light deterioration at 52.1333 s already
exists before SDF, without a same-frame graph or replacement. No new inference
was needed; see [[experiments/point2pose-consumer-impact-and-failure-windows|consumer impact and isolated windows]].

The complete graph-off baseline (`0e6185f`) removes the old graph event spikes,
but retains 2,569/2,004 correct primary poses, 238/58 inaccurate acceptances and
49/794 native losses. Nominal ends lost for 472 frames from 70.7667 s. Neither
condition reaches 95% correct availability; graph-on remains the existing
prototype default. See
[[experiments/point2pose-bounded-global-graph-ablation|the preserved graph comparison]].

Three independent follow-up comparisons each complete both original 2,858-frame
CUDA openings, with identical initialization/source schedules, one attempt and
no process failure/retry. Partial batch gains 198/631 correct rows but loses
317/151; its nominal benefit is real, while accepted precision falls to
85.75%/86.94%. The option remains diagnostic and disabled by default. Six mm
lowers precision to 52.33%/53.17%. Promotion geometry loses 1,022/250 correct poses
and lengthens nominal terminal loss to 566 frames (9.433 sampled seconds).
Both flags are rejected as common corrections. The cap is verified on actual
CUDA storages; retained masks, keyframes and historical maps remain outside it.

Saved pairs distinguish missed RANSAC sampling from correct hypotheses lost in
clustering. At nominal 63.4667 s supported correct 4 mm fits exist but are not
sampled; nearby calls already generate them without retaining a correct cluster.
Pixel rounding worsens median/p95 pair errors. Actual SAM2 prompts are five
raster-sampled positive points, distinct from interior preservation checks;
mask expansion does not establish wrong leaf membership. Hardware calibration,
whole-mask ownership and the separate causal prompt effect remain unverified.
Keep 4 mm, native pixel lifting and original prompts; do not combine changes or replace P2P/policies.

The next isolated own-seed rollback (`96c0943`) completes both original CUDA
recordings without failures/retries. It retains five inliers at 4 mm, coherent
returned-pose statistics and original initialization. Correct accepted poses
within 10/15/20 mm and 5 degrees change from 2,569/2,776/2,805 to
2,183/2,547/2,564 light, and from 2,004/2,047/2,052 to 2,215/2,285/2,308 nominal.
Wrong 10 mm acceptances grow from 238/58 to 386/96. Light's accepted peak grows
to 63.56 mm and longest correct absence to 0.967 s; nominal's absence decreases
to 5.017 s and all-finite point p95 to 22.34 mm. Rollback remains diagnostic,
disabled by default. The failed common result does not admit a partial-batch
combination. Saved refit residuals do not justify an uncalibrated tolerance;
no global threshold widening or four-inlier variant is tested.

The refit comparison proposed testing recovery using current support of the
previous pose and observed candidate consistency, while keeping normal jump rejection and five-inlier/4 mm
gates. Original nominal jump refusals include 24 correct 10 mm candidates but
also wrong hypotheses; blanket bypass is unjustified. Late saved pair families
can lack any supported fit even within 20 mm/5 degrees; that correspondence
problem is distinct from a restrictive guard. This is a proposal, with no
additional correction implemented. Preserve every baseline/failure and keep
truth evaluator-only; no acquisition, policy change, training, full campaign
resume or contact admission follows. See
[[experiments/point2pose-refit-hypotheses|paired results, retained wrong seeds, guards and tail diagnosis]].

The bounded [[experiments/p2p-a3-a4-integration-diagnosis|P2P to A3/A4 diagnosis]]
uses graph-off, partial-only and no rollback at five inliers/4 mm. Targeted saved
images and evaluator geometry verify the primary closed leaf seeds and an observed
static rotation within 0.051 degrees on these two recordings. This does not admit
control: the provider remains local/invalid. The preserved diagnosis first fitted
hinges at 51.3667/39.55 s. The subsequent
[[experiments/p2p-observed-geometry-and-sdf-jacobian|separate observed-geometry intervention]]
freezes the observed static reference, independently timestamps current angle,
uses uncertainty-aware vertical consensus and audits origin-bound propagation.
First fits are now 40.4833/39.9667 s; common-row origin p95 improves 31.23 to
2.60 mm light and 19.92 to 18.97 mm nominal. Nominal loses 25 early hinge rows.
No origin bound reaches 1 cm; complete diagnostic calibration/floor inputs are
missing, although floor/up are authored and rendered camera/FK agreement is now
measured in the shared budget audit.
Angle source age is corrected, but saved publication p95 remains 784/975 ms,
with zero fresh angles under 150 ms. A3/A4 and operational admission remain
unchanged. The shared pre-A4 protocol now specifies the required initialization
and separate pre-articulation admission path; neither is implemented or qualified.
The separate SDF Jacobian vectorization is retained: 2,640 paired CUDA
refinements across 482 original frames preserve poses/decisions exactly and cut
median refinement cost by 3.35/3.29 times. Instrumented pipeline estimates remain
562/665 ms p95; end-to-end 150 ms freshness is not established.

The historical operational source `7460963` still fails all four replay and eight
observer useful-availability gates. Live latency p95 remains 0.784–1.755 s.
Offline latency is reported separately and never changes accuracy or coverage;
it does not repair those historical operational failures.
Use the fixed 95% useful availability and p95 1 cm/5 degree criteria; a provider
that always returns unavailable fails. Preserve every failed attempt and modify
Point2Pose only for evidenced limitations.

Tracking alone establishes neither ownership, a physical hinge nor safe contact.
D must verify continuous relevant sweeps, feedback semantics, response/load
uncertainty, compliance and stop bounds. E must independently score the selected
material contact. Pilot offline and feedback/stop gates precede bounded pilot
interaction; all train/development offline gates precede qualification dynamics.
Both qualification groups and explicit release compatibility precede 6.1 handoff.
The 1 cm / 5 degree and per-door 95% coverage/precision gates are unchanged.

## Resources and material limits

All 50 `engineering-v2` episodes retain original calibration/metadata for static
work and future common train/development qualification. Superseded campaigns,
derived caches, temporary experiment scripts and repeated payloads were removed;
headers, corrected per-door results, significant failures and decisive images remain
under `outputs/b1/perception/evidence/`. Its `cleanup.json` records removals and
relocations. Removed ignored payloads cannot be recovered from Git.
Selected GroundingDINO/SAM3/DINOv3 resources and their image-worker overlay remain.
Prepared assets, corpus/splits, qualification evidence and `knowledge/raw/` are intact.

Static scans preserve ambiguous whole-object ownership and unobserved physical
axes. Human confirmation assigns only the displayed pilot regions to the leaf;
endpoint IK and free-space balls do not validate a path or loaded control.
Current recordings lack torque and the observed monitor rejects loaded execution.
Depth is ideal; camera mounting and hardware feedback/stopping are unvalidated.
Train contains one 15-door residential family and development one right-hand door;
report per-door/handedness results without population claims. The sealed test stays closed.

[[experiments/b1-perception-findings|Historical findings]] retain train fitting with
0/6 development qualification, corrected component scores, interrupted campaigns
and the failed material-tracking exits. They do not justify another training run.
B0 remains historical: the saturated 576-rollout study selected no winner and its
219.95 N event remains under review.

Supported entry points are documented in the root/model READMEs: runtime and
corpus verification, intake/preparation/qualification, recording, frozen image
smoke, static scan diagnosis and Point2Pose smoke/replay/observer/offline diagnostics.
No estimator training or tracker-comparison CLI is supplied. See
[[topics/shared-door-perception|Perception]], [[topics/system-architecture|Architecture]],
[[topics/purdue-b1-robot-and-contact|Purdue contract]] and
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]].
