# Project Status

Current as of 2026-10-05. **6.0B is maintained; prior dynamic trackers are retired;
the CAD-free Point2Pose prototype is implemented but unqualified. Phase 6.0 and
6.1 remain open.** No training, new corpus or sealed-test evaluation was started.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and contact diagnostics. Common zero-yaw setup is maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts and static evidence | Operational/local support, explicit selection, admission and feedback interfaces retained. Static object fusion, ownership alternatives, hinge hypotheses, finite-cover and relevant-space queries maintained. Neither interfaces nor scans admit motion. |
| 6.0C — Panel tracking prototype | Official CUDA pipeline, FK compensation, panel-fixed zones, operational queues and serial evaluator implemented. Both full recorded fixed-camera openings complete at 60 Hz, 2,858 frames each through 63.72 degrees. Primary integration coverage is 80.86% light / 55.35% nominal; conditional error p95 is 7.79 mm / 1.44 degrees and 10.20 mm / 1.24 degrees. Nominal has almost no support above 60 degrees. Full campaign stopped; historical operational useful availability remains 0%. No qualified runtime or loaded contact. |
| 6.0D–H — Contact and qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Action paths | ACT/Diffusion × A1–A4, observed-input/data contracts and execution/replay software maintained. Numerical/CUDA preparation is distinct from pending qualified-provider integration and physical validation. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Keep the full offline campaign stopped. Its original 4,216 completed rows,
502 unprocessed rows and 11 unstarted attempts remain preserved. Native loss/SDF
fixes and the failed moving-camera 4–10 s replay remain documented in
[[experiments/b1-perception-findings|B1 perception findings]].

The short 31–38 s fixed-camera baseline is now extended through the last recorded
frame in both light and nominal, without any tracking/model/threshold change or
acquisition. All 5,716 frames complete; the light prefix reproduces all 421 prior
poses/decisions exactly. The prepared joint permits 90.7 degrees, but these recordings
reach only 63.72 degrees at 78.6167 s after robot-margin stopping and release;
unrecorded angles are not evaluated. Camera, motion and timestamps match between
conditions, but only seven of 30 automatic primary seed pixels are identical.

Primary lost rows are 531 light / 1,274 nominal. There are 206/216 native flag
reactivations, distinct from proven original-material recovery. Native renewal now
occurs at larger views: 90/60 new primary IDs and three/two post-seed graph updates.
No new IDs/keyframes appear while lost. Original map coordinates move at most
1.78/1.90 mm through graph refinement; this is not automatic evidence of corruption.
Above 60 degrees, primary support is 175/275 light but only 2/275 nominal; nominal
ends with an unrecovered 4.4167 s gap and 49.83 mm finite-pose error. Visually good
masks and small conditional pose errors do not establish continuous material
tracking. Full finite/lost/rejected errors and all secondary candidates are retained.

The authorized final-inlier renewal experiment is rejected and reverted. One CUDA
attempt per saved condition completed only 1,352 light / 619 nominal frames before
CUDA OOM / the unchanged TSDF memory-budget guard. All 2,858 scheduled rows per
condition and both terminal errors are retained. Initial support improves, but
light accepts 82 inaccurate poses in its completed prefix versus zero in the same
baseline prefix; rapid reference growth prevents either opening from completing.
The active source again uses the baseline extracted-pair renewal criterion.

Diagnose reference growth and post-graph pose/support consistency from these saved
failures before another change. At 47.1667 s light, the graph replaces a frontend
7.61 mm / 0.47 degree pose with a published 21.16 mm / 14.02 degree pose while
retaining frontend inlier statistics. Original timestamps, all finite errors,
baseline/failure evidence and unchanged models/gates remain preserved. The variant
does not reach nominal's final lost interval or demonstrate material recovery.
Current measured support is required before map growth; frozen poses cannot refresh it.
Neither the strict recovery audit nor first-observed anchors for new points prove
leaf ownership, mechanical-limit coverage or loaded contact. The full campaign,
performance optimization and new acquisition remain stopped. See
[[experiments/b1-perception-findings|recorded-opening evidence and rejected renewal experiment]].

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
