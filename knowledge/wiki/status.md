# Project Status

Current as of 2026-10-05. **6.0B is maintained; prior dynamic trackers are retired;
the CAD-free Point2Pose prototype is implemented but unqualified. Phase 6.0 and
6.1 remain open.** No training, new corpus or sealed-test evaluation was started.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and contact diagnostics. Common zero-yaw setup is maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts and static evidence | Operational/local support, explicit selection, admission and feedback interfaces retained. Static object fusion, ownership alternatives, hinge hypotheses, finite-cover and relevant-space queries maintained. Neither interfaces nor scans admit motion. |
| 6.0C — Panel tracking prototype | Official CUDA pipeline, FK compensation, panel-fixed zones, operational queues and serial 60 Hz evaluator implemented. A new 31–38 s camera-still baseline completes all 421 frames; primary integration support is 420/421, with opening error p95 2.64 mm / 0.295 degrees despite contact occlusion. Point drift and secondary candidate failures remain. The corrected moving-camera 4–10 s replay retains 301 lost primary poses. Full campaign stopped; historical operational useful availability remains 0%. No qualified runtime or loaded contact. |
| 6.0D–H — Contact and qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Action paths | ACT/Diffusion × A1–A4, observed-input/data contracts and execution/replay software maintained. Numerical/CUDA preparation is distinct from pending qualified-provider integration and physical validation. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Keep the full offline campaign stopped. Its original 4,216 completed rows,
502 unprocessed rows and 11 unstarted attempts remain preserved. Native loss/SDF
fixes and the failed moving-camera 4–10 s replay remain documented in
[[experiments/b1-perception-findings|B1 perception findings]].

One authorized 31–38 s replay of existing data now separates a stationary door
from its first opening with the camera fixed, retaining the same automatic primary
candidate and all 421 original timestamps. Primary support is 133/134 static rows
(seed rejected) and 287/287 opening rows; post-seed error p95 is 1.74 mm / 0.116
degrees static and 2.64 mm / 0.295 degrees opening. The contact is covered throughout
opening, but other leaf references maintain the pose. The well-contrasted,
distributed seed does not establish initialization as the cause of prior failures.
Pixel drift and depth sensitivity at relief edges are distinct demonstrated limits.
The native renewal criterion still counts extracted pairs despite declining
material consistency. No post-seed map growth occurs. Secondary candidates retain
72/26 lost frames and 16/19 flag reactivations; none of these returns meets both
original-material support and the fixed pose-error references in this evaluator.

Next, compare one minimal reference/renewal change on this exact window, candidate
and seed, without truth-driven selection. Use current measured, distributed
support before promoting new references; frozen poses cannot refresh support or
grow the map. Recovery must retrieve original material identity before publication
or map updates. The baseline establishes neither long-gap recovery nor larger-angle
opening reliability. No tracking modification, model tuning, performance work,
new acquisition or comparison trial followed this baseline.

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
