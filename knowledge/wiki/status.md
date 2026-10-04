# Project Status

Current as of 2026-10-04 UTC. **6.0B is maintained; prior dynamic trackers are retired;
the CAD-free Point2Pose prototype is implemented but unqualified. Phase 6.0 and
6.1 remain open.** No training, new corpus or sealed-test evaluation was started.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and contact diagnostics. Common zero-yaw setup is maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts and static evidence | Operational/local support, explicit selection, admission and feedback interfaces retained. Static object fusion, ownership alternatives, hinge hypotheses, finite-cover and relevant-space queries maintained. Neither interfaces nor scans admit motion. |
| 6.0C — Panel tracking prototype | Official CUDA pipeline, FK compensation, panel-fixed zones, operational queues and a separate serial 60 Hz evaluator implemented. Native loss/SDF defects are corrected and verified on one complete 4–10 s CUDA replay; primary integration coverage is 59/361 (16.34%), with 301 lost poses preserved. The full campaign stays stopped. The previous four operational replays and eight fresh Isaac cases fail useful availability at 0%. No qualified runtime or loaded contact. |
| 6.0D–H — Contact and qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Action paths | ACT/Diffusion × A1–A4, observed-input/data contracts and execution/replay software maintained. Numerical/CUDA preparation is distinct from pending qualified-provider integration and physical validation. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

Keep the full offline campaign stopped. Source `0adef47` processed 4,216 of the
20,108 planned full-replay frames before the user interrupted the first attempt;
502 rows in that recording and all 11 other attempts are retained as unprocessed
or unstarted. The [[experiments/b1-perception-findings|4–10 s drift diagnosis]]
verifies source timestamps, ROS optical/depth conventions, FK and composition.
The first candidate's native transform freezes for 162 frames with `lost=False`
while the camera moves; frontend drift already precedes world composition.
One subsequently authorized fresh CUDA diagnostic replay completes all 361 frames
of that same interval and reproduces all five native pose sequences exactly.
Telemetry confirms wrong point correspondences, no valid RANSAC cluster and a
previous-pose fallback that leaves `lost=False`. The primary has only 0–2 pairs
consistent with the correct motion during all 161 fallback frames, below the
unchanged five-inlier minimum; the original references remain depth-supported.
A separate defect returns a pre-SDF pose despite accepted refinement, with
statistics from the refined pose. Source `df36b92` corrects both behaviors; one
subsequent fresh CUDA replay verifies every source frame and native contract.
The primary now has 59/361 integration-supported rows and 301 explicit lost poses,
with no recovery from 5.25 through 10 s. Supported error p95 is 2.26 cm / 2.07 degrees
over only 59 non-seed samples; this conditional statistic does not establish
whole-sequence improvement. Saved-image diagnosis now localizes heterogeneous
pixel-identity errors on mostly low-contrast seed references, before registration;
same-time RGB edges and depth support oppose a shared acquisition delay. At 5.25 s,
only two of 17 selected pairs meet the unchanged 4 mm gate although all 30 seed
references remain depth-supported. Native renewal triggers are not reached while
supported; after loss, sampling/promotion are suppressed and 197 registration
attempts all return no cluster, with no jump-guard rejection. The next justified
work is observed reference-quality/renewal with geometric confirmation before map
growth, then a bounded comparison; alternative recovery success is unvalidated.
This diagnosis used only saved evidence, with no new inference. The full campaign
stays stopped; no automatic trial, model tuning or performance optimization follows.

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
