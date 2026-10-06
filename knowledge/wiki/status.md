# Project Status

Current as of 2026-10-06. **6.0B is maintained; prior dynamic trackers are retired;
the CAD-free Point2Pose prototype is implemented but unqualified. Phase 6.0 and
6.1 remain open.** No training, new corpus or sealed-test evaluation was started.

| Area | Current state |
|---|---|
| 4 — Purdue runtime | Seven-joint A1, full-pose A2/A3, WSG32/UMI v1, synchronized head RGB-D/proprioception and contact diagnostics. Common zero-yaw setup is maintained. |
| 5 — Corpus | Complete: 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| 6.0A/B — Contracts and static evidence | Operational/local support, explicit selection, admission and feedback interfaces retained. Static object fusion, ownership alternatives, hinge hypotheses, finite-cover and relevant-space queries maintained. Neither interfaces nor scans admit motion. |
| 6.0C — Panel tracking prototype | Official CUDA/FK/zone/offline runtime. Bounded inlier renewal and one native graph-off ablation each complete both original 2,858-frame openings. Graph-off correct available poses are 2,569 light / 2,004 nominal, versus graph-on 2,250/2,183; inaccurate acceptances 238/58 and losses 49/794. Nominal terminal loss grows 60→472 frames. GPU peaks 7.78/9.61 GiB. Graph-on remains default; no common replacement adopted. Material/ownership, 150 ms freshness, hardware and loaded contact remain unqualified; full campaign stopped. |
| 6.0D–H — Contact and qualification | Feedback/load/stop, independent chosen-contact scoring, bounded interaction and complete qualification remain future work. |
| 6.1 — Action paths | ACT/Diffusion × A1–A4, observed-input/data contracts and execution/replay software maintained. Numerical/CUDA preparation is distinct from pending qualified-provider integration and physical validation. |
| 6.2–7 — Policy data and learning | Not started. No matched B1 policy dataset or learned-policy result. |

## Next action

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

One subsequent full-sequence native graph-off variant (`0e6185f`) changes only
`use_key_frame_graph`; original initialization, models, point selection, numeric
gates, 120-reference budget, replacement/promotion and SDF rules match. Both fresh
CUDA attempts complete all 5,716 original frames without retries or process failure.
Correct available poses change +319 light / -179 nominal versus bounded graph-on.
Light accepted point/normal/conditional-target p95 becomes 11.51 mm/0.05 degrees/
13.95 mm, versus 15.43/1.20/16.66. Nominal becomes 8.15/0.05/10.54, versus
9.28/0.40/7.28: conditional target p95 worsens despite smaller accepted zone errors.

The original event errors fall to 0.95 mm light and 2.23 mm nominal, with no
graph-induced confirmed-landmark revisions. Native pending-point fusion and TSDF
integration continue. However, nominal loses 794 rows and ends continuously lost
for 472 frames from 70.7667 s; finite terminal error is 115.94 mm/7.95 degrees.
All-finite nominal point/normal/conditional-target p95 grows to 100.30 mm/7.69
degrees/34.90 mm. Primary references remain 92 after the last birth at 56.2167 s;
whole 30-point renewals are deferred by the unchanged 120 cap. Late support drops
to four pairs while map/TSDF extent remains fixed. Strict historical material-plus-
pose returns are zero in both conditions; secondary light false acceptances rise.
Neither condition reaches 95% correct available poses. No unified replacement is
adopted; the default graph stays enabled in the unqualified prototype.

Next diagnose the saved late nominal correspondence/cohort and capacity history,
and graph-on publication/landmark consistency at the spike, before another single
controlled intervention. Preserve both baselines, unbounded failures and complete
ablations. Ground truth stays evaluator-only; no gate relaxation, favorable retry,
acquisition, training, campaign resume or contact admission follows. See
[[experiments/point2pose-bounded-global-graph-ablation|full graph ablation and limits]].

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
