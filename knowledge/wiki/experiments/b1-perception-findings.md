# B1 Perception Findings and Direction

## Decision and evidence boundary

Static 6.0B retains GroundingDINO, native SAM3 and calibrated RGB-D.
A later consumer audit removed unused DINOv3 extraction; the historical feature
comparison below remains unchanged.
Earlier full-state estimators and custom trackers failed qualification and are
retired. CAD-free Point2Pose is implemented but unqualified; its selected recipe
and subsequent investigations have dedicated pages linked below. SAM2 remains
an active operational/baseline dependency despite retirement of the old component
comparison. See [[../topics/shared-door-perception|runtime behavior]] and
[[../implementation_phases/phase-6-0-operational-perception-and-contact|future requirements]].

No estimator qualified all six development doors. The original 1 cm/5 degree and
per-door 95% coverage/accepted-precision gates include missing/rejected states;
empty accepted sets fail. All model experiments used RTX 4090. The sealed test
stayed closed. Train fitting, static planes and completed execution are distinct
from qualification, ownership and loaded-contact admission.

Unless named otherwise, early evidence is under
`outputs/evidence/b1/perception/<run-name>/`, including later Point2Pose roots. Each root retains protocols, source/configuration,
corrected scores, failures and decisive images. `e1f98a6`, `2a38858` and pre-cleanup
`57ad483` preserve detailed historical source. October 2 explicitly removed some
ignored payloads, recorded in `evidence/cleanup.json`; those are not Git-recoverable.
October 8 removes no ignored payloads, including diagnostic scripts and images.

## Static 6.0B and reviewed contact regions

Four frozen-model scans read 1,501 observations each, chronologically through 25 s
on the two train pilots in nominal/light. Annotations and later manipulation images
did not enter inference. Pending cues were released at measured completion without
reading another observation. The initial `operational-scan-02` retained competing
objects, unresolved surrounding support and no physical axis (`01b187b`, `6c87045`).

The targeted re-audit (`9f511da`, `3001830`) demonstrated extraction/association loss:
partial borders contracted measured extents, global sampling dropped rare boundaries,
three-plane extraction omitted smaller faces, and first-only frame membership lost
internal seams. Corrections preserve dense support before sampling, re-sample residual
planes, restore only enclosed valid-depth plane inliers and check every same-frame/mask
component pair. Observed separating sides can conditionally exclude surrounding support.
No tolerances or physical gates changed. Twelve decisive observations recover
876–34,849 measured pixels each; more components do not establish more physical objects.

Corrected saved-cue final reconstructions still report `ambiguous_object_ownership`,
zero physical axes and no uniquely fixed surrounding support. Main candidates retain
32/36 white and 38/37 dark material surfaces (light/nominal). Exact alternatives,
rejections and source references are in each
`operational-scan-diagnosis-01/final/<asset>/<condition>/final-report.json`.
Assembly of saved cues does not qualify causal runtime timing.

Human review confirms only the displayed regions: A/C are fixed frame and B is the
leaf bottom; another door can have a bottom frame. This never became an inference
label or a universal rule. Static association remains ambiguous at whole-object scale.

The initial finite-face claim was invalid: each closed collision extremum has nine
repeated vertices but only two distinct positions on a 25.8 mm line. Initial
17/19/14/14 patch counts cannot certify two finite finger faces. `footprint_support`
rejects this degeneracy. The subsequent common correction (`41e4fe6`, `9aa29bd`)
clips/projects canonical distal mesh bands within the unchanged 3 mm classification
tolerance into two finite covers, about 25.8 by 3.4 mm. Assets, tool calibration,
expert extrema/contact rule and Phase 5 results remain unchanged. These covers
approximate geometry, not measured pad area or compliance.

`operational-contact-readiness-01` verifies two human-confirmed local leaf regions
on existing observations only, with no model inference or motion:

| Pilot / observed region | World tangent coordinates | Supporting frames, nominal / light | Best simultaneous two-cover clearance, nominal / light | Maximum observed plane discrepancy, nominal / light |
|---|---|---|---|---|
| White `animated-door-1-88abf40`, upper inset | y=-0.10 m, z=1.50 m | 49 / 48 | 103.7 / 104.4 mm | 2.49 / 1.91 mm |
| Dark `door-2738468b94d74c5f`, central panel | y=0 m, z=1.10 m | 29 / 73 | 315.7 / 316.0 mm | 0.38 / 0.34 mm |

Every verified cover pixel has valid depth and extracted plane membership. Original
SAM membership can fall to 48.7%/90.3% white and 65.1%/100% dark nominal/light;
only enclosed measured inliers are restored. Camera baselines are 111–129 mm.
This is local registration/support, not dynamic rigid membership or safety.
White full-cover support ends at 13.2/13.6 s (light/nominal), even while other areas
of the plane remain visible. Contact must retain its own support time.

CUDA IK succeeds at contact and 30 mm precontact. A 10 mm precontact ball is
observed free, with 3.8–4.6 mm residual ray clearance. These checks do not cover
the hand/arm path, uncertainty, stopping, load or continuous control.

## Retired estimators and trackers

The early SAM3 hotstart comparison retained 20/20 masks with box plus `door`
versus 0/20 with box alone on the same first 20 nominal images per pilot. Native
association removed unmatched box-only tracks despite an initial score near 0.988.
This was retrospective prefix evidence, not full-video/light or material validation;
its worker/orchestration/decord were retired before the current frontend.

The custom material tracker (`2a38858`; source `a3b4587` through `c8313f3`) initialized
local geometry in four complete replays and four fresh parked-arm live schedules,
but produced **zero selected dynamic-support, supported-axis or reacquisition
ticks**. White/dark replays contained 4,718/5,336 observations; live schedules
contained 2,013 each. Independent raw-depth support returned for 121/121 hold
ticks without restoring material support. Explicit contacts never switched on loss.
Endpoint IK/clearance and joint-margin tradeoffs did not qualify a contact ranking
or path. Evidence: `operational-tracking-03`, `operational-tracking-live-02`, with
initial interrupted attempts retained. The corrected neck discretization allowance
was 0.1 + 0.006670 rad (observed peak 0.104564); tool drift was 0.150 mm. It did not
relax dynamic freshness. Stationary live doors did not test articulation.

The geometric full-state prototype (`7468e0a`, reporting `809f575`;
`geometric-pilot-06`) had **0% complete coverage/accepted precision in all four
cases**. Finite rejected panel estimates covered 15.9/37.9% of left/right
manipulation; hinge estimates covered 7.3/21.7%. Incomplete relief support and
fixed-frame inclusion defeated whole-leaf selection; adding points sometimes
worsened dimensions. Semantic p95 200–218 ms exceeded 150 ms. Earlier thin-floor
leakage, replacement/uncertainty failure (`geometric-pilot-04/05`) and both partial
extended evaluations (`ac89744`, `a41c002`, the latter stopped by the user) remain
recorded. Rejected finite estimates and static round trips are not qualified states.

### Learned-state and observability diagnosis

All three custom-estimator runs failed every development door. Essential source
and outcomes are retained in the original per-door reports:

| Experiment | Result and decision |
|---|---|
| Run-01, `522f1ba`, diagnosis `e160f1f` | Stopped at epoch 28; best epoch 18 development position/orientation p95 9.01 cm/13.93 degrees, 0% confidence-valid coverage. Confidence gradients were 22.6× contact gradients; labels/cache alignment were verified. Loss imbalance was demonstrated, generalization was not. |
| Two-door corrected losses, `796797c`, `1bd633e`, `34c2660` | Contact-only fitting left hinge/local coordinates severely wrong. Complete-state supervision reached 98.25/96.70% joint train geometry; this was an overfit check, not held-out evidence. |
| Run-02, `68dc0ec` | Last epoch 15 passed 5/19 train, 0/6 development; development contact p95 8.34–32.09 cm despite 100% confidence coverage. Longer training was not established as a remedy. |
| XYZ/static-memory and confidence probes, `7e2bf56` | Pilots fitted at 7.41/9.93 mm contact p95; `refine-02` fitted 19 train doors at 2.29–4.00 mm. `refine-01` was train-only despite its initial development claim. Independent confidence fitting still accepted inaccurate development geometry, 7.41–32.32 cm p95 and 0% precision. |
| Run-03, `d833b89`, audit `0ea075f` | Last epoch 15 passed 19/19 train with 1.86–3.44 mm contact p95, but 0/6 development with 9.88–54.16 cm. Best epoch 1 also failed all development doors (5.42–51.50 cm). |

At `68dc0ec`/`a48e06f`, all 31,654 cached camera views excluded the top face and
four-frame/0.3 s memory could not retain the scan. Counterfactual RGB/depth/proprio
swaps changed contact p95 by 60.3–72.0/0.76–0.98/2.71–2.82 mm; these are sensitivity
checks, not causal modality percentages. Identical camera poses made pose swaps
uninformative; confidence AUROC 0.570 did not establish rejection. Privileged
substitutions remained diagnostic only.

The 25 s scan and simulated 10-degree mount (`a4ad984`) added seven views at
4/7/10/15/18/21/25 s. Left/right observed top fractions were 57.4/52.5%, bottom
71.3/54.5%; tallest-development inspection reached 24.8/56.4%. These were 101-sample
boundary checks with 5×5 depth neighborhoods/3 cm agreement, not estimator gates.
FK agreed below 0.001 mm/0.000002 rad, but hardware mounting remained unvalidated.
Inspection-only data had no expert hold/release. Calibrated XYZ/static memory and
raw/cache parity improved train fitting without repairing development failure.

## Corrected pretrained component screening

The protocol fixed 800 images and 450 manipulation samples (342 train/108 development).
Inference used predicted masks/boxes and calibrated observations, without asset,
hinge or dimensions truth. GroundingDINO used `door.` at 0.30/0.25 thresholds;
all segmenters used the same metric plane/bounds scoring.

The original scripts confused counters starting at eight with HDF5 row indices,
introducing an eight-row/0.133 s offset. Fifty labeled manipulation rows were actually
release. Corrected repeats (`e1f98a6`) preserved settings and supersede initial
component scores; run-03 and correctly mapped DINO probes were unaffected:

| Method | Split | Available manipulation planes | Normal p95 | Surface-distance p95 | Normal <=5 degrees AND distance <=1 cm, including unavailable as failures |
|---|---|---:|---:|---:|---:|
| Depth geometry | train | 238/342 | 21.17 degrees | 7.83 cm | 56.1% |
| Depth geometry | development | 108/108 | 2.39 degrees | 1.11 cm | 92.6% |
| GroundingDINO + SAM 2 | train | 342/342 | 88.99 degrees | 22.21 cm | 67.0% |
| GroundingDINO + SAM 2 | development | 108/108 | 88.43 degrees | 25.53 cm | 85.2% |
| SAM 3, text | train | 192/342 | 6.10 degrees | 3.38 cm | 43.9% |
| SAM 3, text | development | 56/108 | 1.87 degrees | 1.26 cm | 38.9% |
| SAM 3, predicted box | train | 342/342 | 88.95 degrees | 22.20 cm | 69.9% |
| SAM 3, predicted box | development | 108/108 | 1.79 degrees | 1.26 cm | 89.8% |

SAM2/boxed SAM3 shared 89 development successes, with 3 SAM2-only, 8 SAM3-only and
8 shared failures. The improvement from 92/108 to 97/108 is sample-specific.
Industrial-003 retained severe per-door p95 89.59 degrees/57.78 cm, hidden by pooled
scores. Closed-scan height error remained 9.1–41.0 cm. No plane/mask result supplied
a hinge, contact state or qualified confidence. Median times about 8 ms depth,
83 ms SAM3 text and 85 ms boxed SAM3 excluded cached detection, versus 137 ms SAM2;
these are not comparable optimized end-to-end latencies.

The CAP-Net adapter produced 107/450 fits and 0% development success. It omitted
published multi-instance clustering and used a single-door NOCS fit; this is not a
published-protocol accuracy claim. It was not rerun after alignment correction.
`evidence/comparison-01/` retains aligned/original protocols, frame-index audit,
per-door scores and `sam3-aligned-failures.jpg`.

## Matched DINO diagnostic

Frozen backbones used identical 224-pixel whole-image letterboxing, normalization,
384-channel metric heads, seven initial/four recent views and seeds 6100–6102;
native patch grids differed and CLS/register tokens were excluded. Confidence stayed
frozen/unqualified. A common prespecified 6,000-update extension followed 1,200 updates,
with 3e-4/9e-5/2.7e-5 learning rates in 2,000-update blocks. No best-seed selection
or development early stopping occurred:

| Backbone | Train doors passing geometry in each seed | Development doors passing in each seed | Train mean per-door contact-position p95, seed range | Development mean per-door contact-position p95, seed range | Development pooled contact-position p95, seed range |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 19/19 | 0/6 | 2.28–2.42 mm | 22.98–27.52 cm | 52.66–54.60 cm |
| DINOv3 ViT-S/16 | 19/19 | 0/6 | 1.30–2.57 mm | 24.21–27.58 cm | 45.40–47.07 cm |

DINOv3 improved pooled tails without changing held-out passing-door counts or
consistently improving per-door means. A backbone swap did not repair this tested
architecture; it does not reject DINOv3 as a reusable image-feature component.

## Historical cleanup and initial Point2Pose runtime

October 2 retained all 50 engineering-v2 episodes and selected frozen models,
saved 53 headers from retired campaigns and consolidated 20 completed runs.
`evidence/cleanup.json` records the removals/relocations; retained results do not
promise full executable reproduction of those older experiments. Frozen image
smoke/four static scans in `evidence/cleanup-validation-20261002` validated only
maintained static behavior.

The October 3 CAD-free prototype used pinned upstream
`51856226610df75e5c06e8de545bd27f7c4ba99c`, isolated model dependencies and CUDA TSDF.
Source `7460963` established preparation before acquisition, bounded queues,
complete worker reset and immutable automatic selection. Native five positive
SAM2 prompts were retained after rejecting added border/negative-prompt controls.
An 11-frame smoke had p95 171 ms and 9 timely outputs; concurrent Isaac smoke had
p95 265 ms and 0/12 timely. Neither measured accuracy qualification.

### Frozen pilot replay

`point2pose-replay-04` uses source commit `7460963`, the same recipe for all four
existing recordings and chronological 20 Hz input. All 6,704 sampled rows are
retained, including initialization, unavailable outputs and tracking losses.
The 4,630 sufficient-reference rows are defined by projected source geometry and
measured depth independently of tracker acceptance. Every case fails useful
availability; all useful-error sets and sustained support intervals are empty.

| Pilot / condition | Full / observable rows | Useful output | Complete latency p95 | Raw capture position / rotation p95 | Final state |
|---|---:|---:|---:|---:|---|
| Right / light | 1,573 / 1,442 | 0% | 0.651 s | 30.08 cm / 42.52 degrees | Tracking lost |
| Right / nominal | 1,573 / 1,442 | 0% | 0.651 s | 56.47 cm / 52.53 degrees | Tracked, stale |
| Left / light | 1,779 / 161 | 0% | 2.125 s | 11.69 cm / 14.89 degrees | Tracked, stale |
| Left / nominal | 1,779 / 1,585 | 0% | Unavailable | Unavailable | SAM2 initialization mismatch |

Raw errors are conditional on a finite supported native pose at its source image;
they exclude lost poses and do not demonstrate useful accuracy. They compare the
first automatic hypothesis with the leaf trajectory without truth-based reselection.
Its complete material ownership remains unqualified, so these diagnostics cannot
attribute every error uniquely to registration rather than candidate association.
The full denominators and unchanged official gates remain separate.

CUDA TSDF remains active throughout the three initialized cases, with 11/12/32
observed volume rebuilds. Sampled Point2Pose process peaks are 8.24/8.77/14.52 GB;
no memory-bound failure occurs in this attempt. The failed initialization supplies
no completed-result latency or memory-peak distribution. This is missing evidence,
not zero cost. Zero useful recovery and no sustained useful interval fail the
requirement even where raw native tracking returns.

### Frozen observer matrix

`point2pose-live-02` (`7460963`) completed eight fresh serial parked-arm Isaac
cases: 2,772 inputs, 1,734 independently observable samples, **0% useful
availability in every case**, complete p95 0.784–1.755 s. Raw left-pilot p95 was
1.48–2.24 mm/0.209–0.407 degrees; right camera/combined cases were
19.49/56.70 mm and 1.927/9.135 degrees. These conditional capture errors cannot
compensate for late publication. Covered-zone cases had 42/48 observable samples;
total-occlusion intervals had 31 each and no published localization. Native return
was not useful recovery; synthetic faults did not validate real occluders.

All eight resets changed worker PID/generation and cleared candidates, selections
and queues. CUDA TSDF stayed active; sampled process memory 4.60–6.14 GB,
allocator peaks 2.98–3.92 GB and free device memory at least 11.33 GB are distinct
measures, not an exact combined startup peak. No loaded action was attempted.

## Interrupted 60 Hz campaign and moving-camera diagnosis

`point2pose-offline-60hz-01` (`0adef47`) was stopped by the user after 4,216 rows
in the first recording: 240 preinitialization and 3,976 native rows, through source
row 4215/frame 4223/t=70.25 s. Its 502 remaining rows, 11 unstarted attempts and
20,108-frame full-recording denominator (15,892 unprocessed) remain explicit.
All five initial candidates passed preservation checks; that is not initialization
reliability or ownership. `89c08bd` repaired evaluator SIGINT finalization without
replacing the original attempt. The campaign remains stopped.

The saved 4–10 s audit uses all 361 rows (240–600) while the camera moves
13.15 cm/57.28 degrees against a stationary leaf. Acquisition matching, FK and
composition are correct: simulator/FK discrepancy <=0.327 micrometers/1.11e-6 rad,
native depth backprojection difference about 35 nm, independent world composition
agreement 4.45e-16. Inverting the native pose instead worsens error to
62.71 cm/90 degrees. The contract is current `Ct_M` (not `init_pose`), composed
with acquisition-time `W_Ct` and the immutable seed-zone offset.

Native output freezes from 5.2333–7.9167 s for 162 captures despite `lost=False`;
161 have zero inliers. Primary capture p95 is 20.50 cm/24.47 degrees; integration
supports 178/361 rows with 11 interruptions. Later returned support is not proof
of accuracy. This localized the issue before camera/world composition.

### Registration telemetry and corrected publication

`point2pose-registration-4-10s-01` (`2ea9e61`) contains one 361-frame CUDA repeat,
exactly reproducing original poses for all five candidates. Two separate defects
were demonstrated:

- The primary has 182 no-cluster fallbacks reporting `lost=False`. During the
  161-frame freeze, 9–20 frontend pairs remain but only 0–2 match evaluator-expected
  geometry at 4 mm, despite at least nine original depth-visible references at 8 mm.
  No graph update causes that freeze.
- Ninety accepted SDF refinements return the old cluster pose; the largest discarded
  correction is 9.21 mm/1 degree. SDF is not called during the frozen no-cluster range,
  so correcting SDF alone cannot repair its correspondences.

`df36b92` fixes current loss and returned SDF pose/statistics without relaxing gates.
The one subsequent 361-frame CUDA check (`point2pose-native-fixes-4-10s-01`) verifies
all 1,218 no-cluster fallbacks as lost, 133 returned refinements, nine support-gate
fallbacks, 149 matched statistics and 1,800 contract-consistent states across
candidates. Primary supported rows fall to 59 with conditional p95 22.6 mm/2.07
degrees; 301 finite lost poses remain, with all-finite p95 339.8 mm/42.25 degrees.
Loss from 5.25–10 s is preserved. Better conditional error through rejection is
not recovered tracking or an overall accuracy gain.

### Saved-image material diagnosis

`point2pose-visual-4-10s-01` used existing RGB/depth/traces, without new inference.
27/30 seeds have local contrast std <2 on 21×21 patches; only IDs 0, 1 and 4 exceed
5. At 7 s ID0 agrees within 0.3 pixels, ID29 slides 15.56 pixels/35.49 mm and ID13
drifts 102.54 pixels/114.12 mm while passing native filters. Seed integer lifting
is exact; all 30 points retain measured mask/depth support at 4.3 and 5.25 s.
A geometry-only expected-pixel counterfactual has 30/30 compatible references,
but is not an implemented tracker or material correspondence observation.

RGB/depth edges (Canny 15/45) agree best at original times versus shifts of 1–6
frames; a common -2-pixel correction is inadequate. Later full masks and rejected
candidate coordinates were not saved and remain unknown. Renewal never reaches
its 15-degree trigger before loss (maximum 14.2186 degrees; 17 pairs still exceed
the ten-point trigger), then loss suppresses it for 286 rows: 197 no-cluster and
89 insufficient-pair cases. This supports reference/renewal diagnosis, not a
camera-timing or threshold remedy.

## Camera-still baselines

`point2pose-still-opening-baseline-01` (`112bbe8`) processes all 421 captures from
31–38 s with one automatic seed, no reset and unchanged controls. The leaf is
static for 134 captures, starts opening at 33.2333 s and reaches 4.54 degrees.
All 420 nonseed primary rows are supported/correct: static p95 1.74 mm/0.116 degrees,
opening 2.64 mm/0.295 degrees. Unlike the moving-camera seed, 27/30 references have
useful contrast. Contact is gripper-occluded during 287 opening and 52 stationary
captures despite panel support. Thirty-five secondary native returns pass no strict
original-material recovery check. Depth-edge sensitivity remains; no new primary
IDs or graph updates occur. Evidence includes `annotated-baseline.png`, `point-cases.png`,
`baseline-errors.png` and `recovery-analysis.json`.

The complete extension (`10e7330`, `point2pose-full-opening-baseline-01`) processes
2,858 captures per light/nominal condition, rows 1860–4717 at original 60 Hz,
31–78.6167 s. Both have identical motion/camera/calibration and reach **63.7209
degrees**, not the configured 90.7-degree limit. Light's first 421 poses/decisions
exactly reproduce the short baseline. Truth remains evaluator-only.

| Primary condition | Supported / 2,858 | Correct / inaccurate accepted | Supported position/rotation p95 | All-finite position/rotation p95 | Native lost frames |
|---|---:|---:|---:|---:|---:|
| light | 2,311 (80.86%) | 2,276 / 35 | 7.79 mm / 1.44 degrees | 8.24 mm / 1.44 degrees | 531 |
| nominal | 1,582 (55.35%) | 1,494 / 88 | 10.20 mm / 1.24 degrees | 34.39 mm / 2.65 degrees | 1,274 |

Coverage retains the unsupported seed; errors exclude its constructed zero.
Above 60 degrees support is 175/275 light and 2/275 nominal. Nominal ends with
266 unrecovered frames (4.4167 s), final retained error 49.83 mm/3.26 degrees.
Accepted maxima are 22.37 mm/6.22 degrees light and 36.07 mm/9.07 degrees nominal.
Native loss/return events are 207/206 and 217/216, but strict seed-material returns
are only 9/7; historical-anchor checks pass 10/10. Correct returned poses without
anchor support remain unresolved, not automatically false identities.

The five/six automatic candidates have 30 primary references, 27 with useful
contrast, but only seven seed pixels coincide: this is not an isolated illumination
experiment. New references total 90/60 with 3/2 graph updates; no IDs are born during
loss. All candidates, latched rejections, original masks and complete 60 Hz videos
remain, including decoded final frames and full/recovery-analysis reports.

## Renewal experiments: completion versus accuracy

### Rejected unbounded renewal

`b7602b2` (baseline `59abf31`, reverted at `f2bf69d`) changed the renewal count to
final inliers. `point2pose-inlier-renewal-comparison-01` preserves both failures:
light completed 1,352 frames before CUDA OOM at 53.5333 s, nominal 619 before the
unchanged TSDF memory guard at 41.3167 s. The remaining 1,506/2,239 rows stay in
each 2,858-frame denominator; process interruption is distinct from native loss.

On matched completed prefixes, light support improves 1,097→1,351 but inaccurate
acceptance grows 0→82; nominal changes 590→618 and 0→1. Active populations reach
3,025/2,160 versus baseline 420 across objects, process peaks 21.51/16.42 GiB.
TSDF alone accounts for only about 0.52/1.17 GiB. At 47.1667 s light graph
publication changes 7.607 mm/0.467 degrees to 21.157 mm/14.019 degrees; nominal at
40.1333 s changes 2.493 mm/0.181 degrees to 12.019 mm/3.753 degrees. Light's first
wrong acceptance already precedes that graph event. Both resource and publication
failures reject the variant; there was no favorable retry or memory-limit relaxation.

### Bounded replacement and remaining failures

The growth audit (`94429d9`, baseline `f2bf69d`) in
`point2pose-bounded-renewal-01` exactly preserves its 19-frame prefix. TAPIR causal
state costs 1,966,080 bytes (1.875 MiB) per point plus 4,608 feature bytes; the
largest baseline object holds 120 active references. Early LM accessor failures
remain. Old outliers return after 901 frames, rejecting age-only retirement.

`2b127b4`/`3430808` caps each object at 120 active references, preserving stable IDs,
historical CPU landmarks/keyframes/graph. A promoted replacement requires three
published-pose inlier observations, and an old point more than 15 visible/masked/
valid-depth outlier observations; lost/invisible/invalid depth freezes retirement.
Hull/cell protection and at most three retirements per frame preserve distribution.
Whole batches defer without capacity. This bounds active TAPIR state, not total
history/TSDF or reserved CUDA memory; canonical mechanics are in the topic.

Both full windows complete. Across candidates, maximum active populations are
512/654 and historical populations 2,094/1,830. Sampled peaks are 7.33/9.60 GiB
versus original baseline 6.58/6.52 GiB. Original failed tails stay preserved.

| Primary bounded condition | Supported | Correct / inaccurate accepted | Lost | Supported position/rotation p95 | All-finite position/rotation p95 |
|---|---:|---:|---:|---:|---:|
| light | 2,791 | 2,250 / 541 | 65 | 15.43 mm / 1.27 degrees | 15.71 mm / 1.28 degrees |
| nominal | 2,282 | 2,183 / 99 | 568 | 9.28 mm / 0.79 degrees | 19.02 mm / 2.42 degrees |

Correct supported changes are -26 light/+689 nominal, while all-finite correctness
changes -518/+59. Completion gains over the failed variant are not pure accuracy
gains. Nominal's final loss shrinks to 60 frames but remains unrecovered, with
22.06 mm/3.14 degrees retained error. Complete request p95 775/910 ms is diagnostic,
not a freshness qualification.

Recomputed published-pose support differs from inherited frontend masks on 17/12
accepted primary captures. At 61.2667 s light accepts 17.23 mm/1.03 degrees with
21 declared versus nine actual inliers (two from the declared subset). At
45.6667 s nominal frontend 2.92 mm/0.58 degrees becomes **178.61 mm/32.73 degrees**
after graph publication: ten pairs fit the published map within 4 mm, only one
from the 22 declared. Declared residual reaches 433.79 mm. No bounded accepted
primary/secondary has fewer than five recomputed inliers; metric self-consistency
still does not establish correctness. Original baseline nominal at 49.45 s does
have only one recomputed versus five inherited inliers for a 36.07 mm accepted pose.

Of 25/43 bounded native returns, 13/25 have correct poses; only one/one also passes
five compatible historical anchors, and none passes seed anchors alone. Independent
projected-depth proxies mark 66/62 retirements as near-depth/self-occluded and
24/24 as depth-unknown despite native visibility. These are uncertain physical
proxies, but prevent claiming preservation of every occluded material reference.
Truth never affects renewal. `comparison.json`, stage traces and four-timestamp
`discriminating-comparison.png` retain all candidates, failures and uncertainty.

## Subsequent decisions and current cleanup

Later work has one canonical page per question; earlier evidence remains intact:

| Investigation | Durable decision / source |
|---|---|
| [[point2pose-consumer-impact-and-failure-windows|Consumer impact and failure windows]] | Nominal graph publication changes landmarks/TSDF; light deterioration at 52.1333 s precedes SDF without a same-frame graph/replacement. Actual A3/A4 action scores lack required inputs. |
| [[point2pose-bounded-global-graph-ablation|Graph ablation]] | `0e6185f`: correct poses 2,569/2,004, inaccurate 238/58; nominal lost grows 568→794, terminal loss 60→472. Peaks 7.78/9.61 GiB do not improve overall. Graph ON remains operational default. |
| [[point2pose-isolated-improvements|Partial batches and geometric controls]] | Partial batches retained for explicit diagnostics; 6 mm and pending geometric confirmation rejected. |
| [[point2pose-refit-hypotheses|Refit rollback]] | Common adoption rejected; implementation retired, five inliers/4 mm retained. |
| [[p2p-observed-geometry-and-sdf-jacobian|Observed geometry and SDF]] | Independent static/hinge/angle supports; vectorized Jacobian preserves numerical decisions. |
| [[p2p-selected-registration|Periodic selected registration]] | Nominal tail regression prevents adoption; implementation retired. |
| [[p2p-evaluation-and-performance|Evaluation and compute variants]] | Retain all-query TAPIR/480/four iterations; reduced model/tracker/SVD variants retired. |
| [[p2p-sam3-unified-frontend|Unified frontend]] | Select original single-object SAM3 reconditioning OFF; SAM3.1 retired after complete comparison. |
| [[p2p-sam3-selected-development|Selected development]] | Exact bounded-history equivalence; renewal-only depth gate adopted experimentally with explicit light accuracy/memory tradeoffs. |
| [[p2p-selected-residuals|Selected residuals]] | Preserve recipe; independent material associations and missing rejected-light trace remain unresolved. |

October 8 cleanup retains static B1, SAM2 baseline/operational paths, live/serial
evaluators and shared helpers used by ignored diagnostics. It tracks the selected
opt-in configuration, removes only rejected variant code/tests, and preserves
source reconstruction/migration. `p2p-cleanup-20261008/verification.json` records
exact saved-prefix configuration, masks, poses and decisions for 33 original
frames per condition, plus the existing 12-sample operational CUDA smoke. This
bounded validation establishes neither full-opening equivalence nor 150 ms
freshness. No full replay, optimization, model comparison or ignored-payload
deletion occurred. Historical executable variants remain at `57ad483`.
