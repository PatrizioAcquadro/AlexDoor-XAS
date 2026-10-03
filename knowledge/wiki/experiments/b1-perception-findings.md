# B1 Perception Findings and Direction

## Decision and evidence boundary

The maintained 6.0B uses **GroundingDINO + native SAM3**, calibrated RGB-D/multiview
geometry and DINOv3 image features. Custom DINOv2 estimators, SAM2/CAP-Net comparisons,
legacy dynamic geometry, custom material tracking and SAM3 video comparison are
retired. Future 6.0C will reuse the official CAD-free
[Point2Pose components](https://github.com/tzuyuan/point-to-pose), with distributed
visual references, RGB-D and camera kinematics. Integration is not implemented.
See [[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0]]
for future requirements and [[topics/shared-door-perception|Perception]] for live behavior.

No estimator qualified complete door/contact geometry on the six development doors.
Train fitting, masks, static planes and execution completion are not qualification.
The original 1 cm / 5 degree and per-door 95% coverage/accepted precision gates
remain unchanged, including missing/rejected states and empty accepted sets.
All model experiments below used the RTX 4090; the sealed test remained closed.

Historical protocols, corrected per-door scores, significant initial/interrupted
failures, source references and decisive images are consolidated under
`outputs/b1/perception/evidence/<run-name>/`. Static 6.0B final diagnosis and human
local-role confirmation remain in `operational-scan-diagnosis-01/` and
`operational-contact-readiness-01/` under `outputs/b1/perception/`.
`e1f98a6` and `2a38858` retain earlier tracked source and detailed historical pages.
Removed ignored scripts, weights, recordings and intermediate caches cannot be
restored from Git. Retained results support inspection, not full executable reproduction.

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

### Completed SAM3 video comparison

Box-only hotstart produced an initial mask (score about 0.988) but removed its
unmatched track before export: a single box supplies no continuing detector matches.
On the same first 20 nominal captures per pilot, box plus the existing `door` concept
retained **20/20 masks versus 0/20** with box alone. Native filters and the frozen
checkpoint were unchanged. The original box-only run, corrected prefix results and
initial prompt counts remain separately preserved. This retrospective comparison
establishes neither full-video/light performance, leaf ownership, an axis nor causal
runtime support. Its worker, automatic orchestration and `decord` are now removed.

## Retired custom material tracker — October 1

At baseline `2a38858`, four complete pilot replays and four fresh serial parked-arm
live schedules initialized all requested local geometry but obtained **zero selected
dynamic-support ticks, zero supported-axis ticks and zero material reacquisitions**.
Visibility returned independently. Reports remain in `evidence/operational-tracking-03`
and `evidence/operational-tracking-live-02`; earlier attempts are retained separately.
Source: `a3b4587`, `fec3cf9`, `ae120b4`, `ac3057e`, `3bbde73`, `731fe0e`, `c8313f3`.

| Pilot, nominal/light | Complete replay | Complete parked-arm live schedule |
|---|---|---|
| Dark `door-2738468b94d74c5f` | 5,336 observations / 88.917 s each | 2,013 observations / 33.533 s each |
| White `animated-door-1-88abf40` | 4,718 observations / 78.617 s each | 2,013 observations / 33.533 s each |

White's visual reference was y=-0.10 m, z=1.50 m, with explicit contact y=0,
z=1.10 m and alternative y=-0.10 m, z=1.30 m. Dark used 1.10 m for both visual
reference/contact and retained 1.30 m. The tracker did not switch contacts after loss.
All contact/precontact endpoint solves passed. White upper clearance was about
103 mm versus 66 mm lower, while lower precontact/contact joint margins were
0.168–0.170/0.215–0.218 rad versus 0.121–0.125/0.184–0.188 upper. The 1.30 m
alternative had narrower precontact margins, 0.022–0.027 rad. These comparisons
establish neither a definitive contact ranking nor posture/path qualification.

Live revisited the 7 s neck pose and final pose, holding each two seconds. An
independent raw-depth audit found 121/121 supported upper-hold ticks for white's
high region and 121/121 final-hold ticks for both lower regions; dark recovered
121/121 final-hold ticks for both regions. Material support remained absent.
Stationary live doors did not test articulation; recorded moving doors still yielded
no supported axis. Static visibility could not refresh original material identity.

The initial live guard aborted at 0.100087 rad neck error; that attempt was stopped
for the common resolution correction, not declared a physical failure or completed
pass. Revised guards retained explicit 0.1 + 0.006670 rad allowance, peak neck error
0.104564 rad, tool drift 0.150 mm and dark-door motion 0.000139 rad. Commands stayed
within 0.4 rad/s; measured PhysX velocity peaked at 0.403228 rad/s. This discretization
allowance does not relax 150 ms dynamic freshness or validate loaded control.

The first replay was interrupted for uncertainty/lifetime corrections; the second
completed before static-scene retention was repaired. Numerical rigid-transfer/axis
fixtures were never live transfer validation. The tracker implementation and its
replay/live commands are now retired; independent timing, generation, explicit
selection, axis-free contact and admission contracts remain. Point2Pose replaces
this experimental implementation direction without claiming its exits passed.

## Retired geometric full-state prototype

The corrected two-pilot replay `geometric-pilot-06` (`7468e0a`, reporting `809f575`)
used one common recipe and all three frozen CUDA models. Every pilot/condition had
**0% complete-state coverage and 0% accepted precision**. No complete state passed.
The following manipulation p95 values describe finite **rejected** estimates;
unobserved fields are never zero errors:

| Pilot / condition | Width | Height | Thickness | Hinge origin | Local contact | World contact |
|---|---:|---:|---:|---:|---:|---:|
| Left / nominal | 1.96 cm | 22.15 cm | 1.54 cm | 8.36 cm | 8.33 cm | 1.20 cm |
| Left / light | 1.97 cm | 22.72 cm | 1.48 cm | 8.49 cm | 8.47 cm | 1.07 cm |
| Right / nominal | 11.55 cm | 3.61 cm | unobserved | 12.67 cm | 13.15 cm | 3.25 cm |
| Right / light | 9.48 cm | 1.91 cm | unobserved | 4.19 cm | 4.96 cm | 4.56 cm |

A selected plane failed to represent the complete leaf: white relief faces remained
separate and dark support included the human-identified fixed frame. More retained
points sometimes worsened dimensions. Finite panel states covered only 15.9% left
and 37.9% right manipulation; finite hinge states covered 7.3%/21.7%. Motion was
rejected as discontinuous on 83.9%/61.9%. Small subset angle errors were not motion
qualification. Semantic RPC p95 200–218 ms exceeded 150 ms unrefreshed support;
no accepted-state recovery occurred. Static calibration roundtrips of retained
support agreed at p95 0.66–2.31 mm, not a complete geometry qualification.

Earlier pilots retain initial/corrected distinctions. `geometric-pilot-04` (`cae5afa`)
exposed thin floor leakage; `geometric-pilot-05` (`1252cbe`) was interrupted because
replenishment replaced surviving references and inflated uncertainty. The first
extended evaluation (`ac89744`) partially processed eight episodes before a
degenerate fit; a corrected attempt (`a41c002`) was intentionally stopped by the
user. Neither was a complete common-source evaluation. Their reports/failure records
remain, while scratch scripts, caches and repeated logs were removed. No dynamic
qualification followed. The provider now supports static scan evidence only.

## Custom-estimator and observability results

Run-01 (`522f1ba`, diagnosis `e160f1f`) stopped at epoch 28 after 24.7 minutes;
best epoch 18 failed all six development doors:

| Checkpoint / split | Position p95 | Orientation p95 | Confidence-valid coverage |
|---|---:|---:|---:|
| Best / train | 4.58 cm | 4.45 degrees | 0% |
| Best / development | 9.01 cm | 13.93 degrees | 0% |
| Last / train | 5.91 cm | 5.37 degrees | 0% |
| Last / development | 14.93 cm | 11.29 degrees | 0% |

Confidence gradients were 22.6 times contact-position gradients; reconstructed
labels agreed within 3.7e-8 m and sampled cache/source rows aligned. This diagnosed
loss imbalance, not observability or generalization. Corrected contact loss on two
train doors (`796797c`, seed 6100, 2,318 windows) reached 0.838/0.792 cm position
and 1.232/1.219 degree orientation p95 at epoch 30, but hinge origins were about
1 m wrong, hinge rotations 171.65/175.46 degrees and local contacts 1.243/1.276 m
wrong. Complete-state supervision (`1bd633e`, result `34c2660`) subsequently passed
all nine component p95 gates and 98.25/96.70% joint geometry on those train doors;
that fit was not held-out qualification.

Run-02 (`68dc0ec`) stopped at epoch 15/13.60 minutes, with best development epoch 1:

| Checkpoint / split | Contact position p95 | Contact orientation p95 | Doors passing all gates |
|---|---:|---:|---:|
| Best, epoch 1 / train | 1.43–3.27 cm | 2.32–6.25 degrees | 0/19 |
| Last, epoch 15 / train | 0.40–1.45 cm | 0.88–1.77 degrees | 5/19 |
| Best, epoch 1 / development | 7.88–15.43 cm | 13.30–18.32 degrees | 0/6 |
| Last, epoch 15 / development | 8.34–32.09 cm | 6.72–18.72 degrees | 0/6 |

The last checkpoint accepted inaccurate development states at 100% confidence
coverage. Longer training was not established as a remedy.

At `68dc0ec`/`a48e06f`, all 31,654 cached camera views excluded the prepared top face;
four-frame/0.3 s memory could not retain an earlier scan. Matched counterfactual
swaps changed contact by p95 60.3–72.0 mm RGB, 0.76–0.98 mm depth and 2.71–2.82 mm
proprioception. These are sensitivity tests, not modality-importance percentages;
identical camera poses made pose swaps uninformative. A +1% raw-depth change moved
contact only 0.035/0.067 mm train/dev. Privileged substitutions were diagnoses,
never deployable corrections. Confidence AUROC 0.570 did not establish rejection.

The shared 25 s scan and simulated 10-degree mount (`a4ad984`) added seven views
at 4/7/10/15/18/21/25 s. Left/right top and bottom observed fractions were
57.4/52.5% and 71.3/54.5%; tallest-development inspection yielded 24.8/56.4%.
These used 101 boundary samples, a 5x5 depth neighborhood and 3 cm agreement,
not estimator gates. Camera FK error was below 0.001 mm/0.000002 rad, tool drift
0.151 mm, door motion 0.000102 rad and neck error 0.073 rad. Hardware mounting
remains unvalidated. Inspection-only data had no expert hold/release.

Calibrated XYZ/static memory (`7e2bf56`) and shared scheduling fitted the pilots at
7.41/9.93 mm contact p95 with 98.88/95.08% joint geometry. Contiguous reads cut
epoch time from 99.3 to 5.8 s; raw/cache parity over 1,169 windows stayed within
0.0071 mm/0.0014 degrees. A +1% XYZ change moved contact p95 4.95 mm.
`refine-02` fitted 19 train doors at 2.29–4.00 mm; the original `refine-01` report
incorrectly claimed development evaluation, but actual execution was train-only.
Independent confidence fitting still accepted inaccurate development geometry
(7.41–32.32 cm contact p95, 0% precision). Original reports retain that distinction.

The refreshed 50-episode `engineering-v2` campaign supported run-03
(`d833b89`, audit `0ea075f`):

| Checkpoint | Train contact position p95, range across doors | Train complete geometry | Development contact position p95, range across doors | Development complete geometry |
|---|---:|---:|---:|---:|
| Best, epoch 1 | 1.13–3.10 cm | 0/19 doors | 5.42–51.50 cm | 0/6 doors |
| Last, epoch 15 | 1.86–3.44 mm | 19/19 doors | 9.88–54.16 cm | 0/6 doors |

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

## October 2 cleanup

All 50 `engineering-v2` episodes and selected frozen model resources remain intact.
Superseded `engineering-v1`, `inspection-pilot-01` and `inspection-tallest-01` were
removed after saving 53 calibration/metadata headers and essential results. Temporary
experiment scripts, caches, derived features/masks, duplicate dumps/media and logs
without residual value were removed. Twenty completed run directories were consolidated
under the existing evidence inventory. The local `cleanup.json` records paths,
reasons, moves, retained evidence and reclaimed space; no full-recording hashing was used.

Frozen image smoke and four 0–25 s static scans validate the maintained path in
`evidence/cleanup-validation-20261002/`. They do not validate Point2Pose, dynamic
tracking, a movement/contact admission, hardware or offline release gates. No new
training, collection, extended replay or sealed evaluation was started.
