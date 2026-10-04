# B1 Perception Findings and Direction

## Decision and evidence boundary

The maintained 6.0B uses **GroundingDINO + native SAM3**, calibrated RGB-D/multiview
geometry and DINOv3 image features. Custom DINOv2 estimators, SAM2/CAP-Net comparisons,
legacy dynamic geometry, custom material tracking and SAM3 video comparison are
retired. The 6.0C prototype reuses the official CAD-free
[Point2Pose components](https://github.com/tzuyuan/point-to-pose), with distributed
visual references, RGB-D and camera kinematics. The prototype is implemented,
but useful availability, precision and concurrent runtime are unqualified.
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

## 2026-10-03 — CAD-free Point2Pose prototype

Baseline `main` at `0190d61`. The optional provider now runs the official paper
components at upstream `51856226610df75e5c06e8de545bd27f7c4ba99c`, without CAD,
training, new corpus or sealed-test access. Local worker dependencies and
checkpoints are ignored by Git and isolated from Isaac and 6.0B. The prototype
publishes local diagnostics; legacy full-state validity and policy encoding remain
unavailable. No action or loaded contact is admitted.

The early automatic seed uses the existing right-pilot nominal capture at 4 s.
Official positive SAM2 prompts retain the panel relief and exclude the displayed
frame. An adapter border-pixel check incorrectly rejected this mask, and negative
prompt/full-mask trials produced worse masks. Those changes were reverted.
Superseded attempts 06/07 remain diagnostic failures; they are not evidence that
the original mask included the frame. Final verification uses measured interior
references and preserves ambiguous physical relationships.

The first optimized single-candidate smoke (`point2pose-smoke-11`) reports p95
171 ms and 9/11 timely tracked post-initialization results. The concurrent fresh
Isaac smoke (`point2pose-live-smoke-01`) reports p95 265 ms and 0/12 timely results,
with a 2.31 GB PyTorch allocator peak. CUDA models and TSDF work; the 150 ms gate
fails. These brief smokes do not score pose accuracy or useful provider output.
Resident/process peak and queue-aware latency are measured in subsequent attempts;
PyTorch allocator peak is not total GPU residency.

`point2pose-replay-01` was interrupted after redundant geometric observations
were found to be separate objects. `point2pose-replay-02` completed all four
chronological recordings but produced zero useful outputs in every observable
interval. This is failure, not safe success. The adapter had coupled visual
support to the uniform planar seed, and initialized from the transient reset
view. It was corrected to use distributed native references, independent root
geometry and the first completed common inspection view. Duplicate 6.0B semantic
work after SAM2 initialization was also removed. Two second-attempt reconstructions
hit explicit memory bounds; extent now uses the authors' filtered cloud rather
than outliers they reject. Every failed attempt remains in its original directory.

Real isolated CUDA fusion validates expansion through retained official keyframes,
multiple integration, sparse state preservation and explicit impossible-volume
failure. Numeric regressions cover transforms, acquisition/completion, reset,
measurement versus completed depth, candidate preservation, holes, finite covers,
angular lever arms and physical slide separate from the 1 cm threshold. These
checks do not qualify real stereo/calibration errors, loaded mechanics, physical
occluders, dynamic complete-path clearance or release encoding. Subsequent frozen
pilot/observer reports determine useful availability and precision independently.

See [[../topics/shared-door-perception|prototype behavior]] and
[[../implementation_phases/phase-6-0-operational-perception-and-contact|acceptance requirements]].

Initial live checks exposed one unsupported thin hypothesis invalidating the
whole bundle and a diagnostic neck sweep reaching a mechanical limit.
`point2pose-live-check-01/02` preserve both failures. Interior-support checks now
precede duplicate consolidation and native initialization, and the common camera
sweep respects actual joint limits. This does not alter Point2Pose's prompts,
registration thresholds or robot limits.

The third replay still failed all four recordings: right-pilot tracking was lost
after cold initialization skipped inspection views, while left-pilot volumes
reached the enforced GPU reserve with inactive 6.0B weights still resident.
The first eight-case fresh-process observer matrix also failed useful availability
in every case; one combined-motion initialization failed its SAM2 preservation
check. Some raw left-pilot poses were accurate, but they arrived too late and
cannot count as useful outputs.

Lifecycle revision `7460963` loads both workers before acquisition and releases
the stateless 6.0B worker once the synchronized seed is ready. Startup is reported
separately; copying, candidate preparation, IPC, queue wait and observed-pose
publication processing remain in dynamic timing. Episode reset terminates and
recreates the prepared Point2Pose process and seed worker. Ordinary frame gaps
invalidate requests without model reloads. Warm diagnostic attempts 01/02 exposed
an erroneous reload on every gap; both are retained. Temporary weight staging was
reverted, and inference remains CUDA-only. Warm attempt 03 initialized five
hypotheses and completed real process replacement, but still had 0% useful
availability and 1.046 s p95 latency. GPU sampled process peak was 5.36 GB, with
12.18 GB minimum free memory after releasing the seed worker.

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

`point2pose-live-02` also uses source commit `7460963`. All eight cases complete
in fresh, serial Isaac processes with the arm parked and no loaded action.
The 2,772 input observations and 1,734 independently observable samples remain
in the records. Every useful-availability gate fails at 0%; complete latency
p95 is 0.784–1.755 s, beyond the unchanged 150 ms deadline.

| Pilot / case | Full / observable samples | Complete latency p95 | Raw capture position / rotation p95 |
|---|---:|---:|---:|
| Left / camera | 353 / 233 | 1.726 s | 1.67 mm / 0.261 degrees |
| Left / panel | 354 / 231 | 1.755 s | 1.99 mm / 0.326 degrees |
| Left / combined | 348 / 228 | 1.196 s | 2.24 mm / 0.407 degrees |
| Left / visibility | 346 / 193 | 1.328 s | 1.48 mm / 0.209 degrees |
| Right / camera | 343 / 223 | 1.087 s | 19.49 mm / 1.927 degrees |
| Right / panel | 338 / 212 | 0.784 s | 9.99 mm / 2.538 degrees |
| Right / combined | 332 / 209 | 0.998 s | 56.70 mm / 9.135 degrees |
| Right / visibility | 358 / 205 | 1.714 s | 2.19 mm / 0.392 degrees |

The visibility tests retain 42/48 observable covered-zone samples, all with zero
useful outputs despite other references being visible. Each total-occlusion
interval has 31 samples and no published localization. Native tracking returns
after reappearance without changing the selection, but useful recovery is zero
and every sustained useful interval is empty. Since the baseline is already
unavailable, this is not a successful live loss/recovery validation. Repeat that
validation only after establishing useful pre-occlusion tracking. Synthetic RGB-D
faults do not validate physical occluders.

All eight episode resets terminate the original PID, create a different prepared
worker PID, increment generation and leave candidates, selections and queues
empty. Numerical tests additionally cover invalidation during in-flight work.
Sampled Point2Pose process peaks are 4.60–6.14 GB, PyTorch allocator peaks
2.98–3.92 GB, and sampled free device memory remains at least 11.33 GB during
completed inference with Isaac active. These are separate measurements, not an
exact combined startup/graphics high-water mark. No CPU model or TSDF fallback
is used.

The unchanged 1 cm/5 degree limits also reject raw right-pilot camera/combined
accuracy; raw left-pilot stability does not compensate for late publication.
The subsequent 2026-10-04 offline audit below supersedes performance profiling as
the next action. Complete zone ownership,
hardware calibration/FK and temporal/stop bounds, real occlusion and physical
slide remain unvalidated. Numeric finite-region slip checks do not establish
loaded contact. Official offline/dynamic qualification and policy handoff remain
separate and false.

## 2026-10-04 — Interrupted serial offline campaign and stationary-door drift

The independent 60 Hz evaluator was implemented at `0adef47`, from clean `main`
baseline `3b77122`. It reuses the frozen models/configuration and existing candidate
preparation, with native diagnostic continuation after integration mask rejection.
The RTX 4090 campaign `outputs/b1/perception/point2pose-offline-60hz-01/` started
only `animated-door-1-88abf40/light/attempt-1`. The user requested a controlled stop
before further trials. SIGINT terminated the evaluator and its workers, with no
restart. The original incremental file contains 4,216 intact rows, ending at
row 4,215 / frame 4,223 / image time 70.25 s: 240 pre-initialization rows and 3,976
native-result rows. All five native candidates initialized, and all five initial
mask checks accepted; this single attempt supplies no reliability estimate.

The original interruption path closed workers but did not finalize its reports.
Evaluator-only `finalize_partial.py` preserves `frames.jsonl` unchanged and writes
the 502 remaining rows to `unprocessed.jsonl`, including the interrupted request
at row 4,216. Its report uses all 4,718 expected rows, with observability explicitly
unknown for unprocessed rows. Across the four full recordings, 15,892 of 20,108
frames remain unprocessed: 502 in the started recording and 15,390 in the three
unstarted full recordings. All eight extra initialization windows and the other
three full attempts are unstarted; their seed-dependent prefix lengths are unknown.
None of the twelve planned attempts completed. Original startup, selection,
runtime and native logs remain intact. `89c08bd` adds direct SIGINT finalization,
full-recording counts, campaign stop propagation and interrupted-startup cleanup.

For the partial primary candidate, native flag-based coverage is 3,298/4,718,
integration measured-support coverage is 2,696/4,718, and 678 native lost poses
remain recorded without counting as available tracking. There are 742 unknown
observability rows: 240 before initialization plus 502 unprocessed. Request latency
p95 is 0.967 s over 3,976 completed requests; visual/native startup is 9.770/4.265 s.
Those times do not alter offline accuracy, continuity or coverage. Integration
acceptance here means diagnostic measured support, not the complete operational
controls or contact admission. The historical operational failures above retain
their unchanged 150 ms freshness gate.

### Bounded numerical diagnosis, without further model inference

`diagnose_drift.py`, `drift-diagnosis.json`, `drift-samples.jsonl` and
`drift-4-10s.png` use only the preserved first attempt and original HDF5. All 361
source rows 240–600 / frames 248–608 are present at 60 Hz, including both endpoints.
Every saved timestamp/frame matches its HDF5 acquisition exactly; native indices
0–360 correspond to the same uninterrupted sequence. Scoring uses these source
rows, never completion time. Ground truth enters only this numerical evaluator.

Expected behavior: a stationary material zone keeps a constant world pose even
when its camera-relative pose changes. With map M fixed to the seed optical frame,
the expected native transform is `Ct_M = inverse(W_Ct) * W_Cseed` for this stationary
interval. The integration composes `W_Ct * Ct_M * inverse(W_Cseed) * W_Zseed` once.
Actual upstream f2m registration fits map/source points to current optical points;
the misleading internal name `T_c2w_est` does not reverse that direction.
`estimate_init_pose=false` leaves the initial object transform identity. The graph
inverts poses internally for optimization and inverts them back on output.

Verified controls:

- Leaf translation is constant and rotation changes by less than `7e-18` degrees;
  the camera moves 13.15 cm and rotates 57.28 degrees relative to the seed.
- Camera transforms recomputed from the recorded joint state/calibration match
  every stored transform exactly. The acquisition metadata separately records
  simulator/FK maximum differences of 0.327 micrometers and `1.11e-6` radians.
  This verifies the simulation recording, not hardware calibration.
- ROS optical axes (right/down/forward), image-plane depth in meters and native
  `depth_factor=1` agree. Native and integration backprojection differ by at most
  35 nm on the sampled measured seed points. Projecting those stationary references
  into subsequent depth yields p95 1.08 mm over all 43,657 valid samples, with a
  21.70 mm maximum retained rather than filtered. At least 97 of 128 references
  are depth-supported in every frame. This rules out a gross camera/depth frame
  mismatch on the examined region; it does not prove pixel-perfect render timing.
- Recomposing saved native matrices reproduces saved world poses within
  `4.45e-16` per matrix entry. Supplying the expected native transform instead
  holds each world zone constant to numerical precision. Inverting the native
  output instead makes the primary p95 error 62.71 cm / 90.04 degrees and contradicts
  the verified point-transform contract. No inverse/order correction is justified.
- The frontend log already contains primary p95 error 20.48 cm / 24.42 degrees,
  before graph refinement, IPC or evaluator world composition. The published
  native output has p95 20.50 cm / 24.47 degrees. The downstream composition is
  therefore not the origin of this drift.

Observed primary behavior: from 5.2333 through 7.9167 s its native matrix is exactly
constant for 162 frames while the camera rotates another 24.12 degrees and moves
5.73 cm. All 162 report `lost=False`; the last 161 have zero measured inlier pairs
and are rejected by integration. During that freeze, world error grows from
2.00 cm / 0.69 degrees to 20.48 cm / 24.11 degrees. This establishes the mechanism
for that interval: an outdated camera-relative transform fails to cancel camera
motion. At 7.9333 s measured support returns under the same object/candidate ID,
but the world error remains about 20 cm / 24 degrees. Returned support does not
establish correct recovery. No candidate reselection occurred.

For the primary candidate, all 361 rows are independently observable and finite
with `lost=False`. The following errors exclude only the seed, whose zero is
constructed alignment and does not establish absolute initialization accuracy.

| Primary sample group | Error samples / 360 | Position median / p95 / max | Rotation median / p95 / max |
|---|---:|---:|---:|
| Native flag-based tracking | 360 | 12.45 / 20.50 / 21.05 cm | 13.66 / 24.47 / 26.29 degrees |
| Integration accepted | 178 | 20.35 / 20.60 / 21.05 cm | 23.91 / 24.64 / 26.29 degrees |
| Integration rejected | 182 | 12.30 / 19.39 / 20.48 cm | 13.46 / 22.68 / 24.22 degrees |
| Native lost | 0 | No samples | No samples |

Primary integration coverage is 178/361 (49.31%, with the rejected seed included).
It has 11 interruptions and 11 returns of measured support, the longest interruption
lasting 2.6833 s; its longest supported sampled span is 0.9833 s. Native `lost`
flags indicate a continuous six-second run, but the freeze above shows why this
flag alone does not demonstrate tracking continuity or accuracy.

Every candidate remains reported; smaller errors do not change the automatic
primary choice. These are native tracking errors, excluding finite lost poses
from available tracking while retaining those poses/errors in the detailed JSON.

| Candidate | Native rows / 361 | Integration accepted / 361 | Native lost rows | Native error samples | Native position / rotation p95 |
|---|---:|---:|---:|---:|---:|
| 0, automatic primary | 361 | 178 | 0 | 360 | 20.50 cm / 24.47 degrees |
| 1 | 361 | 167 | 0 | 360 | 37.63 cm / 26.84 degrees |
| 2 | 361 | 289 | 0 | 360 | 9.16 cm / 7.59 degrees |
| 3 | 274 | 10 | 87 | 273 | 1.09 cm / 13.44 degrees |
| 4 | 293 | 99 | 68 | 292 | 3.82 cm / 19.45 degrees |

### Demonstrated boundary and unresolved internal cause

The evidence localizes the large error upstream of world composition, to native
tracking/registration output. It demonstrates unflagged native freezing and biased
poses after support returns. It does not prove that TAPIR alone, SAM2 alone, or
SDF/graph alone caused those errors. Complete material ownership remains unqualified;
ambiguous or changing point association can still affect registration.

The pinned native register has a concrete path that returns its previous pose,
zero inliers and residuals `-1` when no RANSAC cluster succeeds. The frontend sets
`lost` on insufficient input points or a rejected jump, but an unchanged fallback
is not a jump and an invalid residual need not set `lost=True`. This code path is
consistent with the observed freeze; the original payload retained measured-pair
counts but not cluster/fallback/refinement statistics or per-frame correspondences,
so its exact activation and the reason hypotheses failed are not demonstrated.
The degeneracy test checks triangle area, not planar rank; its comment mentioning
coplanarity alone is insufficient to attribute failure to the planar door.

The minimum next intervention is diagnostic-only export of the already computed
frontend correspondence/inlier counts, residuals, cluster/fallback status,
pre/post-SDF pose and graph pose changes. Then inspect only this same 4–10 s input
with frozen models/configuration, if a subsequent bounded trial is authorized.
Keep ground truth solely in scoring. Do not change SE(3) composition, tune gates,
optimize performance or restart the complete campaign based on these results.
No new model inference, training, acquisition, ownership qualification or loaded
contact admission was performed during this diagnosis.
