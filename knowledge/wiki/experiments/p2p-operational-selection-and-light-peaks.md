# SAM3 Selection, Operational Timing and Integrated Light Peaks

## Maintained selection and measurement

Baseline `a37fc18`; selection implementation `402b832`; operational measurement
implementation `6c3f872`. The selected recipe,
models, five inliers/4 mm, A3/A4 and policies remain unchanged. Maintained ingress
uses the existing selection API; explicit static references or a displayed pixel
resolve only a uniquely eligible current measured surface. Inspection completion,
reference support and P2P startup are independent. SAM3 survives empty, ambiguous,
absent and insufficient support; requested identity cannot be replaced. Unit
lifecycle checks and the real recorded CUDA reset preserve episode isolation.

Evidence is in `outputs/evidence/b1/perception/sam3-operational-20261008/`.
`selection.json` explicitly chooses pixel (480,250) in the displayed upper recessed
surface after 25 s, with a recorded criterion. It is a geometric reference only.
The actual seed and alternatives are retained; view/cadence changes can change
that capture. No first-root fallback, ownership qualification or contact ran.

The ordered measurements are a complete original light recording (0–78.6167 s,
all 4,718 acquisitions offered continuously at their original times), then 65 s
with one concurrent Isaac renderer and a stationary leaf. The live camera follows
the maintained inspection; the arm holds its initial pose. It is not a new
manipulation/contact or opening experiment. Boot precedes acquisition. A monotonic
clock mapped to the original capture origin separately accounts for input delay,
snapshot, queue/executor wait, SAM3, geometry/seed, native IPC, binding, publication
and polling delay. Input/update service cost is retained independently of scoring.
Original frame/time survives publication; skipped IDs and unsupported outputs stay
explicit. GPU/RSS samples include coordinator/renderer and both model workers at
250 ms. Sampled post-ready peaks do not certify startup allocation spikes.

## Operational results

| Measurement | Continuous recording | Concurrent stationary live |
|---|---:|---:|
| Inputs / publications / replaced | 4718 / 234 / 4484 | 1331 / 141 / 1190 |
| Offered acquisition rate Hz | 60.00 | 20.53 |
| Seed acquisition / publication s | 25.0167 / 26.9596 | 26.2800 / 28.9614 |
| Steady publications, defined seed capture +5 s | 185 | 109 |
| Effective steady publication rate Hz | 3.79 | 3.24 |
| Complete service p50/p95/max ms | 222/595/703 | 291/324/354 |
| Observation age at publication p50/p95/max ms | 246/610/730 | 315/346/375 |
| SAM3 p50/p95 ms | 108/199 | 136/144 |
| P2P including IPC/export p50/p95 ms | 113/393 | 154/186 |
| Publication p50/p95 ms | 2.36/3.56 | 1.15/2.08 |
| Latest-slot wait p95/max ms | 0.021/51.9 | 0.042/61.8 |
| Poll wait p50/p95 ms | 9.14/15.22 | 14.94/33.77 |
| Geometry maximum during inspection s | 2.910 | 5.608 |
| Simultaneous sampled GPU / RSS peak GiB | 7.727 / 5.631 | 12.128 / 11.937 |
| Steady publications within 150 ms | 0/185 | 0/109 |

The recorded startup is 10.41 s, outside capture freshness. Native initialization
adds 1.93/2.68 s complete service in recording/live. Warmup and stationary inspection
have separate distributions in the reports. Historical dense inspection geometry
maxima above 4 s and this live 5.61 s maximum belong to acquisition/initialization,
not stabilized tracking. The full geometry stage still includes extraction,
static fusion/snapshot, eligibility and binding work; no finer causal cost split
or optimization is claimed.

At steady state SAM3 plus P2P already exceed 150 ms: even the recorded median pair
is about 220 ms; concurrent live is about 290 ms. Reduced waiting through latest
replacement does not remove computation or observation age. Publication and queue
wait are smaller than the main service cost. Recording P2P includes registration
trace export, while maintained live uses its default untraced native worker; the
runs also have different image/motion histories. Their difference cannot be
attributed solely to rendering. No timestamps were refreshed, skipped captures
fabricated or frozen native poses made fresh. Local 150 ms admission stays failed.

## Quality of actual processed outputs and retained failures

On the recorded 31–78.6167 s opening, 180 of 2,858 original acquisitions publish.
153 are integration accepted; 27 have **native loss and zero measured pairs**,
including a terminal missing-tracking tail. The preserved dense integrated opening
had zero native losses and eight degenerate-support refusals; these are different
failure categories. Finite fallback poses remain saved and unavailable. The same three
frozen physical targets have processed-only correct availability 81.67–82.78%
at 10 mm/5 degrees and 85% at 20 mm; all-scheduled availability is only about 5.1%
with skipped captures explicitly unavailable. Accepted primary position p95/max
is 8.63/15.25 mm; all-finite/rejected distributions remain in the report. This does
not transfer dense replay quality to a roughly four-Hz provider, and capture-time
accuracy does not establish usable fresh output.

Live has 117 accepted nonseed processed stationary-zone outputs, all within
10 mm/5 degrees relative to its own measured seed. Position p95/max is 2.47/2.75 mm,
rotation p95/max 0.117/0.219 degrees. This verifies only relative stationary tracking
on these processed captures; it does not validate absolute initialization, the
three original opening targets, material ownership/recovery or contact.

`live/` preserves a complete first 65 s attempt with no eligible reference and no
P2P startup. The maintained observer mistakenly started at the 4 s neck waypoint;
`live-initial-view/` uses the common initial waypoint and starts successfully.
This is a procedure repair, not a SAM3 quality variant or isolated causal proof.
Its operating report and processed quality save before a post-run reset-report
`PublishedTrackingEngine.pending` error. Both workers were actually recreated,
as separate resource-sampled PIDs show, and final cleanup closes them. The reporting
reference is corrected without repeating the 65 s inference. The recorded reset
has complete saved assertions; the live failure remains on disk and is not relabeled
as a fully successful diagnostic command. `audit.json` verifies the unchanged
native config/CUDA math, continuous SAM3 indices, retained ID 0 and reset process
isolation. Neither run qualifies hardware, motion or contact.

## Saved integrated light diagnosis

The audit reads the preserved dense `persistent-sam3-20261008/full-selected-light/`
attempt, not the new sparse run. All 3,217 native outputs, 667 reference births and
636 promotion/map changes are inspected chronologically. Published poses equal the
saved evaluator poses; selected source points equal the previous same-ID map and
registration input exactly. Native integer depth lifts reproduce to numerical
precision, every observed map change has a promotion, and there are zero graph or
post-frontend map changes. Detailed paired IDs/births/promotions and 120 captures in
short windows are in `light-diagnosis/`: ±0.2 s around each peak and ±0.1 s around
each of the eight refusals, with overlaps consolidated.

Membership uses prepared collision hulls plus original truth rigid motion only in
this evaluator, with the inherited 4 mm membership margin. It distinguishes
moving Panel/Handle, fixed Frame, overlap and unknown; it is not runtime ownership.
Reference bias compares stored coordinates with measured birth points transported
to the seed map. Current displacement compares the same-ID TAPIR/depth point with
that rigid birth-transport proxy. It does not establish an independent visual
material chain. Moving-body bias statistics exclude fixed-born/current frame
points; transporting those points as if they were panel material is invalid.

### Angular maximum: 72.0667 s

The 4.0778-degree maximum has 14 measured inliers, all moving-only at birth and
current capture. Their second principal RMS is only **12.42 mm**, versus
208.43/204.04 mm in the immediately neighboring captures. It passes the unchanged
8 mm spatial guard but is poorly conditioned transversely. The displayed points
cluster near the handle and nearly vertical detail; a frame-inlier explanation
is unsupported here. The SAM3 mask contains about 24.1% fixed-only measured pixels,
but its IoU against the preceding mask is 0.99861; the neighboring orientation
errors are 0.754/0.772 degrees.

These inliers were born at 60.7667–70.8 s. Moving-point birth bias median/max is
6.58/11.69 mm; promotion changes are 5.06/9.55 mm; current displacement from birth
transport is 3.41/7.62 mm and 0.45–2.22 pixels. These vectors can cancel or reinforce;
their magnitudes must not be added as scalar errors. Saved same-set numerical
refits separate contributions: stored-current gives 3.760 degrees, birth-stored
current 1.196, birth-observed current 1.430, and stored references with exact birth
transport 4.852. They are evaluator counterfactuals, not proposed operational fits
or proof of correspondence identity. Removing promotion alone is not justified.

Registration selects cluster 0 (score 14 versus 7/6/6). Its pre-SDF pose already
has 23.064 mm / 4.078 degrees primary error. SDF improves this to 14.296 mm /
2.387 degrees and lowers its internal objective, but leaves **zero** measured
4 mm pairs. The existing final five-inlier gate falls back to the pre-SDF cluster
pose with 14 pairs; that is the published angular maximum. Thus SDF's attempted
refinement is not the published angular spike. A different saved cluster has
0.688 degrees error, but evaluator truth cannot choose it in the provider.

### Position maximum: 78.15 s

Primary error is **45.213 mm**, with the other two targets 44.999/46.617 mm and
3.588 degrees common rotation. The six published pairs are four moving-only and
two fixed-only. The four moving correspondences depart from birth transport by
17.04–19.65 mm (6.71–8.10 pixels); their stored birth/update biases have median
6.47/5.65 mm. Fixed IDs **663 and 665** are born on the frame at 78.0 s and promoted
at 78.05 s, before this peak. That is direct saved membership/history evidence,
not a conclusion from a changed mask. The peak mask has about 25.8% fixed-only
pixels and preceding-mask IoU 0.99838. Neighboring primary errors are
12.321/9.794 mm despite 12/5 fixed inliers, so fixed membership alone does not
explain the instantaneous error.

Cluster 0 wins with score 20; a saved alternative has score 14, 15 inliers and
11.116 mm / 0.736 degrees error. The selected pre-SDF pose already has
44.761 mm / 3.805 degrees. SDF changes it to 45.213 mm / 3.588 degrees, and reduces
19 measured pairs (4 moving/15 fixed) to six (4 moving/2 fixed). The final count gate
passes and the second principal RMS is 85.03 mm. The position spike therefore
starts in selected registration; SDF adds only 0.451 mm primary error while reducing
orientation error. No GT-based selection or looser support control is proposed.

### Eight degenerate-support refusals

| Capture s | Before-SDF pairs | Published pairs | Published second RMS mm | Published membership |
|---:|---:|---:|---:|---|
| 74.6167 | 21 | 8 | 5.832 | moving-only |
| 74.6500 | 25 | 5 | 5.686 | moving-only |
| 75.4000 | 23 | 7 | 1.593 | moving-only |
| 75.4667 | 22 | 5 | 2.350 | moving-only |
| 75.9667 | 16 | 7 | 7.367 | moving-only |
| 76.1167 | 18 | 9 | 1.651 | moving-only |
| 77.0667 | 22 | 5 | 3.794 | moving-only |
| 78.0000 | 20 | 5 | 0.723 | fixed-only |

Every output has at least five pairs and no native loss, but fails the unchanged
second-RMS >8 mm integration guard. These are long narrow strips or, at 76.1167 s,
a compact roughly 40×6 mm cloud. Before SDF each selected pose has wider support
(51–295 mm second RMS) with **mixed fixed/moving pairs**. SDF objectively changes
these supports to the narrow published subsets above. For seven refusals it removes
frame pairs and retains moving-only support; the last becomes fixed-only. Wider
pre-SDF support cannot automatically be preferred: it is materially contaminated.
Several rejected finite poses have small evaluator errors, but that does not
make their geometric support sufficient. All refusals remain unavailable; no pose
is frozen to hide them and no threshold is relaxed.

Focused, visually inspected artifacts are `angular-views.png`, `position-views.png`,
`fixed-refusal-views.png` and `eight-refusal-supports.png`; seven RGB-D captures plus
one support-distribution plot. `stage-support.json` independently reproduces the
actual measured pair counts from saved depth/validity and unchanged residual gate.
`mask-and-moving-pair-audit.json` keeps mask membership/IoUs separate from inlier
and moving-pair bias measurements. Neither plot nor count proves that SAM3 history
caused a P2P failure.

## Controlled saved-mask CUDA bridge

Baseline checkout `main` @ `fe41e1f` was clean; it was the only local branch and
remained selected. Exactly one CUDA P2P attempt completed **2,858/2,858** original
light captures from 31 to 78.616667 s. Evidence and local diagnostic adapters are in
`outputs/evidence/b1/perception/p2p-mask-bridge-light-20261008/`. `attempt-1/` retains
the input audit, seed/configuration checks, every native result, registration and
reference traces, progress and final report. There was no inference failure,
retry, SAM3 execution, missing-mask substitution, nominal run, live acquisition,
training or contact. Runtime, models, selected recipe and quality gates were unchanged.

The seed uses the historical `quality-full-light/attempt-1` 31 s acquisition and
saved initial mask. Pose, all 30 sampled pixels/map points, active IDs and the
**entire seed registration record match exactly**, checked before the next request.
Effective numerical configuration, model/source paths and versions, CUDA BF16/TF32,
TAPIR, SuperPoint, renewal, registration, TSDF, five inliers and 4 mm agree with the
historical runtime. Only output directories and the disabled standalone segmenter
configuration are excluded from that configuration comparison. The already
maintained external-mask import repair introduces no numerical control change.
The active NumPy RANSAC constructor resets seed 0; loaded TAPIR/SuperPoint run in
evaluation mode without stochastic inference draws. A historical complete RNG
snapshot was not saved, so binary equality of every unused RNG state is unverified.

All 2,857 subsequent masks come from the preserved dense persistent session that
started SAM3 during inspection and initialized P2P at 25.016667 s. Every mask's row,
capture ID, exact timestamp, shape, storage extent, SAM3 ID 0 and native object ID 0
is checked against the recording and source native mask. Original captures remain
chronological with stride one; output masks also equal the supplied masks. All
postseed masks differ from the historical masks (median/max changed pixels
29,617/52,904 of 576,000); the bridge seed has zero changed pixels. The inference
script accesses observations/calibration only. Truth and prepared collision hulls
are confined to the saved-output evaluators.

Relative to the historical run, only the subsequent mask sequence changes; its
downstream reference/TAPIR/TSDF state is allowed to evolve. Relative to the integrated
run, post-31 s masks are identical but P2P seed and accumulated history differ,
including seed RGB-D/mask/references and pre-31 s tracking/TSDF state. **This second
comparison does not isolate seed time.** It also cannot attribute an individual
later difference solely to that capture's mask.

### Complete opening and denominators

`comparison.json` reuses the existing three frozen physical targets and 10/15/20 mm
with 5 degrees. It includes accepted, all-finite and rejected distributions,
histograms, peaks, chronological gaps, all diagnostic windows and tails for all
targets. The original denominator is 2,858 for every run. Historical/bridge retain
their unavailable 31 s alignment seed in that denominator and exclude it from
correctness/errors; integrated initialization precedes the window. A separate
common 31.016667–78.616667 s comparison has 2,857 nonseed captures in every run.

Availability means correct accepted/scheduled; precision means correct/accepted.
Triplets below are 10/15/20 mm, with the unchanged 5-degree orientation bound.

| Run | Target | Correct availability % | Accepted precision % | Accepted position p95/max mm |
|---|---:|---|---|---|
| Historical 31 s | 0 | 85.83 / 99.79 / 99.93 | 85.86 / 99.82 / 99.96 | 12.163 / 23.822 |
| Historical 31 s | 1 | 87.12 / 99.90 / 99.93 | 87.15 / 99.93 / 99.96 | 11.985 / 24.444 |
| Historical 31 s | 2 | 83.80 / 99.58 / 99.93 | 83.83 / 99.61 / 99.96 | 12.385 / 23.727 |
| Bridge 31 s | 0 | 87.19 / 96.85 / 98.78 | 87.59 / 97.29 / 99.23 | 12.199 / 74.549 |
| Bridge 31 s | 1 | 87.44 / 96.68 / 98.53 | 87.84 / 97.12 / 98.98 | 12.169 / 77.595 |
| Bridge 31 s | 2 | 86.56 / 97.17 / 98.74 | 86.96 / 97.61 / 99.19 | 12.253 / 71.941 |
| Integrated 25 s | 0 | 87.58 / 97.55 / 98.81 | 87.82 / 97.82 / 99.09 | 11.814 / 45.213 |
| Integrated 25 s | 1 | 86.53 / 97.55 / 98.74 | 86.77 / 97.82 / 99.02 | 12.147 / 44.999 |
| Integrated 25 s | 2 | 87.54 / 97.38 / 98.78 | 87.79 / 97.65 / 99.05 | 11.652 / 46.617 |

All runs have zero native losses and zero missing outputs. Historical/bridge/
integrated have 2,857/2,845/2,850 accepted opening outputs and 0/12/8 nonseed
geometric refusals. Bridge primary correct counts are 2,492/2,768/2,823 and wrong
acceptances 353/77/22. On the common nonseed denominator, primary availability is
85.859/99.825/99.965% historical, 87.224/96.885/98.810% bridge and
87.574/97.550/98.810% integrated. Their equal common 20 mm availability makes the
one-frame original integrated advantage explicit; it is not a quality gain.

Primary accepted position p50 is 4.947/5.420/6.493 mm and orientation p50/p95/max is
0.649/1.100/1.658, 0.129/0.604/4.561 and 0.338/0.838/4.078 degrees, respectively.
All-finite primary position p95 is 12.163/12.319/11.824 mm. Bridge rejected-finite
position p50/p95/max is 11.751/25.048/25.385 mm and orientation is
1.186/2.674/2.850 degrees. Integrated rejected-finite position is
8.992/18.765/19.007 mm. Thus bridge improves historical 10 mm availability and
orientation distribution while worsening 15/20 mm acceptance precision and maxima;
there is no uniform improvement.

| Run | Primary correct-gap count / longest observed s at 10, 15, 20 mm | Last 5 s accepted; correct counts at 10/15/20 mm | Final 1 s accepted; correct counts at 10/15/20 mm |
|---|---|---|---|
| Historical | 55 / 2.617; 4 / 0.033; 2 / 0.017 | 301; 8 / 296 / 300 | 61; 0 / 58 / 61 |
| Bridge | 179 / 0.600; 48 / 0.133; 27 / 0.100 | 294; 166 / 235 / 273 | 56; 11 / 20 / 37 |
| Integrated | 166 / 0.317; 49 / 0.117; 31 / 0.033 | 293; 172 / 238 / 269 | 60; 34 / 42 / 48 |

Tail denominators are 301 and 61 original captures. Gap counts include the leading
seed gap where applicable. Historical has a censored 10 mm gap from 76 s through
the end. Bridge's last capture fails all three position bounds; integrated's last
capture fails 10/15 mm. Those final single-capture gaps have zero observed span,
with no recovery or duration beyond the recording inferred. Dense quality gaps
are distinct from native loss and operational skipped acquisitions.

### Matched peaks, refusals and reference history

All three chronological trace audits reproduce native depth lifts, published
poses, selected source points and registration inputs to numerical precision.
Every map-coordinate change has a saved promotion; there are zero graph or
post-frontend map changes. Historical/bridge/integrated have 384/729/667 reference
births and 354/698/636 map changes. Evaluator fixed-only births are **18/469/440**,
including 3/3/9 seed references. The first historical/bridge pose difference already
occurs at 31.016667 s, before any renewal and with 30 references in both. Later
reference histories establish a concrete downstream difference, without proving
that every extra fixed reference causes a particular spike. Membership retains
the inherited 4 mm hull margin and is not operational material ownership.

At **72.066667 s**, bridge has 7.992 mm / 0.091 degrees, 19 pairs (18 moving/one
fixed) and second RMS 226.654 mm; historical has 9.896 mm / 0.134 degrees. The
integrated 23.064 mm / 4.078-degree, 14-moving-pair, 12.415 mm strip and its
pre-SDF fallback do not recur, despite the identical post-31 s masks. Bridge uses
different references born at 58.7–71.0833 s and an SDF-refined cluster 0 with no
count fallback. This is a supported state/history difference, not proof of seed-time causality.

At **78.15 s**, bridge has 8.436 mm / 0.113 degrees with nine pairs (seven moving/
two fixed) and second RMS 239.616 mm, versus historical 13.113 mm / 0.066 degrees.
Integrated retains 45.213 mm / 3.588 degrees with six mixed pairs. Moving-born,
moving-current correspondence displacement median/max is 2.588/4.430 mm bridge
versus 19.162/19.647 mm integrated. Their reference coordinates and selected fits
also differ. Identical fixed-pair counts do not establish identical support or
causality; rigid birth transport remains an evaluator proxy.

All eight original integrated refusals become accepted in bridge, with different
support. Acceptance still does not establish accuracy.

| Capture s | Integrated pairs / second RMS mm | Bridge pairs / second RMS mm | Bridge primary error mm |
|---:|---|---|---:|
| 74.6167 | 8 / 5.832 | 9 / 15.509 | 13.229 |
| 74.6500 | 5 / 5.686 | 17 / 212.716 | 8.802 |
| 75.4000 | 7 / 1.593 | 18 / 230.400 | 7.545 |
| 75.4667 | 5 / 2.350 | 22 / 222.284 | 9.187 |
| 75.9667 | 7 / 7.367 | 15 / 177.636 | 6.834 |
| 76.1167 | 9 / 1.651 | 13 / 270.015 | 11.179 |
| 77.0667 | 5 / 3.794 | 7 / 136.286 | 18.096 |
| 78.0000 | 5 / 0.723 | 13 / 252.907 | 21.102 |

Bridge's 12 different refusals are at 68.9333, 70.6167, 70.7167, 71.1833, 73.4,
76.3333, 76.7, 78.0667, 78.1167, 78.1833, 78.3833 and 78.4667 s. All have 5–8
published measured pairs and second RMS 0.546–6.691 mm. Their pre-SDF supports
have 9–24 pairs, second RMS 32.159–199.430 mm and mixed moving/fixed membership;
SDF leaves narrow, predominantly moving support. All remain unavailable under
the unchanged >8 mm guard. Wider contaminated support is not sufficient grounds
to prefer the pre-SDF pose or relax admission.

Bridge's new angular maximum is **4.561 degrees at 70.516667 s** (25.997 mm
primary). Dense-map selection chooses cluster 2 with saved `3d_dist` 0.9575 versus
1.0 for clusters 0/1. Its eight-pair pre-SDF support has two moving/six fixed pairs
and second RMS 22.022 mm. SDF reduces the internal objective and error to
14.984 mm / 3.763 degrees but leaves four measured pairs. The existing count gate
restores the pre-SDF cluster pose. The support guard does not change the selection.

The new position maximum is **74.549 mm at 78.45 s**, already present before SDF,
with 4.394 degrees, 21 pairs (four moving/17 fixed) and second RMS 35.011 mm.
SDF improves it to 70.686 mm / 3.459 degrees but leaves two pairs; count fallback
restores the selected pre-SDF pose. The four moving pairs have current displacement
median/max only 1.384/2.092 mm, but birth/update bias medians 21.399/21.995 mm.
These vectors must not be added as scalar errors. Better evaluator alternatives
are retained in the traces and cannot be selected by truth in inference.

The final bridge capture is still accepted at 22.567 mm / 2.526 degrees with nine
moving pairs and second RMS **8.017 mm**, just above the unchanged spatial guard;
historical/integrated errors there are 11.792/16.965 mm. The last-second bridge
position p95 is 37.157 mm versus 14.933/30.511 mm historical/integrated. The dense
path therefore retains a materially worse terminal high-tolerance tail even
though its original integrated peaks and refusals disappear.

`event-support-comparison.json`, per-run birth/promotion and paired-point files,
and `paired-metrics.jsonl` retain the matched evidence. Visually inspected
`quality-and-tail.png` and `matched-peak-supports.png` show all finite trajectories,
refusals and corresponding RGB-D inlier support. Native count fallback occurs on
0/31/39 common postseed captures; 17/18 bridge/integrated fallback outputs are
correct at 15 mm. Its association with maxima does not alone justify blanket
fallback rejection or disabling SDF.

### Supported hypothesis and one conditional correction

The controlled historical-to-bridge intervention supports a **subsequent-mask
sequence effect on dense P2P quality and accumulated references**: initialization
is identical, yet results, fixed-born references and registration decisions change.
The integrated-to-bridge difference also supports dependence on **combined P2P
initialization/history**: the same post-31 s masks do not determine the same peaks,
refusals or tail. Restoring the historical seed is not a sufficient quality repair.

The separate contributions of seed time, seed geometry/mask/reference coordinates,
TAPIR/TSDF history, fixed-reference contamination, moving-point bias and cluster
selection remain uncertain. One trajectory per condition does not estimate run
variance. No single quality correction is validated for adoption.

The evidence justifies one focused future candidate: **admit new references only
when their temporal RGB-D/TAPIR motion supports membership in the observed rigid
body**, before they enter/promote into its reference map. Its purpose is to avoid
renewing fixed-frame points as moving-object references. It would need causal
observations with sufficient motion and explicit unsupported cases; evaluator
hulls/truth cannot implement that decision. This is a candidate to qualify, not a
proven filter, and it does not automatically repair moving-only narrow supports
or seed bias. No such correction, threshold change or performance optimization
was implemented. Freshness, sparse loss/recovery, hardware and contact remain
unqualified; this replay evaluates the **dense quality path**, not latency or
operational losses at approximately four Hz.
