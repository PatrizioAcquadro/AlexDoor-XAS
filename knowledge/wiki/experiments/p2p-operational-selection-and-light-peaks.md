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

## Remaining uncertainty and single next comparison

The evidence supports distinct contributions from spatial conditioning, birth and
promotion bias, current correspondence displacement, cluster choice and SDF/count
fallback. It does not separate the original SAM3 inspection history from the changed
P2P initialization/history, or justify one common observable quality correction.
Sparse native loss, fresh-output failure, material recovery and hardware/contact
qualification remain open; the selected recipe stays unchanged.

The smallest next discriminating intervention is **one saved-mask CUDA P2P bridge
replay**, not another SAM/model campaign. At 31 s initialize with the exact historical
31 s RGB-D and seed mask, then feed the saved persistent-inspection masks from the
next original capture through the end, retaining every timestamp and failure.
Verify identical historical P2P seed references/pose before interpreting it.
Compare that single bridge with the preserved historical run to isolate subsequent
mask-history effects with P2P initialization held fixed. Compare it with the existing
25 s integrated run, which uses the same post-31 s masks, to test the combined P2P
seed/pre-31 s state effect. This second difference includes seed time/mask/reference
history; it does not isolate seed time alone. Use the same three frozen physical
targets and original denominators, with GT only in scoring. Full chronological input
from 31 s is necessary because the late peaks depend on accumulated state. This
comparison is specified only; it has not been run. No quality/performance correction,
new model, training or interaction follows automatically.
