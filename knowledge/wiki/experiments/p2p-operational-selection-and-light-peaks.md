# SAM3 Selection, Operational Timing and Integrated Light Peaks

## Maintained selection and measurement

Baseline `a37fc18`; selection implementation `402b832`. The selected recipe,
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

Pending the ordered saved-data audit of 72.0667 s, 78.15 s and eight support refusals.
No quality correction or relaxed guard is introduced.
