# Persistent SAM3 Inspection and Tracking

## Scope and implementation

Baseline `548ab93`; explicit CUDA math milestone `95b831a` and shared runtime
`472f6af`; publication-time preservation `a12fd4b`. The maintained scan,
smoke, live/replay and serial evaluator share `PerceptionSession` and
`configs/point2pose_selected.json`. The same SAM3 state/identity supplies each
processed frame's geometry and P2P masks. Native initialization stays strict,
including measured geometric interiors. Inspection freezes at 25 s; current
full-scene depth remains separate. Models, five inliers/4 mm, actions, policies
and contact thresholds are unchanged. No training, collection or interaction ran.

GroundingDINO, the static box worker and separate SAM2 execution/setup dependency
are retired after migrating their consumers. DINOv3 stays retired. SAM3 internal
tracker components, TAPIR, SuperPoint, registration, TSDF and historical readers
remain. BF16/TF32 are explicit rather than inherited from a SAM2 import.
[[../topics/shared-door-perception|The canonical runtime contract]] describes
selection, missing identity, material recovery, timing and complete reset.

## Historical migration check

Both conditions replayed original rows 1860–1892, 31–31.533333 s, on RTX 4090.
All 33 masks, poses, decisions, reference history and registration traces match
preserved selected evidence exactly. Effective numerical configuration and CUDA
math match; the unused SAM2 checkpoint entry and observational runtime metadata
are the expected differences. `diagnostic_only=False` and no SAM2 import apply.
The final-code comparison repeats this bounded prefix in a separate directory.
All three fixed targets have 96.97% availability at 10/15/20 mm and 5 degrees,
100% accepted precision; the alignment seed remains in the denominator.
This is prefix equivalence, not full-opening or initial-history equivalence.

## Independent inspection-to-tracking check

A separate 20-frame CUDA lifecycle fixture supplies constant-gray RGB only on row
0, then original captures 1–19. The first state is `no_candidate`; row 1 obtains
an identity with SAM3 index 1, no reset or reconditioning. Original timestamps
remain. This fixture assesses search/session continuity only, without geometry
or truth scoring, and is preserved in `empty-first-cuda/`.

Each condition processed all 2,041 original captures from 0 through 34 s. No frame
was resampled or discarded; SAM3 indices 0–2040 and arrived count 2,041 prove one
continuous session. The first light attempt remained `ambiguous_geometry` after
25 s and was interrupted at 28.5 s; original completed/refused/unprocessed rows
remain. Its legacy interrupt label says `user_requested_stop`; a separate note
records the agent's diagnostic stop, not a human request.

Distinct selected attempts explicitly chose the first eligible geometric root
with registered static support, without truth or mask-score/area ownership.
Selection alternatives, static reference and a focused image are saved. Both
initialized at 25.016667 s with all native mask-preservation checks accepted.
There are 540 native outputs, zero native losses and one integration refusal at
the seed for missing measured registration pairs; nonseed outputs remain accepted.
Ownership/material and hardware calibration are unqualified.

The comparable scored interval is **31–34 s only**, 181 original captures per
condition, with the three physical targets frozen by the earlier evaluator. The
new seed camera and evaluator-only rigid motion transport those same targets;
no new zone or per-frame truth alignment substitutes for them.

| Condition | Target | Availability 10/15/20 mm, 5 deg | Accepted precision 10/15/20 mm | Position p95/max mm | Saved selected p95/max mm |
|---|---:|---:|---:|---:|---:|
| light | 0 | 100/100/100% | 100/100/100% | 1.699/2.003 | 1.526/1.771 |
| light | 1 | 100/100/100% | 100/100/100% | 1.687/1.981 | 1.612/1.757 |
| light | 2 | 100/100/100% | 100/100/100% | 1.735/2.051 | 1.513/1.867 |
| nominal | 0 | 100/100/100% | 100/100/100% | 1.770/2.116 | 1.804/2.160 |
| nominal | 1 | 100/100/100% | 100/100/100% | 1.706/2.045 | 1.778/2.165 |
| nominal | 2 | 100/100/100% | 100/100/100% | 1.837/2.208 | 1.828/2.169 |

Rotation p95/max is 0.0828/0.1279 degrees light and 0.1131/0.1467 nominal for all
three targets. Correct-pose gaps and terminal gaps are zero in this short interval.
The saved selected availability is 99.45%, with its 31 s seed excluded from
correctness; the new 25 s seed precedes the scored interval. That denominator
difference is not a quality improvement. Light p95/max position regress slightly;
nominal target2 maximum also increases. Full distributions, peak times, angular
errors, gaps and tails remain in `comparison.json`. These early bounds cannot
predict the saved full-opening failures or late tails.

## Full original openings with inspection history

Both independent selected attempts complete **4,718 original captures, 0–78.6167 s**,
with SAM3 indices 0–4717 and one continuous identity. The recordings include the
original neck/camera scan and a stationary leaf through 25 s; source inspection
is audited separately. Each initializes P2P at 25.016667 s and retains 3,217 native
outputs. Static scene/surface fusion stays frozen after inspection. All masks,
registration traces, failed supports and original acquisition timestamps survive.
The full attempts exactly reproduce their corresponding 0–34 s mask/pose prefixes.

The matched scored opening is **31–78.6167 s, 2,858 scheduled captures**, on the same
three frozen physical targets. Both observed openings end at 63.7209 degrees.
Availability means correct accepted/scheduled; precision means correct/accepted.
Each slash lists 10/15/20 mm with the same 5-degree bound. Saved selected results
retain their original 31 s seed in the scheduled denominator; the new seed lies
before this interval. The denominator difference does not establish improvement.

| Condition | Target | New availability % | New precision % | Saved availability % | Saved precision % | New position p95/max mm | Saved p95/max mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0 | 87.58/97.55/98.81 | 87.82/97.82/99.09 | 85.83/99.79/99.93 | 85.86/99.82/99.96 | 11.814/45.213 | 12.163/23.822 |
| light | 1 | 86.53/97.55/98.74 | 86.77/97.82/99.02 | 87.12/99.90/99.93 | 87.15/99.93/99.96 | 12.147/44.999 | 11.985/24.444 |
| light | 2 | 87.54/97.38/98.78 | 87.79/97.65/99.05 | 83.80/99.58/99.93 | 83.83/99.61/99.96 | 11.652/46.617 | 12.385/23.727 |
| nominal | 0 | 98.25/99.79/100.00 | 98.25/99.79/100.00 | 99.13/99.72/99.86 | 99.19/99.79/99.93 | 8.524/19.557 | 7.435/31.475 |
| nominal | 1 | 98.64/99.83/100.00 | 98.64/99.83/100.00 | 99.13/99.69/99.86 | 99.19/99.75/99.93 | 8.246/18.468 | 7.083/30.199 |
| nominal | 2 | 97.62/99.76/99.97 | 97.62/99.76/99.97 | 98.74/99.65/99.83 | 98.81/99.72/99.89 | 8.834/20.933 | 7.985/32.814 |

There are zero SAM3/native losses in both attempts. Light rejects eight nonseed
outputs at 74.6167, 74.6500, 75.4000, 75.4667, 75.9667, 76.1167, 77.0667 and 78.0000 s:
5–9 measured pairs have degenerate geometric spread. Nominal rejects no nonseed
output. The alignment seed in each attempt lacks measured pairs and is reported
separately, without weakening initialization. Accepted availability on the matched
interval is 99.72% light and 100% nominal; that is not correct-pose availability.

Rotation p95/max is **0.838/4.078 degrees light**, with its maximum at 72.0667 s,
and **0.419/1.280 nominal**, maximum at 57.3500 s, for all three targets. Position
maxima occur at 78.1500 s light and 76.2667 s nominal. All-finite/rejected distributions,
p99, histograms, individual chronological intervals and peak times are retained
in `comparison.json`; finite fallback outputs are never silently counted correct.

| Condition | Target | Longest correct-pose gap 10/15/20 mm, ms | Correct final 5 s at 10/15/20 mm, each of 301 |
|---|---:|---:|---:|
| light | 0 | 317/117/33 | 172/238/269 (of 301) |
| light | 1 | 317/117/33 | 164/239/267 (of 301) |
| light | 2 | 317/117/67 | 173/234/267 (of 301) |
| nominal | 0 | 67/17/0 | 273/296/301 (of 301) |
| nominal | 1 | 67/17/0 | 279/297/301 (of 301) |
| nominal | 2 | 150/17/17 | 264/295/300 (of 301) |

Light ends with one censored missing-correct-pose capture at 10/15 mm for every
target; its observed span is zero, not a claim about unrecorded recovery. At 20 mm
there is no terminal gap. Nominal has no terminal gap at any bound. Light's 10 mm
primary tail improves from the saved 8/301 to 172/301, but 15/20 mm availability,
position maxima and peak rotation regress. Nominal improves its maxima and 15/20 mm
availability while median/p95 position and 10 mm availability regress. The earlier
quality evidence is not transferred to the integrated session.

Effective configuration and BF16/TF32 match the historical-prefix runtime exactly.
The inspection history and P2P seed/history differ: sampled mask IoU against saved
selected masks changes from 0.925 to 0.817 light, 0.941 to 0.854 nominal between
31 s and the end. The measured reference/keyframe histories and degenerate supports
are retained in `history-audit.json`. This distinguishes the new history from a
configuration migration error; it does not identify a single causal mechanism for
the changed errors. No compensating crop, re-prompt, numerical variant or relaxed
threshold was introduced. Ownership/material remain unqualified.

## Timing and residency

Serial nonseed complete request p50/p95 is 214/235 ms light and 215/236 ms nominal;
none of 539 nonseed requests meets 150 ms. SAM3/geometry/P2P and diagnostic export
are timed. The light transition predates the final publication-adapter timing
addition, so its service time is a lower bound for complete operational publication.
There is no serial backlog or dropping. Joint sampled GPU peaks are 6.92/6.91 GiB;
coordinator plus worker RSS peaks are 5.52/5.46 GiB at 250 ms sampling, with no
sampler errors. Allocator and per-process high-watermark fields remain separate;
startup allocation spikes remain unqualified. These short figures are not the
full-opening resource measurement below.

The full attempts measure the unchanged common serial coordinator plus the same
publication adapter, independently from truth scoring. Nonseed p50/p95/max is
227/260/910 ms light and
213/246/1209 ms nominal. **0/3,216 per condition meets 150 ms.** There are no serial
drops or queue waits. Inspection geometry maxima reach 4.525/4.298 s light/nominal;
stage timings preserve
SAM3, geometry/seed, native IPC and identity binding separately. Joint sampled
GPU peaks are 7.77/11.81 GiB and coordinator-plus-workers RSS peaks
6.03/6.10 GiB, with zero sampler errors. These are simultaneous post-ready
samples at 250 ms, not a sum of unrelated process high-watermarks or a startup
spike certification. No causal memory improvement over a different seed is claimed.

A separate CUDA wall-clock ingress and operational replay used the same 33 source
captures/31 s initialization. Each dispatched/published two captures and replaced
31 waiting inputs, retaining one in-flight plus one latest slot. Complete
acquisition-to-publication latencies are 3.270/2.998 s wall-clock and 3.180/2.904 s
replay, including cold seed geometry/P2P, queue waiting and adapter publication.
Only original frame IDs 1868 and 1900 published; skipped IDs remain explicit.
The final publication-time correction was checked independently on CUDA in
`operational-final/`: the same queue counts, complete latencies 3.204/2.931 s
wall-clock and 3.163/2.881 s replay. The seed geometry/static-reference availability
equals its complete publication and remains unchanged by the second capture;
unit checks preserve old scan publication times across snapshots and clear them
on reset. This metadata correction follows the full serial runs and changes no
SAM3/P2P numerical configuration, masks or inference path.
The wall-clock producer uses recorded captures, without an Isaac renderer. This
validates bounded temporal integration, not concurrent rendering or live accuracy.
No output meets 150 ms.

## Evidence and remaining validation

Evidence and read-only comparison tooling are in
`outputs/evidence/b1/perception/persistent-sam3-20261008/`, including the ambiguous
attempt, selected attempts, all new masks, traces, timing and resource samples.
Older recordings, baselines, images, masks, failed tails and readers are intact.
Targeted unit/local integration contracts, lint and formatting pass;
real CUDA is used for the model/TSDF checks above.

Next validate matched operational replay/live with actual rendering and an explicit
observed geometric selection, retaining the same targets, timestamps, drops and
complete publication latency. Diagnose the saved light peaks/degenerate supports
before proposing any separate quality change. The integrated full openings are
not numerically equivalent to the historical 31 s windows. Independent material
recovery, absolute initialization, hardware/FK/floor bounds, 150 ms and action/contact
qualification remain open.
