# Unified SAM3 Video Frontend for Point2Pose

## Decision and protocol

The October 7 complete comparison selected **one original-object SAM3 with
reconditioning OFF** as the experimental reference. It balanced precision,
continuity and complete latency; it did not win every metric. Subsequent
[[p2p-sam3-selected-development|selected development]] separately adopted bounded
SAM3 history and renewal-only depth filtering. Tables here retain the original
unbounded SAM3 OFF configuration. Operational defaults and contact qualification
are unchanged. SAM3.1 and its memory implementation were retired in `c01c3a1`;
source remains in historical Git (`57ad483`), and all ignored evidence is intact.
The current recipe/setup is in the model README and `configs/point2pose_selected.json`.

Baseline `cc9cfcd`; consolidation baseline `2f3ad07`, source `563847a`. The older
[[p2p-evaluation-and-performance|all-query baseline]] uses five light/six nominal
overlapping automatic candidates. Unified paths replace initialization and subsequent
masks together with one full-leaf identity; this is a pipeline comparison, not a
single isolated mask-model effect. P2P retains full-frame TAPIR 480/four iterations
(eight passes)/all-query, SuperPoint, all registrations, graph OFF, partial batches
ON, rollback OFF, five inliers/4 mm, vectorized SDF and 120 active references.

Input: `engineering-v2/animated-door-1-88abf40/{light,nominal}/episode.hdf5`, original
rows 1860–4717 at 60 Hz, 31–78.6166667 s, 2,858 captures per condition, maximum
recorded opening 63.7209 degrees. Original world target definitions come from
`p2p-performance-01/full-target-evaluation.json`; seed-relative alignment scores
cannot replace them. Ground truth enters only evaluators. Missing/failed tails,
conditional errors, material compatibility, complete latency and resources remain
separate. The seed is excluded from correctness/errors and retained in denominators.

The pinned [official SAM3 source](https://github.com/facebookresearch/sam3) is `2345a4ad109ac29c569da749c91d84f10dc08c40`, SAM3
weights `3c879f39826c281e95690f02c7821c4de09afae7`, historical SAM3.1 multiplex weights
`daa63191845a41281374e725f4c9e51c7a824460`. RTX 4090 uses official PyTorch attention
without compilation/FlashAttention3. Workers consume only arrived RGB, never future
lookahead, and pass synchronized masks directly to P2P. Missing original ID0 means
empty mask/loss; new IDs never substitute. Separate SAM2/DINOv3 inference is skipped
only in this path; measured RGB-D geometry, TAPIR and SuperPoint stay unchanged.

## Consolidated four-way comparison

Bounded SAM3.1 OFF changed only native period 16→0; its memory policy, seed0,
`door surface`, initialization and P2P settings stayed fixed. Both complete CUDA
windows finished without retry. Seed masks/poses match ON exactly; no new IDs or
native reconditioning calls appear. First differing masks/poses are at indices
912/272 (46.2/35.5333 s). Packed ON masks cover 1,313/1,151 frames, with six late
snapshots for the rest; this does not imply a full saved ON mask sequence.

SAM3.1 OFF improves light target0 15 mm availability 73.90→97.97% versus ON, but
light target1 at 10 mm regresses. All nominal targets regress: target0 15 mm drops
86.18→83.24%, final correct tail 193→36/301. Its single native loss still permits
a 10.85 s correct-10 mm gap. SAM3 OFF has 57 nominal losses but 87.40% correct
10 mm availability and only 0.150 s maximum correct gap. Native presence alone
is not useful continuity. SAM3 OFF's 43.10 mm nominal accepted peak, weaker light
accuracy than the old baseline and growing memory remain explicit tradeoffs.

Evidence: `p2p-unified-consolidation-01/comparison.json` and `report.md`. Shutdown-only
RSS sampler exceptions are retained; 6,427/6,431 samples cover the final capture,
with all per-frame records intact. Neither inference attempt failed or was rerun.
New instantaneous combined RSS medians rise 6.56→7.14 GiB light and 5.38→5.92 GiB
nominal; bounded SAM buffers do not imply zero whole-pipeline CPU growth.

### All three original physical targets

Each positional bound retains 5-degree rotation. Availability is correct accepted / all 2,858 scheduled frames; precision is correct / accepted nonseed poses. Original seed, timestamps and failures remain. Target definitions are frozen from the older all-query baseline.

| Condition | System | Target | Availability 10/15/20 mm | Accepted precision 10/15/20 mm | Accepted position p95/max (mm) |
|---|---|---:|---:|---:|---:|
| light | Baseline | 0 | 80.41/98.22/99.90% | 80.43/98.25/99.93% | 13.29/21.78 |
| light | Baseline | 1 | 78.69/97.90/99.90% | 78.72/97.93/99.93% | 13.46/21.55 |
| light | Baseline | 2 | 81.88/98.36/99.90% | 81.90/98.39/99.93% | 13.10/22.04 |
| light | SAM3 OFF | 0 | 70.33/96.19/99.86% | 70.35/96.22/99.89% | 14.62/21.26 |
| light | SAM3 OFF | 1 | 69.03/94.05/99.76% | 69.06/94.08/99.79% | 15.23/21.63 |
| light | SAM3 OFF | 2 | 71.66/97.13/99.90% | 71.68/97.16/99.93% | 14.02/20.89 |
| light | SAM3.1 bounded ON | 0 | 68.19/73.90/94.79% | 68.24/73.95/94.85% | 20.03/27.50 |
| light | SAM3.1 bounded ON | 1 | 66.59/73.30/93.60% | 66.63/73.35/93.66% | 20.53/27.66 |
| light | SAM3.1 bounded ON | 2 | 69.03/75.23/95.59% | 69.08/75.28/95.66% | 19.45/27.33 |
| light | SAM3.1 bounded OFF | 0 | 68.82/97.97/99.83% | 68.87/98.04/99.89% | 14.22/20.90 |
| light | SAM3.1 bounded OFF | 1 | 62.35/91.92/99.72% | 62.39/91.98/99.79% | 15.37/21.34 |
| light | SAM3.1 bounded OFF | 2 | 73.55/99.09/99.90% | 73.60/99.16/99.96% | 13.11/20.47 |
| nominal | Baseline | 0 | 68.89/98.15/99.83% | 68.92/98.18/99.86% | 13.25/22.89 |
| nominal | Baseline | 1 | 63.58/97.62/99.79% | 63.60/97.65/99.82% | 13.79/22.30 |
| nominal | Baseline | 2 | 75.26/98.25/99.79% | 75.29/98.28/99.82% | 12.85/23.53 |
| nominal | SAM3 OFF | 0 | 87.40/96.54/97.73% | 89.21/98.54/99.75% | 11.71/43.10 |
| nominal | SAM3 OFF | 1 | 86.21/96.50/97.76% | 88.00/98.50/99.79% | 11.83/40.67 |
| nominal | SAM3 OFF | 2 | 87.82/96.68/97.76% | 89.64/98.68/99.79% | 11.69/45.66 |
| nominal | SAM3.1 bounded ON | 0 | 59.90/86.18/98.92% | 59.94/86.24/98.98% | 17.38/24.63 |
| nominal | SAM3.1 bounded ON | 1 | 59.76/82.54/98.74% | 59.80/82.60/98.81% | 17.69/24.33 |
| nominal | SAM3.1 bounded ON | 2 | 60.08/88.24/99.02% | 60.12/88.31/99.09% | 17.17/24.97 |
| nominal | SAM3.1 bounded OFF | 0 | 56.82/83.24/94.58% | 56.86/83.30/94.64% | 20.19/29.09 |
| nominal | SAM3.1 bounded OFF | 1 | 56.09/81.46/94.30% | 56.13/81.51/94.36% | 20.55/28.93 |
| nominal | SAM3.1 bounded OFF | 2 | 57.21/85.41/95.10% | 57.25/85.47/95.17% | 19.80/29.27 |

### Primary distributions, losses and final tail

Primary is original target0. All three complete distributions, histograms, positional/rotational peak times, correct-gap intervals and terminal gaps remain in `comparison.json` and the complete `report.md`; no seed-relative substitute is used. Accepted distributions are conditional, so all-finite p95/max and loss counts are also shown.

| Condition | System | Accepted position p50/p95/p99/max (mm) | Accepted rotation p50/p95/p99/max (deg) | All-finite position p95/max (mm) | Native losses |
|---|---|---:|---:|---:|---:|
| light | Baseline | 3.87/13.29/15.81/21.78 | 0.178/0.444/0.551/0.793 | 13.29/21.78 | 0 |
| light | SAM3 OFF | 4.37/14.62/17.74/21.26 | 0.472/0.688/0.802/1.044 | 14.62/21.26 | 0 |
| light | SAM3.1 bounded ON | 5.62/20.03/23.25/27.50 | 0.301/0.537/0.664/1.102 | 20.03/27.50 | 1 |
| light | SAM3.1 bounded OFF | 6.42/14.22/15.76/20.90 | 0.496/0.888/1.014/1.218 | 14.22/20.90 | 1 |
| nominal | Baseline | 6.24/13.25/16.42/22.89 | 0.156/0.588/0.748/1.056 | 13.25/22.89 | 0 |
| nominal | SAM3 OFF | 4.45/11.71/15.50/43.10 | 0.104/0.471/0.978/3.850 | 11.83/43.10 | 57 |
| nominal | SAM3.1 bounded ON | 4.88/17.38/19.99/24.63 | 0.114/0.512/0.658/1.048 | 17.38/24.63 | 1 |
| nominal | SAM3.1 bounded OFF | 4.88/20.19/23.38/29.09 | 0.165/0.425/0.634/1.124 | 20.19/29.09 | 1 |

| Condition | System | Maximum correct gap 10/15/20 mm (s) | Last 5 s correct 10/15/20 mm out of 301 |
|---|---|---:|---:|
| light | Baseline | 0.583/0.033/0.017 | 100/294/301 |
| light | SAM3 OFF | 3.100/0.100/0.017 | 2/212/298 |
| light | SAM3.1 bounded ON | 12.200/3.400/0.067 | 0/3/226 |
| light | SAM3.1 bounded OFF | 3.100/0.050/0.033 | 15/279/299 |
| nominal | Baseline | 0.817/0.050/0.017 | 91/294/301 |
| nominal | SAM3 OFF | 0.150/0.117/0.117 | 153/232/241 |
| nominal | SAM3.1 bounded ON | 9.200/0.283/0.067 | 1/193/293 |
| nominal | SAM3.1 bounded OFF | 10.850/0.550/0.150 | 0/36/169 |

### Complete latency and resource peaks

Nonseed complete request time includes SAM inference/IPC, P2P, copies, export and adapter consumption. Startup and seed remain separate in the full report. Baseline has five/six objects; unified paths have one, so baseline differences are complete pipeline comparisons. GPU/CPU simultaneous samples cover SAM and P2P workers; coordinator RSS is separate. Per-process allocator/RSS peaks include startup and are not summed. The old baseline lacks contemporaneous image-worker measures; its separately sampled P2P GPU peak is 9.86/12.09 GiB. No scheduling/live-frequency experiment is inferred.

| Condition | System | Complete p50/p95/max (ms) | Sampled simultaneous GPU/CPU peak (GiB) | SAM/P2P allocator peak (GiB) | SAM/P2P CPU RSS peak (GiB) |
|---|---|---:|---:|---:|---:|
| light | Baseline | 438/554/1544 | unavailable | unavailable/7.17 | unavailable/4.46 |
| light | SAM3 OFF | 232/264/1003 | 14.41/22.86 | 8.39/1.17 | 20.21/2.65 |
| light | SAM3.1 bounded ON | 252/282/1220 | 11.33/5.93 | 5.06/1.17 | 7.50/2.69 |
| light | SAM3.1 bounded OFF | 253/282/1303 | 11.21/7.16 | 5.06/1.17 | 7.50/2.77 |
| nominal | Baseline | 402/644/4069 | unavailable | unavailable/7.93 | unavailable/6.12 |
| nominal | SAM3 OFF | 233/281/978 | 12.92/21.40 | 8.39/1.17 | 18.51/2.92 |
| nominal | SAM3.1 bounded ON | 254/285/1014 | 10.18/5.91 | 5.06/1.17 | 7.50/2.67 |
| nominal | SAM3.1 bounded OFF | 254/283/1003 | 10.83/5.94 | 5.06/1.17 | 7.50/2.70 |

### Memory growth, separate from peaks

SAM current CUDA allocation is summarized over 31–36, 43–48, 55–60 and the last five seconds. CPU entries from old runs are high-water marks including checkpoint loading, not instantaneous RSS; a flat mark cannot prove flat live RSS. New OFF retains separate sampled current RSS in `memory-samples.jsonl`. Current P2P allocation/residency and all four-window quantiles are also retained in JSON and the memory figure.

| Condition | System | SAM window median CUDA allocation (GiB) | SAM CPU high-water first→final (GiB) | P2P process GPU first→final (GiB) |
|---|---|---:|---:|---:|
| light | Baseline | unavailable | unavailable | 4.19→9.86 |
| light | SAM3 OFF | 3.87/4.92/5.98/7.62 | 7.46→20.21 | 1.51→4.68 |
| light | SAM3.1 bounded ON | 4.40/4.40/4.40/4.40 | 7.50→7.50 | 1.51→5.22 |
| light | SAM3.1 bounded OFF | 4.40/4.40/4.40/4.40 | 7.50→7.50 | 1.51→5.26 |
| nominal | Baseline | unavailable | unavailable | 4.35→12.09 |
| nominal | SAM3 OFF | 3.87/4.92/5.98/7.62 | 7.46→18.51 | 1.50→3.19 |
| nominal | SAM3.1 bounded ON | 4.40/4.40/4.40/4.40 | 7.50→7.50 | 1.50→4.07 |
| nominal | SAM3.1 bounded OFF | 4.40/4.40/4.40/4.40 | 7.50→7.50 | 1.50→4.88 |

Figures: `complete-comparison.png`, `three-target-accepted-cdf.png` and `memory-over-time.png` in the consolidation evidence root. The full report names every measure, includes startup/seed and coordinator RSS, and retains all three targets rather than only the primary summary.

## Earlier experiments establishing the decision

### Prompt and integration

`door leaf`, `door slab` and `door without frame` failed to seed either condition;
`door panel` selected decorative insets. Targeted human overlays supported
`door surface` for the full leaf. The original SAM2 primary already covered the
full leaf (about 338k light pixels); it was not just an inset. Small fixed-frame
overlap alone was therefore not a sufficient rejection criterion. Earlier mask-only
summaries are superseded by `full-opening-admission.json`, which requires causal
integration scored on the original physical targets.

Prepared-hull membership is evaluator-only, approximate and excludes overlap or
unclassified pixels. At 31/33/36 s, SAM3.1 retained about 98% of moving-only pixels
with 0.95–4.32% fixed-only inclusion. SAM3 later sampled 8.8/11.4% fixed overlap.
An additional `door panel` pilot retained two light insets but only the upper nominal
inset; it added no full-leaf support. A new native text concept resets state;
multiple IDs are not independent combined prompts. No union/fallback was adopted.

Corrected SAM3.1 frontend pilots completed 301/301 captures per condition, native
median/p95 about 125/128 ms, allocator peak 7.41 GiB. SAM3 pilots completed 601 each,
median 107–109 ms, peak 5.09 GiB with greater fixed inclusion. Integrated SAM3.1
601-frame pilots achieved 100% accepted precision on all three targets at
10/15/20 mm and 5 degrees, 99.67% availability, one confirmation loss followed by
a same-ID return. SAM3 integration pilots had no native losses but retained a
38.19 mm nominal target0 error at 40.8667 s. Prefix success admitted full evaluation,
not operational freshness. Initial adapter/timing failures remain preserved.

### SAM3.1 resource failures and history correction

The offline 16-frame grounding default cached repeatedly overlapping causal
prefixes: both initial attempts failed at frame nine after eight outputs, allocation
3.93→16.17 GiB. One-arrived-frame grounding fixed that input/cache mismatch without
changing model resolution or gates. With that repair, full SAM3.1 still failed the
unchanged TSDF reserve: 1,312 light outputs through 52.85 s and 1,150 nominal through
50.15 s, refusals at 52.8667/50.1667 s. ID0/masks were still present. Growth was
about 8.50 MiB per arrived frame; roughly 144.4 MiB TSDF requests could not preserve
the 20% device reserve. Free memory was 1.43/3.06 GiB at refusal.

Before failure, all three targets had 100% accepted precision at every bound;
full scheduled availability was only 45.84/40.17%, missing tails 25.75/28.45 s.
`sam31-resource-failure-audit.json` retains these resource failures separately
from native loss. Prefix accuracy cannot substitute for full-window completion.

The follow-up (`5f33411`, `21c2abf`, `72532e1`) retained attention-relevant six spatial
memories, fifteen nonconditioning pointers, up to four nearest conditioning frames
and the original conditioning seed. Sixteen recent outputs, their aliases and
persistent confirmation/identity metadata survived; expired outputs, masks and RGB
payloads were released without unbounded CPU offload or changed frame indices.
Reverse propagation/new prompts were outside the forward adapter contract.

All 1,313/1,151 frontend masks/probabilities/IDs/state records and all 1,312/1,150
completed P2P poses/decisions exactly matched the failed prefixes. Allocator peaks
fell from 15.81/14.47 to 5.06 GiB; current SAM allocation stabilized at 4.40 GiB and
peak CPU RSS was 7.51 GiB including checkpoint loading. Both 2,858-frame runs then
completed with one confirmation loss each, but late pose bias remained. This
established a resource fix, not improved material tracking. Detailed equality and
full results remain in `p2p-saved-prefix-equivalence.json` and
`sam31-bounded-full-comparison.json`.

### Isolated SAM3 reconditioning

`f3d8797` changed periodic detection updates 16→0, preserving the already disabled
bbox trigger (-1), prompt, seed, thresholds and native association. Four complete
runs exactly reproduced ON poses, decisions and point traces before comparing OFF.
ON applied 39/26 updates, first at 35.0/31.8 s; OFF applied zero, with first changed
subsequent masks at 35.0167/31.8167 s. Every run kept only ID0. Initial 33-frame
trials had identical masks and no eligible updates; the failed early-application
expectation is retained rather than retuned.

| Condition | SAM3 period | Correct availability 10/15/20 mm | Accepted precision 10/15/20 mm | Maximum correct gap 10/15/20 mm (s) | Last 5 s correct 10/15/20 mm |
|---|---|---:|---:|---:|---:|
| light | 16 (ON) | 57.73/94.16/98.92% | 57.85/94.35/99.12% | 3.817/0.167/0.050 | 9/209/280 |
| light | 0 (OFF) | 70.33/96.19/99.86% | 70.35/96.22/99.89% | 3.100/0.100/0.017 | 2/212/298 |
| nominal | 16 (ON) | 70.40/73.58/73.65% | 95.36/99.67/99.76% | 6.350/6.350/6.350 | 0/0/0 |
| nominal | 0 (OFF) | 87.40/96.54/97.73% | 89.21/98.54/99.75% | 0.150/0.117/0.117 | 153/232/241 |

Native nonseed losses change from 5 to 0 light and 747 to 57 nominal. OFF's last
nominal capture is native-valid; ON loses the terminal 72.2667–78.6167 s. This is
native support return, not qualified same-material recovery. ON nominal accepted
position p95 is 9.83 mm, but all finite poses have p95 68.03 mm: its conditional
precision hides the difficult lost range. OFF has 11.71/11.83 mm respectively.
The light 10 mm tail does not improve: OFF has 2/301 correct final captures versus
9/301 ON, despite better full-window availability. Average gains do not erase this
retained late failure.

OFF improves ON availability at all bounds on all three targets, but not all
tails or baseline comparisons. Full figures/resources are in `final-comparison.json`;
SAM3 ON complete p95 was 282/286 ms, allocator peak 8.41 GiB. Every nonseed request
exceeded 150 ms. Original ON mask-ID persistence coexisted with 747 nominal P2P
losses; semantic ID and native registration support are distinct.

### Fixed-frame inclusion and material limits

The evaluator uses exclusive moving/fixed hull membership with a 4 mm margin at
original capture time. Initial fixed-only points are 0/30 light and 3/30 nominal
for the baseline, versus 3/30 and 4/30 for direct SAM3. Fixed inliers first appear
at 32.65/31.0167 s, before reconditioning. Newly sampled fixed references appear
at 35.4167 s light in both ON/OFF (1/30), nominal 36.15 s ON (12/30) versus
36.2333 s OFF (7/30). Maximum fixed-only frontend inliers are 22/30 ON, 14/15 OFF
and 6/3 baseline (light/nominal). Reconditioning reduces but does not explain all
contamination; initialization and renewal already admit fixed points.

At nominal ON's last frame, 24 measurements include six fixed/fourteen moving/four
other points, but zero frontend/published inliers. Active count is 120, historical
442; no birth occurs. All 747 ON and 57 OFF nominal lost rows have zero renewal:
existing loss/published-support guards prevent replenishment. This is not evidence
to relax five inliers/4 mm.

At 71 s original SAM3 ON includes 10.34/24.08% fixed-only pixels versus SAM2
0.21/0.00%, with moving recall 98.5–98.9%; nominal also includes unclassified floor.
The baseline itself reaches 11.45% fixed-only pixels at the final light capture.
These are approximate observations, not an isolated causal attribution to masks.

None of original SAM3 ON's four light/67 nominal native returns has five compatible
original references, although correct-frame compatibility occurs on 367/373 frames
versus 437/427 baseline. Renewal can remove those IDs, so absence is unresolved
material identity rather than proven substitution. SAM3.1's confirmation return
has 14/19 compatible original inliers; it does not qualify physical-occluder recovery.

`input-mask-audit.json` verifies source rows/frame IDs/times, masks and external
mode. `consumer-transform-audit.json` agrees with transported zone transforms below
7e-16. A missing optional SAM3 `img_ids_np` alias was corrected at `740ee57`; its
initial failed attempt remains. No integration defect explains the remaining
pose errors. Initialized points, one versus multiple candidates, propagated masks,
TAPIR correspondence and TSDF/reference history remain coupled in frontend
comparisons. Later selected residual diagnosis addresses these separately.

## Preserved evidence and historical source

All roots are under ignored `outputs/evidence/b1/perception/`; no October 8 payload deletion:

- `p2p-sam3-unified-01/`: `integrated-{full,pilot}-{light,nominal}` (SAM3.1),
  `integrated-sam3-{full,pilot}-{light,nominal}` (SAM3), protocols, every scheduled
  row, startup/adapter/resource/timing failures, masks and registration traces.
  `full-comparison.png`, `original-target-error-cdf.png`,
  `mask-reference-versus-sam31-light-41s.png` and
  `late-mask-reference-versus-sam3-nominal-2400.png` retain visual evidence.
- `p2p-sam3-followup-01/`: controlled complete results, exact ON/prefix checks,
  `point-membership-*-summary.json`/JSONL and `sam3-controlled-diagnosis.png`.
  **`hull_evaluator.py` remains a dependency of the later residual diagnosis**;
  prepared collision geometry/recorded motion enter that evaluator only.
- `p2p-unified-consolidation-01/`: all four systems/all three targets, distributions,
  gaps/tails/peaks, original-time plots, startup/seed and separate resource measures.
- `p2p-performance-01/full/chunkall/{light,nominal}/attempt-1`: original baseline;
  `full-target-evaluation.json` remains the frozen target-definition dependency.

Implementation milestones: `7404a41` (causal/direct-mask frontend), `740ee57`
(optional alias), `21c2abf`/`72532e1` (SAM3.1 history/completion), `f3d8797`
(SAM3 controls/trace), `563847a` (SAM3.1 OFF consolidation). Their Git history
reconstructs retired implementations; retained local checkpoints/evidence remain
necessary. Current behavior and future acceptance requirements belong in the
perception topic/phase, not these historical recipes.
