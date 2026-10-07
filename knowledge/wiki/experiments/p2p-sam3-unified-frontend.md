# Unified SAM3 Video Frontend for Point2Pose

**Decision, October 7, 2026: retain the adopted all-query reference.** Unified
SAM3 completes both openings but regresses common accuracy and the nominal tail.
SAM3.1 has accurate completed prefixes but fails the unchanged TSDF memory guard.
The implemented path remains opt-in and unqualified; no operational default changes.

## Scope and reference

Baseline: `main` at `cc9cfcd`, with no pre-existing tracked changes. This is an
opt-in serial offline diagnostic, with A3/A4, policies and operational admission
unchanged. The maintained 6.0B image worker remains separate. Existing recordings,
baselines, failures and local models are preserved in ignored storage.

The reference is the adopted all-query TAPIR profile from
[[p2p-evaluation-and-performance|Point2Pose evaluation and performance]]:
full-frame 480, four iterations/eight passes, SuperPoint, all registrations,
graph off, partial reference batch on, rollback off, five inliers and 4 mm,
vectorized SDF Jacobian and 120 active references per object. Other native
parameters, RGB-D geometry, registration and TSDF remain unchanged. The new path
changes semantic initialization and subsequent masks together: one full leaf
identity replaces the reference's five overlapping automatic candidates.

Local evidence: `outputs/b1/perception/p2p-sam3-unified-01/`. Reference:
`outputs/b1/perception/p2p-performance-01/full/chunkall/{light,nominal}/attempt-1`.
Original input is `engineering-v2/animated-door-1-88abf40/{light,nominal}/episode.hdf5`,
rows 1860–4717, 31.0–78.6166667 s, 2,858 frames per condition at original 60 Hz.
The recorded maximum opening is 63.7209 degrees. Replay speed is independent of
acquisition cadence.

## Official frontend and causal adaptation

Use the [official SAM3 implementation](https://github.com/facebookresearch/sam3)
at `2345a4ad109ac29c569da749c91d84f10dc08c40`, including the recommended
[SAM3.1 multiplex builder](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
SAM3 weights remain at `3c879f39826c281e95690f02c7821c4de09afae7`;
SAM3.1 multiplex weights use `daa63191845a41281374e725f4c9e51c7a824460`.
The [EfficientSAM3 release](https://github.com/SimonZeng7108/efficientsam3) supplies
image checkpoints; its video-memory release is listed as future work. No verified
video tracker is available there for this experiment.

`sam3_frontend.py` appends only the just-arrived RGB image and current frame
metadata to native state. Initialization receives one image and one text prompt.
Forward propagation starts at the newest available frame and returns exactly that
frame. It never supplies future RGB or replays earlier outputs using future
lookahead. Native confirmation, association, detection, reconditioning and memory
rules remain. Episode shutdown destroys the state.

SAM3.1's offline 16-frame grounding default retained repeatedly overlapping
prefix batches as the causal available endpoint changed. Both first attempts
failed on frame nine after eight outputs; allocated memory rose from about 3.93
to 16.17 GiB. These attempts and their unavailable tails remain. Setting grounding
to one arrived frame fixes the input/cache mismatch; it does not reduce model
resolution, relax gates or introduce a fallback. Compilation and FlashAttention3
are disabled; the RTX 4090 uses the official PyTorch attention path.

Initial native IDs are immutable. A missing seed ID produces an explicit loss and
an empty mask for that same P2P object. New IDs cannot substitute for the seed.
A same-ID return is recorded separately from proof of original material recovery.
No re-prompting, automatic restart, second segmentation or geometric mask clipping
is used.

## Prompt and bounded GPU pilots

The same text is used in light and nominal. `door leaf`, `door slab` and
`door without frame` produce no seed in either condition. `door panel` selects
decorative insets rather than the entire mobile leaf. `door` and `door surface`
produce whole-leaf candidates. Human inspection of targeted overlays and the
reference SAM3-to-SAM2 masks supports retaining `door surface`.

The old primary SAM2 initialization already covers the full leaf (about 338k
pixels in light), with overlapping alternatives; it was not only a decorative
inset. Small fixed-frame overlap alone is therefore not an automatic rejection.
Earlier mask-only rejection summaries are preserved but superseded by
`full-opening-admission.json`, which requires causal tracking and successful P2P
integration evaluated on the original physical targets. Five inliers/4 mm and
all numeric acceptance thresholds remain fixed.

Prepared collision hull membership is used only in a separate evaluator to
inspect moving-only recall and fixed-frame contamination. It is approximate,
excludes overlapping/unclassified pixels and is not exact rendered segmentation
truth. At targeted 31/33/36 s images, SAM3.1 `door surface` retains about 98% of
visible moving-only pixels; fixed-frame-only selected pixels are 2.64/3.06/0.95%
in light and 4.32/4.29/3.83% in nominal. SAM3 `door surface` reaches about 8.8%
light / 11.4% nominal fixed-frame overlap at later sampled images.

Both corrected SAM3.1 frontend pilots complete 301/301 frames. Native median/p95
is 125.5/127.7 ms light and 125.3/127.5 ms nominal; CUDA allocator peak is 7.41 GiB.
SAM3 completes both 601-frame pilots at roughly 107–109 ms median and 5.09 GiB
allocator peak, with more fixed-frame inclusion. No latency optimization is run.

An additional SAM3.1 `door panel` pilot completes 301 frames per condition. Light
tracks two decorative insets; nominal maintains only the upper inset after native
confirmation. It does not add full-leaf support beyond `door surface`. The native
API uses one active text concept per state; adding a new text prompt resets that
state. Multiple object IDs for a concept do not establish a combined independent
text-prompt frontend. No union, replacement or extra fallback is integrated.

## Direct P2P integration and full-opening admission

`offline_episode(..., sam3_frontend={"version": "sam3.1", "prompt": "door surface"})`
selects the new worker. The first eligible observed RGB-D face supplies the local
zone, while the unchanged complete semantic mask supplies the native P2P object.
Every later mask is synchronized to original capture frame/time and sent directly
as `Frame.mask`. `external_masks=True` disables the separate native SAM2 segmenter;
TAPIR and SuperPoint remain on CUDA. Missing or unsynchronized masks fail explicitly.

DINOv3 is not loaded or inferred in this path. Its image tokens feed
`geometry.patch_anchors` and the static Surface appearance fields. P2P consumes
measured RGB-D points, masks and geometry, not those descriptors. Disabling only
that optional extraction retains identical measured points, normals and membership
on the behavioral check. The maintained static scan keeps its existing features.

Both integrated 31–41 s pilots complete 601 frames with seed at original row1860.
All three original physical targets have 100% accepted precision at 10/15/20 mm
and 5 degrees, with 99.67% availability. Native confirmation suppresses the mask
at 31.0167 s, and the same ID returns at 31.0333 s. The reference has 99.67–99.83%
correct availability. Complete median/p95 measured latency is 251/285 ms light
and 237/275 ms nominal, versus 434/537 and 514/680 ms for the preserved reference
on the same prefix. This admits complete offline evaluation, not operational
freshness or contact.

Full runs must retain every original scheduled row and terminal failure. Evaluation
uses the three original reference world target poses, transformed by new
`camera_from_map` through the original observed seed camera pose. Expected motion
uses recorded hinge/angle only in the evaluator. A new seed-relative score is
reported separately and cannot substitute for those physical target scores.
Native loss, same-ID returns, original-reference geometric compatibility, accepted
and all-finite error distributions, tails, complete latency and memory are separate.
The new timing runner includes upstream SAM3 in adapter availability. Its initial
interrupted measurement attempt is preserved with the correction reason.

## SAM3.1 complete-window attempts and memory impediment

Both original SAM3.1 attempts preserve all 2,858 scheduled rows. Light yields
1,312 native results through 52.85 s; the request at 52.8667 s fails the unchanged
TSDF device-memory budget. Nominal yields 1,150 results through 50.15 s; the
50.1667 s request fails the same guard. Remaining rows, including terminal tails,
are explicitly unavailable. The frontend still reports original ID0 and a leaf
mask at each failed P2P request; this is a resource failure, not a mask loss.

Correct one-frame grounding removes the initial overlapping-prefix OOM, but
native SAM3.1 history still grows at about 8.50 MiB per arrived frame after frame300.
The official builder retains `save_image_features=True`, with CPU output offload
and past-history trimming disabled. Those defaults remain unchanged. Sampled
combined process memory, native allocator growth and the exact guard refusal are
saved in `sam31-resource-failure-audit.json`. TSDF expansion requests are only
about 144.4 MiB, but the existing 20% device reserve must also remain available;
free GPU memory is 1.43 GiB light / 3.06 GiB nominal at refusal. Neither the reserve
nor tracker history is modified to force completion.

All three original physical targets retain 100% accepted precision at 10/15/20 mm
and 5 degrees before failure. Full scheduled correct availability is only 45.84%
light / 40.17% nominal, with missing tails spanning 25.75/28.45 s. Same-prefix
comparison and full-window completion are reported separately. **Do not adopt
SAM3.1 for complete openings on the current 4090 recipe.** Native ID persistence
and accurate early outputs do not compensate for missing late observations.

## Independent SAM3 integration comparison

SAM3 uses the identical `door surface` prompt, RGB-D and P2P controls. An initial
adapter attempt exposed its optional `img_ids_np=None` field; that failure is
preserved. The corrected adapter (`740ee57`) updates that alias only when present,
with an essential regression check for both official metadata forms.

Both SAM3 601-frame integration pilots complete with one stable leaf ID and no
native tracking losses. Original-target precision is 99.67–100% across the three
points and unchanged 10/15/20 mm and 5-degree thresholds. Isolated incorrect poses
remain, including a 38.19 mm nominal primary error at 40.8667 s. Frontend allocations
stay around 4.3 GiB; measured whole-pipeline median/p95 is 232/296 ms light and
224/274 ms nominal. `sam3-full-opening-admission.json` admits its complete-window
comparison independently; there is no runtime switch or fallback from SAM3.1.

Both SAM3 openings complete all 5,716 scheduled frames, including the last original
capture. Light loses five native frames in four episodes near the end; nominal
loses 747 frames in 68 episodes, with 67 native returns and a terminal loss from
72.2667 s through 78.6167 s. Both semantic runs retain ID0 without a reported mask-ID
loss. Native P2P support and mask identity are separate outcomes.

Measured fixed-frame inclusion becomes substantial in late SAM3 masks. At 71 s,
fixed-only fractions are 10.34% light / 24.08% nominal, versus 0.21% / 0.00% for
reference SAM2; moving-only recall stays about 98.5–98.9%. Nominal also selects
unclassified floor regions. At the last light frame the reference itself includes
11.45% fixed-only pixels, so contamination is not universally unique to SAM3.
These approximate observations explain why the small initial border alone was
not a sound rejection. They do not isolate the causal effect of masks from changed
initialization/reference histories. All input masks remain unmodified.

Original-reference compatibility at five measured inliers/4 mm is present on
367 light / 373 nominal correct SAM3 frames, versus 437 / 427 reference frames.
None of SAM3's four light or 67 nominal native returns has five compatible original
references. Reference renewal can remove those IDs, so absence is not proof of a
wrong material identity; it prevents claiming strict original-material recovery.
SAM3.1's initial confirmation return does retain 14/19 compatible original inliers;
no real physical-occluder recovery is thereby qualified.

`input-mask-audit.json` verifies every original capture row/frame/time and arrived
frame count, unchanged initial masks and six current-mask snapshots per complete
SAM3 run, external mask mode, no SAM2 initialization export and only seed ID0.
`consumer-transform-audit.json` checks the stored transported zone against native
map transport on accepted frames: maximum coefficient difference is below 7e-16.
The original physical-target evaluator therefore tests the same rigid transform
used by the adapter, with immutable seed offsets, rather than substituting a new
zone's seed-relative score. Operational freshness/ownership/contact admission is
still separate. Essential causal/loss/geometry and legacy numerical checks pass;
the optional SAM3 alias correction has its own regression check.

## Adoption decision and remaining work

**Reject common replacement by either tested version.** SAM3 is integrable and
about twice as fast in complete request time, but loses light precision at all
three bounds and nominal 15/20 mm availability; the last nominal five seconds
have no supported pose. Its nominal conditional error distribution improves
because many difficult frames are unavailable. SAM3.1's better early errors
cannot qualify a late range it never processes. All variants still have zero
complete requests within 150 ms.

Keep the current all-query, graph-off, partial-on, rollback-off reference with five
inliers/4 mm and all other adopted controls. The unified API remains an explicit
experimental comparison, without an automatic version switch, fallback, relaxed
gates or policy change. Its full evidence, failed startup/adapter attempts,
interrupted timing attempt and original baselines stay preserved.

Next-phase work may inspect SAM3.1 history/cache consumers before any memory
intervention and examine late semantic ownership/reference drift. General cleanup,
latency optimization and live frequency tests were not started. A3/A4, policy
integration, operational freshness, contact and physical qualification remain
unchanged. No acquisition, training, other-door campaign or sealed-test run occurs.


## Full-window original physical-target results

Every denominator includes the original seed and unavailable tail; the seed is
excluded from correctness and error statistics. Availability means correct,
accepted results divided by every scheduled frame. Precision means correct
results divided by accepted nonseed results. Rotation is limited to 5 degrees at
every positional bound. Tables below use original target0; all three original
physical targets are separately reported afterward.

| Condition | System | Accepted / scheduled | Correct 10 mm | Correct 15 mm | Correct 20 mm | Accepted precision 10/15/20 mm |
|---|---|---:|---:|---:|---:|---:|
| light | Reference | 2857/2858 | 80.41% | 98.22% | 99.90% | 80.43/98.25/99.93% |
| light | SAM3 | 2852/2858 | 57.73% | 94.16% | 98.92% | 57.85/94.35/99.12% |
| light | SAM3.1 | 1310/2858 | 45.84% | 45.84% | 45.84% | 100.00/100.00/100.00% |
| nominal | Reference | 2857/2858 | 68.89% | 98.15% | 99.83% | 68.92/98.18/99.86% |
| nominal | SAM3 | 2110/2858 | 70.40% | 73.58% | 73.65% | 95.36/99.67/99.76% |
| nominal | SAM3.1 | 1148/2858 | 40.17% | 40.17% | 40.17% | 100.00/100.00/100.00% |

### Accepted error distributions

Position p50/p95/p99/maximum is in mm; rotation is in degrees. Complete JSON also
retains all-finite distributions, histograms, peak timestamps and new-seed-relative
scores separately. SAM3.1 statistics are conditional on its shorter completed
prefix, not evidence of accurate late tracking.

| Condition | System | Position p50/p95/p99/max (mm) | Rotation p50/p95/p99/max (deg) |
|---|---|---:|---:|
| light | Reference | 3.87 / 13.29 / 15.81 / 21.78 | 0.178 / 0.444 / 0.551 / 0.793 |
| light | SAM3 | 8.12 / 15.16 / 19.57 / 27.58 | 0.162 / 0.449 / 0.638 / 3.243 |
| light | SAM3.1 | 2.38 / 4.51 / 5.68 / 8.57 | 0.129 / 0.364 / 0.460 / 0.770 |
| nominal | Reference | 6.24 / 13.25 / 16.42 / 22.89 | 0.156 / 0.588 / 0.748 / 1.056 |
| nominal | SAM3 | 3.58 / 9.83 / 12.90 / 38.19 | 0.140 / 0.406 / 0.834 / 3.069 |
| nominal | SAM3.1 | 1.43 / 3.58 / 6.15 / 7.98 | 0.056 / 0.194 / 0.288 / 0.424 |

### All three unchanged physical targets

| Condition | System | Original target | Correct 10 mm | Correct 15 mm | Correct 20 mm | Accepted position p95/max (mm) |
|---|---|---:|---:|---:|---:|---:|
| light | Reference | 0 | 80.41% | 98.22% | 99.90% | 13.29 / 21.78 |
| light | Reference | 1 | 78.69% | 97.90% | 99.90% | 13.46 / 21.55 |
| light | Reference | 2 | 81.88% | 98.36% | 99.90% | 13.10 / 22.04 |
| light | SAM3 | 0 | 57.73% | 94.16% | 98.92% | 15.16 / 27.58 |
| light | SAM3 | 1 | 55.21% | 93.46% | 98.95% | 15.31 / 27.66 |
| light | SAM3 | 2 | 60.50% | 94.54% | 98.95% | 15.05 / 27.50 |
| light | SAM3.1 | 0 | 45.84% | 45.84% | 45.84% | 4.51 / 8.57 |
| light | SAM3.1 | 1 | 45.84% | 45.84% | 45.84% | 5.04 / 8.89 |
| light | SAM3.1 | 2 | 45.84% | 45.84% | 45.84% | 4.10 / 8.24 |
| nominal | Reference | 0 | 68.89% | 98.15% | 99.83% | 13.25 / 22.89 |
| nominal | Reference | 1 | 63.58% | 97.62% | 99.79% | 13.79 / 22.30 |
| nominal | Reference | 2 | 75.26% | 98.25% | 99.79% | 12.85 / 23.53 |
| nominal | SAM3 | 0 | 70.40% | 73.58% | 73.65% | 9.83 / 38.19 |
| nominal | SAM3 | 1 | 70.57% | 73.55% | 73.65% | 9.80 / 37.44 |
| nominal | SAM3 | 2 | 70.29% | 73.55% | 73.65% | 9.92 / 38.98 |
| nominal | SAM3.1 | 0 | 40.17% | 40.17% | 40.17% | 3.58 / 7.98 |
| nominal | SAM3.1 | 1 | 40.17% | 40.17% | 40.17% | 3.51 / 7.89 |
| nominal | SAM3.1 | 2 | 40.17% | 40.17% | 40.17% | 3.70 / 8.07 |

### Tails and continuity

Correct observations in the final five seconds and longest original-time correct
pose gap are separate from native ID loss. An incorrect finite accepted pose is
unavailable for correctness even if the tracker says it is present.

| Condition | System | Last 5 s at 10 mm | At 15 mm | At 20 mm | Maximum correct-pose gaps 10/15/20 mm |
|---|---|---:|---:|---:|---:|
| light | Reference | 100/301 | 294/301 | 301/301 | 0.583 / 0.033 / 0.017 s |
| light | SAM3 | 9/301 | 209/301 | 280/301 | 3.817 / 0.167 / 0.050 s |
| light | SAM3.1 | 0/301 | 0/301 | 0/301 | 25.750 / 25.750 / 25.750 s |
| nominal | Reference | 91/301 | 294/301 | 301/301 | 0.817 / 0.050 / 0.017 s |
| nominal | SAM3 | 0/301 | 0/301 | 0/301 | 6.350 / 6.350 / 6.350 s |
| nominal | SAM3.1 | 0/301 | 0/301 | 0/301 | 28.450 / 28.450 / 28.450 s |

### Latency and memory

Steady complete request time includes image inference/IPC, P2P, measured copies,
trace export and adapter consumption; model startup and seed are separate. SAM3
variants have one whole-leaf object, while the reference has five overlapping
objects. These are pipeline comparisons, not isolated model speedups. A second
100 ms sampler measures simultaneous SAM/P2P process GPU residency for the new
path; reference residency is its P2P worker after the old image worker is released.
Startup residency of the old image worker was not jointly sampled; its missing
allocator/CPU entries are unavailable rather than zero. Allocator
peaks and per-process CPU RSS peaks are named separately and are not sums of
simultaneous peaks. No frame scheduling or queue/frequency experiment is run. The serial evaluator
has one synchronous request in flight, so it does not measure a live acquisition
backlog. Original input cadence and every failed tail are retained.

Pre-acquisition visual/P2P startup is 7.20/3.03 s light and 7.82/3.22 s nominal for
SAM3; exact measured values remain in each `startup.json`. These cold startup
costs and the first seed request (3.26/3.51 s) are separate from steady request
quantiles. SAM3.1 visual/P2P startup is 10.37/3.04 s light and 10.40/3.13 s nominal;
its seed requests are 1.59/1.80 s.

| Condition | System | Complete p50/p95/max (ms) | Sampled process GPU peak (GiB) | SAM / P2P allocator peak (GiB) | SAM / P2P CPU peak RSS (GiB) | Requests <=150 ms |
|---|---|---:|---:|---:|---:|---:|
| light | Reference | 438 / 554 / 1544 | 9.86 | not sampled / 7.17 | not sampled / 4.46 | 0/2857 |
| light | SAM3 | 233 / 282 / 988 | 16.28 | 8.41 / 1.17 | 18.53 / 2.68 | 0/2857 |
| light | SAM3.1 | 252 / 288 / 586 | 20.17 | 15.81 / 1.15 | 10.99 / 2.45 | 0/1311 |
| nominal | Reference | 402 / 644 / 4069 | 12.09 | not sampled / 7.93 | not sampled / 6.12 | 0/2857 |
| nominal | SAM3 | 234 / 288 / 1289 | 12.70 | 8.41 / 1.17 | 18.51 / 2.87 | 0/2857 |
| nominal | SAM3.1 | 249 / 284 / 472 | 18.51 | 14.47 / 1.02 | 9.21 / 2.47 | 0/1149 |

## Evidence and figures

The ignored evidence root contains `integrated-{full,pilot}-{light,nominal}` for
SAM3.1 and `integrated-sam3-{full,pilot}-{light,nominal}` for SAM3. Each attempt
retains protocol, startup/configuration, per-frame receipts, masks, native traces,
failures, every scheduled row and evaluator reports. Full comparison JSON retains
paired completed-prefix metrics as well as the whole original window. The archived
reference CUDA attempts are reused unchanged; they are not rerun for favorable
results or overwritten.

`full-comparison.png` shows original-target errors, scheduled correct availability,
request latency and named memory measures over original capture times.
`original-target-error-cdf.png` shows conditional distributions for all three
unchanged physical targets. `mask-reference-versus-sam31-light-41s.png` compares
old primary SAM3-to-SAM2 and SAM3.1 at the same frame.
`late-mask-reference-versus-sam3-nominal-2400.png` shows the late 71 s collider audit
with moving-only/fixed-only/unclassified membership. Collision truth and recorded
motion are used only in evaluators, never in RGB workers or P2P inference.

Implementation commits: `7404a41` (causal frontend/direct P2P masks), `740ee57`
(optional SAM3 frame alias). Results/documentation are recorded in the follow-up
local commit; none is pushed.
