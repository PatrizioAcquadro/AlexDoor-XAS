# Unified SAM3 Video Frontend for Point2Pose

**Experimental reference, October 7, 2026: unified SAM3 reconditioning OFF.**
Use one SAM, one original leaf object, TAPIR and SuperPoint for the next frontend
development. The older all-query baseline, every previous attempt and operational
defaults remain preserved. This choice balances precision, continuity and complete
latency; it does not require winning every metric against the older multi-object
baseline. Tracking and contact qualification remain open.

The only new comparison is bounded SAM3.1 with periodic reconditioning changed
from 16 to 0. Its existing useful-memory policy, original initialization, prompt
and all P2P controls remain fixed. Full original light/nominal CUDA replays and the
four-way saved-evidence comparison are complete. The final choice remains SAM3 OFF;
bounded SAM3.1 OFF improves parts of light but regresses nominal on every original
target. Detailed current results and the resource tradeoff are consolidated below.

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
identity replaces the reference's five light / six nominal overlapping automatic candidates.

Local evidence: `outputs/b1/perception/p2p-sam3-unified-01/`. Reference:
`outputs/b1/perception/p2p-performance-01/full/chunkall/{light,nominal}/attempt-1`.
Original input is `engineering-v2/animated-door-1-88abf40/{light,nominal}/episode.hdf5`,
rows 1860–4717, 31.0–78.6166667 s, 2,858 frames per condition at original 60 Hz.
The recorded maximum opening is 63.7209 degrees. Replay speed is independent of
acquisition cadence.

## Consolidated four-way decision

The consolidation starts from `main` at `2f3ad07`, with no pre-existing tracked
changes. Source control/trace support is `563847a`. Historical tables call the
older multi-object all-query baseline "reference"; the **new experimental
reference is SAM3 OFF**, not a change to operational defaults. Saved baseline,
SAM3 OFF and bounded SAM3.1 ON results are reused and exactly reproduced by the
same physical-target evaluator. Every previous result/failure remains intact.

Only bounded SAM3.1 OFF is new: the native periodic control changes 16→0.
Both original CUDA attempts process all 2,858 frames, seed row 1860 to row 4717,
without selective retry. Startup runtime fields are identical except period and
observation metadata; the complete P2P configuration is identical except output
paths and measured startup time. `sam3_memory.py` and every P2P rule are unchanged.
Seed masks and initial P2P poses match ON exactly. OFF makes zero native
reconditioning calls/updates and creates no additional IDs. First differing masks
and P2P poses are at 46.2 s light / 35.5333 s nominal (indices 912/272, both multiples
of 16). This verifies application, not just a configuration label. Saved full-prefix
mask hashes cover 1,313/1,151 frames; six saved ON mask snapshots cover the rest of
its full window. No second ON experiment is introduced.

**Use SAM3 OFF for the next phase**, with one SAM, one original object ID0,
`door surface`, the preserved seed0 recipe, TAPIR Large/480/four iterations/all-query,
SuperPoint, graph off, partial batch on, rollback off, five inliers/4 mm,
vectorized Jacobian and 120 active references. Operational defaults remain unchanged.
The opt-in frontend setting preserves the original diagnostic seed and trace:

```python
sam3_frontend = {
    "version": "sam3",
    "prompt": "door surface",
    "seed": 0,
    "detection_reconditioning": False,
    "trace_reconditioning": True,
}
```

The choice balances metrics. SAM3.1 OFF improves light primary 15 mm availability
73.90→97.97% versus ON and the final primary tail 3→279/301. Its light target1
10 mm result regresses, so light gains are not universal. Nominal OFF loses
availability at every bound on all three targets versus ON; primary 15 mm falls
86.18→83.24% and the final tail 193→36/301. Its native loss stays only one frame,
but biased accepted poses create a 10.85 s primary 10 mm correctness gap.
SAM3 OFF has 57 nominal native losses yet a much shorter 0.150 s maximum correct
10 mm gap and 87.40% correct availability. Native presence alone is not useful
continuity. SAM3 OFF's 43.10 mm nominal primary accepted peak and weaker light
accuracy versus the old baseline remain; this choice does not claim universal
accuracy, recovery or provider qualification.

SAM3.1 ON/OFF wins the memory comparison: SAM allocation stays about 4.40 GiB
and allocator peaks stay 5.06 GiB. SAM3 OFF's current SAM allocation grows
3.65→7.84 GiB; CPU high-water marks reach 20.21/18.51 GiB. Those costs are accepted
explicitly for this finite next-phase experimental reference. Complete p95 is
264/281 ms SAM3 OFF versus 282/283 ms new bounded OFF and 554/644 ms old baseline.
No nonseed request meets 150 ms. No memory optimization is added to SAM3 here.

Evidence root: `outputs/b1/perception/p2p-unified-consolidation-01/`.
`comparison.json` and `report.md` retain all three targets, original-time gaps/tails,
accepted/all-finite distributions, peak times, startup/seed, simultaneous memory,
per-process peaks and growth. Both RSS sampler threads raised a retained
shutdown-only error when workers closed; 6,427/6,431 samples cover the last original
capture, with every per-frame resource record intact. Neither inference replay
failed or was rerun. New instantaneous combined RSS window medians increase
6.56→7.14 GiB light / 5.38→5.92 GiB nominal; bounded SAM buffers do not imply zero
whole-pipeline CPU growth. Sampled peaks remain sampled, not exact allocator peaks.

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

## SAM3.1 forward-history correction

The October 7 follow-up starts from `5f33411`, with no pre-existing tracked changes,
in `outputs/b1/perception/p2p-sam3-followup-01/`. Original attempts and defaults
remain unchanged. `bounded_memory=True` is a separate opt-in causal lifetime policy.
The pinned native attention consumes six recent spatial memories, fifteen recent
nonconditioning object pointers and up to four closest conditioning frames. Image
features are useful inputs to that attention and remain enabled. The original
conditioning frame is retained independently, along with recent conditioning
fallbacks. No model threshold, memory encoder, P2P cap or TSDF reserve changes.

The audit demonstrates growth in stored nonconditioning outputs (mask memory,
propagation image features/positions, masks and pointers), cached full-resolution
output masks, and CPU normalized RGB history. The native grounding/backbone caches
are already bounded after the previous one-frame correction. The forward policy
releases expired outputs and their per-object aliases together, retaining sixteen
recent outputs and the selected conditioning history. It retains current output
scores/suppression plus persistent identity, confirmation and occlusion metadata.
Input sequences retain the seed template and current payload with original global
indices/counts, instead of accumulating CPU images or CUDA stage tensors. A read of
released input fails explicitly; reverse propagation, old-frame interaction and
new prompts are outside this adapter contract. Nothing is offloaded to grow RAM.

Fresh CUDA runs compare the entire previously reached prefixes, including the
last SAM request before each original P2P failure: 1,313 light / 1,151 nominal
frames. Packed masks, probabilities, tracker scores, seed/current IDs, loss/state
and capture/count fields are exactly equal on every paired frame. Frontend allocator
peaks fall from 15.81/14.47 GiB to 5.06/5.06 GiB; final allocated memory is 4.40 GiB
in both conditions. Corrected peak CPU RSS is 7.51 GiB (including checkpoint loading),
with bounded RGB payloads. This establishes frontend equivalence before the independent full P2P evaluation.
The complete GPU replays both finish 2,858/2,858 original frames. All 1,312
light / 1,150 nominal completed P2P prefix poses and native/integration decisions
match the preserved failed attempts exactly (maximum matrix difference zero).
Each complete run has only the original one-frame confirmation loss. Memory growth
was an implementation resource defect; resolving it does not resolve late pose bias.

| Condition | Primary correct availability 10/15/20 mm | Accepted precision 10/15/20 mm | Maximum correct gap 10/15/20 mm (s) | Last 5 s correct 10/15/20 mm out of 301 |
|---|---:|---:|---:|---:|
| light | 68.19/73.90/94.79% | 68.24/73.95/94.85% | 12.200/3.400/0.067 | 0/3/226 |
| nominal | 59.90/86.18/98.92% | 59.94/86.24/98.98% | 9.200/0.283/0.067 | 1/193/293 |

All three frozen physical targets regress versus the adopted reference, with
5-degree rotation retained. Full accepted primary p95 is 20.03/17.38 mm and
0.54/0.51 degrees. Complete request p50/p95 is 252/282 ms light and 254/285 ms
nominal; zero nonseed requests meet 150 ms. SAM allocator peak remains 5.06 GiB.
Sampled simultaneous SAM/P2P GPU peak is 11.33/10.18 GiB and host RSS is 5.93/5.91
GiB, excluding pre-acquisition checkpoint-loading peaks. Per-process peaks are
reported separately. Keep the memory fix opt-in, but reject this SAM3.1 recipe
as the shared tracking replacement because complete late accuracy regresses.
Detailed results are in `sam31-bounded-full-comparison.json` and
`p2p-saved-prefix-equivalence.json`; both original resource failures remain intact.

## Isolated SAM3 reconditioning diagnosis

Saved complete receipts verify all original frame/row/time and arrived-frame
indices, only seed ID0, unchanged source/current masks, and external-mask P2P with
no separate SAM2. The older route supplies five light / six nominal observed
geometry masks, then SAM2 positive prompts. Its primary whole-leaf seed masks
have 97.26%/93.29% IoU with the direct SAM3 masks. Initialization, candidate count,
subsequent mask model and reference evolution therefore remain distinct factors;
the older route is diagnostic evidence and is not reintroduced.

The streaming adapter calls the native full visual-grounding propagation on the
newest available frame, preserving association, confirmation, suppression and
memory. The interactive wrapper's repeated-pass cache-fetch behavior is not an
arrived-frame streaming interface. Saved receipts and current-mask audits reveal
no integration error explaining the complete SAM3 regression.

The original SAM3 diagnostic changes periodic detection reconditioning from
16 to 0, leaving its already-disabled bbox trigger (-1), every other heuristic,
detection/association threshold, checkpoint, `door surface`, seed0 and P2P
controls unchanged. The same opt-in control now supports SAM3.1's native multiplex
`add_new_masks` call without changing its memory policy or native heuristics.
`trace_reconditioning=True` reports eligible matches,
actual native reconditioning mask calls and newly created detection IDs separately;
it does not add prompts or substitute IDs. Initial 33-frame ON/OFF trials have
identical masks and no eligible reconditioning. An initial gate expecting an early
application fails and is retained as a negative result, without threshold tuning.
All four controlled full replays complete 2,858/2,858 frames from the original
initialization. ON reproduces every original SAM3 pose, decision and point-trace
field exactly in both conditions. ON applies 39 light / 26 nominal reconditioning
updates, first at 35.0/31.8 s. OFF applies zero. The first changed subsequent mask
is at 35.0167/31.8167 s; seed masks are identical. Every run retains only seed ID0,
with no newly created detection IDs. This isolates matched-detection updates to the
existing object from object creation and initial selection.
ON and OFF both complete the full window, so their common comparison includes all
2,858 captures. In contrast, SAM3.1 old/new common prefixes end at 52.85/50.15 s;
its newly completed tails are evaluated separately from the exact prefix equality.

### Complete controlled accuracy, gaps and tails

The following are the same original physical target0, with 5-degree rotation and
all 2,858 scheduled rows in each denominator. Precision is conditional on accepted
nonseed outputs; gaps include inaccurate and unavailable rows. The final five
seconds contain 301 original captures. No missing tail is censored out.

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

All three frozen physical targets are evaluated independently in
`final-comparison.json`. OFF improves ON at every bound in both conditions, but
versus the reference it loses light 10/15 mm and nominal 15/20 mm availability on
all three targets. Its nominal 10 mm gain is real. OFF's three-target availability:

| Condition | Original target | 10 mm | 15 mm | 20 mm |
|---|---:|---:|---:|---:|
| light | 0 | 70.33% | 96.19% | 99.86% |
| light | 1 | 69.03% | 94.05% | 99.76% |
| light | 2 | 71.66% | 97.13% | 99.90% |
| nominal | 0 | 87.40% | 96.54% | 97.73% |
| nominal | 1 | 86.21% | 96.50% | 97.76% |
| nominal | 2 | 87.82% | 96.68% | 97.76% |

### Frame points, inliers and renewal

The evaluator classifies measured 3D points against exclusive moving/fixed prepared
collision hulls with a 4 mm margin and original capture-aligned motion. Overlap and
unclassified points are excluded from both categories. This is approximate physical
membership, not exact rendered instance segmentation; truth never enters workers.
ON's exact point-trace reproduction permits reuse of the saved original membership
audit, while OFF is classified separately.

At initialization, reference seed points include 0/30 light and 3/30 nominal fixed-only
points; direct SAM3 includes 3/30 and 4/30. SAM3's first measured fixed inliers appear
at 32.65/31.0167 s, before reconditioning. The first newly sampled fixed references
appear at 35.4167 s light in both ON/OFF (1/30 fixed), and at 36.15 s nominal ON
(12/30 fixed), versus 36.2333 s OFF (7/30 fixed). Reference first additions occur at
36.4/36.5667 s with 1/30 and 2/30 fixed. Maximum fixed-only frontend inliers are
22/30 ON, 14/15 OFF and 6/3 reference (light/nominal). Thus initialization and later
reference sampling already admit static points independently of reconditioning;
the isolated control reduces, but does not eliminate, contamination.

At the final nominal ON frame, 24 frontend measurements contain six fixed-only,
fourteen moving-only and four other points, but frontend/published inliers are zero.
The active budget is 120 with 442 historical references; no references are born.
All 747 lost ON nominal rows have zero renewal events; OFF's 57 lost rows likewise
have none. `BoundedRenewal.after_frame` skips lost objects and published support
below five. This explains why renewal cannot replenish the terminal loss. It is
the existing support guard, not evidence to relax it or the 4 mm threshold.

### Complete latency and resources

Whole-request time includes SAM inference, IPC, P2P and trace export; startup/seed
are separate. All nonseed requests exceed 150 ms. GPU is sampled simultaneous
SAM/P2P use; SAM allocator and process RSS peaks are separate measurements. The
reference lacks a simultaneous SAM/P2P measure because its initialization worker
was released before acquisition; its P2P sampled peaks are 9.86/12.09 GiB.

| Condition | Variant | Accepted target0 p95 mm / degrees | Complete p50 / p95 ms | Sampled combined GPU GiB | SAM allocator GiB | SAM / P2P CPU RSS peaks GiB |
|---|---|---:|---:|---:|---:|---:|
| light | reference | 13.29 / 0.44 | 438 / 554 | unavailable | unavailable | unavailable / 4.46 |
| light | sam3-on | 15.16 / 0.45 | 232 / 282 | 16.27 | 8.41 | 18.51 / 2.69 |
| light | sam3-off | 14.62 / 0.69 | 232 / 264 | 14.41 | 8.39 | 20.21 / 2.65 |
| light | sam31-bounded | 20.03 / 0.54 | 252 / 282 | 11.33 | 5.06 | 7.50 / 2.69 |
| nominal | reference | 13.25 / 0.59 | 402 / 644 | unavailable | unavailable | unavailable / 6.12 |
| nominal | sam3-on | 9.83 / 0.41 | 231 / 286 | 12.71 | 8.41 | 18.51 / 2.90 |
| nominal | sam3-off | 11.71 / 0.47 | 233 / 281 | 12.92 | 8.39 | 18.51 / 2.92 |
| nominal | sam31-bounded | 17.38 / 0.51 | 254 / 285 | 10.18 | 5.06 | 7.50 / 2.67 |

OFF is not a universal memory improvement: nominal combined GPU rises slightly,
and light SAM CPU RSS rises from 18.51 to 20.21 GiB. Native SAM3 CPU image history
still grows across this finite recording; this experiment deliberately does not
combine a memory intervention with the reconditioning control. SAM3.1's separate
bounded lifetime correction solves that version's demonstrated growth.

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

## Earlier causes and adoption decision, before consolidation

**Earlier decision, superseded by the consolidation above: retain the old
all-query reference; neither complete corrected recipe is a shared replacement.** Preserve the minimal SAM3.1 memory correction and SAM3 OFF as opt-in
diagnostics. OFF is the better tested single-SAM3 diagnostic, but its gains do not
compensate for light accuracy, nominal 15/20 mm gaps and weaker final targets versus
the reference. SAM3.1 prefix precision was real; full completion exposes late bias
that cannot be attributed to pruning because every previously reached pose and
decision is exactly equal. Completion and common-prefix quality are separate claims.

Demonstrated causes are expired SAM3.1 payload retention, a causal adverse effect
of SAM3 matched-detection reconditioning on later masks/support, static-frame point
inclusion from seed and renewal, and blocked renewal during native loss. Streaming
arrival indices/times, full native propagation and mask transfer show no integration
error explaining the regression. The controlled ON/OFF seed/model/configuration
and original ON reproduction rule out a changed initialization as the ON/OFF cause.

Open hypotheses concern the residual effects of direct-mask initialization/query
selection, one object versus the older five/six candidate contexts, subsequent
mask propagation, TAPIR correspondences and TSDF/reference evolution. They are not
isolated by comparing two different complete frontends. Frame contamination alone
does not explain every failure, and stable ID0 does not prove material identity.
The next justified diagnostic is observed-only seed/renewal ownership with the
single-SAM/single-object path retained; no such new intervention is implemented here.

Keep all-query TAPIR/480/four iterations, SuperPoint, graph off, partial batch on,
rollback off, five inliers/4 mm and 120 active references. The unified API retains
baseline defaults. No SAM2 or multiple-candidate route is reintroduced into it.
Original resource/startup/adapter/timing failures, the early negative control and
the smoke sampler shutdown error remain preserved. General cleanup, further latency
optimization, live resampling, A3/A4 and policies are deferred. No acquisition,
training, other-door campaign or sealed-test evaluation occurs.

## Historical full-window original physical-target results

These tables retain the original unbounded SAM3.1 failures and original SAM3 ON
results. The complete corrected comparisons above and `final-comparison.json`
supersede their adoption conclusion without replacing their evidence.

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
variants have one whole-leaf object, while the reference has five light / six nominal overlapping
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

The follow-up evidence root is `outputs/b1/perception/p2p-sam3-followup-01/`.
`final-comparison.json` contains all three original targets, common-prefix metrics,
complete gaps/tails/resources, update events and exact ON reproduction checks.
`point-membership-*-summary.json` and per-frame JSONL retain membership/renewal
audits. `sam3-controlled-diagnosis.png` shows all finite target0 errors (unavailable
outputs dotted), fixed-frame frontend inliers and evaluator-only moving published
inliers over the original times. The native gate still counts all compatible points.

Implementation commits: `563847a` (SAM3.1 period control/multiplex trace), `7404a41` (causal frontend/direct P2P masks), `740ee57`
(optional SAM3 frame alias), `21c2abf` (expired SAM3.1 history), `72532e1` (complete
bounded results), `f3d8797` (isolated SAM3 control/trace). The final controlled results
are recorded in the documentation follow-up commit. All commits are local.
