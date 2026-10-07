# Unified SAM3 Video Frontend for Point2Pose

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

The complete light and nominal runs are in progress. General repository cleanup,
latency optimization and live frequency tests belong to the next phase.
