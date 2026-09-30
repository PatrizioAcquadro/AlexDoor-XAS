# B1 Approved DINOv3 and SAM 3 Screening

## Scope and acquisition

Follow-up to [[experiments/b1-perception-model-comparison|the initial comparison]],
at baseline `main` @ `9dff5f3`, after the user obtained both access approvals.
Both official downloads succeeded and their SHA256 values match Hub metadata:

- SAM 3 native `sam3.pt`, revision `3c879f39826c281e95690f02c7821c4de09afae7`,
  3,450,062,241 bytes, with configuration and license.
- DINOv3 ViT-S/16, revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`,
  86,406,384-byte `model.safetensors`, with configuration, preprocessing and license.

All model execution used the RTX 4090. SAM 3 source revision is
`2345a4ad109ac29c569da749c91d84f10dc08c40`; dependencies are isolated in the ignored
comparison directory. The Isaac environment, maintained estimator, recordings,
original caches and run-03 checkpoints were not changed. New small diagnostic
heads were fitted on train only; no full-corpus training, new collection or sealed
test access occurred. These experiments do not qualify a replacement estimator.

## Frame alignment finding

The original sample's `raw_frames` and `context_frames` store frame counters.
The component runners used them as HDF5 row indices. Recorded counters start at
eight, giving an eight-row/0.133-second offset from the intended cached-head
samples in every episode. Within each geometric, SAM 2, CAP-Net and SAM 3 result,
RGB, depth and scoring truth refer to the same row, so those component comparisons
remain paired. Their direct per-image comparison to the cached current head was
not exact and must not be claimed as such. The run-03 training and full replay are
unaffected.

A final phase audit also found that 50 of the 450 legacy manipulation-labeled
component rows are actually in phase 4. Therefore geometry, SAM 2 and both SAM 3
variants were repeated at the correct source rows with unchanged settings and
explicit counter/phase assertions. **The aligned repeat is the primary result
below**; original attempts remain preserved. Historical CAP-Net screening was not
repeated and must retain this additional sampling limitation.

For the DINO probes, counters are explicitly mapped to source rows. Recomputed
metric inputs match the original cache; DINOv2 features differ by relative L2
`1.30e-5`, consistent with floating-point execution differences. Minimum valid
depth support is 32.0%. The failed preparation and offset component runs are
retained. Evidence: `frame-index-audit.json`, `aligned-component-protocol.json` and
`dino-cache-validation.json`.

## SAM 3: finding the door versus segmenting a supplied region

Both trials use the same 800 component images: 350 inspection and 450 manipulation
views, covering all 25 train/development doors and both conditions. Frozen SAM 3
uses its native 1008-pixel processor, BF16 autocast and default confidence 0.5.
The highest scored mask feeds the unchanged metric plane/bounds estimator.

The first protocol uses the common text `door`. Its frequent train omissions
motivated a second, separately recorded protocol using the **same predicted
GroundingDINO boxes supplied to SAM 2**, as positive geometric prompts with the
processor's implicit `visual` text. No annotated box, mask, dimensions or hinge
enters inference; no per-door prompt or threshold tuning was performed.

Error p95 values use available planes only. Success fractions count missing
outputs as failures. All table rows below use the corrected frame mapping.

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

The corrected paired development comparison has 89 successes shared by SAM 2
and boxed SAM 3, three exclusive to SAM 2, eight exclusive to SAM 3 and eight
shared failures: **97/108 versus 92/108**. Train improves from 229/342 to 239/342.
This supports boxed SAM 3 as a candidate visual component for the prototype,
with SAM 2 retained as a reference. The small development sample does not
establish universal model superiority. These aligned results supersede the
opposite small aggregate difference seen in the offset preliminary attempt.

Text-only SAM 3 misses every manipulation view of `modern-door` and
`industrial-003`, and 16/18 of `industrial-004`. Supplying observed boxes restores
availability. For industrial-002, boxed SAM 3 improves normal p95 from SAM 2's
88.72 to 1.71 degrees, but its surface-distance p95 remains 1.26 cm. Conversely,
industrial-003 has two severe boxed-SAM-3 failures: its per-door normal and
surface-distance p95 are 89.59 degrees and 57.78 cm. They disappear in the pooled
1.79-degree statistic. Development closed-scan height errors remain 9.1–41.0 cm.
Visual review of the same middle-push view for all six development doors confirms
text omissions and mostly similar box-conditioned masks; this single view per
door is illustrative and does not replace the full sample's failure metrics.
Review of the two opposing worst cases (`sam3-aligned-failures.jpg`) shows that
either segmenter can emphasize a frame/side surface instead of the main panel
face; the other segmenter succeeds on that same image.

Depth geometry alone has the best development plane success fraction here, but
fails to produce a plane on 104/342 train views and does not recover full borders
or hinge/contact state. No single component is a sufficient estimator. Masks
should guide association while metric and temporal consistency must detect
incorrect surface selection; a mask cannot be treated as ground-truth identity.

Median aligned SAM 3 text-plus-geometry replay time is about 83 ms. Box-conditioned
SAM 3 takes about 85 ms **excluding the cached GroundingDINO box computation**;
it cannot be compared as an end-to-end speedup against the aligned 137 ms SAM 2
pipeline. Depth-plane fitting alone is about 8 ms in this replay. These are not
optimized online throughput measurements. Sparse image screening does not
evaluate SAM 3 video tracking. Plane normal and surface distance do not measure
full contact position, hinge state or confidence.

## Matched DINOv2 versus DINOv3 diagnostic

Both frozen ViT-S backbones receive the same whole-image 224-pixel letterbox and
ImageNet normalization. CLS and DINOv3 register tokens are excluded. Native patch
grids are 16x16 for DINOv2 and 14x14 for DINOv3, both with 384 channels. The
unchanged `MetricMemoryEstimator` accepts both through its adaptive pooling.

Each selected manipulation endpoint retains seven initial views and four causal
recent observations: 342 train and 108 development windows. The same head
initialization seeds (6100–6102), sampled window sequence, train-only normalization,
depth branch, optimizer and articulated-state loss are used for each backbone.
Confidence output is frozen and unqualified. Backbones are never updated.

The initial fixed 1,200-update trial left substantial train error for both models.
A separately specified common extension uses 6,000 updates, with learning rates
3e-4, 9e-5 and 2.7e-5 for successive 2,000-update blocks. All original results and
all intermediate reports are retained. No development early stopping, best-seed
selection or model-specific tuning is used. This remains a small head-fit probe,
not a full-corpus run or a benchmark of all DINOv3 sizes and resolutions.

Final step-6,000 results, retaining all three seeds:

| Backbone | Train doors passing geometry in each seed | Development doors passing in each seed | Train mean per-door contact-position p95, seed range | Development mean per-door contact-position p95, seed range | Development pooled contact-position p95, seed range |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 19/19 | 0/6 | 2.28–2.42 mm | 22.98–27.52 cm | 52.66–54.60 cm |
| DINOv3 ViT-S/16 | 19/19 | 0/6 | 1.30–2.57 mm | 24.21–27.58 cm | 45.40–47.07 cm |

The seed-averaged mean per-door development position p95 is 24.76 cm for DINOv2
and 25.81 cm for DINOv3. DINOv3 improves pooled tail position error, but not the
mean per-door measure or the number of passing doors. Development pooled contact
orientation p95 is 8.61–10.97 degrees for DINOv2 and 7.40–12.07 for DINOv3.
Replacing the frozen backbone alone does not resolve generalization in this
matched probe, even once both heads accurately fit every train door in the sample.
This does not rule out DINOv3 in a different architecture or at another resolution.

## Component decision and next architecture

**DINOv3 is not a justified backbone-only fix; boxed SAM 3 is a provisional
candidate for visual assistance.** Both approved models are available and run on
the RTX 4090. Keep GroundingDINO to supply observed boxes in the tested SAM 3
path, and retain SAM 2 for comparison. The measured five additional development
plane successes justify testing SAM 3 inside the geometric prototype, but its
rare severe failures forbid treating its masks as reliable measurements by
themselves. Text-only SAM 3 is too incomplete here; its video tracking remains
untested. A separate DINO backbone is not required for the first geometric
prototype. Keep DINOv2 as the comparison baseline until a specific learned
association or residual component demonstrates a benefit from DINOv3.

The next implementation should explicitly reconstruct and retain observed metric
geometry: select the panel face, associate its borders across the initial scan,
track it through motion, infer supported hinge/articulation state, and check local
surface geometry at the intended contact. Use pretrained masks as evidence to
combine with depth and temporal consistency, not as guaranteed panel identity.
Missing or ambiguous boundaries must produce an unavailable/uncertain state,
not inferred dimensions taken from asset truth. Any learned part should address
a measured association or residual error.

First validate this observed-only prototype per door on the existing recordings,
including full contact/hinge errors and loss/reacquisition. Train a new complete
estimator only after the prototype exposes a concrete learning need. The current
component results do not pass the full 1 cm/5-degree contract, do not validate
confidence, and do not justify another collection campaign. The prototype and
its closed-loop behavior have **not** been implemented or qualified by this study.

## Evidence and validation

Ignored local directory: `outputs/b1/perception/comparison-01/`. Source model
downloads are in `models/dinov3/` and `models/sam3/`. Key artifacts:
`approved-models-protocol.json`, `sam3-box-protocol.json`,
`dino-extended-protocol.json`, `aligned-component-protocol.json`,
`aligned-component-summary.json`, `sam3-aligned-score.json`,
`sam3-box-aligned-score.json`, `sam3-mask-comparison-aligned.jpg`,
`sam3-aligned-failures.jpg`,
`dino-probe-summary.json`, `dino-probe-curves.png`, `dino-probes/`,
`dino-probes-extended/`, `approved-models-validation.json` and the frame/cache
audits. Each aligned component variant completed 50 episodes/800 images; masks,
finite plane normals, frame counters, phases and completion flags were checked.
All six extended probes completed;
the original run-03 best/last hashes remain unchanged. Scripts, isolated
dependencies, failed preparation and console logs are retained locally.

Official interfaces checked 2026-09-29:
[DINOv3 ViT-S/16 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
and [SAM 3 code](https://github.com/facebookresearch/sam3). These sources describe
model capabilities; the suitability decision above follows the local experiments.
