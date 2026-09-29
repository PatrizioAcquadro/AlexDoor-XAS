# B1 Perception Model Comparison

## Question and boundary

At `main` baseline `d833b89`, compare the current estimator, public pretrained
components and an observed RGB-D geometric approach before choosing another
training recipe. The user subsequently requested a stop before gated-model use
until they can authenticate. **DINOv2 versus DINOv3 remains untested.** This is
component screening, not a qualified replacement or a complete architecture ranking.

All inference used the RTX 4090. No optimizer updates, full training, new episodes
or sealed test access occurred. The existing best/last checkpoint files were
hashed before and after replay and remained unchanged. Experiments and isolated
third-party dependencies are ignored under `outputs/b1/perception/comparison-01/`;
the maintained runtime and Isaac environment were not changed.

## Current run: fitting versus generalization

The refreshed `engineering-v2` campaign contains all 50 train/development episodes
and `features-v2` contains 44,342 sampled frames. The completed `run-03` stopped
normally for `development_stagnation` after 15 epochs, 11,820 optimizer steps and
1,180 seconds. Epoch 1 remains the best development checkpoint. Full replay used
the existing evaluator and unchanged gates on 25,190 train and 6,452 development
windows; its metrics select contact/push/hold within those windows.

| Checkpoint | Train contact position p95, range across doors | Train complete geometry | Development contact position p95, range across doors | Development complete geometry |
|---|---:|---:|---:|---:|
| Best, epoch 1 | 1.13–3.10 cm | 0/19 doors | 5.42–51.50 cm | 0/6 doors |
| Last, epoch 15 | 1.86–3.44 mm | 19/19 doors | 9.88–54.16 cm | 0/6 doors |

At epoch 15, train contact orientation p95 is 0.45–0.84 degrees per door;
development is 3.58–11.43 degrees. These are geometry results: the confidence
head is not qualified, and no checkpoint passes the complete offline release
contract. The new observations and metric recipe enabled accurate train fitting.
They did not establish generalization. This evidence does not isolate a single
cause such as backbone quality, shape coverage or head inductive bias.

## Fixed component screening

`sample.json` and `protocol.json` were fixed before candidate results: all 25
train/development doors, both nominal/light episodes, seven closed inspection
views and three uniformly spaced frames in each contact/push/hold phase. This is
800 images, including 450 manipulation images (342 train, 108 development).
Annotations are used for sampling/scoring only. Candidate inference uses observed
RGB, calibrated metric depth/validity, intrinsics and camera pose from calibration/FK.
No simulator mask, mesh, door dimensions, hinge pose or identity enters inference.

Methods:

- **Current best/last:** unchanged frozen DINOv2 plus trained metric-memory head,
  with the existing seven inspection and four recent cached observations.
- **Geometry:** calibrated RGB-D backprojection, a vertical-plane prior,
  256 three-point RANSAC candidates, 1 cm inlier threshold and PCA refinement.
  At most 12,000 observed points; at least 200 inliers and 20% support.
- **Segmentation hybrid:** GroundingDINO Tiny with common text `door.`, highest
  scored box (0.30 box/0.25 text threshold), SAM 2.1 Hiera Small best predicted-IoU
  mask, then the same metric plane estimator. This uses predicted masks.
- **CAP-Net adapter:** official frozen checkpoint, DINOv2/FeatUp and SAM 2.1
  Hiera Large features; predicted GroundingDINO ROI replaces the upstream
  annotated crop. Its 24,576 metric points are normalized as upstream, retaining
  pixel/point alignment through radius filtering. All points predicted as
  `hinge_door` feed the upstream NOCS/Umeyama fit, assuming one manipulated door.
  The adapter omits upstream multi-instance mean-shift clustering. It is a
  deployment-oriented screening adapter, **not a reproduction of the published
  evaluation or a definitive accuracy judgment on CAP-Net**.

CAP-Net's checkpoint uses `pose_score_net` while its public code uses `pose_net`;
only that prefix was mapped, with a strict all-keys/all-shapes load. Local FeatUp
construction uses its bundled DINOv2 architecture and the complete checkpoint
weights, avoiding redundant backbone downloads. CUDA extensions were built in
isolated output directories using GCC 11. Compatibility patches, source revisions,
failed attempts and successful logs are preserved locally.

For all methods the common orientation metric is the sign-invariant angle of the
predicted panel normal. Surface distance measures the perpendicular distance of
the true contact point from the predicted plane. **Neither metric measures the
full 3D contact-position error, hinge accuracy or complete contact orientation.**
A correct plane can still leave the contact point tens of centimeters wrong along
that plane. Availability means an output exists, not that it is trustworthy.
Geometry/hybrid dimensions use the union of seven closed-scan inlier clouds and
0.5%/99.5% extents; CAP-Net/current dimensions use per-frame predictions.

## Results and interpretation

Pooled metrics on the fixed manipulation sample; p95 errors use available
outputs only. Success fraction counts unavailable outputs as failures.

| Method | Split | Available outputs | Normal error p95 | Surface distance p95 | Normal ≤5° AND surface ≤1 cm |
|---|---|---:|---:|---:|---:|
| Current best | train | 342/342 | 2.10° | 1.90 cm | 80.4% |
| Current best | development | 108/108 | 8.18° | 51.29 cm | 3.7% |
| Current last | train | 342/342 | 0.38° | 0.17 cm | 100.0% |
| Current last | development | 108/108 | 10.54° | 53.50 cm | 7.4% |
| Depth geometry | train | 232/342 | 7.96° | 3.85 cm | 55.6% |
| Depth geometry | development | 108/108 | 2.24° | 1.26 cm | 88.9% |
| GroundingDINO + SAM 2 + geometry | train | 342/342 | 89.05° | 22.20 cm | 69.0% |
| GroundingDINO + SAM 2 + geometry | development | 108/108 | 1.85° | 1.25 cm | 88.0% |
| CAP-Net observed-ROI adapter | train | 94/342 | 62.79° | 26.03 cm | 1.8% |
| CAP-Net observed-ROI adapter | development | 13/108 | 59.06° | 41.88 cm | 0.0% |

The CAP-Net adapter produced a hinge-door fit on only 107/450 manipulation
images (13/108 development). On the other 343 it predicted fewer than 200
hinge-door points, before any instance-clustering or pose-fit decision. No
development image satisfied both component tolerances. This provides no evidence
for using these frozen weights as a direct replacement on the current close-up
views; it does not rule out a better adapter or adaptation to this domain.

Replay median timings were about 23 ms for geometry and 139 ms for the SAM 2
hybrid. CAP-Net was about 421 ms including the shared ROI pipeline (which also
computed an unused SAM 2 Small mask). These include relevant preprocessing/HDF5
access and exclude model loading; they are not optimized online throughput
measurements. Current cached-head timings are omitted because they exclude its
backbone and would not be comparable.


The geometric baseline estimates development surface normals and plane locations
well on most sampled views. Adding a pretrained mask increases train output
availability but can select a narrow panel side or frame surface. Visual review
of saved RGB/mask overlays confirms these ambiguities. For example, industrial-002
has hybrid normal p95 near 88 degrees, even though the pooled development p95 is
only 1.85 degrees: rare failures on one door disappear in a pooled statistic.
Per-door metrics and unavailable samples must remain visible.

The simple geometric extent estimator still fails height: maximum absolute
per-door development height error is 5.1–35.7 cm without segmentation and
10.0–42.9 cm with segmentation. New top observations do not automatically make
visible inlier-cloud extents equal to the complete panel boundary. This result
supports improving multi-view boundary association, not another camera change
based on this experiment alone.

## DINOv3 and other candidates

[DINOv3](https://github.com/facebookresearch/dinov3) is a pretrained visual feature
extractor, not a complete articulated-door estimator. The official
[ViT-S/16 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
reports dense-feature improvements; it does not establish our 1 cm/5 degree gate.
Our official config download returned HTTP 401 with restricted-access text and no
local authenticated session. [SAM 3](https://huggingface.co/facebook/sam3) had the
same access condition. These are access blockers, not model failures. No gated
weights were obtained, and no access conditions were bypassed. The user will
complete account access and local authentication later.

DINOv3 uses 16-pixel patches versus 14 for the present DINOv2 ViT-S. Changing the
model name alone would not be a controlled comparison: preprocessing, token grid,
register-token handling and feature caches must be adapted, and the downstream
head must be fitted to the new features. A future matched trial should hold
recordings, split, head capacity, depth path and optimization budget fixed.
GroundingDINO is a text-conditioned detector; its name does not mean this
experiment already compared DINOv2 and DINOv3.

[FoundationPose](https://github.com/NVlabs/FoundationPose) is a useful rigid-object
pose candidate when a CAD model or reference reconstruction is available. It was
not run: using prepared door meshes at inference would change this benchmark's
observed-only problem, and rigid pose alone does not provide the required
articulation/contact state. This is a suitability decision, not measured failure.

## Proposed direction and next checkpoint

The current evidence favors **investigating an explicit geometric estimator with
pretrained visual assistance**, rather than spending another full run on the same
head. It does not select a deployable hybrid: the measured simple hybrid has
serious failures, and DINOv3/SAM 3 are still pending.

After the requested authentication handoff, complete the matched backbone check
and segmentation comparison. Then develop one common observed-only prototype:
associate panel face/borders across the initial scan, retain its metric geometry,
track the panel through manipulation, and reject ambiguous or unsupported states.
Estimate hinge/contact state from observed boundaries/motion and a common physical
model; never substitute annotated dimensions or hinge truth. If learning remains
needed, use it for part/boundary association or bounded residuals before asking a
network to regress every metric quantity freely. Verify complete geometry and
uncertainty per door before another full training run. Existing episodes remain
usable; this comparison does not justify a new collection campaign.

## Evidence and sources

Local files: `baseline.json`, `current-sample.json`, `geometry-score.json`,
`component-score.json`, `geometry-selfcheck.json`, `segmentation/`, `capnet/`,
`mask-diagnostics.jpg`, `model-access.json`, `source-revisions.json` and scripts/logs
in `outputs/b1/perception/comparison-01/`. These ignored artifacts are retained
locally and are not recoverable from the documentation commit alone.

Official sources checked 2026-09-29:
[CAP-Net code/weights](https://github.com/ShaneHuangHZ/CAPNet),
[SAM 2](https://github.com/facebookresearch/sam2),
[GroundingDINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny),
[DINOv3 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
[SAM 3](https://github.com/facebookresearch/sam3), and
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
The existing perception contract remains in
[[topics/shared-door-perception|Shared Door Perception]].
