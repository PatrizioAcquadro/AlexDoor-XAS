# B1 Perception Findings and Direction

## Decision and evidence boundary

The maintained direction is **GroundingDINO + SAM 3**, explicit calibrated
RGB-D/multiview geometry, and **DINOv3 when learned visual features are needed**.
The custom DINOv2 estimators, SAM 2 reference, CAP-Net adapter and their training,
refinement and comparison workflows are retired. This supersedes the earlier
recommendation to keep SAM 2/DINOv2 as operational comparison baselines.
No candidate has qualified complete door/contact geometry or confidence on the
six development doors. The new geometric pipeline is now a diagnostic prototype;
a backbone swap, plane estimate or train fit does not close Phase 6.0.

All model experiments below used the RTX 4090. Train/development identity splits
and physical gates were unchanged; the sealed test was not used. The original
records are now under `outputs/b1/perception/evidence/`, retaining run names.
Protocols, per-door results, metrics, diagnoses, aligned masks and representative
images are preserved. Source at Git `e1f98a6` retains the tracked implementation
and detailed historical pages. Removed ignored scripts, weights, caches and
intermediate payloads are **not recoverable from Git**; these records support
inspection of the results, not a claim of fully executable reproduction.

## Geometric prototype — September 30, 2026

The implementation is described in [[topics/shared-door-perception|Shared Door
Perception]]. Local `geometric-smoke-02` passed actual frozen CUDA inference through
all three models; cold/warm worker-model times were 369/174 ms on the RTX 4090.
This is model execution evidence, not a latency or complete-state gate pass.

The first pilot (`geometric-pilot-01`) replayed left `door-2738468b94d74c5f` and
right `animated-door-1-88abf40`, nominal/light. Both doors accepted zero complete
states. Missing hinge, missing thickness and ambiguous panel/jamb association were
explicitly rejected. The right nominal episode also exposed large panel-orientation
errors. Common corrections preserve normal continuity and require observed extent
edges, without changing the gates or substituting nominal dimensions. Pilot-02 was
interrupted after a report-format defect was found; its partial evidence is preserved.
`geometric-pilot-03` completed all four episodes and 20,108 chronological
observations. Both doors still accepted zero complete states. On finite rejected
partials, pooled panel-rotation p95 was 0.051 degrees left and 0.591 degrees right;
height span errors reached 15.45/16.40 cm respectively. These do not certify full
dimensions: edge evidence, hinge and stable associations remain unresolved, and
discontinuous motion is now rejected explicitly. All four episodes produced zero
accepted rigid-motion hinge fits. Worker latency p95 was 220–243 ms, above the
150 ms freshness window for an unrefreshed model result. Intermediate estimates
require current depth support, not a freshly stamped cache.

One unchanged recipe is fixed for the full 50-episode campaign. No simulation will
run unless all 25 train/development doors pass offline. No training, collection or
sealed-test evaluation was started. Baseline source: `4df9a09`; current prototype
sources and protocol identities are retained with the new runs.

The first full-campaign attempt (`geometric-evaluation-01`, source `fe61935`) ended
after eight complete episodes and a partial ninth (`modern-door-2fb8d024/light`).
A degenerate rigid-fit consensus returned no solution; its caller incorrectly
unpacked it. The common fix rejects that fit, preserving thresholds and gates.
The interrupted attempt, complete reports and failure record remain intact. A
fresh attempt (`geometric-evaluation-02`, source `a41c002`) used the corrected
source and was subsequently stopped by the user. Its worker and evaluation
process are stopped; completed reports, partial status and logs are preserved
with `failure.json` identifying an intentional interruption. Neither attempt is
a complete common-source evaluation. No further campaign will start until the
main causes are clarified and corrected, with verification on both pilots first.
Reset also clears the public estimate, and the prototype IO rejects unsafe contact
commands before simulator access. These software guards do not validate dynamics.

### Pilot diagnosis after the user stop

`geometric-pilot-diagnosis-01` inspects existing nominal recordings and pilot-03
clouds only; it does not evaluate another campaign or use annotations for inference.
The selected planar support grows across the seven scan views, so single-view
clipping alone does not establish missing evidence. Reprojection of the retained
main-plane cloud into those views agrees with measured optical depth at p95
0.73–1.95 mm left and 0.66–2.31 mm right, where projected depth is valid. This
checks retained support, not every edge or a complete calibration qualification.

The provider derives dimensions from one plane's inliers. Relief, side faces and
other parallel faces become separate hypotheses; object extent is therefore not
the same as planar support extent. A coarse raw-depth inspection of every fifth
observation during the scan finds support beyond retained vertical bounds, including
before the first semantic sample. These points are only proximity candidates:
their ownership by the leaf or fixed frame has not been certified. Two targeted
RGB-D overlays mark upper/lower candidates and separate profile support for human
review. Do not merge frame points into the leaf or label these candidates complete
silhouette evidence before that distinction is resolved.

The current static fusion also keeps only the first surface's DINO anchors/features;
later-view descriptors update but later anchors are discarded. This is a concrete
memory limitation when the first view loses overlap during motion. Its contribution
to the zero rigid-motion fits remains to be measured independently of boundary
visibility. Proposed common corrections are object-level association of observed
leaf faces, causal retention of scan support and multiview feature anchors, with
metric verification and continued rejection of ambiguity. They have not yet been
implemented or verified. Current evidence does not justify new training.

Control-local requirements and the unchanged complete-state contract are separated
in [[topics/shared-door-perception|Shared Door Perception]]. Human review is pending;
no pilot correction run or extended restart has been launched after the stop.

## Custom-estimator sequence

Run-01 (`522f1ba`, diagnosis `e160f1f`) stopped at epoch 28 after 24.7 minutes;
best epoch 18 failed all six development doors. Pooled manipulation diagnostics:

| Checkpoint / split | Position p95 | Orientation p95 | Confidence-valid coverage |
|---|---:|---:|---:|
| Best / train | 4.58 cm | 4.45 degrees | 0% |
| Best / development | 9.01 cm | 13.93 degrees | 0% |
| Last / train | 5.91 cm | 5.37 degrees | 0% |
| Last / development | 14.93 cm | 11.29 degrees | 0% |

At the inspected shared weights, confidence gradients were 22.6 times the contact
position gradients. Annotation reconstruction agreed within 3.7e-8 m; three
source/cache samples per episode aligned. These checks diagnosed unequal loss
scales without proving full observability or resolving generalization.

The same two train doors (one per hand, nominal/light; 2,330 frames and 2,318
windows, seed 6100) were then fitted without development selection. The corrected
contact loss (`796797c`) passed at epoch 30/2,190 updates: position p95
0.838/0.792 cm, orientation 1.232/1.219 degrees and 100% confidence coverage.
However, compensating primitive errors remained: hinge origins approximately
one meter wrong, hinge orientation 171.65/175.46 degrees, and local contact
position 1.243/1.276 m. Correct composed contact alone was insufficient.

Complete-state supervision (`1bd633e`, result `34c2660`) reached epoch 17 with
contact p95 0.872/0.964 cm, orientation 1.256/2.749 degrees and all nine component
p95 gates passing. Joint geometric coverage was 98.25/96.70%; confidence coverage
100%. This was a train-only fit, not held-out qualification.

Run-02 (`1bd633e`, diagnosis `68dc0ec`) stopped at epoch 15/13.60 minutes;
best development checkpoint remained epoch 1. Per-door manipulation p95 ranges:

| Checkpoint / split | Contact position p95 | Contact orientation p95 | Doors passing all gates |
|---|---:|---:|---:|
| Best, epoch 1 / train | 1.43–3.27 cm | 2.32–6.25 degrees | 0/19 |
| Last, epoch 15 / train | 0.40–1.45 cm | 0.88–1.77 degrees | 5/19 |
| Best, epoch 1 / development | 7.88–15.43 cm | 13.30–18.32 degrees | 0/6 |
| Last, epoch 15 / development | 8.34–32.09 cm | 6.72–18.72 degrees | 0/6 |

The last checkpoint accepted inaccurate development states at 100% confidence
coverage. Longer training was not established as a remedy.

## Observation, metric-depth and confidence diagnoses

At `68dc0ec`/`a48e06f`, replay covered 25,006 train and 6,498 development windows.
Every prepared-panel top face was outside the camera frustum in all 31,654 cached
views. Four-frame/0.3-second memory could not retain an earlier inspection.
Counterfactual swaps used 2,000 windows per checkpoint, two matched train-donor
assignments and a separate within-episode temporal intervention. RGB replacement
changed contact by p95 60.3–72.0 mm, depth by 0.76–0.98 mm and proprioception by
2.71–2.82 mm. A +1% depth change moved contacts by only 0.035/0.067 mm train/dev.
These are sensitivity diagnostics, not modality-importance percentages; identical
camera poses made camera swaps uninformative.

Train failure was concentrated in composed contact position: 22.71% of manipulation
windows failed, pooled p95 12.44 mm, while the other components met their individual
tolerances. Privileged substitutions/debiasing suggested systematic local-contact
error but were never deployable corrections. Gradients on 19 inspected batches
were aligned, not evidence for another blanket reweighting. Confidence AUROC was
0.570 on correlated train frames. No per-door correction was applied.

The common 25-second scan and simulated 10-degree upward mount correction
(`a4ad984`) retained seven views at 4/7/10/15/18/21/25 seconds. Two complete train
pilots and a separate tallest-development-door scan established partial boundary
visibility. At the seven views, depth-consistent top/bottom fractions were:

| Door | Hand | Hold angle | Observed top boundary | Observed bottom boundary |
|---|---|---:|---:|---:|
| door-2738468b94d74c5f | Left | 57.99° | 57.4% | 52.5% |
| animated-door-1-88abf40 | Right | 61.32° | 71.3% | 54.5% |

Fractions use 101 boundary samples, a 5x5 depth neighborhood and 3-cm agreement;
they are observability diagnostics, not estimator gates. The tallest door had
24.8%/56.4% top/bottom visibility. That inspection-only episode is retained under
`datasets/b1/perception/inspection-tallest-01` and is not a manipulation demonstration.
Camera FK error stayed below 0.001 mm/0.000002 rad; maximum parked-tool drift was
0.151 mm, door motion below 0.000102 rad and neck error below 0.073 rad. Hardware
mounting/calibration remains unvalidated.

Metric/static-memory correction (`7e2bf56`) used calibrated XYZ and separate RGB,
static and recent paths. `metric-fit-01` exposed scattered HDF5 reads; contiguous
reads/context retention reduced epoch time from 99.3 to 5.8 seconds. Fixed-rate
`metric-fit-02` stagnated (14.9/21.9-mm best contact p95); the shared plateau
schedule in `metric-fit-03` passed both pilot doors at epoch 26/962 updates:
7.41/9.93-mm position, 1.30/1.70-degree orientation and 98.88/95.08% joint geometry
coverage. Confidence remained unqualified. Raw/cache replay matched all 1,169
windows within 0.0071 mm and 0.0014 degrees; +1% XYZ changed contact by p95 4.95 mm.

Legacy `refine-02` fitted all 19 train doors in one epoch (2.29–4.00-mm contact p95).
The earlier `refine-01` incorrectly reported `development_evaluated=true`; actual
execution was train-only, and that original report remains unchanged. Separate
confidence fitting preserved geometry but still accepted inaccurate development
states: 7.41–32.32-cm contact p95, 0% accepted-state precision, qualification false.
Refinement/confidence fitting therefore did not solve held-out geometry.

## Refreshed campaign and pretrained screening

The refreshed campaign has 50 complete nominal/light episodes with the common
inspection (`engineering-v2`). Run-03, audited at `d833b89`/`0ea075f`, again
separates train fitting from development failure:

| Checkpoint | Train contact position p95, range across doors | Train complete geometry | Development contact position p95, range across doors | Development complete geometry |
|---|---:|---:|---:|---:|
| Best, epoch 1 | 1.13–3.10 cm | 0/19 doors | 5.42–51.50 cm | 0/6 doors |
| Last, epoch 15 | 1.86–3.44 mm | 19/19 doors | 9.88–54.16 cm | 0/6 doors |

Component protocols fixed 800 image samples across 50 episodes and 450
manipulation samples (342 train/108 development). No simulator masks, asset
identity, dimensions or hinge truth entered inference. GroundingDINO supplied
the highest-scored `door.` box at 0.30 box/0.25 text thresholds. The same metric
plane/bounds scoring was used across segmenters; masks were predictions.

The original component scripts confused frame counters starting at eight with
HDF5 row indices, creating an eight-row/0.133-second offset. RGB/depth/truth within
a component stayed paired, but direct cached-head/image comparisons were not
exact. Fifty of the 450 labeled manipulation rows were actually release phase.
Geometry/SAM 2/SAM 3 were repeated at corrected rows with unchanged settings
(`e1f98a6`). Run-03 training/full replay and correctly mapped DINO probes were
unaffected. **Aligned results supersede the initial component statistics.**

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

SAM 2 and boxed SAM 3 shared 89 development successes: 3 exclusive to SAM 2,
8 exclusive to SAM 3 and 8 failures shared. Boxed SAM 3 therefore improved
92/108 to 97/108 on this sample, not universal model superiority. Industrial-003
still had two severe boxed-SAM-3 failures: per-door p95 89.59 degrees/57.78 cm,
hidden by pooled p95. Development closed-scan height errors remained 9.1–41.0 cm.
Text-only masks often omitted doors; depth alone missed 104/342 train planes.
Neither planes nor masks establish hinge/contact geometry or qualified confidence.

Median replay times were about 8 ms for depth geometry, 83 ms for SAM 3 text and
85 ms for boxed SAM 3 **excluding cached detector computation**, versus 137 ms
for the SAM 2 pipeline. These are not comparable end-to-end optimized latencies.
Video tracking was not evaluated.

The historical CAP-Net adapter produced 107/450 fits and 0% development plane
success. It used predicted ROIs, upstream point normalization and a single-door
NOCS fit, omitting multi-instance clustering. This was not the published protocol
or a definitive CAP-Net accuracy claim. It was not rerun after the row/phase audit.

## Matched DINO diagnostic and retained next steps

Frozen ViT-S backbones used the same 224-pixel whole-image letterbox, normalization,
384-channel features, metric head, seven initial/four recent views and seeds
6100–6102. Native patch grids differed (16x16 DINOv2, 14x14 DINOv3); CLS/register
tokens were excluded. Confidence remained frozen/unqualified. The fixed
1,200-update trial was followed by a common prespecified 6,000-update extension,
with 3e-4/9e-5/2.7e-5 learning rates in 2,000-update blocks, no best-seed selection
or development early stopping. All three final seeds are retained:

| Backbone | Train doors passing geometry in each seed | Development doors passing in each seed | Train mean per-door contact-position p95, seed range | Development mean per-door contact-position p95, seed range | Development pooled contact-position p95, seed range |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 19/19 | 0/6 | 2.28–2.42 mm | 22.98–27.52 cm | 52.66–54.60 cm |
| DINOv3 ViT-S/16 | 19/19 | 0/6 | 1.30–2.57 mm | 24.21–27.58 cm | 45.40–47.07 cm |

DINOv3 improved pooled tail error but not mean per-door error or passing-door
counts; swapping the backbone alone did not repair the tested architecture.
This does not reject DINOv3 for learned association or residual features.

The next pipeline must recover metric planes/borders, maintain multiview static
geometry, track articulation, measure local contact, and represent ambiguity or
loss explicitly. Masks guide association; metric/temporal consistency must reject
wrong surfaces. Require full per-door geometry and dynamic loss/reacquisition
before freezing one shared provider. No new training or collection is implied.
See [[topics/shared-door-perception|the maintained boundary and fixed gates]].

Selected official weights/configuration/licenses are in `models/perception/`;
its README records revisions, including native SAM 3 format. Evidence includes
`comparison-01/aligned-component-protocol.json`, `aligned-component-summary.json`,
`frame-index-audit.json`, all per-door scores, DINO protocols/results and
`sam3-aligned-failures.jpg`. Original offset scores remain historical evidence.
The local `cleanup.json` records relocations, preserved episodes and explicit
payload removals. Five interrupted RGB-D payloads were removed by user decision;
their calibration, metadata and failure records were retained.
