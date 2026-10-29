# B1 Perception Findings and Direction

## Question and method

Which earlier perception results support the maintained B1 direction, and which approaches failed
qualification?

Historical CUDA studies use frozen train/development identities and independent evaluators.
Complete-state qualification requires per-door 95% coverage and accepted precision at 1 cm/5
degrees; missing/rejected states remain failures and empty accepted sets fail. The sealed test stays
closed. The current SAM3/P2P implementation is canonical in [[../topics/shared-door-perception]],
rather than the retired workflows below.

## Result and decision

**No earlier estimator qualifies all six development doors.** Custom complete-state learning fits
train data but fails held-out geometry:

| Experiment | Result and decision |
|---|---|
| Run-01, `522f1ba`, diagnosis `e160f1f` | Stopped at epoch 28; best epoch 18 development position/orientation p95 9.01 cm/13.93 degrees, 0% confidence-valid coverage. Confidence gradients were 22.6× contact gradients; labels/cache alignment were verified. Loss imbalance was demonstrated, generalization was not. |
| Two-door corrected losses, `796797c`, `1bd633e`, `34c2660` | Contact-only fitting left hinge/local coordinates severely wrong. Complete-state supervision reached 98.25/96.70% joint train geometry; this was an overfit check, not held-out evidence. |
| Run-02, `68dc0ec` | Last epoch 15 passed 5/19 train, 0/6 development; development contact p95 8.34–32.09 cm despite 100% confidence coverage. Longer training was not established as a remedy. |
| XYZ/static-memory and confidence probes, `7e2bf56` | Pilots fitted at 7.41/9.93 mm contact p95; `refine-02` fitted 19 train doors at 2.29–4.00 mm. `refine-01` was train-only despite its initial development claim. Independent confidence fitting still accepted inaccurate development geometry, 7.41–32.32 cm p95 and 0% precision. |
| Run-03, `d833b89`, audit `0ea075f` | Last epoch 15 passed 19/19 train with 1.86–3.44 mm contact p95, but 0/6 development with 9.88–54.16 cm. Best epoch 1 also failed all development doors (5.42–51.50 cm). |

The original four-frame memory could not retain the 25 s scan; calibrated XYZ/static memory and
cache parity improved train fitting without repairing development failures. Privileged modality
substitutions were diagnostics, not policy inputs.

Corrected component screening fixes an eight-row/0.133 s counter/index offset and supersedes initial
component scores:

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

Pooled scores hide severe per-door failure (Industrial-003 p95 89.59 degrees/57.78 cm); planes/masks
do not supply a hinge or qualified contact state. CAP-Net’s incomplete single-door adapter produced
107/450 fits and 0% development success; this is not published-protocol accuracy.

Static 6.0B saved-cue reconstructions retain ambiguous whole-object ownership and zero physical
axes. Human review confirms only displayed local leaf/frame regions. Degenerate contact extrema are
lines, not finite pads; projected distal mesh bands supply geometric covers, not measured
compliance. Local footprint/depth support and endpoint IK do not certify continuous hand/arm travel,
stopping or load. The custom material tracker produced zero selected
dynamic-support/axis/reacquisition ticks and is retired.

The moving-camera P2P audit verifies acquisition/FK/composition but exposes native no-cluster
`lost=False` fallbacks and discarded SDF refinements. `df36b92` corrects publication/loss semantics
without repairing the later lost trajectory. Stationary-camera short tracking succeeds on 420
nonseed primary rows; complete original openings retain:

| Backbone | Train doors passing geometry in each seed | Development doors passing in each seed | Train mean per-door contact-position p95, seed range | Development mean per-door contact-position p95, seed range | Development pooled contact-position p95, seed range |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 19/19 | 0/6 | 2.28–2.42 mm | 22.98–27.52 cm | 52.66–54.60 cm |
| DINOv3 ViT-S/16 | 19/19 | 0/6 | 1.30–2.57 mm | 24.21–27.58 cm | 45.40–47.07 cm |

Unbounded renewal fails after 1,352/619 rows (CUDA OOM/TSDF guard); 1,506/2,239 suffix rows remain
missing. Bounded replacement completes both windows but trades fewer losses for inaccurate
acceptance, including the nominal 178.61 mm graph-publication peak. Subsequent
[[point2pose-consumer-impact-and-failure-windows|consumer diagnosis]],
[[point2pose-bounded-global-graph-ablation|graph ablation]],
[[point2pose-isolated-improvements|isolated changes]] and
[[p2p-sam3-selected-development|selected SAM3 development]] own their results. Completion and
self-consistent support are not qualification.

## Limits and evidence

Conditional errors exclude constructed seed zero while complete availability retains scheduled rows.
Native recovery does not establish original-material identity. The recorded maximum is 63.7209
degrees, not the configured 90.7-degree limit. Ideal depth, limited families/handedness,
uncalibrated hardware and unqualified freshness/contact remain material limits; no new training or
inference is performed by this summary.

Early evidence lives under `outputs/evidence/b1/perception/`: `run-*`, `comparison-01`,
`operational-scan-diagnosis-01`, `operational-contact-readiness-01`, `operational-tracking-03`,
`operational-tracking-live-02`, `point2pose-native-fixes-4-10s-01`,
`point2pose-full-opening-baseline-01`, `point2pose-inlier-renewal-comparison-01` and
`point2pose-bounded-renewal-01`. Original per-door reports, protocols, images, corrected/original
scores, masks and failures remain authoritative. Git: `e1f98a6`, `2a38858`, `df36b92`, `112bbe8`,
`10e7330`, `2b127b4`, historical source `57ad483`.

Storage/removal provenance is canonical in [[../topics/system-architecture|architecture]] and the
repository evidence guide. Earlier removed ignored payloads are listed in
`outputs/evidence/b1/perception/cleanup.json`; Git cannot restore them.
