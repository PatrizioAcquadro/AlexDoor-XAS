# Isolated Point2Pose Improvements After Graph-Off Diagnosis

Work starts from clean `main @1689623`, with the complete graph-off baseline
preserved. The isolated partial-batch option is implemented at `fac02e5`;
observational prompt/promotion telemetry is implemented at `4c6dc25`. Outcomes
below separate saved-data diagnostics from complete CUDA comparisons. The paired
partial-batch and 6 mm comparisons are complete; the independent promotion
comparison remains in progress. No combined variant or tracking default is adopted.

## Fixed controls and comparisons

Use the original light and nominal recordings, original automatic 31 s
initialization and every original 60 Hz frame through 78.6167 s. Each replay
condition/variant has one attempt, including failure and unavailable tails.
Models, candidate selection, SDF, reference replacement, 120-reference cap,
1 cm/5 degree evaluator limits and all unrelated checks remain fixed. Truth
stays in evaluators. The three independent graph-off comparisons are native
batch-prefix admission, 4→6 mm geometric support and enabling the existing
pending promotion geometry check. The latter retains its existing 3-observation,
8 mm spread and view-diversity settings. Pixel discretization and SAM2 prompts
are audited before considering any intervention.

Ignored evidence is under
`outputs/b1/perception/point2pose-isolated-improvements-01/`.
All earlier graph-on, graph-off and failed unbounded attempts remain preserved;
ignored payloads are not recoverable from Git. No policy/provider replacement,
new recording, extended parameter search, training, campaign or push is involved.

## Saved correspondence evidence

On the same primary map/observations, the expected correct pose has at least
five compatible pairs on 2,399 light / 1,833 nominal rows at 4 mm, versus
2,854/2,254 at 6 mm. This includes 3/14 native-lost rows at 4 mm and 49/206 at
6 mm. These are geometry diagnostics, not counterfactual accepted poses.
The fixed returned pose crosses five pairs at 6 mm on 18/88 correct rows and
3/16 inaccurate rows. A higher support count alone is therefore insufficient.

The original shared NumPy RANSAC stream can be reconstructed from seed zero,
100 draws per eligible remaining pool and saved cluster inlier removal. The
inspected executed pipeline has no other NumPy randomness. Numerical extraction
reproduces every saved cluster in all 31 checked primary calls. This separates
raw sampled hypotheses from retained clusters, selection, refinement and SDF.

At nominal 63.4667 s, the sampled hypotheses contain no correct supported 4 mm
fit, while the bounded evaluator-only enumeration finds 14. The previous pose
is correct (0.708 mm/0.054 degrees) and supports five pairs. Other cases already
sample a correct supported hypothesis but do not retain a correct cluster:
light 70.65 s and nominal 63.3833 s are examples. Merely increasing RANSAC
iterations would not address both mechanisms. At nominal 70.7667 s, the correct
pose has zero 4 mm pairs and five 6 mm pairs; correct sampled support appears
only at 6 mm. At 74.2 s/end it has only 1/0 pairs at 4 mm and 2/1 at 6 mm,
so neither fixed threshold restores late correspondence evidence.

The fit enumeration uses evaluator-compatible points only to demonstrate
existence; its truth-selected hypotheses never enter inference. It is not an
implementable oracle recovery or a trajectory continuation. Some hypotheses
within the 1 cm/5 degree zone limits can support more points than the exact
expected map transform; these are distinct diagnostics.

## Promotion and pixel evidence

Reconstructing native graph-off pending observations reproduces the fused
coordinates of all 540 light promotions within 0.003 mm and 390 nominal
promotions within 0.013 mm. Three light promotions have median object-frame
spread 8.78–9.05 mm; one nominal promotion has 15.71 mm. The existing geometry
check is disabled, so these pass visibility/stability promotion. Enabling only
that existing check is an evidence-justified isolated replay. One nominal
promotion at 50.75 s uses nonconsecutive good observations because the native
reset-on-bad option is false; this fact alone does not justify changing it.

The current depth lift truncates pixel coordinates; mask/promotion and IPC
measured-depth checks round them. The floor calculation reproduces all 161,293
light and 125,763 nominal saved primary pairs within 1 micrometre. On frozen
pairs, rounding recovers five-point correct-pose support on 183/103 frames but
loses it on 101/55. Median pair error worsens 7.46→7.88 mm light and
6.82→7.11 mm nominal; p95 worsens 36.03→36.51 and 50.78→51.23 mm. A blanket
rounding correction is not justified. Integer initial query pixels are unchanged
by either method. Seed-material projected pixel p95 later reaches 8.05/7.29 px;
quantization does not account for all accumulated tracking error.

The worker's distributed interior references are preservation checks, not the
actual native SAM2 prompts. Native initialization samples five positive points
at linearly spaced indices of the source-mask raster. New diagnostic traces
observe the actual `add_new_prompt` calls and save the source masks, without
changing them. Absolute initialization/leaf ownership is not established by the
relative trajectory metric, whose seed pose is zero by construction.

See [[point2pose-bounded-global-graph-ablation|the graph-off baseline]] and
[[point2pose-consumer-impact-and-failure-windows|consumer metric boundaries]].

## Complete partial-batch comparison

Both original CUDA attempts complete all 2,858 frames without retries, missing
rows or process failure. Runtime controls, initial masks/maps/IDs/selection and
source schedules match the graph-off baseline. Actual unique CUDA query,
feature and causal storages obey 120 references per candidate.

| Condition/system | Correct available | Inaccurate accepted | Native lost | Accepted point p95, mm | Rotation p95, degrees | Process peak, GiB |
|---|---:|---:|---:|---:|---:|---:|
| light / graph-off baseline | 2,569 | 238 | 49 | 11.51 | 0.53 | 7.78 |
| light / partial batch | 2,450 | 407 | 0 | 11.89 | 0.84 | 8.41 |
| nominal / graph-off baseline | 2,004 | 58 | 794 | 8.15 | 0.73 | 9.61 |
| nominal / partial batch | 2,484 | 373 | 0 | 12.30 | 0.48 | 10.41 |

The nominal gain of 480 correct poses is real, as is the light loss of 119.
Eliminating every native loss does not establish reliable availability: inaccurate
acceptances increase by 169/315. Both p95 point errors remain above 1 cm.
The partial option remains an isolated diagnostic, disabled by default.
There are no native flag returns because neither primary ever becomes lost;
this cannot be called strict material recovery. Peak total queries are 599/720,
while per-candidate capacity stays 120. CPU historical records remain retained.

Actual original SAM2 calls are now observed in both conditions. Primary source
retention is 99.12%/98.94%, mask IoU 0.431/0.423, with 192,131/195,441 extra
pixels. Two positive prompts lie one pixel from the source-mask boundary, at
image top/bottom. Only 5/30 light and 6/30 nominal seed query pixels lie inside
the original automatic component. Targeted RGB inspection shows substantial
expansion onto door relief and other plausible same-leaf regions, so neither
extra pixels nor overlapping automatic components alone proves contamination.
The original initialization stays fixed; absolute ownership/calibration and a
causal prompt ablation remain unverified. The early static/first-opening p95 is
1.8/1.6 mm and 1.85/2.93 mm, while late tracking degrades. Initial component
ambiguity and accumulated metric/tracking error are separate findings.

The native frontend supports tentative-point fallback, but no primary baseline
row actually uses it in either complete graph-off recording. Its existence is
not a demonstrated explanation of these primary errors. The capability and
all its thresholds remain unchanged.

## Complete 6 mm comparison

Both original CUDA attempts complete 2,858 frames with unchanged seed outputs,
source times and all native configuration except `inlier_thres=0.006`.
Ground truth remains evaluator-only. Correct and inaccurate counts exclude the
constructed seed; accuracy is the relative zone point/full rotation within
1 cm/5 degrees. Native loss and integration rejection are separate checks.

| Condition/system | Correct available | Inaccurate accepted | Native lost | Accepted point p95, mm | Rotation p95, degrees | Process peak, GiB |
|---|---:|---:|---:|---:|---:|---:|
| light / graph-off baseline | 2,569 | 238 | 49 | 11.51 | 0.53 | 7.78 |
| light / 6 mm | 1,495 | 1,362 | 0 | 20.67 | 0.50 | 7.84 |
| nominal / graph-off baseline | 2,004 | 58 | 794 | 8.15 | 0.73 | 9.61 |
| nominal / 6 mm | 1,518 | 1,337 | 2 | 22.10 | 0.61 | 9.91 |

Accepted-pose precision falls from 91.52%/97.19% to 52.33%/53.17%.
Correct availability loses 1,074/486 rows while inaccurate acceptances increase
by 1,124/1,279. The two nominal native losses each last one original frame
(16.7 ms), but the final correct-pose gap lasts an observed 13.5 s. Neither
native return is correct or passes the strict historical 4 mm material/pose audit.
Thus 6 mm recovers some compatible frozen-pair support, but worsens actual full
trajectories and is rejected as a tracking correction. Keep 4 mm.

Actual query populations peak at 498/595 and every candidate stays at or below
120. Measured TSDF CUDA storage peaks at 0.657/1.812 GiB. The threshold comparison
includes its effect on subsequent shared RANSAC draws, masks/reference history
and TSDF, not only a counterfactual final gate on frozen pairs. Few saved stage
audits already show wrong poses with strong 4 mm support before SDF; a threshold
or refinement-only explanation is insufficient. Full stage records remain in
`stage-audit.json`.
