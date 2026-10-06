# Isolated Point2Pose Improvements After Graph-Off Diagnosis

Work starts from clean `main @1689623`, with the complete graph-off baseline
preserved. The isolated partial-batch option is implemented at `fac02e5`;
observational prompt/promotion telemetry is implemented at `4c6dc25`. Outcomes
below are saved-data diagnostics. Full CUDA comparisons are in progress; no
combined variant or tracking default is adopted.

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
