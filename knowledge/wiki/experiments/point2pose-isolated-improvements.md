# Isolated Point2Pose Improvements After Graph-Off Diagnosis

Three independent comparisons complete all six original CUDA openings: 17,148
frames, one attempt per condition/variant, no inference retry or process failure.
Partial batch admission gains correct nominal poses but costs precision; 6 mm
and enabling the native promotion geometry check worsen both complete conditions.
No common tracking default or combined variant is adopted.

Work starts from clean `main @1689623`, with no pre-existing tracked user edits.
Only `main` was available, so work stays there. Implementation commits are
`fac02e5` (partial batch option) and `4c6dc25` (observational prompts/promotion
telemetry); saved diagnosis is `29f194b`, paired partial/threshold findings
`c919ea8`. All preserved baselines and failed unbounded attempts remain available.

## Fixed controls and metric boundaries

Each comparison starts from the bounded graph-off baseline with the original
31 s automatic initialization and all 2,858 original 60 Hz frames through
78.6167 s. The recorded opening ends at 63.7209 degrees; it does not cover the
configured 90.7-degree limit. Models, candidate selection, native point checks,
SDF, retirement, 120 references/candidate and unrelated controls remain fixed.
Original initial masks, map/query coordinates, IDs, poses and candidate selection
match exactly. Actual CUDA storage counts, chronological times, graph-off counters
and native semantic configuration deltas pass for all six attempts.

The isolated changes are:

- `allow_partial_reference_batch=True`: admit the fitting prefix of the fully
  filtered native batch; zero capacity still defers. No change to its selection,
  ordering, promotion, uncertainty or other new-point checks.
- `register.params.inlier_thres=0.006`: only the geometric threshold changes.
- `pipeline.params.pending_use_geom_check=True`: only the existing check changes;
  retain three observations, 8 mm median spread and disabled view diversity.

Native configuration overrides are temporary; every original YAML is restored
and final bytes match all saved originals. Partial admission remains disabled
by default. Graph-on remains the existing prototype default; the experimental
results do not validate combinations with it. ACT/Diffusion policies and the P2P
provider are unchanged. No parameter search, new acquisition or training occurs.

Correct availability means accepted integration and relative zone point/full
rotation error within 1 cm/5 degrees. The constructed seed is excluded from
correct/incorrect counts and error distributions; the complete scheduled
2,858-frame denominator is retained. Inaccurate accepted poses fail either bound.
Native loss and other integration refusals are distinct; table columns are not
an exhaustive partition. Accepted precision is correct/accepted non-seed poses.
Relative seed zero does not establish absolute initialization accuracy or leaf
ownership. Acquisition cadence is not inference rate or 150 ms freshness.

## Complete primary comparison

Point/rotation p95 are conditional on accepted integration. GPU columns are
observed per-process resident peaks, including models and allocator effects.

| Condition/system | Correct available | Inaccurate accepted | Native lost | Point p95, mm | Rotation p95, degrees | Accepted precision, % | GPU peak, GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| light / original graph-on | 2,276 | 35 | 531 | 7.79 | 1.44 | 98.49 | 6.58 |
| light / bounded graph-on | 2,250 | 541 | 65 | 15.43 | 1.27 | 80.62 | 7.33 |
| light / bounded graph-off | 2,569 | 238 | 49 | 11.51 | 0.53 | 91.52 | 7.78 |
| light / partial batch | 2,450 | 407 | 0 | 11.89 | 0.84 | 85.75 | 8.41 |
| light / 6 mm | 1,495 | 1,362 | 0 | 20.67 | 0.50 | 52.33 | 7.84 |
| light / promotion geometry | 1,547 | 1,301 | 8 | 19.47 | 0.62 | 54.32 | 7.67 |
| nominal / original graph-on | 1,494 | 88 | 1,274 | 10.20 | 1.24 | 94.44 | 6.52 |
| nominal / bounded graph-on | 2,183 | 99 | 568 | 9.28 | 0.79 | 95.66 | 9.60 |
| nominal / bounded graph-off | 2,004 | 58 | 794 | 8.15 | 0.73 | 97.19 | 9.61 |
| nominal / partial batch | 2,484 | 373 | 0 | 12.30 | 0.48 | 86.94 | 10.41 |
| nominal / 6 mm | 1,518 | 1,337 | 2 | 22.10 | 0.61 | 53.17 | 9.91 |
| nominal / promotion geometry | 1,754 | 107 | 994 | 10.57 | 0.57 | 94.25 | 8.63 |

The original unbounded attempt remains failed: light completes 1,352 native rows
with 1,269 correct/82 inaccurate and 1,506 unavailable scheduled rows before CUDA
OOM (observed peak 21.51 GiB); nominal completes 619 with 617/1 and 2,239 unavailable
before the TSDF guard (16.42 GiB). Its favorable conditional prefix errors are
not full-opening success. No failed tail is removed or retried.

| Condition/system | Longest native loss, observed s | Longest correct-pose absence, observed s |
|---|---:|---:|
| light / bounded graph-off | 0.067 | 0.183 |
| light / partial batch | 0.000 | 0.250 |
| light / 6 mm | 0.000 | 3.400 |
| light / promotion geometry | 0.017 | 4.133 |
| nominal / bounded graph-off | 7.850 | 7.850 |
| nominal / partial batch | 0.000 | 0.317 |
| nominal / 6 mm | 0.017 | 13.500 |
| nominal / promotion geometry | 9.417 | 9.417 |

Recovered gaps use acquisition time from first unavailable row to recovery.
Terminal gaps are censored at the last original sample. Graph-off nominal loses
472 final frames from 70.7667 s: observed span 7.850 s, sampled duration
472/60 = 7.867 s. Promotion geometry loses 566 final frames from 69.2 s:
observed span 9.417 s, sampled duration 9.433 s. Partial batch has no native loss
but correct-pose gaps still reach 0.250/0.317 s. At 6 mm nominal has two one-frame
native losses (16.7 ms each), yet ends with 13.5 observed seconds without a
correct pose. Removing native loss cannot substitute for accurate availability.
All-finite nominal point p95 is 100.30 mm graph-off and 129.03 mm with promotion
geometry, rather than their favorable accepted-only errors.

## Partial batch: useful gain, incomplete quality

Relative to graph-off, partial admission gains 198 correct light rows and loses
317 (net -119); nominal gains 631 and loses 151 (net +480). Inaccurate acceptances
increase by 169/315. The native capacity stall is removed when a filtered batch
can only partially fit, but maintaining the original point checks does not make
its references reliable. Both accepted point p95 remain above 1 cm. Keep the
option as an isolated diagnostic, not the common default.

There are 182/263 partial-batch events across all candidates. Actual unique CUDA
query/feature/causal rows obey the cap in every frame. Neither primary becomes
native-lost, so neither has a native return; this is not evidence of strict
original-material recovery. The first partial light attempt predates actual
promotion-call telemetry; its missing telemetry is not zero promotions. Original
seed/control checks and saved registration/map evidence remain available.

## Geometric threshold versus RANSAC hypotheses

On frozen primary pairs, the expected correct pose has at least five compatible
pairs on 2,399 light / 1,833 nominal rows at 4 mm, versus 2,854/2,254 at 6 mm.
Native-lost rows with this support rise from 3/14 to 49/206. A fixed returned pose
crosses five pairs at 6 mm on 18/88 correct rows but also 3/16 inaccurate rows.
These are geometry eligibility diagnostics, not counterfactual accepted poses.

The original shared NumPy stream is reconstructed from seed zero, 100 draws per
eligible remaining pool and saved cluster inlier removal. The inspected executed
pipeline contains no other NumPy randomness. All 31 checked calls (7 light,
24 nominal) reproduce every saved retained cluster. Raw sampled fits, clustering,
selection and SDF are scored separately with evaluator-only truth.

At nominal 63.4667 s, no correct supported 4 mm raw hypothesis is sampled, but
14 exist in a bounded enumeration of evaluator-compatible four-point subsets.
The previous pose is correct (0.708 mm/0.054 degrees), with five current 4 mm
pairs. Nine of ten inspected nominal missing-hypothesis calls admit a correct
supported fit in that bounded enumeration. Conversely, light 70.65 s and nominal
63.3833 s already sample a correct supported fit but retain no correct cluster.
More RANSAC draws alone would not address both mechanisms. Oracle-selected fits
never enter inference; their existence is not a deployable recovery algorithm.
Enumeration is limited to evaluator-compatible subsets; a negative result does
not prove the absence of every fit inside the evaluator pose limits.

At nominal 70.7667 s the exact expected pose has zero 4 mm pairs and five at
6 mm. At 74.2 s/end it has only 1/0 at 4 mm and 2/1 at 6 mm. Neither threshold
restores late support for the exact expected transform. Some fits inside the 1 cm/5-degree bounds can
support more points than the exact expected transform; these are separate tests.

In actual complete 6 mm trajectories, only 8/4 correct rows are gained versus
1,082/490 lost. Precision falls to 52.33%/53.17%; reject 6 mm as a tracking
correction and keep 4 mm. At nominal 76.6167 s the native returned cluster is
wrong by 65.00 mm/4.40 degrees: four pairs satisfy 4 mm, five satisfy 6 mm. SDF
loses all support and the native cluster fallback is returned. This illustrates
an actual inaccurate acceptance admitted by the wider support threshold.
Other saved wrong poses already have strong 4 mm support before SDF: light
6 mm at 54.6833 s has 15.07 mm point error and 14 such pairs. The threshold
also changes later sampling/reference/TSDF history; a final-gate-only or
SDF-only explanation is insufficient.

## Actual promotion impact

Saved original graph-off reconstruction reproduces all 540/390 primary fused
promotions within 0.003/0.013 mm. Three light promotions have median spread
8.78–9.05 mm; one nominal promotion has 15.71 mm. The original geometry gate is
disabled, so they pass visibility/stability confirmation. One nominal promotion
uses nonconsecutive good observations because reset-on-bad is false; this alone
does not justify changing that additional control.

In the isolated enabled-gate replay, 24 light / 22 nominal failed check attempts
block 15/11 distinct secondary points. No primary check fails in either changed
trajectory; all 510/330 observed primary promotions pass. First blocked secondary
checks occur at 38.4167/36.35 s, followed by primary pose divergence at
38.4667/36.3833 s. The original primary bad promotions cannot simply be removed
from a frozen map to predict the new trajectory. Shared RANSAC draws and altered
candidate/reference histories can contribute; their separate causal effects
were not isolated by another intervention.

The complete measured outcome loses 1,022/250 correct primary poses, adds
1,063/49 inaccurate acceptances and lengthens nominal terminal loss. Reject this
flag as a common correction on these data; the existing default stays disabled.
This does not show that geometric confirmation is intrinsically useless, or
that median spread alone verifies material/pose quality. Tentative-point fallback
exists, but no primary original graph-off row actually uses it; it cannot explain
these baseline primary failures. Its controls remain unchanged.

Primary native/correct/strict-historical-4-mm returns are 8/1/0 light and
53/37/0 nominal with promotion geometry, versus 0/0/0 partial and 0/0/0 light,
2/0/0 nominal at 6 mm. First-observation material anchors assume rigid leaf
membership and cannot certify ownership. Reactivation is not strict recovery.

## Pixels and original initialization

Native depth lift truncates pixels; mask/promotion and IPC measured-depth checks
round. Truncation reproduces all 161,293/125,763 original primary saved pairs
within 1 micrometer. On fixed UV/map pairs, rounding recovers five-point correct
support on 183/103 frames but loses it on 101/55. Median pair error worsens
7.46→7.88 / 6.82→7.11 mm; p95 worsens 36.03→36.51 / 50.78→51.23 mm.
The median 3D shift is 2.40/2.35 mm, p95 10.66/11.83 mm. No blanket rounding
correction or extra replay is justified. Integer seed queries are unchanged.
Later seed-material projected pixel p95 reaches 8.05/7.29 px; quantization is not
the complete explanation of observed mismatch.

Actual `SAM2.add_new_prompt` calls use five positive raster-linspace points,
including two at image top/bottom one pixel from the source boundary. The
worker's distributed interior references are preservation checks, not those
prompts. Observed source masks/prompts and initial outputs match the original
baseline. Primary source retention is 99.12%/98.94%, IoU 0.431/0.423, with
192,131/195,441 extra pixels. Only 5/30 and 6/30 seed query pixels lie inside
the original automatic component.

Targeted RGB inspection shows expansion onto door relief and other plausible
same-leaf regions. Extra pixels or overlapping source components alone do not
prove contamination. There is no ownership truth mask; absolute seed accuracy
and a causal prompt intervention remain unverified. Keep original prompts and
initialization. Relative graph-off static p95 (133 rows) is 1.80/1.61 mm; early
opening p95 (287 rows) is 1.85/2.93 mm. Later degradation is measured, but initial
component ambiguity and accumulated correspondence/map bias cannot be fully
separated causally without an independent ownership/calibration check.

## Memory and every candidate

Component peaks are measured separately and need not occur together; do not
sum them as a coincident process peak. CPU RSS is not measured. Historical
reference/keyframe/dense-point counts remain observable and unbounded by the
active cap; retained frame masks can also occupy CUDA memory.

| Condition/system | Torch peak, GiB | Max total queries | Causal state, GiB | Frame masks, GiB | TSDF sum, GiB | Historical references |
|---|---:|---:|---:|---:|---:|---:|
| light / bounded graph-off | 5.27 | 520 | 0.952 | 0.923 | 0.583 | 2,670 |
| light / partial batch | 6.06 | 599 | 1.097 | 1.202 | 0.602 | 2,831 |
| light / 6 mm | 5.42 | 498 | 0.912 | 1.148 | 0.657 | 3,155 |
| light / promotion geometry | 5.22 | 494 | 0.905 | 0.890 | 0.619 | 2,480 |
| nominal / bounded graph-off | 5.74 | 630 | 1.154 | 0.798 | 1.832 | 1,890 |
| nominal / partial batch | 7.15 | 720 | 1.318 | 1.687 | 1.400 | 2,939 |
| nominal / 6 mm | 6.24 | 595 | 1.089 | 1.416 | 1.812 | 3,330 |
| nominal / promotion geometry | 5.64 | 611 | 1.119 | 0.811 | 1.258 | 1,950 |

All candidates obey 120 live references and their actual unique CUDA storage
counts, while stable IDs and historical records are preserved. This establishes
completion/resource behavior for the recorded range, not an indefinite memory
bound. Secondary totals below are candidate-frame sums, not additional unique
physical leaves; ownership ambiguity is retained.

| Condition/system | Secondary correct available | Secondary inaccurate accepted | Secondary native lost |
|---|---:|---:|---:|
| light / bounded graph-off | 4,258 | 4,296 | 2,818 |
| light / partial batch | 3,678 | 4,917 | 2,827 |
| light / 6 mm | 4,070 | 7,353 | 0 |
| light / promotion geometry | 4,421 | 4,178 | 2,371 |
| nominal / bounded graph-off | 3,304 | 1,127 | 7,770 |
| nominal / partial batch | 3,237 | 2,304 | 6,324 |
| nominal / 6 mm | 3,948 | 3,395 | 2,411 |
| nominal / promotion geometry | 3,605 | 1,157 | 7,156 |

No common variant resolves secondary errors. Per-candidate counts, all finite
poses, every interval and all source rows remain in the saved comparison.

## Retain, reject and next problem

Keep the bounded cap/storage compaction, original 4 mm threshold, truncation and
original SAM2 initialization. Keep partial admission as a tested diagnostic
option with its real nominal benefit and documented precision cost. Do not adopt
6 mm or enabled median-spread promotion as a common fix. No combination has been
tested or enabled; graph-on remains the existing unqualified prototype default.

Next address preservation/selection of supported correct 4 mm hypotheses, first
on saved nominal 63.3833–63.4667 s: distinguish missing sampling from destruction
by clustering/refinement and test one observable candidate-retention change.
Current-pair support of the previous pose is an evidence-backed hypothesis to
inspect, not a validated fix. Then address accumulated reference/map bias when
exact expected-pose support becomes scarce. Strong self-consensus on a wrong pose
must not be treated as proof of identity. Do not substitute P2P or resume policies.

One attempt per condition does not establish statistical repeatability; light
and nominal have different original seeds, so this does not isolate lighting.
Absolute calibration/ownership, reliable material recovery, contact/occlusion,
150 ms operational freshness, hardware and official release remain unqualified.

Ignored evidence is under
`outputs/b1/perception/point2pose-isolated-improvements-01/`: protocols, frames,
original/configured/restored controls, correspondence/RANSAC audits, actual
promotions/prompts, memory/return audits and `full-sequence-comparison.png`.
A first partial-light evaluator failed only while serializing a NumPy integer;
the evaluator-only repair and original failed log remain saved. Inference was
not retried. Ignored payloads are not recoverable from Git.
See [[point2pose-bounded-global-graph-ablation|the starting graph-off baseline]],
[[b1-perception-findings|preserved earlier attempts]] and
[[point2pose-consumer-impact-and-failure-windows|consumer metric limits]].
