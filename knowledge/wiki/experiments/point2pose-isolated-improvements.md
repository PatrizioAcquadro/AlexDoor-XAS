# Isolated Point2Pose Improvements After Graph-Off Diagnosis

## Question and method

Which single changes improve support after the graph-off diagnosis without hiding complete-sequence
regressions?

Three independent variants—partial reference batches, 6 mm inlier threshold, native promotion
geometry check—each run one full light/nominal CUDA opening. All six attempts complete 17,148
original frames; original seed, masks, point order, 120-reference cap and unrelated controls remain
fixed. No combined variant is tested.

## Result and decision

The complete primary comparison uses all 2,858 captures per condition; correctness requires 1 cm and
5 degrees, with seed excluded from correct/error counts.

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

Partial batches improve nominal availability but cost precision; keep them as an explicit
experimental option, not evidence of a qualified common default. Reject 6 mm and promotion-geometry
activation: both worsen complete-condition outcomes. Saved audits show sampling/refit, reference
promotion and correspondence history interact; more pairs or fewer losses alone are insufficient. No
common tracking replacement is adopted.

## Limits and evidence

Failed unbounded runs retain 1,506/2,239 missing suffix rows, CUDA OOM/TSDF refusals and original
resource counts. A NumPy serialization repair did not repeat inference. Single attempts, ideal depth
and different seeds do not isolate lighting or qualify hardware, original-material recovery,
contact, freshness or a full A3/A4 reference.

Evidence: `outputs/evidence/b1/perception/point2pose-isolated-improvements-01/`, `controls`,
correspondence/RANSAC/promotion/pixel audits, full candidate/memory reports and
`full-sequence-comparison.png`. Git: `fac02e5`, `4c6dc25`, `29f194b`, `c919ea8`.
