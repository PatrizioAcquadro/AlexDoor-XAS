# Point2Pose Evaluation and Isolated Performance

## Question and method

How do evaluation tolerance and isolated compute controls change full-window quality and cost?

Historical multi-object comparisons retain original 2,858-row openings, full denominators and three
fixed observed targets. Positional diagnostic bounds 10/15/20 mm each retain 5-degree rotation; they
do not change the 4 mm registration gate. Complete timing includes image frontend, P2P and
diagnostic work.

## Result and decision

Retain all-query TAPIR, Large/full-frame 480, four iterations/eight passes and SuperPoint for the
selected experimental path. Chunk 64 remains a historical narrow-10 mm comparator; larger query
blocks are not numerically equivalent. Isolated controls and decisions:

| Control | Decisive outcome | Extension decision |
|---|---|---|
| Chunk 256 | Median savings 8.73/11.40% light/nominal. | Light misses fixed 10% rule; preserve pilot. |
| All queries | Median savings 14.16/18.62%, all primary poses correct. | Extend both complete openings. |
| SAM2 Small | Light saves 7.86% with secondary 3/4 initialization refusals; nominal saves 22.44% but primary mismatch yields zero integration availability (seed mask IoU 0.363). | Reject; native completion cannot bypass initialization. |
| Crop 480 | Fixed 600×600 crop from original 960×600 image; complete median ~13.9% slower both conditions, light query peak 370→473. | Reject speed extension. |
| Crop 384, four iterations | Eight actual passes retained; median 9.2/15.9% slower. | Pixel reduction alone insufficient. |
| Crop 256, four iterations | Four passes (pyramid changes too); savings 7.31% light, -0.35% nominal. | No common extension. |
| Crop 384/256, two iterations | Four/two actual passes; savings 7.94/3.69% and 13.69/3.81%. | No common extension. |
| Crop 256, one iteration | One pass; median savings 12.79/5.85%, so original rule fails. About 20% p95 saving in both; nominal two poses exceed 10 mm, all pass 15/20 mm. | Separate prespecified >=15% p95 extension recorded in `full-admission.json`; median failure remains. |
| Simplified SVD | Median/p95 239/329 ms light, 306/407 nominal. Nominal accepts 483/600, loses 109, correct gap 1.383 s versus 16.7 ms; 15/20 mm availability 80.37% despite 100% conditional precision. | Reject continuity regression; no full extension. |

Complete p95 remains 553.79/644.39 ms, with zero timely complete observations; subtracting
diagnostic time does not meet 150 ms. Target transport exposes distinct lever-arm errors and assumes
original local coordinates—it does not evaluate actual A4 actions or complete A3 references.

Retired model/tracker/SVD variants are not maintained options. The later single-object frontend and
selected recipe belong to [[p2p-sam3-unified-frontend]] and [[p2p-sam3-selected-development]].

## Limits and evidence

Availability retains missing/rejected/lost rows; precision is conditional on acceptance.
Full-angle/original-material/hardware/contact qualification is absent. Original travel ends at
63.7209 degrees. Model-only or stage-only latency cannot replace complete publication age.

Evidence: `outputs/evidence/b1/perception/p2p-performance-01/`, especially
`full-target-evaluation.json`, isolated compute comparisons, original target records and failed
variants. Historical executable source: `57ad483`; retirement `c01c3a1`.
