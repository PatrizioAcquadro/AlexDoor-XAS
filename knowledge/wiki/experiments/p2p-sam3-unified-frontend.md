# Unified SAM3 Video Frontend for Point2Pose

## Question and method

Which single persistent SAM frontend best balances fixed-target quality, continuity and
full-pipeline resources?

October 7 comparison: SAM3/SAM3.1 with periodic reconditioning ON/OFF, one original object, all
2,858 original captures per condition and three frozen physical targets. P2P retains graph OFF,
partial batches ON, rollback OFF, five inliers/4 mm, all-query TAPIR and vectorized SDF. Changing
initialization and subsequent masks together makes this a pipeline comparison.

## Result and decision

**Select original SAM3 with detection reconditioning OFF as the experimental reference.** It
improves ON availability across all bounds/targets but does not win every metric or tail. Primary
ON/OFF results retain all scheduled frames:

| Condition | SAM3 period | Correct availability 10/15/20 mm | Accepted precision 10/15/20 mm | Maximum correct gap 10/15/20 mm (s) | Last 5 s correct 10/15/20 mm |
|---|---|---:|---:|---:|---:|
| light | 16 (ON) | 57.73/94.16/98.92% | 57.85/94.35/99.12% | 3.817/0.167/0.050 | 9/209/280 |
| light | 0 (OFF) | 70.33/96.19/99.86% | 70.35/96.22/99.89% | 3.100/0.100/0.017 | 2/212/298 |
| nominal | 16 (ON) | 70.40/73.58/73.65% | 95.36/99.67/99.76% | 6.350/6.350/6.350 | 0/0/0 |
| nominal | 0 (OFF) | 87.40/96.54/97.73% | 89.21/98.54/99.75% | 0.150/0.117/0.117 | 153/232/241 |

SAM3.1 unbounded attempts fail after 1,313/1,151 masks, leaving only 45.84/40.17% full scheduled
availability despite perfect prefix precision. Bounded history exactly matches these prefixes,
reduces allocator peaks to 5.06 GiB and completes both runs; that is a resource repair, not
material-tracking improvement. SAM3.1 OFF still regresses nominal quality and is retired.

Selected SAM3 OFF retains a nominal 43.10 mm accepted peak, weaker light accuracy than the older
baseline and memory growth. Fixed-frame seed/renewal contamination precedes reconditioning; mask
presence and semantic ID are distinct from native/geometric support. Subsequent
[[p2p-sam3-selected-development|bounded memory and renewal filtering]] are separate interventions.

## Limits and evidence

All nonseed requests exceed 150 ms. The seed stays in availability denominators and is excluded from
errors. Resource failures remain distinct from native losses; native return is not original-material
recovery. One door, ideal depth and differing seed histories do not qualify hardware, contact or
full A3/A4 inputs.

Evidence roots under `outputs/evidence/b1/perception/`: `p2p-sam3-unified-01/`,
`p2p-sam3-followup-01/`, `p2p-unified-consolidation-01/`. Full three-target distributions,
resources, failed prefixes and material audits remain in their comparison JSON/reports. Git:
`7404a41`, `740ee57`, `21c2abf`, `72532e1`, `f3d8797`, `563847a`; historical source `57ad483`.
