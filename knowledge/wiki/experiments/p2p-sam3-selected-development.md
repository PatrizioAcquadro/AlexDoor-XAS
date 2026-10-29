# Selected SAM3 OFF: Memory and Quality

## Question and method

Can bounded causal SAM3 memory preserve behavior, and does applying the already configured renewal
depth gate improve the selected trajectory?

Two separately attributed interventions preserve original 31 s initialization and both 2,858-capture
histories. Memory-only `b911887` retains native useful conditioning/pointers and indices without
unbounded CPU offload. Quality `bc6fca5` applies the previously inactive native 10 mm/radius-2
depth-edge filter only to selected renewal points, leaving initialization and masks unchanged.

## Result and decision

Memory-only replays have zero differing masks, poses, decisions, reference history or registration
traces. SAM final allocation falls 7.84→3.66 GiB and peak 8.39→4.21 GiB; CPU high-water falls to
7.46 GiB. No nonseed request meets 150 ms. **Retain bounded memory for the selected recipe.**

The independent quality comparison retains original targets and all 2,858 scheduled captures:

| Condition | Original target | SAM3 OFF availability % 10/15/20 mm | New gate availability % 10/15/20 mm | New gate accepted precision % 10/15/20 mm |
|---|---:|---:|---:|---:|
| light | 0 | 70.33/96.19/99.86 | 85.83/99.79/99.93 | 85.86/99.82/99.96 |
| light | 1 | 69.03/94.05/99.76 | 87.12/99.90/99.93 | 87.15/99.93/99.96 |
| light | 2 | 71.66/97.13/99.90 | 83.80/99.58/99.93 | 83.83/99.61/99.96 |
| nominal | 0 | 87.40/96.54/97.73 | 99.13/99.72/99.86 | 99.19/99.79/99.93 |
| nominal | 1 | 86.21/96.50/97.76 | 99.13/99.69/99.86 | 99.19/99.75/99.93 |
| nominal | 2 | 87.82/96.68/97.76 | 98.74/99.65/99.83 | 98.81/99.72/99.89 |

The renewal gate improves all three target availabilities, but light position maximum grows
21.26→23.82 mm and rotation p95 grows 0.69→1.10 degrees; nominal rotation p95 grows 0.47→0.68
degrees despite better position and no native losses. Light 10 mm final-five-second target-0 correct
support remains only 8/301. **Retain the gate as the selected experimental recipe with these
tradeoffs.**

## Limits and evidence

Light per-call rejection totals/identities are unavailable because an observer joined acquisition ID
to native index. All 354 admitted light references pass the configured gate; nominal retains full
rejection events. Missing counts are not zero and no inference replay repaired instrumentation.
Bounded history does not bound total P2P map/TSDF state or qualify material identity,
hardware/contact or freshness.

Evidence: `outputs/evidence/b1/perception/p2p-sam3-selected-01/`; complete memory equivalence,
`quality-comparison.json`, `quality-birth-gate-audit.json`, `report.md` and original
target/gap/resource records. Current settings are canonical in `configs/point2pose_selected.json`;
further diagnosis: [[p2p-selected-residuals]].
