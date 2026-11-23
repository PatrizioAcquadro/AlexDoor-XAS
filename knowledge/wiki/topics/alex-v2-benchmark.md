# Alex V2 Benchmark

B0 used a fixed-base IHMC Alex V2 torso, six right-arm joints and a
collision-derived tool point. Physics ran at 120 Hz and control at 60 Hz.
Position-only differential IK limited translation to 0.02 m per tick. A2/A3
recorded rotations while the controller commanded translation. Success was the
first 45° crossing. [[decisions/calibrated-position-only-alex-v2-execution|Execution and contact measurement]]

The five poses shared one simulated door family. D0 was nominal. D1/D2 used
yaw offsets of ±0.05 rad and D3/D4 used ±0.10 rad, with centimeter translations.

## Development phases

| Phase | Contribution and evidence |
|---|---|
| 1.1 · Runtime foundation | Established the package, host checks and simulator startup with external assets. Data and policy imports stayed independent of Isaac. [[topics/system-architecture|Current architecture]] |
| 2.1 · Scripted execution and recording | Recorded pre-action samples and the terminal response through approach, contact, push and release. A1–A4 exports shared each physical episode and outcome. The standalone numerical export was retired at `70fe3cf`. [[decisions/door-relative-task-and-matched-representations|Matched representations]] |
| 3.1 · Learned baselines | Trained state-only ACT and Diffusion on A2/A3 and evaluated them in matched closed-loop rollouts. [[experiments/phase-3-unified-evaluation|Evaluation protocol and results]] |
| E1.1 · Alex V2 migration | Calibrated joint order and the offset tool point, and isolated raw contacts from the exact door panel. [[decisions/calibrated-position-only-alex-v2-execution|Calibration decision]] |
| E2.1 · Local stabilization | Hardened contact entry, terminal settling and force reporting. Across four N50 cells, 144/144 local smoke rollouts succeeded with zero adapter rejections. This measured rollout stability. |
| E3.1 · Cluster compatibility | Returned the first two checkpoints from Gilbreth for a workstation loading check. [[experiments/gilbreth-nested-scale-sweep|Compatibility pilot and training sweep]] |
| E4.1 · Scale dataset | Built one matched dataset with nested training subsets and fixed holdouts. [[decisions/one-scale-master-with-nested-views|Dataset membership and normalization]] |
| E5.1 · Nested sweep | Completed the seed-0 training matrix on Gilbreth A100 GPUs. [[experiments/gilbreth-nested-scale-sweep|Training result]] |
| E6.1 · Unified evaluation | All 576 rollouts succeeded, leaving no winner on this benchmark. One 219.95 N event remains `REVIEW_REQUIRED`. [[experiments/phase-3-unified-evaluation|Evaluation]] · [[experiments/act-a3-n50-seed-112-force-diagnostic|Force diagnostic]] |

Git `9c16e3d` preserves the historical source and `7f1fc8c` preserves the
curated evaluation reports. The ignored B0 datasets, checkpoints and rollout
payloads were removed. The [[topics/purdue-b1-robot-and-contact|Purdue B1 runtime]]
uses its own calibration, data and checkpoints. B0 outcomes cover one simulated
door family and one training seed. Held-out geometry and hardware safety were
outside the study.
