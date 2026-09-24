# Alex V2 Benchmark

Historical B0 record, preserved at Git `9c16e3d`; code, calibration, D0–D4 scenes,
dataset payloads and compatibility readers were retired during the B1 cleanup.
The active successor is [[topics/purdue-b1-robot-and-contact|Purdue B1]].

B0 used a fixed-base IHMC Alex V2 torso, six right-arm joints and a calibrated
collision-derived tool point. Physics ran at 120 Hz and control at 60 Hz.
Position-only differential IK limited translation to 0.02 m per tick; A2/A3
rotation was represented but not actuated. Success was the first 45° crossing.

The five poses shared one door family: D0 nominal, D1/D2 yaw ±0.05 rad and
centimeter translations, D3/D4 yaw ±0.10 rad and centimeter translations.
Raw GPU contacts were filtered by the exact panel actor; net robot force and
geometric contact were not substitutes for measured panel force.

The completed 550-episode study trained sixteen ACT/Diffusion × A2/A3 ×
N50/N100/N250/N500 cells. All 576 evaluation rollouts succeeded, selecting no
winner. One 219.95 N event remains `REVIEW_REQUIRED`; see
[[experiments/phase-3-unified-evaluation|Unified Evaluation]] and
[[experiments/act-a3-n50-seed-112-force-diagnostic|Force Diagnostic]].

These results concern one simulated door family, state-only inputs and seed-0
training. They do not establish held-out geometry generalization, hardware
safety, physical deployment or Purdue checkpoint compatibility.
