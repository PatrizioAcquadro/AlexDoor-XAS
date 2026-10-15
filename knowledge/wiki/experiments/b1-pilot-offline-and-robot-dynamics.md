# Pilot Checkpoint Diagnosis and Robot-Only Dynamics

Baseline: `e26436b`, branch `PatrizioAcquadro/b1-policy-pilot`. This follow-up to the
[[b1-act-diffusion-a1a2-pilot|physical-data pilot]] separates offline policy errors
from robot feedback/load/stopping. It changes no encoder, normalization, policy,
action meaning, common setup, gain, cadence, task threshold or training budget.
Geometry/TSDF work stays in its own checkout. No operational startup is repeated.

## Persistent evidence and reproduction

New local evidence is outside the managed worktree, at
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1-policy-diagnosis-20261015/`.
`offline-complete/` contains the full report, predictions/targets, window indices,
clipping attribution and the reconstructed ACT baseline check. `offline/` and
`offline.log` retain an initial reporting failure: an empty proprioceptive group
produced NaN during JSON serialization after ACT A1 inference. The grouping fix
changes reporting only; the completed run uses identical models/seed.

Original episodes, checkpoints, normalization, training histories and operational
refusals remain untouched under the active worktree's
`outputs/generated/b1-policy-pilot/act-a1a2-pilot/`. They are ignored payloads,
not recoverable from Git. Keep this checkout until those inputs are archived.

Run `scripts/pilot_diagnosis.py --root <original-pilot> --output <new-directory>`
with the Isaac Lab Python runtime and this checkout's `src` on `PYTHONPATH`.
CUDA is mandatory for model inference. Runs use a GPU occupancy check and the
local `/tmp/alexdoor-xas-gpu.lock`; no other process is stopped. The lock alone
does not coordinate unrelated tools, so inspect GPU use before every launch.

## Offline method

All four existing final checkpoints are evaluated without optimization. Every
sixth tick starts a 12-action chunk on all three recorded episodes; end padding
is excluded. Development contributes 518 starts / 6,200 valid action predictions.
Attribution uses the target action's recorded phase and zero-based chunk position,
including transitions. These overlapping predictions are descriptive correlated
samples, not independent trials. Positions 0–5 are the nominal executed prefix;
6–11 are predicted but discarded at six-tick replanning.

The historical 256 development windows are included for an ACT regression check.
Their ACT predictions reproduce original initial/final z-score losses within
1e-6 using the exact original seed-17 architecture initialization, with no training.
Diffusion uses seed 23 / ten-step DDIM in the expanded chronological query stream;
its noise assignment differs from the original 256-window stream. Do not treat
its changed sampled values as checkpoint changes or select among seeds.

Baselines are the zero action and a single componentwise median fitted only to
all train actions. A1 errors are per joint in degrees (native radian MAE also
retained). A2 uses world translation norm in mm and relative SO(3) angle in degrees.
No ACT z-score versus Diffusion min-max loss ranking is made. Labels remain the
existing observed-state A1 target deltas and full-pose A2 world deltas.

### Development errors

| Path / comparator | Mean joint MAE, averaged over 7 joints | A2 translation mean / p95 | A2 rotation mean / p95 |
|---|---:|---:|---:|
| ACT A1 | 0.361 deg | — | — |
| Diffusion A1 | 1.108 deg | — | — |
| Zero A1 | 0.419 deg | — | — |
| Train median A1 | 0.326 deg | — | — |
| ACT A2 | — | 11.98 / 26.34 mm | 1.593 / 1.898 deg |
| Diffusion A2 | — | 16.61 / 42.02 mm | 2.934 / 8.503 deg |
| Zero A2 | — | 7.40 / 27.21 mm | 0.638 / 1.673 deg |
| Train median A2 | — | 6.39 / 27.19 mm | 0.661 / 1.711 deg |

The JSON is authoritative for exact quantiles. ACT A1 beats zero on five joints,
but beats the train median on only shoulder Z and wrist Z. Wrist X and gripper Y
MAE are 0.607 / 0.634 deg versus zero 0.495 / 0.387 deg. Diffusion A1 is worse
than both baselines on all seven joints. Both A2 policies lose to these baselines
on aggregate means. A low zero-command error does not demonstrate task progress
or justify sending zero during a manipulation task.

### ACT A2 regression

ACT A2 train translation/rotation means are 1.88 mm / 0.256 deg (right identity)
and 3.68 mm / 0.588 deg (left), versus development 11.98 mm / 1.593 deg. This is
an observed train/development gap, with one development identity and one seed.

The original-window z-score L1 regression 0.937818 → 1.006923 is reconstructed
from native per-column errors and unchanged train scales. Translation Y MAE
increases 4.04 → 9.04 mm, rotation Y 0.513 → 0.821 deg and rotation Z
0.980 → 1.301 deg; translation X improves 7.31 → 3.34 mm. The aggregate scalar
therefore masks opposing component changes.

| Development phase | Initial → final translation mean | Initial → final rotation mean |
|---|---:|---:|
| approach | 25.69 → 22.15 mm | 1.690 → 1.723 deg |
| contact | 8.94 → 7.09 mm | 0.933 → 1.086 deg |
| push | 9.66 → 10.87 mm | 1.179 → 1.644 deg |
| hold | 12.24 → 13.83 mm | 1.349 → 1.854 deg |
| release | 10.40 → 8.41 mm | 0.915 → 0.958 deg |

Final translation means across chunk positions span 11.82–12.19 mm; rotation
means span 1.574–1.615 deg. There is no monotone horizon deterioration that alone
explains the regression. Phase/position intersections and all other paths are
retained in the full report. This diagnoses predictions, not physical tracking.

### Clipped development features

The unchanged +/-10 z-score clip affects 121,195 / 4,027,694 entries (3.0090%),
51 distinct dimensions, and every development row. 37 static depth cells and
one static red / one static green cell clip on every row: 121,017 entries,
99.853% of clipping. Remaining entries are transient recent RGB cells and neck
position/velocity. Train clipping is approximately 0.0006%, mainly neck transients.

Static inspection features have almost no within-episode variation and only two
train identity values. Small train variance makes development offsets large in
z-score units: static green cell (4,7) reaches 103.66; static red (3,7), 11.75.
Depth is stored divided by the bound far distance, actually 100,000 m in these
recordings; one clipped static depth dimension uses the 1e-8 standard-deviation
floor. Other clipped depth dimensions also have very small train variance.
This is a demonstrated input-distribution/clipping issue. It does not establish
that clipping, the encoder, or a particular image cell caused ACT A2 regression:
no input ablation or preprocessing change was made.

## Evidence versus hypotheses

Demonstrated: train learning, inadequate development action accuracy against
simple baselines, component/phase-specific ACT A2 regression, and extensive static
feature saturation on development. One short budget, stochastic Diffusion sampling,
limited identity diversity, action imbalance and clipped visual offsets are
possible contributors; their causal effects remain unisolated. No checkpoint is
qualified for learned door manipulation, and the earlier zero-command operational
refusals remain infrastructure/prerequisite checks, not policy failures.

## Robot-only 6.0D boundary

The separate `scripts/verify_pilot_dynamics.py` protocol uses the unchanged common
robot setup, controller, two 120 Hz load samples per 60 Hz command and existing
`ResidualLoadModel` / `PurdueIO.stop`. Calibration and verification are distinct;
force references are evaluator-only. Dynamic results and remaining qualification
are recorded below after inspection of their raw traces. No geometry provider,
door contact, learned command, training, extended collection or sealed test is
introduced. Local observed geometry and 6.0E, together with the required 6.0D
checks, must precede bounded diagnostic interaction 6.0F.
