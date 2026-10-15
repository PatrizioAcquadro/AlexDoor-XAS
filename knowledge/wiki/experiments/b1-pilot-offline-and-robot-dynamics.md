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

The subsequent [[b1-pilot-preprocessing|fixed-unit preprocessing pair]] isolates
one input correction on these same historical episodes. It reproduces the ACT A2
baseline exactly and reports prefix mean gains alongside release/p95 regressions;
this original diagnosis and its checkpoint values remain unchanged.

## Evidence versus hypotheses

Demonstrated: train learning, inadequate development action accuracy against
simple baselines, component/phase-specific ACT A2 regression, and extensive static
feature saturation on development. One short budget, stochastic Diffusion sampling,
limited identity diversity, action imbalance and clipped visual offsets are
possible contributors; their causal effects remain unisolated. No checkpoint is
qualified for learned door manipulation, and the earlier zero-command operational
refusals remain infrastructure/prerequisite checks, not policy failures.

## Robot-only 6.0D protocol and results

`scripts/verify_pilot_dynamics.py` completed one Isaac/CUDA run on RTX 4090:
1,080 control ticks / 2,160 feedback samples, 18 s of simulation. Physics is
120 Hz, control and cameras 60 Hz. Common floor, initial arm/neck configuration,
robot/grippers/camera, gains, A1/A2 dispatch and physical limits are retained.
Commissioning fixtures are moved to X=3 m; no corpus door is spawned and no robot
contact is detected. The script bypasses policy/geometry admission explicitly for
this known robot-only diagnostic; it cannot supply an operational admission source.

The protocol is written before starting Isaac. Calibration uses two seconds of
settling, a two-second unloaded cosine excursion, a one-second hold and a one-second
known 2 N world-X perturbation. Joint excursion amplitudes are 0.015–0.030 rad.
The last 0.5 s of the known load verifies residual sign; only unloaded residuals
fit per-joint bounds, with a declared additional 0.02 N m numerical allowance.
A provisional box of initial joints +/-0.08 rad (clipped to physical limits),
0.25 rad/s and 5 rad/s² is declared before verification. It is a diagnostic
candidate envelope, not evidence that every state inside it was covered.

Verification then uses a distinct 80%-amplitude unloaded trajectory, the same
trajectory under 4 N, moving stops with and without that maintained load, and
one unloaded 4 mm A2 world-translation cycle. The injected force remains in the
evaluator and enters neither expected robot torque nor estimator inputs. Its
moment about the actual tool point is computed separately at each physics sample.
The estimator receives only robot feedback, inverse dynamics, Jacobian, joint
state and a fixed world normal. Native PhysX feedback/model is the declared error
model: no measured hardware friction/noise or injected sensor-error distribution.

`dynamics/feedback.npz` and `tracking.npz` retain the full robot trace; the exact
executed script is `dynamics/probe-source.py`. `report.json` separates run completion
from qualification. `analysis.json`, generated by the retained `analyze_dynamics.py`,
reconstructs every estimate/rejection exactly from saved feedback and is
authoritative for conditional coverage. The original report's empty-set `all()`
booleans at phases with zero estimates are vacuous; analysis reports `null` there.
The maintained script fixes that reporting issue and additionally exports normal
lower bounds; no physical trial was repeated to change the outcome.

### Residuals and load estimation

The 2 N calibration selects external sign -1: mean projected-reference error is
0.00162 N m versus 1.62343 N m for +1. Provisional unloaded residual bounds span
0.02001–0.02683 N m across joints. They are not globally verified error bounds.

All 240 samples of the separate unloaded cosine trajectory estimate successfully.
Maximum dynamic residual is 0.00546 N m; estimated force error is at most 0.0172 N,
and conservative force upper bound at most 0.401 N, on this tested trajectory.
The 4 mm A2 cycle also yields all 240 estimates; world-position tracking error
reaches 1.741 mm and applied joint-target error 0.00476 rad.

Under 4 N, only 147/240 samples stay in the provisional joint box. Their maximum
force error is 0.0684 N, upper bound reaches 4.446 N and normal lower bound spans
3.823–3.893 N. Reference force is covered on both substeps of those valid samples;
93 joint-envelope exits remain unavailable, not successes. Applied joint-target
error reaches 0.10384 rad in the loaded cycle versus 0.00843 rad unloaded.

Across verification, 1,110/1,440 samples estimate and 330 refuse with
`outside_verified_load_envelope`. Exit counts overlap: 315 position, 104 speed,
nine acceleration. Evaluator-corrected residuals exceed at least one provisional
bound on 64 samples, all in loaded stopping/return, reaching 0.06504 N m on return.
The calibration cannot be transferred to those states/movements. Do not widen
bounds from this verification or discard its failures.

Both samples are necessary: among 73 fully available loaded-motion tick pairs,
the first sample has the higher bound in 61. Endpoint-only monitoring misses up
to 0.00105 N there and 0.00410 N on return. These small observed differences are
not a bound for untested disturbances. Recorded timestamps verify every 1/120 s
interval, and per-substep counts/errors are retained.

### Stop behavior with physics advancing

Both tests call existing `PurdueIO.stop`, clear pending commands and freeze targets
at measured joints without advancing time. Physics then advances for 60 control
intervals / 120 feedback samples (one second), with targets asserted unchanged.
Stop displacement/rotation maxima include both physics samples.

| Stop condition | Peak joint speed at stop | Maximum tool travel | Final travel at 1 s | Maximum rotation |
|---|---:|---:|---:|---:|
| Unloaded moving | 0.03015 rad/s | 1.576 mm | 0.0366 mm | 0.242 deg |
| Maintained 4 N perturbation | 0.03076 rad/s | 106.845 mm | 106.845 mm | 19.172 deg |

The loaded stop has no available load estimates throughout its 120 post-stop
samples, exits the provisional joint/speed/acceleration envelope and still has
0.0823 rad/s peak joint speed at one second. Unloaded stopping is a local measured
result; it is not a verified general stopping bound. Frozen targets do not imply
zero physical motion under sustained load. After removing the perturbation, the
scripted return reaches measured 0.7864 rad/s and 18.3824 rad/s². The common
0.5 rad/s target-step cap was unchanged; it does not bound actual disturbed
motion. The run completes diagnostically but fails to establish a usable loaded
tracking/stop envelope. No favorable repetition or threshold/gain change follows.

The physical test establishes target persistence during the declared hold. It
does not independently qualify the runner's command-refusal latch, timeout/fault
behavior with continuing physics, arbitrary stopping durations, contact dynamics,
force saturation, singular postures, hardware signals or other configurations.
Existing numerical load/admission/stop contracts pass; their tests are separate
from this measured robot result. `verified`, `stop_verified`, provider
`offline_passed`/`dynamic_passed` and contact rollout readiness remain false.

## Concrete next boundary

The separate [[b1-loaded-hold-and-stop|loaded hold/stop follow-up]] now addresses
the local controller limitation and verifies the ready-pose 0/4 N regime. The
failures above remain valid for this original controller. Neither run supplies a
general verified loaded calibration or contact-stop receipt; all remaining 6.0D
checks still precede contact.

The geometry branch must provide supported observed local ownership, finger
footprints, sweep/clearance and response evidence, and pass independent 6.0E
scoring/replay. **Those local-geometry/6.0E prerequisites together with the required
6.0D feedback/load/stop checks must pass before bounded diagnostic interaction
6.0F.** No second geometric provider, door contact, new training, extended
collection or sealed-test run was added here.
