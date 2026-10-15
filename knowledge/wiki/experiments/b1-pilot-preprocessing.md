# B1 Pilot Fixed-Unit Preprocessing

Baseline: `0b986e6`, branch `PatrizioAcquadro/b1-policy-pilot`. This experiment
isolates one input transformation on the three unchanged historical pilot episodes.
The newer controller, perception, geometry, action meanings and labels are unchanged.
No demonstrations are regenerated and no contact rollout is authorized by this result.

## Evidence and diagnosis

Persistent local evidence is at
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1-policy-preprocessing-20261015/`.
Original episodes, prepared data, statistics and checkpoints remain in the existing
pilot worktree. `original-stats.json`, the original checkpoint reference/hash and
`baseline-source/` preserve the preprocessing reference. These ignored payloads
are not recoverable from Git; retain both locations.

The saved-data audit reproduces 37/64 static depth cells clipping on every one of
3,103 development observations. Overall clipping is 121,195/4,027,694 entries
(3.0090%). The recorded validity interval is approximately 0.01–100,000 m.
Changing depth units alone leaves nonfloored z-scores equal within 4.2e-12 and
all 37 static depth cells still clipped. Variance estimates have only two train
identities; static RGB and nearly constant neck features also amplify offsets.

## One correction

The explicit opt-in `rgbd-grid-fixed-units-v2` recipe uses fixed observation units:
RGB/255 and valid coverage, mean valid depth/5 m, joint angles in radians and
velocities in rad/s. Observation normalization stores center zero and scale one,
not inverse variances estimated from this small pilot. Train-only min/max/count
remain descriptive; action normalization is unchanged and fitted only on train.
The 5 m scale reuses the existing encoder's nominal metric scale; it is declared
before training, independent of development and of the camera far plane.

Depth is not capped at 5 m or invalidated above it. The original sensor mask still
uses the original near/far interval. Invalid pixels contribute neither depth nor
coverage; RGB, coverage, temporal support, pooling and proprioception are unchanged.
The existing +/-10 policy clip is identical in v2 training and inference. The v1
path retains its historical training/inference clip asymmetry for faithful baseline
reproduction. No clip threshold is increased. Recipe-bound contracts reject loading
old checkpoints under v2, and v2 refuses empirical observation normalization.
The legacy default remains explicit compatibility behavior; no checkpoint is migrated.

On the unchanged saved feature tensors, v2 reduces clipping to zero on all three
episodes. The largest absolute input is 7.926. This demonstrates improved input
conditioning only; it does not establish prediction improvement or generalization.

## Prespecified bounded comparison

Use `scripts/compare_pilot_preprocessing.py prepare --root <original-pilot>
--output <persistent-evidence>` once, retaining the original data. It converts only
the depth columns in a new lightweight prepared-data copy; actions, timestamps,
nondepth features and split membership are checked equal. Original raw HDF5 files
remain the authoritative recordings; the copy does not duplicate their 40 GB.

CUDA verification replays every causal 10 Hz encoder acquisition from reset,
including inspection and history, and checks 60 Hz time/proprioception joins.
The finite float32 pooling tolerance is 3e-6 absolute plus 2e-6 relative.
An initial raw replay failed before training: moving the unit conversion inside
the float32 spatial sum differed by up to 0.77 mm on the first checked frame.
That attempt is retained. V2 now preserves v1's pooling arithmetic and applies
its linear unit conversion after pooling, matching the saved-feature conversion
without relaxing the original consistency tolerance.
The numerical raw-observer round trip separately covers preparation/export,
training tensors, inference and checkpoint binding.

Then run only `baseline` and `fixed` modes: ACT A2, seed 17, the original 1,298-input
architecture, horizon 12, batch 32, learning rate 1e-4, three epochs of 32 batches
(96 updates each). Initialization, ordering, actions and budget are held fixed;
use the final checkpoint without development selection. Preserve original final.pt
and both new initial/final checkpoints. Epoch resume states retain completed work.

Evaluate every sixth tick with end padding excluded. Report translation norm in
mm and relative SO(3) angle in degrees, by target phase and chunk position, including
phase-position intersections. Positions 0–5 are the nominal executed prefix;
6–11 are discarded at replanning. This is an offline schedule, not physical execution.
Comparators are zero and a componentwise median of all train actions only.
Correlated overlapping windows are descriptive samples, not independent trials.

All GPU stages require the shared `/tmp/rtx4090-exclusive.lock` plus the historical
`/tmp/alexdoor-xas-gpu.lock`, three continuously idle compute minutes and process/
utilization inspection before launch. Monitor foreign compute during each stage.
Retain interference attempts, stop only our worker, wait again and resume only
incomplete cells or epochs. Free memory alone is not admission.

## Completed CUDA evidence

Implementation: `6169199`, with arithmetic/resume correction `7aa83ba`.
Both paired runs completed exactly 96 CUDA updates on RTX 4090. All initial model
weights are identical across the pair, and the rerun baseline's final weights,
action/observation statistics and architecture match the original checkpoint
**exactly**. The original checkpoint hash remains unchanged. All 96 minibatches
have identical actions and padding across variants. No development checkpoint
selection, additional seed, overfit trial or larger budget was introduced.

The corrected raw replay covers all 2,445 causal encoder acquisitions across the
three original recordings, including inspection/history from reset; maximum saved
versus raw feature difference is 2.3842e-7 in encoded units, below the unchanged
consistency tolerance. Recorded time/proprioception joins are exact. Synthetic
raw-observer/export/batch/checkpoint round trips pass for both recipes and both
families, including refusal to load the opposite recipe. Only ACT A2 is trained.
Numerical regressions also preserve sensor validity independently of policy scale
and protect the pooling-roundoff correction.

`accepted-*.json` and `attempts/*/gpu.jsonl` retain the accepted stage boundaries
and monitoring: no foreign compute or new graphics process was observed during
any accepted stage. The initial failed raw attempt remains separate; no training
was performed in it. Ordinary Xorg/shell/portal/TeamViewer and existing browser
GPU processes are allowed and recorded. The initial supervisor's extra 15% display
utilization cutoff was removed before training because it rejected ordinary desktop
activity; the frozen protocol requires idle compute, process/utilization inspection
and both shared leases, not a utilization threshold. That clarification and the
initial logs are retained. No scientific criterion, clip or training budget changed.
The supervisor queues on the shared lease, releases it on completion and preserves
accepted cells. Long resource waits are not algorithm timing measurements.

## Development predictions

Every sixth tick produces 518 chunks. Full-horizon scoring contains 6,200 valid
predictions; the actually consumed schedule prefix (positions 0–5) contains 3,103
actions, each recorded action once. This remains offline prediction analysis.
The discarded suffix contains 3,097 overlapping predictions. JSON/NPZ retain all
predictions, targets, phases, positions, native column errors and quantiles.

### Executed prefix: overall errors

| Predictor | Mean translation (mm) | p95 translation (mm) | Mean rotation (deg) | p95 rotation (deg) |
|---|---:|---:|---:|---:|
| ACT v1 | 11.927 | 26.121 | 1.5897 | 1.8975 |
| ACT fixed units | 6.443 | 29.707 | 0.9001 | 2.0853 |
| Zero | 7.396 | 27.208 | 0.6377 | 1.6704 |
| Train median | 6.385 | 27.178 | 0.6604 | 1.7100 |

Fixed units reduce prefix mean translation by 46.0% and rotation by 43.4% versus
the paired ACT baseline. Both p95 errors **worsen** (26.12→29.71 mm and
1.897→2.085 deg). Train median has slightly lower mean translation (only 0.058 mm difference,
practically a near tie here) and lower mean rotation; zero also has lower mean
rotation error. Reduced clipping alone would not establish this
prediction benefit, and the observed mean benefit is not uniform superiority.

### Executed prefix: target phase

Each predictor cell is **translation mm / rotation deg**, using mean errors.

| Phase | Actions | ACT v1 | ACT fixed units | Zero | Train median |
|---|---:|---:|---:|---:|---:|
| approach | 360 | 21.91 / 1.699 | 20.76 / 1.635 | 21.13 / 1.366 | 20.32 / 1.445 |
| contact | 180 | 6.99 / 1.075 | 3.86 / 0.705 | 4.01 / 0.278 | 2.25 / 0.289 |
| push | 2203 | 10.83 / 1.644 | 4.01 / 0.789 | 5.26 / 0.501 | 4.09 / 0.515 |
| hold | 180 | 13.83 / 1.854 | 1.73 / 0.200 | 7.80 / 0.937 | 7.23 / 0.955 |
| release | 180 | 8.39 / 0.958 | 14.94 / 1.682 | 9.09 / 0.912 | 9.96 / 0.944 |

Push accounts for 2,203/3,103 prefix actions (71%). Gains concentrate in push and
hold; release regresses from 8.39 to 14.94 mm and 0.958 to 1.682 deg. Approach
remains approximately 21 mm. Keep the release and upper-tail regression alongside
the aggregate benefit; do not qualify a manipulation policy from the mean alone.

### Chunk position

Mean translation mm / rotation deg; positions **0–5** form the consumed prefix.
All phase-position intersections are retained in `summary.json` and each variant's
`physical-errors.json`; no future/discarded prediction is counted as executed.

| Position | ACT v1 | ACT fixed units | Zero | Train median |
|---|---:|---:|---:|---:|
| 0 | 11.99 / 1.615 | 6.44 / 0.918 | 7.39 / 0.638 | 6.38 / 0.660 |
| 1 | 12.07 / 1.589 | 6.59 / 0.897 | 7.39 / 0.637 | 6.38 / 0.660 |
| 2 | 11.94 / 1.577 | 6.53 / 0.884 | 7.39 / 0.637 | 6.38 / 0.660 |
| 3 | 11.82 / 1.574 | 6.41 / 0.885 | 7.40 / 0.638 | 6.38 / 0.660 |
| 4 | 11.84 / 1.587 | 6.35 / 0.900 | 7.40 / 0.638 | 6.39 / 0.661 |
| 5 | 11.91 / 1.596 | 6.34 / 0.917 | 7.40 / 0.639 | 6.39 / 0.661 |
| 6 | 11.96 / 1.602 | 6.35 / 0.925 | 7.41 / 0.639 | 6.39 / 0.661 |
| 7 | 11.97 / 1.599 | 6.43 / 0.928 | 7.40 / 0.638 | 6.39 / 0.661 |
| 8 | 11.98 / 1.596 | 6.46 / 0.928 | 7.41 / 0.639 | 6.39 / 0.661 |
| 9 | 11.99 / 1.594 | 6.42 / 0.933 | 7.41 / 0.639 | 6.39 / 0.661 |
| 10 | 12.09 / 1.595 | 6.45 / 0.939 | 7.41 / 0.639 | 6.40 / 0.662 |
| 11 | 12.19 / 1.593 | 6.52 / 0.945 | 7.42 / 0.640 | 6.40 / 0.662 |

Both mean errors improve against ACT v1 at every chunk position, including all six
consumed positions. Full-horizon means become 11.978→6.441 mm and 1.593→0.917 deg;
discarded-suffix means become 12.030→6.438 mm and 1.597→0.933 deg. The benefit is
therefore present on the consumed prefix, not confined to discarded predictions.

Train fit becomes slightly worse: prefix translation changes 1.859→2.106 mm on
the first train identity and 3.688→3.976 mm on the second; rotation changes
0.256→0.341 and 0.590→0.659 deg. Final train normalized L1 changes 0.2641→0.3079.
These are the observed effects of one recipe under one fixed short budget; they
do not identify which individual visual/proprioceptive feature caused the changes.

## Decision and remaining limits

Implemented and numerically demonstrated: one fixed-unit input transform, zero
clipping on the saved episodes, and consistent preparation/training/inference.
Demonstrated on this development identity: lower mean ACT A2 prediction error,
especially push/hold, with worse release and p95 errors and no aggregate win over
the train median. Retain v2 as an explicit opt-in pilot recipe; preserve v1/default
compatibility and all original checkpoints. No further training variant or broader
campaign is launched from this result.

There are only two train identities, one development identity and one seed. Broad
generalization, sealed-test performance, contact execution and physical validity
under the newer controller remain **not demonstrated**. The newer controller is
unchanged, and the comparison uses only the same historical controller's episodes.
Before comparative rollouts, verify demonstrations with the chosen controller and
collect new ones if necessary; also retain the outstanding load/contact/geometry
admission requirements. Offline improvement does not transfer controller validity.

## Residual follow-up

[[experiments/b1-pilot-approach-release|The targeted approach/release diagnosis]]
checks saved alignment, actual minibatch exposure, transition/input ambiguity and
historical tracking compensation before one fixed-budget sampling experiment.
The preprocessing pair above and its v2 opt-in decision remain unchanged.
