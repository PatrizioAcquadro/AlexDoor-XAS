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

## Results and limits

Paired CUDA training and raw-frame verification are pending an exclusive window.
There is one development identity and one training seed. No test split is used.
Even a positive offline result cannot transfer physical validity of demonstrations
from the historical controller to the newer controller. Before comparative rollouts,
verify demonstrations with the chosen controller and collect new ones if necessary;
retain the outstanding load/contact/geometry admission requirements.
