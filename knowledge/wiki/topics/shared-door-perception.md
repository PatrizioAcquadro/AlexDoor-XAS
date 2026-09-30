# Shared Door Perception

Phase 6.0 remains **unqualified**. The custom DINOv2 estimators and their training,
feature-cache and comparison commands are retired. The selected direction is
GroundingDINO + SAM 3, explicit calibrated RGB-D/multiview geometry and DINOv3
when learned visual features are useful. That pipeline is **not implemented**.
[[experiments/b1-perception-findings|Perception findings]] records the measured
train/development gap, corrected component screening and limitations.

## Recording and storage

`recording/b1.py` and `recording/b1_runtime.py` retain synchronized metric RGB-D,
valid-depth masks, arm/neck proprioception, intrinsics and optical camera poses.
`b1.rgbd.v1` stores N+1 observations including reset/terminal, N applied commands,
separate annotations, calibration and a factual outcome. Observation selection
excludes asset identity, split, expert angle, phase and simulator labels.

`scripts/collect_perception.py` uses the frozen train/development corpus and one
fresh Isaac process per episode, sequentially on CUDA. A worker failure stops
admission. `--resume` skips only validated complete episodes; existing output is
never silently overwritten. Qualification records are not changed. The collector
is retained for future authorized work; no collection was started by cleanup.

Raw data remain under `datasets/b1/perception/`: 50 episodes each in
`engineering-v1` and `engineering-v2`, two complete `inspection-pilot-01` episodes,
and the separate `inspection-tallest-01` diagnostic. The first campaign lacks the
later scan views; do not silently combine it with the refreshed recipe. Engineering
recordings are not the final matched policy dataset. The inspection-only diagnostic
has no expert hold/release and is not admitted as a manipulation episode.

`recording.b1.episode_paths` checks complete campaign membership, recorded corpus
identities, conditions, duplicates and train/development isolation without loading
visual features. The frozen family assignment remains authoritative. Five
interrupted original payloads were explicitly removed; their metadata/calibration
and failure evidence remain in `outputs/b1/perception/evidence/cleanup.json`.

Selected weights and auxiliary resources live in `models/perception/`; historical
results live in `outputs/b1/perception/evidence/`. Their READMEs and local cleanup
record describe relocations. Removed ignored payloads cannot be recovered from Git.

## Inspection and calibration

`configs/perception_inspection.json` retains the common 25-second bounded scan,
seven sample times and simulated 10-degree upward camera mount adjustment. The
arm holds its parked pose while the neck scans; contact, door motion, tool drift
and tracking checks can stop inspection. Physical neck limits are unchanged.
This is a simulated mounting proposal, not validated hardware calibration.

The same inspection recipe is recorded with observations and bound to future
policy releases. `load_inspection`/`validate_inspection` check its trajectory.
The runtime mount transform and recorded calibration stay aligned. RGB-D depth
is ideal rendered optical-axis depth, not a reproduced real ZED stereo-error model.

## Model-independent policy boundary

`perception.contracts.DoorEstimate` describes timestamp, validity/reason, confidence,
world hinge frame, panel rotation, signed articulation, dimensions and world contact
pose. Positions are meters; rotations are matrices; articulation is radians.
`B1Observer` combines a provider's static/recent visual encoding with current nine
joint positions and nine velocities. It shares the same path for raw preparation
and live execution, with causal frame/time checks, reset and freshness rejection.

A provider exposes `binding`, a finite NumPy `encoding`, `reset()` and `update(sensor)` returning
`DoorEstimate`. It receives only recorded sensor keys and owns device conversion,
inference and observation-dependent validity. Between inference ticks, only a
fresh cached estimate/encoding may be reused, with current proprioception. Loss
invalidates cached policy inputs. No production provider, fallback or registry is
supplied yet; numerical test providers do not qualify perception.

`b1.perception.release.v2` declares named SHA256 artifact identities,
`config.visual_dims` (static/recent widths), `warmup_s`, `max_gap_s`, inspection,
and mandatory true `offline_passed`, `dynamic_passed`, `frozen` flags. The binding
is immutable JSON. `verify_artifacts(paths)` verifies exact names and file bytes
before a future provider loads them. The provider must bind that verified recipe;
release declarations themselves are not evidence of qualification. Release v1 and
the model-specific loader are retired; no qualified v2 release has been produced.

Policy observation/action schemas and matched data retain their existing semantics.
No B1 policy dataset or trained policy artifact requires conversion. ACT/Diffusion,
A1–A4 adapters, normalization and stop-only execution remain maintained components,
but final raw/live equivalence and physical integration await the qualified provider.

## Fixed acceptance boundary

Per development door, during contact/push/hold: valid coverage at least 95%,
contact-position p95 at most 0.01 m and orientation p95 at most 5 degrees. Preserve
complete-state checks: hinge origin, dimensions and local/world contact position
within 1 cm; hinge, local/world contact and panel rotation plus wrapped signed angle
within 5 degrees. Confidence must accept accurate states and reject missing or
incorrect states; accepted-state precision must reach 95% per door. Report coverage
and errors separately, with worst per-door results for both handednesses.

Fit learned quantities/normalization on train only; use development for selection
without changing gates or the frozen identity split. Never use sealed-test
qualification evidence as training or model-selection input. Plane-only component
success is not complete geometry qualification. Offline success must be followed
by dynamic loss/reacquisition and observed-geometry execution under unchanged
control/safety rules before Phase 6.0 can close.

## Retained collection interface

For a separately authorized future collection, from the repository root:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/collect_perception.py \
  --output datasets/b1/perception/NEW_CAMPAIGN \
  --inspection configs/perception_inspection.json --device cuda:0
```

Choose a fresh output; use `--resume` only for that same campaign. Optional
`--asset-id`, `--condition` and `--smoke` bound the selection. `--inspection-only`
requires `--inspection` and produces a diagnostic without expert manipulation.
There is no maintained estimator training, preparation, evaluation or launch CLI.
