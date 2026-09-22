# Project Status

Current as of 2026-09-22. Subphase 4.0 is implemented and GPU-verified.
Subphase 4.1 is in progress: all four articulated synthetic doors pass GPU
reset/drift/opening checks. No common setup or expert maximum is qualified yet.
Code and tests define executable behavior.

## Current Runtime

The sole registered robot environment is `AlexDoor-DoorPush-Purdue-v0`: fixed-base
Purdue Alex003, measured pedestal, WSG32/UMI v1, seven right-arm joints and head
ZED X Mini Wide. It currently uses small synthetic panel/frame/handle fixtures
for operational commissioning, not the four-door setup search or a training task.

- A1 addresses all seven joints; A2 actuates tool translation and rotation;
  A3 transforms an explicitly supplied frame into A2.
- Gravity compensation uses the model, with external PD gains and physical limits.
- Raw contact diagnostics resolve imported rigid owners and distal-finger points.
  Exact panel partners are authorized; other contact surfaces/partners are rejected.
  Force reports include normal components only.
- RGB-D at 960 × 600 is captured with arm/neck proprioception, time and frame IDs.
  Valid depth contains no segmentation or simulator object labels. The Gym policy
  tensor remains proprioception-only until Phase 6; capture is reusable separately.

See [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] and
[[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]] for
interfaces, numerical defaults, verification and approximation limits.

## Migration Boundary

B0 robot execution, calibrated scripted preset and simulator asset loader are
retired. Existing datasets, checkpoint loading, model training and historical
results remain available. Calibration/manifests needed to interpret the historical
offline contracts are retained; they are not Purdue configuration.

Full generation and learned-policy evaluation commands now fail clearly before
simulator startup or output creation. B0 checkpoints are never reinterpreted as
Purdue-compatible models. Generic data/adapter components remain tested for their
software contracts, without claiming B1 learned execution.

## Verification

The dependency/CUDA preflight passes on the RTX 4090. All 332 software tests pass,
including historical data/model contracts, as do Ruff and wiki-link/index checks.
GPU checks pass three stable resets, seven-joint/full-pose control, loaded distal
contacts and forbidden-contact detection, and synchronized metric head RGB-D.
The local pose targets remain within 0.0204 mm / 0.00209 degrees during their
0.5 s holds; this is synthetic commissioning evidence, not hardware accuracy.

The complete run is in `~/.cache/alexdoor-xas/verification/purdue-final/`.
Targeted final checks in sibling `purdue-final-rgbd/` and `purdue-final-contacts/`
verify reset renderer settling and force direction. Numerical traces, raw contact
records and representative images are retained. The separate D0 door-only GPU
smoke also passes; it establishes no B1 corpus qualification.

The launcher still warns that `setup_conda_env.sh` is absent. Its actual Python,
Isaac provenance, Alex assets, ZED dependency and CUDA preflight pass. No runtime
installation or driver modification was made.

## Maintained Entry Points

- `scripts/check_env.py` — supported runtime and Purdue/pedestal/ZED preflight.
- `scripts/verify_purdue_runtime.py --viz none --device cuda:0` — complete operational gate.
- `scripts/verify_benchmark_scene.py` and `scripts/verify_adapters.py` — route to that same gate.
- `scripts/verify_dataset_interface.py` — retained historical dataset interface checks.
- `scripts/train_policy.py` — offline training on existing supported data.
- `scripts/prepare_phase4_1_assets.py`, `scripts/normalize_phase4_1_door.py`,
  `scripts/verify_phase4_1_doors.py` — retained legacy preparation, with the limits below.

`eval_policy.py`, `run_scripted_baseline.py`, `verify_policy_rollout.py` and
`verify_scripted_baseline.py` are explicit migration stops, not available B1 workflows.

## Legacy Preparation and Storage

Preparation still assumes the B0 anchor, simple panel/jamb/header collision boxes,
non-colliding handles, 0–90-degree stops and the previous mass/damping template.
Its physics inspection now instantiates only the door: it measures reset, drift
and applied-torque response, not collision-consistent robot qualification.
Phase 5.0 must validate and replace these limitations before B1 corpus admission.
The ignored worklist and all local assets were preserved.

Historical D0–D4 layers, datasets and policy outputs are unchanged. Verification
reports and images live in the project cache; no generated dataset, corpus,
training run or media was added to the tracked output tree.

## Next Phases

4.1 selects a common floor pose, contact point and fixed neck view on the four
synthetic doors. It freezes controller/force/tolerance choices before collected
assets. Phase 5 then validates preparation infrastructure and admits a real corpus.
Phase 6 integrates observations and all learned A1–A4 paths, then a demonstration
pilot. Phase 7 owns training and generalization evaluation.

There is no active gaze, wrist-camera requirement, corpus download, physical
robot control, hardware-safety claim or sim-to-real qualification in 4.0.

## Historical Results

The completed scale study used 550 matched A2/A3 episodes across D0-D4 with nested N50, N100, N250, and N500 training views. Sixteen ACT/Diffusion x A2/A3 x data-size cells were evaluated over 576 rollouts. Every rollout succeeded, so the benchmark did not select a policy family, representation, or dataset size.

One ACT-A3-N50 rollout at seed 112 produced a reproducible 219.95 N one-tick peak. Two +/-1 mm door-position perturbations reduced the peak, but the original cell remains `REVIEW_REQUIRED`.

These are historical scientific conclusions, not active workflows. See [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] and [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]. Git retains removed implementation and evidence files.

## Version Notes

- 2026-09-22 — Implemented Purdue operational integration and retired B0 execution, preserving historical readers and isolated door preparation.
- 2026-09-22 — Completed preparatory cleanup and revised the B1 nine-subphase plan.
- 2026-08-13 — Recorded the historical B0 system and approved held-out-door study.
