# Project Status

Reviewed 2026-10-29 on `PatrizioAcquadro/b1-observed-a3a4`. This review changes
software organization and documentation; scientific results retain the dates and
limits in their experiment records. The sealed test remains closed.

## Current capabilities

| Area | Delivered capability | Remaining boundary |
|---|---|---|
| Runtime | Fixed-base Purdue Alex003, WSG32/UMI v1, seven right-arm joints and synchronized head ZED RGB-D/proprioception | Simulation commissioning; no physical-hardware control or validation |
| Corpus | 32 qualified doors in 12 reviewed families; frozen 19 train / 6 development / 7 test split | 29 redistributable, two local-only, one private/noncommercial |
| Perception | Shared persistent SAM3, observed static fusion, explicit reference selection and externally masked CAD-free Point2Pose | Whole-object/material identity, physical hinge, 150 ms freshness and loaded-contact qualification remain open |
| Learning software | Observed-input B1 data/checkpoints, train-only normalization, ACT/Diffusion and A1–A4 adapters | Phase 6.1 integration/physical validation and final matched Phase 6.2 dataset remain incomplete; no qualified learned-policy result |
| Cluster runtime | Official kit-less NVIDIA image and separate project environments verified on A100/H100 | Bounded deployment/trainer checks do not qualify policies or migrate the custom simulator |

The maintained workflow/consumer map is in [[topics/system-architecture]].
[[topics/perception-model-setup]] owns commands; each recorded diagnostic requires
one explicit `--episode`, live requires `--asset` and `--case`, and each invocation
starts one attempt in fresh output. No diagnostic completion implies qualification.

## Scientific limits and decisions

The selected recipe remains `configs/point2pose_selected.json`: bounded SAM3 with
trace reconditioning ON/detection OFF; P2P graph OFF, partial batches ON, rollback
OFF, all-query TAPIR and renewal-only depth filtering. It is maintained but
unqualified. [[experiments/p2p-persistent-session-integration|Persistent full openings]]
retain mixed accuracy changes and serial p95 260/246 ms with zero timely requests.
[[experiments/p2p-operational-selection-and-light-peaks|Operational publication]]
retains age p95 610/346 ms and explicit skips/losses. These clocks and denominators
are distinct.

Temporal renewal admission, TSDF readback/snapshot/double-buffer candidates,
SAM3 compilation and RANSAC point buffers did not justify a maintained replacement.
Their negative outcomes, sources and evidence remain in the indexed experiment
pages. Local speedups do not establish complete-provider benefit.

Qualification still requires per-door 95% coverage/accepted precision and
1 cm/5 degrees, counting missing estimates as failures. Tracker acceptance,
endpoint IK and evaluator truth cannot substitute for original-material support,
calibration/FK/floor bounds, physical hinge evidence, continuous sweeps, measured
load and independent stopping evidence. See [[topics/shared-door-perception]] and
[[topics/pre-a4-initialization-and-hinge-budget]]. Ideal depth, limited
family/handedness coverage and no hardware validation remain material limits.

## Open work

1. Establish reliable observed geometry/material support and useful publication
   timing before claiming a qualified provider; use the existing cost and failure
   diagnoses to define any separately authorized experiment.
2. Complete the shared pre-A4 initialization/admission requirements and validate
   observed A1–A4 execution before final matched data and Phase 7 evaluation.
3. Keep the separate `PatrizioAcquadro/b1-policy-pilot` paused. Its failed A1/A2
   executions are independent evidence and do not qualify this branch or A3/A4.

Historical B0 evaluation selected no winner; the 219.95 N event remains under
review. Maintenance preserves recordings, models, prepared assets, masks, traces,
failures and user-owned raw research; it does not authorize campaign resumption.
