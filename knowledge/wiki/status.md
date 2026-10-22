# Project Status

Scientific state as of 2026-10-18; cluster runtime verification added on
2026-10-22. Static 6.0B and the complete B1 software core are
maintained. CAD-free Point2Pose remains implemented but unqualified; Phases 6.0,
6.1 and final matched-data production remain open. The sealed test stays closed.

| Area | Maintained capability and limit |
|---|---|
| Purdue runtime | Fixed-base Alex003, WSG32/UMI v1, seven right-arm joints, head ZED RGB-D and synchronized proprioception. No repository command controls physical hardware. |
| Corpus | 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test split. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| Perception | One persistent SAM3 inspection/tracking frontend; observed static fusion, explicit reference selection and externally masked Point2Pose. Identity, hinge and 150 ms freshness remain unqualified. |
| Policy software | ACT/Diffusion, A1–A4 math/adapters, observed-input datasets, train-only normalization and checkpoint contracts. Software availability does not qualify physical execution. |
| Cluster training | Official NVIDIA kit-less runtime and separate application environments verified on Gilbreth A100 and Gautschi H100. Bounded runtime/trainer checks do not qualify policies or migrate the custom simulator. See [[topics/cluster-training|deployment and limits]]. |

## Selected perception and decisions

The common recipe remains SAM3 with bounded history, trace reconditioning ON,
detection reconditioning OFF; Point2Pose graph OFF, partial reference batches ON,
rollback OFF, all-query TAPIR and renewal-only depth filtering. Cleanup changes no
model algorithm, native patch, selected configuration or scientific threshold.

The complete persistent recordings improve some accuracy measures and regress
others; serial p95 is 260/246 ms, with zero requests within 150 ms. Recorded
operational publication has age p95 610/346 ms and retains skips and native losses.
These clocks and denominators are distinct. See
[[experiments/p2p-persistent-session-integration|full serial evidence]],
[[experiments/p2p-operational-selection-and-light-peaks|operational quality and failures]]
and [[experiments/p2p-sam3-cost-diagnosis|cost diagnosis]].

The temporal body/static renewal criterion was rejected before runtime activation:
its waiting budget cannot distinguish the hypotheses and useful support collapses.
Its implementation is retired; source and historical analysis dependencies remain
at `24a2aa7`. The complete result is in
[[experiments/p2p-temporal-renewal-admission|the rejected temporal experiment]].

## Maintained workflows and evidence

Perception offline/replay requires one explicit `--episode`; live requires an
explicit pilot `--asset` and `--case`. Each invocation starts one attempt. Historical
live matrices, twelve-attempt offline orchestration and the concurrent smoke alias
are retired at `24a2aa7`. Process completion remains separate from diagnostic
quality and qualification. See [[topics/perception-model-setup|supported commands]].

Recordings, models, prepared assets, corpus records/splits, failed and interrupted
attempts, masks, timestamps, traces and ignored analysis scripts are preserved.
Historical scripts importing retired experimental modules use their recorded Git
revision. Generic saved-evidence readers remain maintained. `knowledge/raw` is
user-owned and untouched. No payload was deleted by this cleanup.

Current technical contracts belong in [[topics/system-architecture|architecture]],
[[topics/shared-door-perception|perception]],
[[topics/episode-and-dataset-contracts|data]] and
[[topics/learned-policy-stack|policies]]. Experiment pages retain methods, negative
results, attribution and limits; Git retains ordinary implementation history.

## Qualification and next work

Unchanged qualification requires per-door 95% coverage/accepted precision and
1 cm / 5 degrees. Missing estimates remain failures. Material ownership, physical
hinge evidence, calibration/FK uncertainty, continuous sweeps, measured contact/load
and independent stopping evidence cannot be substituted by tracker acceptance,
endpoint IK or evaluator truth. The shared pre-A4 requirement applies to all
representations; see [[topics/pre-a4-initialization-and-hinge-budget]].

The main scientific limits remain ideal depth, incomplete hardware calibration,
no physical robot validation and narrow family/handedness coverage. The historical
B0 576-rollout study selected no winner; its 219.95 N event remains under review.
Maintenance does not authorize new training, simulation, collection or experiments.

## Separate pilot and retained performance experiments

The A1/A2 learned pilot lives on `PatrizioAcquadro/b1-policy-pilot`; it remains
paused and incomplete. Its failed learned executions are separate from this
base checkout and do not qualify A3/A4 or the final matched benchmark.

TSDF read-only inputs, deferred snapshots and double buffering, official SAM3
compilation and RANSAC point buffers were evaluated without adopting a runtime
replacement. Their original sources/failures and the baseline remain preserved:
[[experiments/tsdf-readonly-launch-inputs]],
[[experiments/tsdf-deferred-snapshot-audit]],
[[experiments/tsdf-rebuild-double-buffer]],
[[experiments/sam3-official-compilation]] and
[[experiments/p2p-ransac-point-buffers]]. Local savings do not establish operational
freshness or justify repeating failed scientific cells during maintenance.
