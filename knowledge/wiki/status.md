# Project Status

Current as of 2026-10-18. Static 6.0B and the complete B1 software core are
maintained. CAD-free Point2Pose remains implemented but unqualified; Phases 6.0,
6.1 and final matched-data production remain open. The sealed test stays closed.

| Area | Maintained capability and limit |
|---|---|
| Purdue runtime | Fixed-base Alex003, WSG32/UMI v1, seven right-arm joints, head ZED RGB-D and synchronized proprioception. No repository command controls physical hardware. |
| Corpus | 32 qualified doors, 12 reviewed families and frozen 19 train / 6 development / 7 test split. Rights: 29 redistributable, two local-only, one private/noncommercial. |
| Perception | One persistent SAM3 inspection/tracking frontend; observed static fusion, explicit reference selection and externally masked Point2Pose. Identity, hinge and 150 ms freshness remain unqualified. |
| Policy software | ACT/Diffusion, A1–A4 math/adapters, observed-input datasets, train-only normalization and checkpoint contracts. Software availability does not qualify physical execution. |

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

## A1/A2 pilot — paused and incomplete

The separate pilot on `PatrizioAcquadro/b1-policy-pilot` remains paused at the
user's request. No automatic restart, simulation, training or experiment is active.
Four ACT/Diffusion × A1/A2 latest checkpoints retain 30,000 updates and five eligible
train records; normalization remains fitted only on the original two train records.

The original `selected-v10` panel is **24/24 complete, all failed**, across 26
physical attempts including interruptions. Best controlled opening reaches 99.54%
on ACT A1 left/dim with valid hold, followed by a verified gripper-domain refusal
before release. The frozen latest-30k `qualified-latest-v11` panel has **2/24
completed outcomes, both failed; one interrupted attempt and 21 unallocated slots**.
Its ACT A2 interruption retains 144 learned control rows, exit130 and a healthy
two-second continuing-physics stop. No checkpoint completes the task on both doors.

Cumulative authorization remains 40 local GPU hours, 32 remote GPU hours, 200,000
updates, 1 TiB evidence and 24 demonstration attempts. The pause ledger retains
13.827548 local GPU hours, 3,503 remote GPU seconds, at most 120,364 updates,
392.827 GiB evidence and 11 demonstration attempts. Failed/interrupted work stays
charged; no allocations or optimizer updates are reserved.

Maintained commands require explicit protocols; the encoder must agree with data
and checkpoints. New production uses `rgbd-conv-v1`; historical grid inputs remain
readable. The initial numerical protocol and completed preprocessing comparison
launchers are retired at `8bbeefe`. See [[topics/b1-policy-pilot|current interfaces]]
and [[experiments/b1-end-to-end-pilot-resume|complete outcomes, budgets and pause]].


## Bounded ACT A1 operational-check investigation

New user authorization on 2026-10-18 permits the scoped position-calibration and
right-door contact-margin diagnosis, focused tests and bounded reference GPU
execution. The full panels and other policy combinations remain deferred; no
training resumes. The fixed ACT A1 checkpoint and every previous outcome remain.
Directional material-margin accounting and numeric refusal diagnostics are
implemented. Saved right-reference replay has no first operational refusal;
strong-load position support is rejected by measured error/stop failures. The separately tested 1 N position support passes unchanged error/response/stop
bounds; strong-load extensions remain blocked. Fresh complete reference execution
remains under assessment. See
[[experiments/b1-operational-tolerances|current bounded evidence and limits]].
