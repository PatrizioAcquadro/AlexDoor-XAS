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

The full panels on `PatrizioAcquadro/b1-policy-pilot` remain paused. The user has
subsequently authorized only the bounded ACT A1 operational-check investigation
below; no automatic panel restart or training is active.
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

The scoped Goal 1 operational-tolerance follow-up is complete under the user's
2026-10-18 criterion: justified/corrected uncertainty and explicit decisions on
known small ACT A1 excursions, without requiring learned task success. A separate
runtime-speed goal is now active below. The original full learned-success panels remain paused and
incomplete; checkpoints, outcomes and physical requirements remain unchanged.

One replay of 524 saved ACT A1 captures exactly reconstructs the 83.618 mm
frame-origin bound. Admission now transports primitive errors directly to the
actual finite footprint, credits the geometry tolerance already spent by material
coverage, and separates material interval motion from robot tracking response.
For the preserved terminal proposal's clearance, even a conservative 20 mm
endpoint-travel envelope leaves at least 47.755 mm material reserve. No next
learned action or complete task is executed after that correction.

A single upper-shoulder-Z robot probe validates the existing 4.25 mrad weak-load
rule on the missing side. All ten known terminal position-excursion samples
obtain estimates, including the latest30k left 20.683 microradian excess. The
strong-load/combined-domain restrictions and independent future motion/stop
guard remain: the saved left proposal path can still refuse without contact-free
stop evidence. All 4,899 retained strong-load out-of-position samples refuse.
The earlier reference cycle, policy failures and interruptions retain their
original classifications. See
[[experiments/b1-operational-tolerances|evidence, new recipe and qualification limits]].

The investigation retains nine GPU launches in total, including the previous
failed/interrupted attempts. The two closure checks add 304 allocation-wall
seconds (log creation to last write plus five-second tails), bringing the
investigation increment to 0.996111 local GPU hours above the
preserved pause ledger. No remote allocation or training update is added.
Detailed counters and analysis failures remain in the separate closure report.

## ACT A1 runtime speed — in progress

The fixed 30,000-update ACT A1 diagnostic completes all 1,631 commands, inspection,
recording and healthy stop in 466.902 seconds. This improves the historical
roughly 10.7–11.1-minute runtime, but misses the required 360-second full-process
target. The full 7,217-command reference completes in 1,476.799 seconds versus
about 32.2 minutes previously, missing the 1,200-second target. Opening, hold,
release, physical limits and stop pass; all compared numeric reference columns
remain exact. This is reference execution, with zero learned commands/task success.

Changes reuse bounded HDF5 chunk caches, avoid repeated within-capture depth
selection and use lossless LZF for local rollout images. Complete saved-input
rewrites retain all 47 columns/2.42 GB exactly; fixed-input GPU ACT predictions
remain exact. A live float32 JSON metadata defect is corrected without changing
any geometry or safety decision. The ownership refusal, external-CUDA
interruption and 148-command serialization failure remain separate outcomes.
Further validated source changes reuse joint frames within one clearance query,
build static indexes only for final condition banks, avoid unused perception
preloads during single-run shutdown, and mask captured robot depth on CUDA.
The latter preserves all 201 saved diagnostic/reference masks exactly; complete
updated timing remains pending. Thread/CPU-affinity alternatives are unadopted;
no training, panel restart,
physics/sensing-rate change or safety recalibration is performed. See
[[experiments/b1-runtime-speed|measured costs, evidence and remaining targets]].
