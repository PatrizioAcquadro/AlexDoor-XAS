# Project Status

Current as of 2026-10-19. Static 6.0B and the complete B1 software core are
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

## Data infrastructure

The saved-data pipeline now streams preparation/export, stores exact reusable
frames and joins, and loads large contiguous/compressed/VDS inputs selectively.
Across the declared complete two-episode workload, prepared size falls from
6.06 GiB to 106.5 MiB and peak process-tree RSS from 11.36 to 1.17 GiB. Batch
prefetch keeps the bounded ACT input consumer within baseline variability; the
standalone CPU loader is slower than materialized RAM. Values, train-only
normalization and prior outcomes remain exact. See
[[experiments/b1-data-pipeline|fixed workloads, results, failures and handoff]] and
[[topics/episode-and-dataset-contracts|compatibility and commands]]. Expansion,
the maintained temporal path is described in
[[topics/act-a1-observations-and-time|ACT A1 temporal observations]]. Training,
collection and physical validation remain separate.

Temporal ACT A1 consumption now shares one strict model/checkpoint contract across
normal and fresh intermediate startup metadata. Quality distinguishes complete
scope coverage, reviewed usable windows and explicitly qualified short subtasks;
source-row event sampling preserves real history and phase-crossing labels.
The existing right/left/C admissions and interrupted-prefix outcome remain.
[[topics/act-a1-four-door-expansion|Four-door expansion]] is a concrete train-only
proposal; new geometries/start criteria still require physical qualification.
No new collection, training, learned success or final-test access accompanies this
software/data closure. See [[topics/act-a1-observations-and-time|current interfaces]].

## A1/A2 pilot — paused and incomplete

The full panels on `PatrizioAcquadro/b1-policy-pilot` remain paused. The user has
subsequently authorized bounded ACT A1 operational-check and two-door diagnostics
below; both are closed. No automatic panel restart or training is active.
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

## ACT A1 runtime speed — measured targets met

On the same `c7c62f8` product source, the complete fixed 30,000-update ACT A1
1,631-command diagnostic takes **344.736 s**, within 360 s. The full 7,217-command
reference opening/hold/release takes **1,065.262 s**, within 1,200 s. Both include
inspection, final recording flush, continuing-physics stop and shutdown. Full
sensor/clock/checkpoint/recording audits pass, with no losses or physical
violations. All 44 compared non-image reference columns and all 7,217 admit/refuse
outcomes match the earlier accepted reference. The long cycle is reference
execution, with zero learned commands and no learned task-success claim.

Changes remove repeated geometry/FK/index work and recording overhead while
preserving 120 Hz physics, 60 Hz control/render, 10 Hz consumed RGB-D, sensor
fidelity and original physical criteria. Local raw images use more storage;
old compressed recordings remain readable. The user explicitly authorized a
separate correction allowing retained globally ambiguous observations to support
existing local material-witness tests. Global ambiguity, three-view/identity/
finite-footprint/root/causal requirements remain; original refusals and all
interruptions are preserved.

No native-model algorithm change, training, dataset expansion, full panel,
remote allocation or push was added. The reconciled cumulative local allocation
estimate is 16.368/40 GPU hours, including 1.545 hours for this speed goal;
all failed and interrupted allocations remain charged. The full learned panels above remain
paused and incomplete. Use this branch for future explicitly authorized bounded
diagnostics; broad qualification is separate. See
[[experiments/b1-runtime-speed|complete measurements, preserved outcomes and limits]].


## ACT A1 two-door diagnostic — complete, 0/2 task successes

The October 19 bounded run uses the unchanged audited 30k ACT A1 checkpoint/current
runtime, nominal seeds 7100/23, chunk 12/replan 6 and 9,000 commands available after
inspection on each existing training door. Right executes 6,238 commands in 979.767 s,
reaches 45.325 deg controlled opening and valid hold, then refuses a load-domain
wrist-position excursion before release. Left executes 368 commands in 205.022 s,
never contacts, and refuses a prospective weak-position motion/stop envelope.
Both have zero physical violations/contact losses and healthy two-second stops.
The right 10.014-microradian calibration excess and left 12.2-microradian future
envelope excess do not demonstrate physical danger. Left also retains 19 wrist
target clips; right retains material-tracking drift beyond the narrow training
corridor. Operational censoring prevents conclusions about later learned release
or temporal causation. No retry, recording fix, training, collection or panel restart.

Both full recordings and independent scoring audits pass. Existing expensive speed
benchmarks were not repeated;344.736 s remains a short learned prefix, while the
1,065.262 s complete cycle is reference-only. Local allocation is now 16.700309/40 h,
with 423.988/1024 GiB retained evidence and no owned GPU workload. A proposed combined
four-door data/causal-history task, then development-selected ACT A1 training and
full-task evaluation, is documented but unstarted; final-test data stay closed.
See [[experiments/b1-act-a1-two-door-diagnostic|outcomes, interpretation and next task]].

## ACT A1 calibration and stopping — simulation correction complete

The October 19 correction at `2bea7df` addresses both diagnostic stops with
side-specific empirical/load support and a separately selected loaded-stop
profile; normal ACT motion and physical/task criteria are unchanged. The explicit
local recipe is `outputs/evidence/act-a1-calibration-stop-20261019/runtime-recipe.json`.
Legacy recipes/default stopping retain their earlier behavior.

The original perturbed 10 N stress stops fail at 23.849/16.626 mm. The corrected
profile passes 12 new matrix cases, four frozen independent cases and two corrected
corpus-door contact probes; two invalid initialization attempts are preserved.
Maximum corrected matrix stop travel is 8.627/5.301 mm. Saved decisions support
both original terminal states and future intervals at required 10 N; weak or
out-of-domain cases still refuse. Total: 32 physical attempts, 277.741 s charged,
cumulative 16.777459/40 h; no owned workload remains.

Proceed to the separately scoped data/observation pipeline. This is finite
simulation support, not global all-speed/workspace, hardware or full ACT task
qualification. Original 0/2 diagnostic outcomes and paused panels remain intact;
no policy training, collection or architecture changes occurred.
See [[experiments/b1-act-a1-calibration-stopping|validation, failures and handoff]].

The [[topics/act-a1-observations-and-time|ACT A1 temporal data path]] now removes
the static-reference progress clock and has an explicit checkpoint/incremental
entry point through guarded A1 execution: history 4 s, chunk12/replan6. Saved-event
replay covers ordering, replanning, reset and per-row stale-input stopping while
retaining existing admission, controller and physical stops. Legacy flat policies
remain supported; revised temporal semantics require v2 checkpoints.

Representative distinct/overlapping/mixed batch construction falls from
207/158/173 to 104/65/76 ms at 120×192, and 860/721/770 to 308/185/223 ms at
240×384. Prefer 240×384, retaining 120×192 as the cheaper alternative: complete
prefetched batch32 work peaks at 7.39 GiB allocated VRAM and 4.24 GiB RSS, including
resident AdamW moments, with no optimizer updates. Reuse stays off by default:
120×192 fails the existing strict numerical criterion on the longer trace;
240×384 passes that fixture and saves about 4% steady inference time.

These saved-input and numerical execution results do not establish learned task
completion or hardware dispatch qualification. The separately authorized native
teacher collection tranche is now complete below; trained checkpoints, physical
thresholds and final test remain unchanged.

## ACT A1 native teacher collection — bounded tranche complete

The maintained collection entry point produces qualified complete nominal right
and left train demonstrations (51.75/52.55 deg controlled opening), a physically
reconstructed loaded hold/release continuation and a recoverable controlled
interruption. Every completed case includes real release and 2 s ordinary
stabilization, followed by a separate healthy safety stop. C uses fresh camera,
collision and controller state, actual reset-conditioning physics, a new local
generation and 1 s pre-roll; unavailable history stays masked. D retains 60 states
and 60 delivered commands, one missing terminal state and zero expert windows.

All five attempts remain: the first right physically passes but lacks a ready
inspect→approach window; one affected repeat follows an explicit .4 s real
pre-roll correction. The three admitted files have exact N+1 command/state
alignment, all seven measured simulated joint_output channels, 120 Hz native
feedback, 60 Hz state and original 10 Hz RGB-D. They retain 7,537/7,996/528 valid
training windows and 11 crossings per present transition. A1 labels, causal
live/offline histories and actual-data CUDA gradients pass, with unchanged model
weights and zero optimizer updates. Raw feedforward effort remains a separate
diagnostic with per-joint SI units. Historical 68 ambiguous weak-load samples
remain diagnostic; source/family grouping and final-test closure remain fixed.

Five launches consume 1,229.03 s, peak at 7.73 GiB process-tree RSS / 6.87 GiB total
GPU memory, and retain about 28.8 GiB. Final CUDA verification takes 17.75 s, plus a 5.51 s exact tensor-reach check, with
1.96 GiB RSS and 1.65/3.67 GiB allocated/reserved CUDA memory. All declared budgets
hold. No training, remote allocation or learned physical-task success is added.
The next data expansion starts from the three admitted records and repeats the
same physical/tail/torque/transition gates. See
[[topics/act-a1-observations-and-time|canonical collection, all attempts and handoff]].
