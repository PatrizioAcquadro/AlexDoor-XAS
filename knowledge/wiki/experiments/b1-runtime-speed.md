# ACT A1 Runtime Speed

Both measured runtime targets pass on `PatrizioAcquadro/b1-policy-pilot`, with
product code at `c7c62f8`. The goal fixes ACT A1 at 30,000 updates and uses
`operational-tolerances-20261018/goal1-closure/runtime-recipe.json`. No training,
architecture/action-space expansion or full scientific panel was resumed.
The reviewed baseline is `2cbd144`; all subsequent work and outcomes are preserved.

## Complete measurements

`outputs/evidence/act-a1-runtime-speed-20261018/benchmark-plan.json` was frozen
before optimization. Both clocks run from process launch through final recording
flush, the original continuing-physics stop and shutdown. The unchanged
180-second exclusive-GPU preflight is outside the clock. Targeted nonphysical
component probes use five seconds, with the same locks and resource guards.

| Workload | Earlier complete process clock | Final process clock | Limit | Result |
|---|---:|---:|---:|---|
| 1,512 inspection ticks +1,631 ACT A1 commands | 466.902 s | 344.736 s | 360 s | Pass |
| Inspection +7,217 reference commands, opening/hold/release | 1,476.799 s | 1,065.262 s | 1,200 s | Pass; reference only |

Earlier historical runtimes were about 10.7–11.1 and 32.2 minutes, respectively;
their clocks omit some startup/shutdown and are approximate comparisons. The
38-minute figure for 9,000 commands remains a projection, not a completed run.

`optimized-diagnostic-v4` completes every required command with 17.2120 degrees
controlled opening, 6.8648 N peak native force, zero contact losses/physical
violations and a healthy 240-sample stop with 0.2410 mm maximum tool travel.
This bounded prefix has no hold/release or learned task-success claim.

`optimized-reference-v2` completes all 7,217 reference commands, 51.0242 degrees
controlled opening, valid hold/release, zero losses/violations and a healthy
240-sample stop with 0.02134 mm maximum travel. Learned commands and learned
task success remain zero/false. All 44 compared non-image columns, applied
commands, capture clocks/calibration match the previous completed reference exactly; all 7,217 admit/refuse
outcomes are also preserved. Physical simulation was not required to
be bitwise identical; this equality is additional measured evidence.

Both full recording/sensor audits pass: 524/1,455 complete 600×960 RGB-D captures,
3,144/8,730 control states, 3,143/8,729 total commands including inspection and
6,286/17,458 rows in each native feedback/evaluator stream. Physics remains
120 Hz, control/render 60 Hz and consumed RGB-D 10 Hz. Causal joins, original
physical criteria and the checkpoint remain unchanged. No new seed, favorable
retry, shortened workload or early timing boundary is used.

## Implemented changes and verification

| Change | Preserved contract and measured check |
|---|---|
| Recording | Retained dataset handles and at most 128 copied numeric rows per stream; existing 60-tick flush, finish/close and fail-incomplete behavior remain. All 47 columns and 2,419,788,862 bytes in the saved diagnostic rewrite are exact, including NaN payloads, joins and commands. Input/return mutation, partial close and storage failure regressions pass. |
| Image layout | Local development rollouts use uncompressed whole-frame image chunks and retain LZF for numeric columns. The complete rewrite takes 41.80 s versus 48.25 s with buffered LZF and 65.38 s with the older gzip path. File size is about 2.417 GB versus 0.798 GB LZF: an explicit storage-for-speed tradeoff. Old compressed readers and final dataset writers remain supported. |
| Robot masking | Captured CUDA depth uses the same complete robot volumes in float64, four shapes at a time; current joints, calibration, validity and optional pixel domains remain inputs. All 201 saved masks and all 253 masks from the retained refusal match CPU/CUDA exactly. GPU boundary, missing-depth, domain and changed-state checks pass; no image is cached across captures. |
| Static workspaces | All rays are validated before indexes are built only for final banks. At most four independent builders join before the complete map is published; failures publish no partial map. All 102 background/reference views, 49,339,458 points, tree indices, pixel mappings, source clocks and 36 queries are exact; view arrays stay immutable. Construction takes 3.05 versus 8.65 s. No persistent cache is introduced. |
| Robot geometry | Parsed URDF origins stay within the existing immutable model snapshot. Joint frames, shape bounds and pair gaps are reused only inside the current clearance query; every pair, SAT check, travel reserve and early refusal remains. All 201 saved margins/pairs, full robot matrices and continuous sweep balls are exact; small negative, empty-filter and perturbed cases pass. |
| Admission | Surface bases and candidate inverses are reused only within one fixed admission query. Finite-radius arithmetic and every uncertainty source remain. All 6,440 recorded-state budgets are exact, including unavailable registration/calibration/speed cases; the next state/time creates a fresh query. |
| FK | Pose-only requests skip the discarded Jacobian; default Jacobian calls remain unchanged. All 603 nominal/perturbed CUDA poses and default Jacobians are exact. A bounded four-posture cache owns exact arm-joint keys, validates all nine current inputs, applies fresh root calibration and returns independent results. The 2,412-request CUDA pattern is exact; invalid-input, changed-root/head, mutation and eviction checks pass. |
| Shutdown | The single-run CLI clears perception without preloading an unused next episode. Reset/worker cleanup remains complete; later acquisitions create fresh state. Reset falls from about ten seconds to 0.228 s. |

Three fixed saved-input GPU checks reproduce ACT features/predictions exactly.
The combined 266 recording, geometry, causal, contact and documentation
regressions pass. In the final diagnostic, 26,133 FK requests require 5,166
forward calculations; the reference has 105,268 requests and 22,855 calculations.

The passive `phase-comparison.json` separates startup, inspection, perception,
policy inference, admission, safety, physics/render/native telemetry,
recording/independent evaluation and shutdown. Method clocks are inclusive and
nested, not additive. The final reference records 418.65 s in observer updates,
206.07 s in admission, 281.98 s in physics/render/native telemetry and 29.36 s in
the complete recording/evaluator callback. ACT inference in the diagnostic is
still a small cost; the reference executes no learned predictions.

## Explicitly authorized local-support correction

The retained `optimized-diagnostic-v2` refusal was reproduced on its saved
253-capture prefix; the previously successful prefix was reproduced separately.
Globally ambiguous scan associations retained real measurements that the local
material-witness query omitted. Searching all 4,000 current anchor points did
not produce three assigned-view witnesses. The original anchor does have three
independent finite two-finger supports at 18, 21 and 25 seconds when retained
unassigned observations are included, with the same semantic identity.

Automatic review rejected expanding these admission inputs under the original
speed-only scope before any mutation. The user then explicitly authorized the
local correction and its verification. Commit `1d7b1ee` includes those retained
measurements only in local witness queries and assigns their actual publication
clocks without refreshing old acquisitions. Global associations and the separate
coverage atlas remain unchanged. Three calibrated views, semantic identity,
finite footprints, root consistency and future-support rejection remain required.

Both recorded CUDA prefixes retain their actual native masks, anchors, geometry
and global associations. The old successful terminal candidate is unchanged;
the refused prefix gains the three real witnesses above. This can change
admission and is distinguished from behavior-preserving performance work. The
original refusal is preserved, not relabeled as success.

## Retained outcomes and limits

| Retained case | Process time | Actual learned/reference commands | Outcome |
|---|---:|---:|---|
| `baseline-diagnostic` | 268.815 s | 0 learned | Ownership refusal after inspection |
| `optimized-diagnostic` | Invalid parent boundary | 0 learned | External RustDesk CUDA interruption during inspection; partial recording and healthy stop |
| `optimized-diagnostic-infra-resume1` | 229.079 s | 148 learned | Strict JSON rejected nested NumPy float32 metadata; corrected native Python scalars preserve values and nonfinite rejection |
| `validated-diagnostic` | 466.902 s | 1,631 learned | Complete workload; target missed |
| `validated-reference` | 1,476.799 s | 7,217 reference | Complete accepted reference; target missed |
| `optimized-diagnostic-v2` | 157.024 s | 0 learned | Retained ownership refusal; diagnosed before another physical attempt |
| `optimized-diagnostic-v3` | 390.457 s | 1,631 learned | Complete repaired workload; target missed |
| `optimized-diagnostic-v4` | 344.736 s | 1,631 learned | Diagnostic runtime target and correctness pass |
| `optimized-reference-v2` | 1,065.262 s | 7,217 reference | Reference runtime target, original hold/release and correctness pass |

The original interrupted parent clock of 142.924 seconds is invalid because its
child was still cleaning up. Its separate correction charges 161.99262 seconds
including the tail; the raw negative snapshot stays preserved. The helper now
signals only the owned runtime and waits for actual cleanup. The one identical
infrastructure resumption remains declared separately.

Unapplied alternatives remain evidence: 16 runtime threads/CPU affinity, ROI
masking, CUDA-graph FK, native batched RANSAC and SAM3 pixel lookup. The latter
two preserve 705 native results/inliers/remaining masks/RNG states and 523 input
tensors, respectively, but are not part of either final measured execution.
Helper failures/interruption are retained separately: wrong RANSAC wrapper
keyword, a 20-to-seven-joint fixture mapping error, and an oversized synthetic
SAT uncertainty probe stopped at about 13 GiB RSS. There was no kernel OOM or
physical workload in that CPU probe; the operational reserve was never changed.

Both scoped speed targets are met. This does not qualify Point2Pose generally,
complete a learned opening/hold/release task or resume the paused full panels.
No training, remote allocation, dataset expansion, threshold recalibration or
push was performed. Use this branch for subsequent explicitly authorized
bounded diagnostics; scientific qualification remains separate.

Evidence and allocation closeout are in
`outputs/evidence/act-a1-runtime-speed-20261018/completion-report.json`, with
per-case audits, `reference-full-preservation.json`, profiles and all earlier
outcomes. See [[b1-operational-tolerances|physical limits]],
[[../topics/episode-and-dataset-contracts|recording contracts]] and
[[b1-end-to-end-pilot-resume|the separate paused scientific pilot]].
