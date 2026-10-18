# ACT A1 Runtime Speed

The authorized 2026-10-18 runtime goal fixes ACT A1 at 30,000 updates and uses
`operational-tolerances-20261018/goal1-closure/runtime-recipe.json`. No training,
action-space/architecture expansion or full scientific panel is active. The
initial branch is `PatrizioAcquadro/b1-policy-pilot` at `2cbd144`, with a clean
tracked/non-ignored tree. Earlier outcomes and ignored evidence remain unchanged.

## Workload and measurement

`outputs/evidence/act-a1-runtime-speed-20261018/benchmark-plan.json` was frozen
before optimization. The diagnostic includes all 1,512 inspection ticks, exactly
1,631 learned commands and the original two-second physical stop. The reference
contains all 7,217 recorded commands through the same operational checks and
original opening/hold/release acceptance; it cannot count as learned success.

The external clock starts before launching Isaac and ends after child exit,
final recording close/flush, safe stop and shutdown. Exclusive-GPU preflight is
outside this clock. Historical runtime clocks omit startup and part of shutdown,
so the earlier figures below are approximate comparisons. The old 9,000-command
38-minute figure remains a projection, not a completed benchmark.

| Workload | Earlier measured runtime | New complete process time | Target | Status |
|---|---:|---:|---:|---|
| 1,631-command ACT A1 diagnostic | about 10.7–11.1 min | 466.902 s / 7.78 min | 360 s | Complete workload; target missed |
| 7,217-command reference | about 32.2 min | 1,476.799 s / 24.61 min | 1,200 s | Complete reference cycle; target missed |

The completed diagnostic uses `facad1d`: controlled opening 17.2334 degrees
versus 17.2595 degrees previously, zero contact losses/physical violations, and a
healthy 240-sample stop with 0.2379 mm maximum tool travel. All capture/control/
native counts, image dimensions, intrinsics and capture/control times match the
earlier prefix. Live commands differ slightly, consistent with different live
observations; no bitwise physical-simulation requirement or per-run tolerance is
introduced. This short horizon has no hold/release or learned task success.

The complete reference uses `8d72689` product code: all 7,217 reference commands,
51.0242 degrees controlled opening, valid hold/release, zero contact losses or
physical violations, and a passing continuing-physics stop. Learned commands and
learned task success remain zero/false.

## Measured changes and correctness

HDF5 dataset reopening discarded bounded chunk caches and recompressed partial
image tiles. Keeping handles alive reduces the fixed 120-row/20-capture write
sample from 6.47 to 1.03 seconds under profiling, including flush/close. All row
validation, separate clocks, 60-tick flushes and completion semantics remain.
The robot occlusion filter also selects depth once per capture instead of twice
per mesh: 54 fixed saved captures take 6.30 versus 3.17 seconds and produce
identical complete masks. No sensor data is cached across captures.

When captured depth is already on CUDA, robot masking now evaluates the same
complete link-local volumes in float64, in four-shape batches, then returns the
complete Boolean mask before publication. Current joints, calibration errors,
validity and optional pixel domains remain inputs on every call. The NumPy path
remains available for recorded CPU inputs. Across 201 diagnostic/reference
captures all masks are exact: 11.40 seconds on CPU versus 4.70 seconds on CUDA,
with 237 MB peak allocated CUDA memory in the component probe. Actual-GPU
boundary, missing-depth, domain and changed-posture regressions pass. This is a
component result; complete updated runtime measurements remain necessary.

Static workspace loading now validates all recorded rays before building indexes
only for the final condition banks. Previously it built and discarded the same
per-record indexes before rebuilding the merged banks. Construction takes 7.34
versus 14.20 seconds in the paired saved-input check. All 81 views, 40,318,928
indexed points, pixel coordinates, immutable arrays, source clocks and 18
free/unknown/occupied queries remain exact. There is no persistent cache.

Self-clearance also reuses joint origins/axes within one query. Every new query
recomputes them from the current matrices and reevaluates travel/model bounds.
On 201 recorded diagnostic/reference postures, all minimum margins and critical
pairs remain exact; the isolated queries take 1.34 versus 1.97 seconds.

Local rollout images now use existing lossless LZF/shuffle instead of gzip level
4. The isolated sample is 39% faster with 31% larger files. On the full recording,
reader-inclusive rewrite time is 57.16 versus 65.38 seconds; the output is 798 MB
versus the historical 577 MB. Both complete rewrites preserve every one of the
2,419,788,862 uncompressed bytes in all 47 columns: 524 RGB-D captures, 3,144
control states, 3,143 total commands including inspection, and 6,286 rows per
native stream. Missing-depth NaN payloads, causal joins and applied commands
remain exact. Old gzip files remain readable without a schema change.

Three fixed recorded-input GPU checks reproduce features and ACT predictions
exactly, including the first saved prediction. The checkpoint is unchanged.
Focused recording, geometry, temporal-support and contact regressions pass.
Numerical equality is used for identical recorded inputs, while live acceptance
uses the original physical criteria.

## Retained blockers and profiling decisions

The fresh pre-optimization baseline refuses `ambiguous_leaf_ownership` after all
1,512 inspection ticks and zero policy commands; its 268.815-second duration is
not a completed diagnostic. The first optimized attempt is interrupted during
inspection by an external RustDesk CUDA process. Its partial recording and
healthy stop remain. The initial monitoring helper exited before its runtime
child finished cleanup; its 142.924-second parent clock is explicitly invalidated
in the separate interruption audit. The helper now signals the uniquely owned
runtime handler and waits for actual shutdown. One identical infrastructure
resumption is declared and retained separately.

That resumption issues 148 commands before strict JSON rejects a NumPy float32
camera-coordinate scalar nested in pose-error source metadata. A regression
reproduces the producer defect. Emitting native Python coordinate scalars fixes
recording without changing values, ownership/contact decisions or non-finite
rejection. The 148-command failure is not relabeled; the subsequent complete
diagnostic is a separate outcome after this causal repair.

The complete diagnostic spends 164.58 seconds in observer updates, including
110.12 seconds waiting for perception, about 103 seconds in physics/render/native
telemetry, 65.42 seconds in motion admission and 36.98 seconds in safety checks.
These method clocks are nested and must not all be added. ACT inference is only
0.643 seconds. A full replay of the same 524 captures profiles four fixed native
tracking/registration frames without physics, policy commands or training.

A 300-step paired GPU probe measures 8.767/8.567 seconds with 32/16 runtime threads;
both stops pass. The small 2.3% difference does not justify another runtime
setting; default threads remain. Pinning numerical geometry to performance cores
adds no gain and is not adopted. An ROI filter candidate retains 54 exact masks
but saves only about six projected diagnostic seconds; it remains unadopted.
The single-run CLI now clears perception state without preloading an unused next
episode. Profiling found about 10 seconds spent loading that unused worker during
stop. Explicit `preload_on_reset=False` keeps all reset/worker termination rules;
a later acquisition creates fresh model state, and the default preload behavior
for other clients remains unchanged. The initial workload and causal publications
are unchanged; the end-to-end gain still requires measurement.

No observation, force, stopping, material or physical gate is weakened.

## Current limit and next action

The speed goal is incomplete: both measured full-process targets are missed. Continue profiling measured dominant
costs; do not claim success from component gains, an early refusal, opening alone
or the reference role. The full learned panels remain paused.

See [[b1-operational-tolerances|the closure recipe and physical limits]],
[[../topics/episode-and-dataset-contracts|recording contracts]] and
[[b1-end-to-end-pilot-resume|the original scientific outcomes]].
