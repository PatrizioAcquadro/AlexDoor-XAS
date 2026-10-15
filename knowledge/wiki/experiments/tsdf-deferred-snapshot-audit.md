# TSDF Deferred Snapshot Failure Audit

2026-10-14. Point 3 only, restored baseline `1f9cb20` on
`PatrizioAcquadro/b1-observed-a3a4`. **No production optimization is adopted.**
The final-only snapshot sketch fails the partial-failure state contract. This is
a control-flow rejection, not a measured CUDA performance result or a rejection
of every possible separation of CUDA integration and CPU snapshots.

The requested timing diagnosis remains incomplete: Isaac Sim occupies the RTX
4090, and an exclusive GPU was not available. No CUDA workload, timed comparison,
chronological prefix or complete replay was started by this audit. The user was
asked to release that GPU or authorize stopping the competing process; it was
not terminated by this experiment.

## Baseline and preservation

The previous [[tsdf-readonly-launch-inputs|read-only launch-input candidate]]
remains in `ae3bcec`; its rejection and restored baseline remain in `1f9cb20`.
The complete prior evidence is preserved at
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1/perception/tsdf-readonly-20261014/`.
It was read, not overwritten. The old `/1671` worktree was externally removed
during read-only inspection. Its existing branch, still at the identical commit,
is now checked out in `/home/pacquadr/.codex/worktrees/782a/AlexDoor-XAS`.
There were no tracked user changes in either inspected checkout.

Private upstream sources were copied into this worktree, while weights and
runtimes reference the existing maintained installation. No model, configuration,
threshold, reference, resolution, map content, scheduler, rebuild policy,
encoder, dataset, training, feedback, shared contract or A3/A4 code changed.
The ignored `TODO.md` contains the remaining implementation checklist.

## Cost boundaries identified in source

The old measurements established 17/6 downloads at the largest light/nominal
rebuilds and 122.7/61.6 ms before their final download. These are historical
potentially removable costs, not new measured savings. The old 390.7/227.1 ms
residual host category must not be relabeled as snapshot-copy time.

The next targeted measurement must separate these operations:

| Operation | Maintained source and behavior |
|---|---|
| Outer observed extent | `PanelSDFBuilder.integrate_keyframe`: inverse pose, transform, finite-point selection, extent/radius, native filter, observed bounds and envelope update |
| Replayed point preparation | Native `SDFBuilder.integrate_keyframe`: float32 conversion, homogeneous transform and `_filter_points` for every old keyframe and the current keyframe; this work remains even when snapshots are deferred |
| Point filtering | Median, distance, percentile, radial cutoff and selected-point allocation in `_filter_points`; both outer and native calls remain |
| Images and pose | Depth cast/division, mask CUDA-to-CPU conversion, depth copy/masking, RGB selection or zero-image allocation, inverse pose and intrinsics conversion |
| Fusion preparation and launch | `TSDFVolume.integrate`: float RGB conversion, packed color calculation, argument casts, PyCUDA argument allocations/transfers, synchronous kernel invocation and argument readbacks; its host duration is not pure kernel time |
| Volume allocation | Three initial CPU grids, three CUDA grids, initial HtoD, module setup and grid calculation inside the existing memory guards |
| Snapshot download | `get_volume`: complete TSDF and color DtoH, into reusable volume CPU buffers |
| Snapshot construction | Metadata copies plus separate complete `tsdf.copy()` and `color.copy()` into `obj.sdf`; these are distinct from the downloads and point/image preparation |

No new timing is assigned to these categories. Baseline source separation alone
cannot quantify them or establish a speedup. The fixed target volumes remain
186 x 304 x 294 light and 288 x 321 x 294 nominal.

## Actual consumers and failure boundary

A successful rebuild has no reader between the native integrations in the
history loop. Registration, pending-reference/keyframe SDF gates, optional bbox
refinement and mesh export run outside that inner loop. On return they require
the completed same-frame map. The selected register reads `obj.sdf` through
`SVDClusterRANSACRegister._query_sdf_signed`; keyframe manager queries also read
its arrays. Bbox and mesh paths can separately call `get_volume` and weights.

The intermediate snapshots still have an observable failure role:

- Growth clears `obj.sdf_volume` and `obj.sdf`, then replays old keyframes in order.
  Every successful native integration increments `sdf_num_integrated`, downloads
  TSDF/color and creates an independent CPU snapshot.
- A rejected old keyframe returns `False` but does not stop that loop. If the
  current keyframe is rejected after replay, the last old-keyframe snapshot
  remains available; `rebuilds` has already incremented and history is not
  appended. Flushing only when the final call returns `True` breaks this case.
- An integration exception leaves the last completed CPU snapshot. The CUDA
  volume may already contain writes from the failed integration. Counters,
  envelope, history and `rebuilds` retain their existing partial updates.
- A download exception occurs after the integration count increments and before
  replacement of `obj.sdf`. A failure while copying the snapshot can leave the
  new metadata or new TSDF without the color entry. These are existing statement
  boundaries, not reasons to add retries, rollback or a new session policy.
- `worker.main` sends an error and continues its request loop. The normal engine
  and capture session latch failure, but that does not close the direct worker
  interface. A later worker request can perform CPU SDF queries before another
  integration. It is therefore insufficient to argue that the operational
  session stops, or that every possible CUDA error poisons the context.

A final or exception-time download of the live volume does not reconstruct the
previous completed snapshot after a failed in-place write. A success-only test
would miss this difference. No retry, reconstruction, device rollback,
checkpoint scheme or change to worker failure policy is introduced here.

## Executed bounded control-flow probe

Local executable evidence is in
`outputs/evidence/b1/perception/tsdf-snapshot-20261014/`:
`audit_failures.py`, `control-flow-failure-audit.json` and its log. It executes the
actual native builder class, the baseline panel adapter and the actual native
signed-query function against small simulated storage. This is a CPU unit-level
control-flow probe, **not CUDA fusion equivalence, a performance benchmark or a
maintained candidate implementation**. The final-only sketch exists only in that
ignored diagnostic file, leaving native and tracked production sources unchanged.

Five preset cases compare eager snapshots with a final/exception-time flush:
normal success, rejected current keyframe, rejected old keyframe, exception before
writing and exception after writing. There is one execution per case, with no
replacement or favorable retry. The first four preserve outputs, integration
count, history and rebuild count. The fifth preserves those counters but changes
both the retained snapshot and its subsequent native signed query:

| After the second integration writes and raises | Baseline | Final-only sketch |
|---|---:|---:|
| Completed integration count / snapshot count | 1 / 1 | 1 / 1 |
| Snapshot value and following valid native query | 0.8999999762 | 0.6999999881 |
| Live volume value | 0.6999999881 | 0.6999999881 |

This deterministic counterexample rejects unconditional final-only deferral under
the requested failure semantics. No gain or memory reduction is claimed.

## Remaining work and decision

**Decision for this sketch: reject; keep the restored baseline.** Adopted measured
savings are zero because production code is unchanged; candidate CUDA savings
and memory/error behavior have not been measured. The broader point-3 experiment
is not complete and has no evidence-backed performance adoption decision.

`protocol.json` records three preset baseline/candidate local repetitions per
condition, alternating B/V, V/B, B/V after one untimed warmup per path. Exact
volumes, colors, weights, native queries, history and refusal/error state must pass
before a performance result is eligible. Full fixed-input replays require a
median target reduction of at least 10% and non-overlapping measured timing
ranges. Timed ingress follows only after fixed-input equivalence. Failed runs,
losses and regressions must remain. The failing sketch does not pass this gate,
so no complete light/nominal or timed replay is authorized by its timing potential.

`capture_keyframes.py` is prepared, syntax-checked but not executed. Once an
exclusive GPU is available, baseline chronological prefixes through 67.733333 s
light and 53.083333 s nominal can capture the actual retained keyframes, prior
bounds and native configuration without pretending an unavailable checkpoint
exists. Those inputs can support the missing cost split. A maintained optimization
is conditional on proving its failure-state contract; this audit does not justify
silently adding a recovery policy or selecting a second optimization.

Remaining ordinary SAM3, TAPIR, CPU registration, startup/contention and material
recovery costs are separate. There is no new complete-provider latency result,
no hardware/contact qualification and no 150 ms promise.
