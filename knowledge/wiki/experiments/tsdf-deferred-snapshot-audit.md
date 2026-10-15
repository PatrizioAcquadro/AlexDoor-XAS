# TSDF Deferred Snapshots: CUDA Cost and Failure Audit

2026-10-14 audit, completed on 2026-10-15 with real RTX 4090 CUDA. Baseline
`1f9cb20`, branch `PatrizioAcquadro/b1-observed-a3a4`; initial audit commit
`f960c2f`. **Reject final-only CPU snapshot deferral for adoption.** It provides a
large, repeatable local success-path gain, but changes the retained snapshot and
actual worker queries after a failed in-place integration. Production remains at
the restored baseline. This does not reject the broader rebuild direction.

## Scope and preserved evidence

Only point 3 was investigated. The previous
[[tsdf-readonly-launch-inputs|read-only launch-input candidate]] remains in
`ae3bcec`, and its restoration/rejection remains in `1f9cb20`. Its source and
complete old evidence were read, not overwritten. The original `/1671` worktree
was externally removed during the initial inspection; its existing branch at the
identical baseline was reused in `/782a`, with no tracked user changes.

Current executable probes, all rows, captured keyframes, full comparison arrays,
failures and summaries are under
`outputs/evidence/b1/perception/tsdf-snapshot-20261014/`. A preserved copy also
lives at that relative path in `/home/pacquadr/Desktop/AlexDoor-XAS/`, beside the
unchanged `tsdf-readonly-20261014/` evidence. The initial pre-CUDA audit is retained.

The single sketch separates the native integration body from its CPU snapshot
tail. It retains every integration in its original order and materializes once
on rebuild return or exception, including when the current keyframe is rejected.
The native filtering, images, kernel, seven `InOut` argument handlers, weights,
color, grid, bounds, resolution and rebuild policy stay unchanged. The previously
rejected read-only-input optimization is not combined with it. Models, references,
thresholds, map inputs, scheduler, encoder, dataset, training, feedback, common
contracts and A3/A4 are unchanged.

The sketch is a local subclass of the actual installed native class; it never
edits the upstream installation or becomes an active production option. The
consumer audit did not establish unconditional redundancy, so no unsafe
production/setup modification was introduced. Its complete executable
implementation is preserved in `local_rebuild.py`, with the earlier CPU
counterexample in `audit_failures.py`.

## Identical inputs and predetermined local comparison

A baseline chronological prefix captures each actual target rebuild: 2,564 native
acquisitions through 67.733333 s light, and 1,685 through 53.083333 s nominal. All
trace lines, complete output-packet bytes and update/next-capture volume/query
checks equal the preserved old baseline, including losses and refused decisions.
These prefixes acquire fixtures; their timings include fixture export and are
not used as performance measurements. They are not complete replay comparisons.

Each local cell runs in a fresh CUDA process with the same captured keyframes,
native configuration and current keyframe. The previous real CUDA volume is
reconstructed before the timer; its TSDF, color, bounds, origin and integration
count must match the captured previous state exactly. Its CPU/CUDA allocation
release is therefore included in the measured rebuild. The target is 17 ordered
integrations over 186 x 304 x 294 light and six over 288 x 321 x 294 nominal.
Final TSDF/color/weight bytes, bounds, origin, dimensions, metadata, counts,
integration sequence, native signed queries and validity all match the captured
target and both success paths. Direct full-array byte comparisons also pass on
every normal pair. Same-frame materialization completes before return.

One preset untimed warmup per path is followed by three pairs, ordered B/V,
V/B, B/V. Every row is retained. All times below are milliseconds:

| Target | Baseline repetitions 1, 2, 3 | Final-only repetitions 1, 2, 3 | Paired savings 1, 2, 3 |
|---|---:|---:|---:|
| Light | 650.2, 654.1, 643.3 | 288.7, 281.1, 281.7 | 361.5, 373.0, 361.6 |
| Nominal | 468.5, 474.6, 452.0 | 268.4, 273.4, 268.9 | 200.1, 201.1, 183.1 |

Median event duration is **650.2 -> 281.7 ms light** and **468.5 -> 268.9 ms
nominal**. Differences of medians are 368.5 ms (56.7%) and 199.6 ms (42.6%);
median paired savings are 361.6/200.1 ms. Both candidate ranges are below their
baseline ranges, comfortably beyond the observed within-path variation. Three
pairs do not establish a population confidence interval or an extreme percentile.

Warmups, excluded as specified before measurement, are 654.8/281.2 ms light and
453.2/270.9 ms nominal (baseline/sketch; exact raw rows own the values). One
separate detailed baseline profile per target is also excluded from the three
comparison pairs. There are 34 completed local cells: 12 timed cells, four
warmups, two detailed profiles and 16 refusal/error cells. There is no favorable
timing retry or discarded completed timing row.

## What the host work actually contains

The following disjoint exclusive host categories sum within each detailed event;
these are not sums of quantiles. Detailed totals are 659.95/462.76 ms light/nominal.

| Baseline target category | Light ms | Nominal ms |
|---|---:|---:|
| Point transforms and observed extent | 45.4 | 44.8 |
| Point filtering | 81.8 | 44.2 |
| Images, masks, pose and packed color | 32.4 | 15.1 |
| Volume allocation and initial uploads | 71.7 | 117.7 |
| CUDA calls, argument preparation, transfers and waits | 27.7 | 10.1 |
| TSDF/color snapshot downloads | 135.9 | 76.2 |
| Snapshot metadata and release of prior arrays | 3.8 | 1.7 |
| Snapshot array allocation and copies | 255.7 | 144.7 |
| Remaining control/profiler glue | 5.6 | 8.3 |

Point preparation includes both the panel adapter's observed-extent transform,
finite selection, radius/bounds and filter, and the native float32 homogeneous
transform/filter repeated for every replayed keyframe. Filters contain median,
distance, percentile, radial cutoff and selected-point allocation. Images include
depth casts/division, mask CUDA-to-CPU conversion, depth copy/masking, pose and
intrinsics conversion, RGB selection and packed-color construction. These remain
in the final-only sketch.

Volume allocation includes initial CPU grids, device grids, initial HtoD, cached
module setup and launch-grid preparation under the existing memory guards.
CUDA-call time includes argument casts/allocations, HtoD/DtoH handlers, driver
launch and synchronization; it is not pure kernel time. Snapshot copies include
NumPy allocation plus copy. Metadata time can include freeing the prior snapshot.
Timers add no fences inside integration; the native synchronous boundary remains.

The historical 390.7/227.1 ms residual host times came from different whole-worker
trials and cannot be repartitioned numerically using these new local timings.
Here, points alone cost **127.2/89.0 ms**, and images another **32.4/15.1 ms**.
Only the separately measured snapshot categories support a snapshot-cost claim.

In these profiles, downloads before the final snapshot cost **128.0/63.8 ms**;
pre-final snapshot array allocation/copies cost **240.8/120.4 ms** light/nominal.
They explain the substantial local opportunity, without treating cross-run timing
differences as an exact additive causal decomposition.

Actual target DtoH accounting, excluding verification reads:

| Target | DtoH calls baseline -> sketch | DtoH bytes baseline -> sketch | Removed full-snapshot bytes |
|---|---:|---:|---:|
| Light | 153 -> 121 | 2,339,193,812 -> 211,330,004 | 2,127,863,808 |
| Nominal | 54 -> 44 | 1,332,275,064 -> 245,086,584 | 1,087,188,480 |

Complete TSDF/color downloads fall from 17/6 to one, removing 32/10 calls. The
other seven argument readbacks per launch remain. The same logical byte counts
are removed from snapshot array copies; this is payload size, not a claim about
all memory-bus transactions. HtoD calls/bytes and CUDA allocation-size sequences
are exactly unchanged: 122/45 allocations per light/nominal cell.

## Memory and remaining cost

| Target | Timed sampled host RSS peak, baseline MiB | Sketch MiB | Driver sampled peak, baseline/sketch MiB |
|---|---:|---:|---:|
| Light | 1593.7–1594.2 | 1576.6–1577.1 | 976 / 976 |
| Nominal | 1692.7–1693.4 | 1673.7–1674.2 | 1088 / 1088 |

These ranges retain all three repetitions. RSS is sampled every 5 ms around the
rebuild; driver process memory every approximately 100 ms across the local cell.
Post-timer array export/digests are excluded from the native timer. The separate
process high-water mark includes verification and is not used as the timed peak.
The figures include the interpreter, captured frames, masks and TSDF state;
SAM3/TAPIR residency and full-provider memory are absent. Sampling is not an
allocation-lifetime proof.

Both paths retain the same three CUDA grids, three CPU volume workspaces and two
final CPU snapshot arrays. Torch peaks for the captured masks remain
9.338/3.296 MiB light/nominal. There is **no demonstrated persistent GPU-memory
saving**; the modest local RSS difference is about 17–20 MiB. Initial allocation,
point/image preparation, all fusion launches and the final snapshot still remain.
The sketch's successful event medians are still 281.7/268.9 ms. No complete-provider
latency or 150 ms availability is inferred from these isolated events.

## Partial failures and the actual next worker request

The successful inner rebuild loop has no consumer between integrations, but each
intermediate `obj.sdf` is an independent last-completed CPU snapshot. CUDA fusion
writes the live volume in place. The native counter increments only after fusion
returns; readback and snapshot construction then have their own existing partial
failure boundaries. `obj.sdf` can therefore intentionally differ from a device
volume touched by a failed call.

One preset refusal/error case per path and target verifies the boundary:

| Case | Baseline vs final-only sketch |
|---|---|
| Current keyframe rejected after replay | Exact outputs, snapshot, queries, history and counters |
| An old keyframe rejected during replay | Exact skipped integration, later integration order, snapshot, queries and counters |
| Exception before integration #2 writes | Exact last-completed snapshot, queries, error and partial counters |
| Exception after real CUDA integration #2 writes | Same error, live TSDF/color/weight volume, count 1 and snapshot count 1; different CPU TSDF/color snapshot and subsequent signed queries, in both light and nominal |

The post-write exception is deliberately injected after actual synchronous CUDA
fusion, without an illegal device access or poisoned context. It tests the
fallible call boundary; it does not estimate natural hardware-error frequency.
The earlier five-case CPU control-flow probe is retained separately and is not
substituted for this CUDA evidence.

Consumers include `SVDClusterRANSACRegister._query_sdf_signed` and keyframe-manager
SDF gates; optional bbox/mesh paths can separately read current volume/weights.
A rejected final keyframe must still expose the successfully replayed old state.
Flushing only on a final `True` would also violate this refusal path.

`worker.main` sends an error and continues receiving requests. The normal engine
and capture session latch failure, but the direct worker interface remains open.
This was verified with two fresh real CUDA workers and three original raw
acquisitions each. After the second acquisition executes native model/pipeline
work, a constructed fixture invokes the captured light rebuild and injects the
post-write exception. It leaves the native index at 1; the third request reuses
that index and successfully executes **29 valid native SDF query calls** in both
paths. The first query has identical 1,495 input points and identical validity,
but different signed values because it consumes the different retained snapshot.
This is an actual consumer, not just a dictionary comparison or an assumed read.
The constructed fixture does not qualify operational recovery or reproduce a
natural late-session failure frequency.

An exception-time download of the current device grid cannot recover the last
completed CPU state after an in-place write. No rollback, retry, checkpoint or
changed worker/session recovery policy is added to conceal that difference.

## Failures, reproducibility and final decision

The GPU was initially occupied by Isaac on October 14; no CUDA measurement ran
then. On October 15 no other compute process was present at startup. During a
light before-write error check, a foreign `squadbot-av-phase1` compute process
appeared. The 100 ms guard terminated our own probe and retained its partial
output under `failed-concurrent-compute/`. A subsequent preflight refused to
start while another foreign process was active, retained under
`failed-preflight-compute/`. After 30 continuous seconds without a compute
process, only missing checks/cells resumed. Completed light timings were never
replaced. No foreign compute process was observed in any completed timed cell
or either worker-reentry check. Ordinary desktop graphics remained present.

Use the maintained pinned installation from `scripts/setup_point2pose.py`.
The private native SDF source is byte-equal to the restored baseline; source,
configs, setup and maintained tests have no tracked implementation diff.
The preserved `.pkl` fixtures and scripts can be copied into a fresh evidence
folder at the same repository depth, then run with:

```bash
PYTHONPATH=src models/perception/point2pose/runtime/bin/python \
  outputs/evidence/b1/perception/tsdf-snapshot-20261014/run_local.py
```

Use a fresh copy because the launcher refuses to overwrite existing run logs.
`capture_keyframes.py` recreates the fixtures from the original chronological
prefixes when needed; `worker_reentry.py` owns the direct-worker fault check and
`analyze_local.py` performs full-byte comparisons and keeps every result.
`local-summary.json`, `worker-reentry-summary.json`, `protocol.json` and
`completion-audit.json` own the raw boundaries and gates. Ignored evidence and
`TODO.md` remain uncommitted, as required; only the durable wiki record is tracked.

**Reject this final-only variant despite its clear local speedup.** Success
identity and the benefit gate pass; the error-state gate fails, so full fixed-input
and timed-ingress replays remain closed. No complete replay or timed-ingress run
was started, no quality/loss regression was removed, and no second optimization
was attempted. Adopted savings are zero because production is unchanged;
the measured successful-path savings above are not safe operational savings.

The experiment is complete at this rejection. Keep the restored baseline and all
evidence. Any separately authorized successor must first preserve the last
completed state across a failed in-place write. SAM3, TAPIR, CPU registration,
material recovery and hardware/contact qualification remain separate concerns.
