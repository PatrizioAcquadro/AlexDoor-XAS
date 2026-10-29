# Bounded B1 Data Pipeline

The 2026-10-19 infrastructure study preserves the existing ACT A1 information,
observation recipe, task behavior and scientific outcomes. Storage/reader code is
`ea5d829`; writer, CLI, prefetch and compatibility are `7a392be`. Baseline is
`a0602a2` on `PatrizioAcquadro/b1-policy-pilot`. No simulation, collection,
checkpoint selection or optimizer update is part of this work.

## Method and fixed acceptance

Existing right/left nominal recordings supply complete preparation and saved-input
writer replay. Loader workloads contain 512/2,048 commands per direction, the same
115,218 observation values, batch 8, seed 641 and action/sequence lengths 16/64.
Complete preparation covers 7,052 learned commands plus both terminal observations
and reads the original inspection prefixes. Three loader/ACT runs and two writer
runs retain every attempt. New-process first passes and subsequent OS-cache-warm
passes remain separate; the OS cache was not purged, so no cold-device claim is made.

Before implementation, `targets.json` fixed: loader incremental RSS ≤256 MiB and
ready ≤0.2 s; two-episode preparation ≤1.5 GiB/123 s/128 MiB output; export ≤0.5 s;
four writer windows <18 s without storage growth; batch p95 ≤6 ms and complete
ACT mean step ≤11.1 ms. The ACT limits derive from measured 6.1–6.5 ms compute
and the 7.11–9.08 ms baseline range. Resource budget is 20 GPU-context minutes,
60 GiB new artifacts, no optimization or simulation.

## Measured result

| Identical workload | Before | After |
|---|---:|---:|
| Complete two-episode preparation | 115.80 s | 106.25 s; isolated repeat 106.03 s |
| Prepared output, raw retained separately | 6.06 GiB | 106.5 MiB |
| Preparation peak process-tree sum RSS, launcher included | 11.36 GiB | 1.17 GiB |
| Export only | 1.875 s | 0.087 s |
| Ready, 4,096 commands/H64, median | 1.670 s | 0.0188 s |
| Loader worker peak RSS, same workload | 3.67 GiB | 191.4 MiB |
| Loader CPU batches/s, same workload | 2,332 | 225 |
| Loader batch p95, same workload | 0.495 ms | 5.694 ms |
| Real normalized factory setup including validation | 9.26 s | 4.95 s |
| Real normalized factory worker peak RSS | 4.19 GiB | 737 MiB |
| Real normalized factory batches/s | 701 | 205 |
| Four writer windows, write + final flush | 18.58–18.80 s | 13.74–13.84 s |
| Writer output | 2,651,801,219 bytes | 2,649,996,372 bytes |
| ACT fixed raw-input mean step, 32 steady steps/run | 7.11–9.08 ms | 8.34–8.67 ms with prefetch |
| Actual normalized factory → ACT CUDA, same workload | 10.52–10.91 ms | 8.98–9.13 ms |

Raw RGB-D write replay preserves the same image resolution/rate and data; its
benchmark uses unavailable feedback on both versions. Separate saved-input checks
exercise actual measured feedback and native 120 Hz streams. Final flush grows
from about 2 ms total to 192–197 ms, while total write+flush improves. Writer worker
RSS grows from 108 to 133 MiB. Fsync is reported separately and remains within the
observed range. The pure CPU loader is slower than an already materialized RAM
corpus. This tradeoff stays explicit; prefetch keeps the measured ACT consumer
within its original variability and declared limit. GPU transfer medians remain below
0.5 ms across these pipelines and PyTorch peak allocation stays about 94.5 MiB for this model/batch.
These are input costs, not learning or physical-task results.

## Evidence and limits

Every prepared command observation, terminal, clock, action and metadata value is
bitwise equal. Normalized minibatches and statistics are also bitwise equal at both
corpus sizes; normalization membership and the input/annotation boundary remain
unchanged. Four saved writer windows retain all 136 compared columns byte for byte.
An additional real prefix retains all 132 observations and 262 native feedback
rows bitwise, crossing the buffers and preserving its interrupted completion flag.
Two failed rollouts and one historical interruption remain unchanged and excluded.
Targeted regressions cover NaN payloads, signed zero, missing feedback, failures,
interruptions, duplicate row order, field selection, A1–A4 sampling, padding,
file eviction/reopen, storage errors, prefetch and indexed-to-dense compatibility.
Two real A1→A2 close/reopen cycles over seven mixed dense/VDS records peak at
225.2 MiB worker RSS. The actual maintained prepare/index CLI paths are exercised.

Direct historical VDS reads remain memory bounded but take about 28 ms/batch.
Whole-image compression, wide HDF5 point gathers and synchronous ACT input failed
the input-time target. Their logs remain preserved. The selected layout stores
lossless reusable frame blocks and joins, and overlaps one next normalized batch.
Three initially contended prefetch runs are retained and excluded; both workloads
were repeated in isolation. No threshold was changed to make a candidate pass.

Raw references require their retained source for inspection/relocation; explicit
`--copy-raw` is the self-contained transfer path. A future encoder/history choice
requires its own binding and causal join preparation from the original data. No
future corpus throughput, temporal-policy quality or manipulation qualification
is inferred. See [[../topics/episode-and-dataset-contracts|maintained storage and
compatibility]] and [[../topics/b1-policy-pilot|pilot commands and boundaries]].

The [complete Italian report](../../../outputs/evidence/data-pipeline-20261019/report.md),
[fixed targets](../../../outputs/evidence/data-pipeline-20261019/targets.json),
[all measurements](../../../outputs/evidence/data-pipeline-20261019/measurements.jsonl),
[full process-tree resources](../../../outputs/evidence/data-pipeline-20261019/process-tree-resources.json)
and [bitwise/reopen validation](../../../outputs/evidence/data-pipeline-20261019/validation.json)
retain denominators, unsuccessful candidates, costs and the concrete handoff.

All frozen numerical acceptance checks pass in the [completion record](../../../outputs/evidence/data-pipeline-20261019/completion.json).
Measured GPU-job intervals, including CPU waits and retained attempts, total
11.83 minutes; launcher startup is included in process-tree/CLI measurements and
excluded from the other intervals. New evidence is about 25.76 GiB. The scientific
pilot ledger is unchanged. Neither budget is evidence of learning improvement.


## Existing generated dataset compaction

The same-day follow-up replaces the three-episode legacy `rgbd-grid-v1` dataset
at `outputs/generated/b1-policy-pilot/act-a1a2-pilot/data` in place. File allocation
falls from 41,554,046,976 bytes (38.70 GiB) to 6,021,120 bytes (5.74 MiB), reclaiming
38.69 GiB after removing the verified redundant files. Separately retained original
recordings remain unchanged and accessible through relative HDF5 `raw` links;
prepared observations use indexed storage. The final figure excludes those originals.
Every raw column/attribute and all 10,158 prepared observations, actions, clocks,
terminal values and metadata match exactly. A1/A2 batches and train-only statistics
also match before and after replacement. Historical evidence and benchmark payloads
remain preserved; this does not claim that every historical dataset was compacted.
The [migration report](../../../outputs/evidence/data-pipeline-migration-20261019/report.md)
and [exact validation](../../../outputs/evidence/data-pipeline-migration-20261019/migration.json)
record the scope and reclaimed blocks. Moving the prepared files requires their raw
sources or a self-contained `--copy-raw` export.


## Temporal pipeline and resolution study

The following saved-input measurements are historical. Their recorded source
revisions and evidence remain valid; the maintained benchmark now derives its
workloads from eligible rows rather than historical fixed indices. This maintenance
does not rerun GPU timings or establish performance equivalence.

## Reviewed baseline measurements (historical)

Evidence: [complete local measurements and attempts](../../../outputs/evidence/act-a1-temporal-implementation-20261019/summary.json),
[matched native detail](../../../outputs/evidence/act-a1-temporal-implementation-20261019/matched-native-detail.png).
One saved **training** episode, 183 identical selected frame indices, nine startup/
nominal/transition/terminal snapshots, four-second history and identical seeded
CNN weights at every resolution. No trained policy was selected. Model weights
were initialized for cost/gradient checks; these are not learning comparisons.

Declared budget: one RTX 4090, 18 GiB VRAM, 8 GiB process RSS, 15 minutes of GPU
measurements. All runs stayed within it. Each resolution has a measured batch-1
memory probe before batch 32, three forward/backward passes with no optimizer,
and 12 contiguous inference queries per full/reuse path. Prior failed harness
attempts and intermediate graph/cache measurements remain, including the wrong
initial depth interval, missing import/contract wiring and the ineffective first
cache implementation. Only affected measurements were repeated after code fixes.

| Resolution | Prepare 183 frames | Batch-32 load / transfer | Batch-32 fwd+bwd median (min–max), ms | Peak allocated / reserved VRAM, GiB | Peak process RSS, GiB |
|---|---:|---:|---:|---:|---:|
| 60×96 | 1.92 s | 118 / 5 ms | 9.9 (9.7–40.3) | 0.56 / 0.94 | 1.66 |
| 120×192 | 1.93 s | 269 / 19 ms | 24.1 (23.7–44.9) | 1.88 / 2.52 | 2.15 |
| 240×384 | 2.01 s | 865 / 73 ms | 94.8 (92.2–96.5) | 7.42 / 9.60 | 4.22 |

Preparation includes disk read/decode, GPU transfer/pooling, return transfer and
frame-store writing. Loader costs include joins and bounded frame maintenance;
model measurements use float32, the intended batch 32, and the final current-state
readout. Action/torque statistic fitting is checked separately, not fitted to the
real benchmark's unreviewed labels. First convolution/backward calls remain in
the reported variability; three samples are insufficient for robust tail estimates.

| Resolution | Complete batch-1 full path p50 / p95, ms | Reuse p50 / p95, ms | First model forward | Native history fill |
|---|---:|---:|---:|---:|
| 60×96 | 11.70 / 13.79 | 13.06 / 14.96 | 134 ms | 247 ms |
| 120×192 | 16.75 / 17.42 | 16.21 / 18.89 | 133 ms | 244 ms |
| 240×384 | 31.16 / 32.76 | 29.85 / 34.31 | 141 ms | 248 ms |

The complete saved-input path includes native image read/decode, CUDA pooling and
return transfer, history join/load, required transfer to ACT, visual computation/
feature-cache maintenance, temporal context and action return transfer. New-frame
preprocessing is about 8.5–9 ms on this fixture; model construction is about 12 ms.
History fill and first forward are separate startup costs. This is saved-stream
wall time, not measured sensor-to-actuator hardware latency or physics throughput.

### Historical inference reuse decision (12 queries)

The bounded cache stores only deterministic visual features (64 entries), never
training latents. `reuse=True` keeps host snapshots and transfers/encodes misses
only; temporal context is always recomputed. It invalidates on reset, encoder
weight version/storage changes, device/dtype or preprocessing changes. Training
and grad-enabled calls recompute with correct gradients.

Full/reuse outputs agree within maximum absolute **6.2e-7** in model output on the
final measured graph. The first two queries fill the interleaved visual history;
all 12 samples are reported above. Across the following ten queries, 240×384
median falls from 31.16 to 29.50 ms, while p95 changes from 32.00 to 32.34 ms.
At 120×192 median gains are small and tail latency regresses; 60×96 is slower with
reuse. Keep reuse **off by default**. It is a measured median-cost option for the
240×384 fixture, not a proven deadline improvement.

### Resolution decision and limits

60×96 visibly merges finger/panel edges. 120×192 retains more contour detail and
is feasible at batch 32 with substantial VRAM margin; 240×384 retains more fine
geometry but costs about four times the backward time and allocated VRAM.
These observations justified the original 120×192 compromise, not learning
superiority; the representative delivery measurements below supersede that default. Even native hold/release images show arm self-occlusion; resolution
cannot recover hidden contact. No resolution is certified for millimetric contact.

The proposed 16.7 ms deadline applies only if future hardware uses a synchronous
60 Hz dispatch schedule. It is not a universal gate for this offline pipeline.
The 120×192 saved path already has p95 above that budget before live admission/
IO costs; startup also exceeds it. Hardware real-time readiness is **unvalidated**.
No control rate, asynchronous execution or dispatch compensation was introduced.

## Representative input delivery

New evidence is preserved separately in
[`act-a1-temporal-integration-20261019`](../../../outputs/evidence/act-a1-temporal-integration-20261019/).
The reviewed measurements above repeat five distinct decisions. The new matched
before/after profiles use three fixed batch-32 workloads: 32 distinct decisions
spread across the saved training trajectory; 32 consecutive, overlapping
decisions; and eight spread decisions plus 20 consecutive decisions and explicit
duplicates. Every workload retains all 23 RGB-D images, 241 robot tokens and 12
targets. Diagnostic labels remain ineligible for training.

Profiles precede optimization and are kept with five raw timing repetitions per
pattern/resolution. They identify repeated visual gathers/stacks/copies as the
main cost, followed by small numeric HDF5 reads. The target fixed before changing
the loader was a 50% reduction in matched median construction time, with the same
64-frame cache, one delivered plus one prefetched batch, 8 GiB RSS and 18 GiB
allocated VRAM budget. OS page cache was not flushed; application frame-cache
cold costs must not be called cold-disk measurements.

The builder now exposes numeric support without materializing images. Each batch
joins each distinct decision once and writes distinct frames directly into final
batch storage, restoring every duplicate and original sample order. The 64-frame
LRU never grows with a batch. At most 8 MiB of numeric values are retained per
open episode; larger numeric datasets remain lazy. Targets, internal holes,
eligibility, true tail padding, masks, preprocessing and normalization retain
single-sample semantics. The existing one-next-batch prefetch helper is reused;
there is no expanded window dataset or GPU feature cache during backward work.

| Resolution | Distinct windows, before → after | Overlapping windows, before → after | Mixed with duplicates, before → after |
|---|---:|---:|---:|
| 120×192 | 207 → 104 ms | 158 → 65 ms | 173 → 76 ms |
| 240×384 | 860 → 308 ms | 721 → 185 ms | 770 → 223 ms |

The original 50% target is met in **5/6** cases. Distinct 120×192 reaches 49.8%
and remains just below it; the target was not relaxed. The first copy-only
optimization and its weaker 120×192 results remain in the evidence. Numeric
retention addresses the measured residual reads; joins themselves were not
replaced by a second causal implementation. Cold frame reads for the mixed batch
fall from 243 to 237; direct final-buffer materialization supplies most of the
benefit. Distinct batches still read 646 frames, overlapping batches 47. A cold
646-read batch evicts 582 entries from the fixed 64-frame LRU.

[`benchmark_temporal_pipeline.py`](../../../scripts/benchmark_temporal_pipeline.py)
measures **eight complete batches per workload**, with one bounded CPU prefetch
thread, normalization, collation, H2D, synchronized float32 forward/backward and
cleanup. Untimed per-sample comparisons establish exact order, duplicates,
actions, masks and normalization before timing. The source labels stay diagnostic;
seeded initial model weights and backward passes are not policy training.

| Resolution | Complete wall / 8 batches: distinct / overlap / mixed, ms per batch | Median H2D | Median GPU forward/backward | Peak allocated / reserved VRAM | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 120×192 | 129 / 77 / 91 | 20 ms | 24–25 ms | 1.88 / 2.51 GiB | 2.18 GiB |
| 240×384 | 355 / 223 / 274 | 85–88 ms | 94 ms | 7.39 / 9.34 GiB | 4.24 GiB |

These wall averages include first delivery and pending-reader cleanup; they are
not sums of overlapped stage medians. Loading remains the throughput bottleneck.
One-time first model backward calls and every per-batch duration remain in the
JSON. The approximately fourfold image/compute cost of 240×384 remains explicit;
its measured memory and delivery cost are practical within the declared bounded
workstation budget, supporting the information-preserving default. 120×192
remains supported as the cheaper alternative. Neither has learning evidence.

AdamW's two FP32 moments were actually allocated without `optimizer.step` and
remained resident during the batch measurements: **12.50 MiB**, with 6.25 MiB of
parameters and corresponding gradients. A prospective foreach temporary of about
one parameter-sized buffer is an **estimate**; optimizer-step latency and its
transient peak were not measured. No optimizer update occurred.

### Incremental reuse: fill, steady operation and numerical result

The actual `IncrementalObservationBuilder` → `TemporalActPolicy.predict` path
replays the same native saved events at 151 decisions, six control ticks apart.
Each query includes new native RGB-D reads/decode, CUDA preprocessing/return,
numeric event ingress, causal construction, normalization, required transfers,
ACT and action return. Three trials per mode alternate order. The first history
and prediction cost is separate (120×192: 366–378 ms; 240×384: 405–415 ms). The
next **40 queries / one whole 4 s history cycle** are retained as cache fill;
**110 steady queries per trial, 330 per mode** remain. Serial queries are correlated;
empirical p95 values do not establish a deadline guarantee.

| Resolution | Full recomputation steady p50 / p95 | Reuse steady p50 / p95 | Untimed strict numerical comparison |
|---|---:|---:|---|
| 120×192 | 16.38 / 16.97 ms | 16.40 / 16.88 ms | Fails in 86/150 queries, 109/12,600 components |
| 240×384 | 41.10 / 41.88 ms | 39.31 / 40.47 ms | Passes in 150/150 queries, 12,600 components |

The table reports the median of the three trial p50/p95 statistics; complete
trial ranges and all fill/steady samples remain in
[`summary.json`](../../../outputs/evidence/act-a1-temporal-integration-20261019/summary.json).
Equivalence comparisons are separate, untimed passes with the existing
`atol=1e-6, rtol=1e-5` criterion. Maximum absolute output differences are 2.27e-6
at 120×192 and 1.40e-6 at 240×384 with identity diagnostic action scaling. The
first strict 120×192 assertion failure and the complete census are both kept;
no tolerance was changed to pass it. Small absolute differences alone do not
establish equivalence under the declared criterion or a physical safety result.

**Keep reuse off by default at both resolutions.** At 120×192 it provides no
meaningful latency gain and fails the strict numerical criterion. At 240×384 it
passes this saved-trace check and reduces median/tail time by about 4%; it remains
an explicit measured option, not a qualified default for a future learned
checkpoint or hardware producer. Use the preferred 240×384 path with full
recomputation for collection consumption and later ACT work. No asynchronous execution,
control-rate change or physical threshold adjustment was introduced.
