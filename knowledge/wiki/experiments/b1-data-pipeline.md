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
