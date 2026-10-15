# TSDF Rebuild Double Buffer

2026-10-15. Branch `PatrizioAcquadro/b1-observed-a3a4`; continuation baseline
`b498d1f`. **Evaluation complete: retain the maintained baseline.** The candidate
is numerically equivalent at fixed inputs and saves 35.2%/28.8% on the captured
light/nominal rebuilds with bounded temporary memory. The timed operational
comparison does not justify adoption: ordinary p95 latency does not improve,
nominal quality deteriorates across all three targets, and the nominal buffered
history performs no post-seed volume expansion. Its shorter maximum stall is
not a measured speedup of the buffered rebuild. No maintained code is changed.

## Protocol and adoption criteria

`adoption-criteria-continuation.md` was saved before the remaining tests; commit
`a633202` records the criteria and completed memory gate. Require bounded memory
with zero live pool arrays after retirement, and at least a 5% improvement in
steady p95 service, p95 publication age, publication frequency or maximum
publication interval in at least one condition, consistent with update history.
Assess both conditions together. A broad latency/quality regression or a gain
obtained by losing useful tracking defeats adoption. The three frozen material
poses are scored at 10/15/20 mm and 5 degrees, with processed and scheduled
denominators, precision, terminal five seconds, gaps and losses retained.
This optimization alone is not required to reach 150 ms; one timed pair is not
a population confidence statement. Different timed histories are permitted.

Only the missing light buffered, nominal buffered and nominal baseline memory
cells ran, followed once each by light baseline, light buffered, nominal buffered
and nominal baseline. The valid light memory baseline, all local/error/worker
gates and both full fixed-input comparisons were reused without repetition.
The recipe, initialization protocol, random seed, source times, selection,
models, geometry, thresholds and control remained fixed. No training or contact.

The continuation held `/tmp/rtx4090-exclusive.lock` throughout, with 30 seconds
without compute contexts before each cell and foreign-compute monitoring during
each process. No foreign compute was observed in these seven completed cells.
Graphical desktop contexts remained permitted by the existing protocol. No
other process was stopped. All four resets confirm exited workers and empty
queues/publication state; the GPU lease is released.

## Already valid correctness and local benefit

The recovered-source 24 local CUDA pairs and eight actual-worker error/reentry
pairs pass, including partial TSDF/color downloads, completed snapshots,
bbox/mesh reads and later integration/queries. Both full chronological sequences
have 3,217 inputs and zero differences in packets, registration, references,
history, volumes, metadata and signed queries. Light has 17 updates/33 volume
audit rows; nominal has 10/19; each compares 3,217 reference-array files.
`exact-equivalence.json` and the existing local/worker summaries own these gates.

| Captured rebuild | Baseline median | Buffered median | Reduction |
|---|---:|---:|---:|
| Light, 17 integrations | 659.2 ms | 427.4 ms | 35.2% |
| Nominal, six integrations | 481.7 ms | 343.0 ms | 28.8% |

Three preset pairs per condition have disjoint timing ranges. Every complete
native download, HtoD/DtoH call and byte count, CUDA allocation sequence and
integration order remains unchanged. Full CPU snapshot allocations fall from
34/12 to four arrays light/nominal, removing their repeated full-array copies.
These are demonstrated local savings, with zero adopted production savings.

## Repeated-rebuild memory

Each cell completes 12 identical rebuilds; cycle zero is warmup and the final
eight assess stability. No allocator trimming or forced GC. All output digests
match within each cell and between paths.

| Condition/path | Peak RSS, MiB | Final-eight retired RSS range, MiB | RSS slope, MiB/cycle |
|---|---:|---:|---:|
| Light baseline, preserved | 1713.0 | 5.8 | -0.651 |
| Light buffered | 1840.0 | 6.3 | +0.178 |
| Nominal baseline | 1893.0 | 10.5 | -0.370 |
| Nominal buffered | 2105.6 | 5.4 | -0.006 |

Exactly two pool arrays remain alive while `obj.sdf` owns the completed buffered
snapshot. After object retirement, every pool weak reference is dead in every
cycle. RSS left behind therefore does not represent live snapshot buffers: it
includes fixtures, runtime and allocator residency. The test does not attribute
all remaining RSS to the allocator. The small positive light slope stays inside
a 6.3 MiB oscillation and is not progressive pair-sized retention. Buffered
global CUDA-free tail ranges are 0/0.75 MiB light/nominal; desktop graphics can
also change global free memory. Added transient CPU residency is about 127/213
MiB here; the candidate is not a memory-reduction optimization.

## Four timed operational comparisons

Each run receives all 4,718 source inputs; opening quality uses all 2,858 inputs
from 31 seconds onward. Steady timing/frequency begins five seconds after the
actual seed, per the unchanged protocol. Slashes below separate p50/p95/maximum
or the explicitly named quantities. Times are milliseconds.

| Metric | Light baseline | Light buffered | Nominal baseline | Nominal buffered |
|---|---:|---:|---:|---:|
| Seed capture, s | 27.350 | 26.150 | 27.300 | 25.283 |
| Processed / all inputs | 238 / 4718 | 246 / 4718 | 234 / 4718 | 236 / 4718 |
| Opening processed / scheduled | 199 / 2858 | 201 / 2858 | 196 / 2858 | 196 / 2858 |
| Replaced / skipped | 4480 / 4480 | 4472 / 4472 | 4484 / 4484 | 4482 / 4482 |
| Steady publications | 193 | 200 | 191 | 199 |
| Service p50/p95/max | 223.2 / 276.0 / 567.0 | 218.8 / 280.4 / 1040.5 | 229.2 / 274.2 / 424.8 | 230.0 / 292.3 / 361.2 |
| Publication age p50/p95/max | 244.5 / 302.3 / 596.4 | 245.1 / 305.8 / 1062.2 | 246.4 / 295.1 / 443.7 | 247.6 / 312.3 / 377.1 |
| Publication Hz | 4.154 | 4.201 | 4.103 | 4.100 |
| Publication interval p95/max | 284.6 / 584.1 | 294.7 / 1051.1 | 283.9 / 434.0 | 300.7 / 366.9 |
| Stalls >0.5 / >1 / >2 s | 1 / 0 / 0 | 1 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Native lost, nonseed | 13 | 52 | 0 | 27 |
| Accepted, nonseed | 107 | 161 | 200 | 184 |
| Post-seed volume changes | 3 | 2 | 2 | 0 |
| Peak RSS / GPU, MiB | 5796.4 / 7910 | 5742.8 / 7918 | 5693.3 / 10738 | 5728.6 / 9928 |
| Steady publications <=150 ms | 0 | 0 | 0 | 0 |

Peaks are simultaneous process-tree post-ready samples at 250 ms, with zero
sample errors. Startup spikes remain unqualified. GPU residency differs with
model allocator/history state; the CPU snapshot pool does not reduce CUDA
allocation or transfer volume. Full phase accounting remains in each report.

Light frequency rises 1.1%, while service/age p95 worsen 1.6%/1.2%. Its largest
buffered stall at 50.6 s includes 879 ms SAM3 and 161 ms P2P; it is retained in
the total. Volume-change services are 426/567/477 ms baseline and 410/441 ms
buffered, but at different times, extents and history lengths. Those are not
matched causal pairs. Nominal maximum publication interval falls 15.4%, while
service/age p95 worsen 6.6%/5.8% and frequency is unchanged (-0.06%). The nominal
buffered resource trace has no post-seed volume-bound change, versus two in the
baseline: its reduced maximum does not exercise the optimization being adopted.

## Frozen three-target quality

All triples below are 10/15/20 mm with 5 degrees. Correct counts are identical
between processed-only and all-scheduled scoring. The denominators remain
explicit; missing, rejected, lost and wrong accepted cases are never discarded.
Precision is correct/accepted. The tail is correct/scheduled over the final
five seconds (301 scheduled inputs). Quality is at capture time, not a
freshness-compensated contact guarantee.

| Run / target | Accepted opening | Correct counts | Processed correct, % | Scheduled correct, % | Accepted precision, % | Tail correct |
|---|---:|---:|---:|---:|---:|---:|
| Light baseline / 0 | 100 | 34/34/34 | 17.09/17.09/17.09 | 1.19/1.19/1.19 | 34.00/34.00/34.00 | 0/0/0 |
| Light baseline / 1 | 100 | 34/34/34 | 17.09/17.09/17.09 | 1.19/1.19/1.19 | 34.00/34.00/34.00 | 0/0/0 |
| Light baseline / 2 | 100 | 34/34/34 | 17.09/17.09/17.09 | 1.19/1.19/1.19 | 34.00/34.00/34.00 | 0/0/0 |
| Light buffered / 0 | 149 | 131/141/145 | 65.17/70.15/72.14 | 4.58/4.93/5.07 | 87.92/94.63/97.32 | 0/0/1 |
| Light buffered / 1 | 149 | 131/138/146 | 65.17/68.66/72.64 | 4.58/4.83/5.11 | 87.92/92.62/97.99 | 0/0/1 |
| Light buffered / 2 | 149 | 130/141/146 | 64.68/70.15/72.64 | 4.55/4.93/5.11 | 87.25/94.63/97.99 | 0/0/1 |
| Nominal baseline / 0 | 195 | 156/191/195 | 79.59/97.45/99.49 | 5.46/6.68/6.82 | 80.00/97.95/100.00 | 11/22/23 |
| Nominal baseline / 1 | 195 | 159/191/194 | 81.12/97.45/98.98 | 5.56/6.68/6.79 | 81.54/97.95/99.49 | 11/22/23 |
| Nominal baseline / 2 | 195 | 148/185/194 | 75.51/94.39/98.98 | 5.18/6.47/6.79 | 75.90/94.87/99.49 | 10/17/23 |
| Nominal buffered / 0 | 169 | 139/161/163 | 70.92/82.14/83.16 | 4.86/5.63/5.70 | 82.25/95.27/96.45 | 2/3/3 |
| Nominal buffered / 1 | 169 | 138/157/163 | 70.41/80.10/83.16 | 4.83/5.49/5.70 | 81.66/92.90/96.45 | 2/2/3 |
| Nominal buffered / 2 | 169 | 145/161/162 | 73.98/82.14/82.65 | 5.07/5.63/5.67 | 85.80/95.27/95.86 | 2/3/3 |

Light improves from 34 correct opening outputs per target to 130–146, but has
52 native lost outputs versus 13, and only one final-five-second correct output
at 20 mm per target. Nominal deteriorates across all targets: 20 mm correct
counts 195/194/194 become 163/163/162, with 27 native losses versus zero.
Final-five-second 20 mm correct counts fall from 23 to three per target.

| Run | Accepted position peaks for targets 0/1/2, mm | Accepted rotation peak, degrees | Longest all-scheduled correct gap at 20 mm, s |
|---|---:|---:|---:|
| Light baseline | 784.7/743.2/828.4 | 60.72 | 38.42/38.42/38.42 |
| Light buffered | 71.2/65.3/77.5 | 5.29 | 4.03/4.03/4.03 |
| Nominal baseline | 19.2/20.4/20.4 | 2.41 | 0.47/0.77/0.57 |
| Nominal buffered | 58.8/58.9/61.3 | 5.80 | 3.40/3.40/3.40 |

Nominal accepted position peaks rise from about 19–20 mm to 59–61 mm, and
rotation from 2.41 to 5.80 degrees. All threshold-specific wrong-accepted counts,
finite-pose errors, censored gaps and times remain in `quality.json` and
`continuation-operational-detail.json`. These samples do not qualify safe contact.

The same initialization protocol yields actual seeds 1.2/2.017 seconds earlier
in the buffered light/nominal runs. Only 15/13 published input IDs are shared
between each complete pair. Frame substitution changes later tracking, accepted
keyframes and volume growth. Fixed-input identity rules out a demonstrated
buffer-arithmetic discrepancy; timed outcomes do not isolate the cause of every
quality difference. They are retained as operational outcomes, without retries
or treating improved light quality as an established buffer-induced gain.

## Decision and next bounded work

**Do not integrate the prototype. Keep the maintained baseline.** Memory and
fixed-input correctness pass, but the combined operational evidence fails the
predeclared benefit/quality requirement. The shorter nominal maximum coincides
with absent rebuilds and weaker tracking, while ordinary p95 latency is worse
and nominal quality declines across all targets. This is not a rejection for
failing 150 ms, nor proof that two-buffer arithmetic damages tracking. No
prototype-to-maintained transition is performed because adoption is rejected.
Both maintained extent logic and installed native SDF source remain byte-equal
to the preserved baseline.

The main recurring cost is ordinary per-input SAM3 followed by P2P: steady
medians are approximately 108–113 ms and 109–115 ms respectively, before queue
and publication age. Existing fixed-input profiles further separate TAPIR
(~38–39 ms) and registration (~53–66 ms inclusive); sparse rebuild savings do
not remove that recurrent service cost. The next bounded investigation should
address ordinary service and tracking under the resulting sampled ingress,
while keeping the current recipe and evidence as controls. No new variant is
implemented by this evaluation.

## Snapshot mechanism and preserved evidence

The candidate keeps two CPU TSDF/color pairs inside one serial CUDA rebuild.
Each synchronous native download targets the spare pair; only complete arrays
and metadata replace `obj.sdf`. Partial writes leave the previous completed
snapshot intact. Native scratch destinations are restored in `finally`, so
later bbox/mesh/direct reads cannot overwrite returned snapshots. The pool is
retired on return/refusal/error; later requests cannot reuse escaped arrays.
Native CUDA state, integration order and error/recovery policy remain unchanged.
The prototype uses evidence-only AST instrumentation, not maintained hooks.

All current results, fixtures, logs, monitors, scripts and the candidate live in
`outputs/evidence/b1/perception/tsdf-double-buffer-recovery-20261015/`, independent
of disposable worktrees. `complete-summary.json`, `operational-summary.json`,
`continuation-operational-detail.json`, per-run reports/quality/reset files and
`decision.json` own the complete result. The ignored TODO/evidence remain
uncommitted. Commits `cc920dd`/`806976e` own prior valid local/full milestones;
`a633202` owns the memory checkpoint and adoption criteria.

The original worktree removal lost raw local/interrupted-prefix payloads from
the earlier experiment; original source/session extracts and reports survive in
`recovery.json`/`original-tmp/`. Historical reported numbers do not substitute
for the fresh recovered-source gates. The original reported 388/426-frame
interruptions cannot be described as preserved raw measurements. The later
seven-cycle RustDesk-interrupted memory attempt is genuinely retained under
`failed-concurrent-compute/rustdesk-memory-attempt-1/`, and its valid light
baseline was reused. No completed valid experiment was repeated in this turn.
