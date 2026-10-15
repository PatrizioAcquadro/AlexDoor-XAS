# TSDF Rebuild Double Buffer

2026-10-15. Branch `PatrizioAcquadro/b1-observed-a3a4`, baseline `99fa48e`.
The one authorized successor to [[tsdf-deferred-snapshot-audit|final-only
snapshot deferral]] retains a complete CPU download after every integration.
**Evaluation checkpoint: production remains at baseline. Fresh local gates and
both complete 3,217-input fixed pairs pass. Repeated-rebuild memory now passes
for both paths and conditions; the valid light baseline and interrupted attempt
are preserved. Four timed-ingress runs are in progress under the shared GPU lock.
Adoption awaits their combined latency, availability, quality and memory result.**

## Persistent recovery

The continuation starts in the primary checkout on
`PatrizioAcquadro/b1-observed-a3a4` at `b498d1f`, with no tracked changes. The
original candidate survived in `/tmp/tsdf_candidate.py` and was copied
byte-for-byte into
`outputs/evidence/b1/perception/tsdf-double-buffer-recovery-20261015/` alongside
original preparation/analysis scripts, independent copies of both captured
fixtures and unchanged baseline sources. This persistent directory has no
dependency on a disposable worktree.

The original candidate source is recovered; the runnable harnesses are
reconstructed from the original generators and the original session's explicit
half-download patch. `recovery.json` distinguishes them. Searches of the primary
checkout, surviving worktrees and desktop trash found no original local arrays,
worker records or interrupted replay payloads. The original session extract and
committed report preserve the two interruption reports at 388/426 frames; their
missing raw data cannot be recreated or claimed preserved. All newly produced
attempts remain in the recovery directory.

Both other active chats finished GPU work and agreed to use
`/tmp/rtx4090-exclusive.lock`. `validate_suite.py` holds that lock for the entire
local/fixed/timed suite, including gaps between processes; concurrent-compute
guards remain enabled. Because raw local evidence was missing, the reconstructed 24 local correctness
pairs, eight real-worker pairs and preset local benefit comparison had to pass
afresh. Those gates and full equivalence have now passed as recorded below.

## Fresh recovered-source local validation

The reconstructed harness now completes all 24 local CUDA pairs exactly,
including full-array comparisons, refusals, integration/download errors, actual
half downloads, bbox/mesh reads and later native integration/query behavior.
All eight real-worker pairs also match exactly; every next request executes 29
SDF queries. The 66 local CUDA cells and 16 workers have no observed foreign
compute in their retained monitors. These are new results, independent of the
missing original payloads.

The preset three timing pairs pass again, without favorable retries:

| Captured rebuild | Baseline median | Buffered median | Reduction |
|---|---:|---:|---:|
| Light, 17 integrations | 659.2 ms | 427.4 ms | 35.2% |
| Nominal, six integrations | 481.7 ms | 343.0 ms | 28.8% |

Both timing ranges are disjoint. Sampled host RSS peaks remain higher by about
one pair: 1708.7–1711.4 / 1835.4–1838.6 MiB light and 1883.8–1884.3 /
2091.5–2091.8 MiB nominal, baseline/buffered. Downloads, HtoD/DtoH byte/call
counts and CUDA allocation sequences are identical. Complete fixed-input identity
and repeated-rebuild memory pass below; timed quality/availability remains open
at this checkpoint.
`local-summary.json`, `correctness-summary.json`, `worker-reentry-summary.json`
and `local-evidence-audit.json` in the recovery directory own these results.

## Fresh complete fixed-input equivalence

Both baseline/buffered pairs complete all 3,217 chronological inputs with zero
differences. Every packet, registration decision/pose, reference/map/history
record and each of the 3,217 reference-array files per condition is compared.
Full TSDF/color/weight arrays, completed snapshots, metadata and signed queries
are checked at initialization, updates and their follow-up audit points:

| Condition | Registration rows | Reference-array files | Update rows | Volume/query audit rows | Differences |
|---|---:|---:|---:|---:|---:|
| Light | 3,217 | 3,217 | 17 | 33 | 0 |
| Nominal | 3,217 | 3,217 | 10 | 19 | 0 |

`exact-equivalence.json` owns this result. Full inputs complete under the same
continuous GPU lease without an interruption. This opens repeated-rebuild memory
and the four unchanged timed-ingress runs; operational adoption is still pending.

The complete fixed runs have no observed foreign compute in their retained
monitors. Later, `RustDesk` PID 1550141 opened a 24 MiB CUDA context during the
light buffered memory test. The unchanged guard stopped only this experiment,
after seven completed memory cycles. The completed 12-cycle light baseline,
all full fixed results and local gates are retained. Timed ingress had not
started. `failed-concurrent-compute/rustdesk-memory-attempt-1/` preserves the log,
GPU monitor, source versions and orchestration errors; no other process was
stopped. Future memory rows are journaled immediately to preserve a partial
repetition. Resume only after a new explicit exclusive window, using
`resume_exclusive.py`; do not rerun completed fixed pairs or the local gate.

## Single change and ownership

Two pairs of CPU arrays hold TSDF and color: one completed snapshot and one
preparation destination. The synchronous native `get_volume()` downloads into
the preparation pair. Only after both arrays and metadata are complete does one
assignment replace `obj.sdf`; the next integration prepares the other pair.
An incomplete or failed download never replaces the completed snapshot.

Reuse is limited to the inside of one serial rebuild, including its final
current-keyframe attempt. The existing pipeline/worker has no consumer between
these integrations. The pool is retired on return, refusal or exception. Its
last completed arrays can remain owned by `obj.sdf` or a caller; they are never
reused by the next request. Snapshots already delivered before the rebuild stay
independent, too. This does not create a general reusable-snapshot API or allow
concurrent readers of internal, unpublished iterations.

Registration and keyframe gates consume `obj.sdf`. Bbox and mesh paths instead
call native `get_volume()` and, for bbox, `get_weight_volume()`. Native CPU
scratch destinations are restored in `finally` after each redirected download;
later direct reads cannot overwrite an escaped completed snapshot. CUDA volume
state and weights remain live, including any partial state after a failed
integration, exactly as in the baseline. No rollback, retry or recovery-policy
change is introduced. The CUDA-only scope leaves native CPU fusion unchanged.

The experimental subclass is in the ignored evidence directory; it does not
edit the installed native source, setup, runtime or configuration. Model weights,
thresholds, geometry, reference selection, integration order, cadence, selected
recipe, policy, training and A3/A4 are unchanged.

## Correctness established before performance

The original captured inputs are reused unchanged: light at 67.733333 s has 17
ordered integrations over 186 x 304 x 294; nominal at 53.083333 s has six over
288 x 321 x 294. Each fresh CUDA process reconstructs and exactly verifies the
previous native grid before the target call.

All 24 paired local CUDA cases match in full arrays/digests, native signed
queries, validity, metadata, bounds, dimensions, weights, integration order,
history, counters, rebuilds, refusals and errors. Cases include success, refusal
of the current or an old keyframe, errors before/after CUDA integration #2,
before/after either download, actual half-array CUDA downloads followed by an
exception, and a color-read exception when the first pair is reused at #3.
The completed snapshot is also checked after actual bbox/colored-mesh reads,
and after the next native integration and SDF query. Pre-rebuild snapshots stay
stable. Post-timer target digests refer to the target state; fault-case full-array
exports also compare the native scratch state after that next integration.

Eight paired real-worker fault checks use three original raw acquisitions per
fresh worker. The second request executes normal inference, then invokes the
captured light target with the constructed error. The third request succeeds
and executes 29 native SDF queries in each of the 16 workers. Request results,
retained snapshots, native indices, counters and query values/validity match
exactly for errors before/after integration, before/after each download and
after each half-array download. These are controlled failure-boundary fixtures,
not a measured natural hardware-error rate or operational recovery qualification.

## Historical original local CUDA performance

One excluded warmup per path, then three serial pairs ordered B/V, V/B, B/V
per target; every completed row is retained. One separate detailed baseline
profile is excluded. The same prior published snapshot is held by an audit
reference in both paths to verify its lifetime, including during timing; this
adds equal host residency relative to the previous experiment's local harness.
Memory and total timing are compared within this new pair, not subtracted from
historical measurements. Reconstruction and post-timer verification are excluded.

| Target | Baseline ms, all three | Buffered ms, all three | Median baseline -> buffered | Reduction |
|---|---:|---:|---:|---:|
| Light | 654.1, 650.9, 662.9 | 411.7, 420.7, 415.4 | 654.1 -> 415.4 | 36.5% |
| Nominal | 452.3, 449.5, 465.0 | 356.1, 339.0, 339.2 | 452.3 -> 339.2 | 25.0% |

Median paired savings are 242.4/110.5 ms light/nominal. Every buffered event is
below every corresponding target baseline event. Three repetitions establish a
clear local effect in these targets, not an extreme percentile or population CI.
The separate baseline profile measures array allocation/copies at 251.7/145.0 ms
and downloads at 132.2/76.1 ms. Savings are measured directly; they do not equal
the unsafe final-only deferral's removed-copy-plus-download savings.

| Rebuild operation | Light baseline -> buffered | Nominal baseline -> buffered |
|---|---:|---:|
| Full TSDF/color downloads | 17 -> 17 | 6 -> 6 |
| DtoH calls / bytes | 153 / 2,339,193,812, unchanged | 54 / 1,332,275,064, unchanged |
| CUDA allocations | 122 -> 122 | 45 -> 45 |
| Full snapshot array allocations | 34 -> 4 | 12 -> 4 |
| Cumulative full snapshot bytes allocated | 2,260,855,296 -> 265,982,976 | 1,304,626,176 -> 434,875,392 |
| Full snapshot array copies / payload bytes | 34 / 2,260,855,296 -> 0 | 12 / 1,304,626,176 -> 0 |

HtoD calls/bytes and CUDA allocation-size sequences match exactly too. Metadata
copies and native CPU volume workspaces remain; counts above concern full TSDF
and color arrays, not all NumPy allocations or physical memory-bus transactions.

| Timed sampled peak | Light baseline / buffered | Nominal baseline / buffered |
|---|---:|---:|
| Host RSS, MiB | 1708.6–1710.9 / 1835.7–1838.3 | 1883.6–1884.1 / 2091.5–2091.7 |
| Driver process GPU, MiB | 976 / 976 | 1088 / 1088 |

RSS is sampled every 5 ms during the target and process GPU usage about every
100 ms. The pool increases the transient RSS peak by approximately one pair,
127/208 MiB; it does not demonstrate a memory reduction. The pool is retired at
exit, without persistent CUDA allocations. These isolated cells exclude full
SAM3/TAPIR provider residency. Sampling does not qualify startup spikes.

## Repeated-rebuild memory

The already completed light baseline is reused. Only the missing light buffered,
nominal buffered and nominal baseline cells run in this continuation. Each has
12 identical rebuilds; cycle zero is warmup and the final eight assess stability.
No allocator trimming or forced garbage collection is used. Full output digests
remain identical within each cell and between baseline/buffered paths.

| Condition/path | Peak RSS, MiB | Retired RSS range over final eight cycles, MiB |
|---|---:|---:|
| Light baseline, preserved | 1713.0 | 5.8 |
| Light buffered | 1840.0 | 6.3 |
| Nominal baseline | 1893.0 | 10.5 |
| Nominal buffered | 2105.6 | 5.4 |

Exactly two pool arrays remain alive while the completed buffered snapshot is
owned by `obj.sdf`; after object retirement all pool weak references are dead in
every cycle. Thus retained process RSS does not mean retained snapshot objects.
It includes fixtures, runtime and allocator residency; the experiment does not
attribute all remaining RSS to the allocator. The buffered cells' global CUDA
free-memory tail varies by 0/0.75 MiB light/nominal, without progressive volume
retention. Baseline global free-memory fluctuations include graphical desktop
allocations and are not a count of this process's live CUDA objects.

The added transient CPU pair costs approximately 127/213 MiB in these repetition
cells, within this workstation's available memory. This is bounded reuse, not a
memory-reduction claim. `memory-gate-passed.json` owns the completed gate.

## Adoption criteria, declared before remaining tests

`adoption-criteria-continuation.md` in the evidence directory records the criteria
before any new memory or timed run. Complete fixed-input identity is already
established and is not repeated. Memory must stabilize with zero live pool arrays
after retirement. Require at least a 5% improvement in at least one steady primary
operational outcome: p95 service, p95 publication age, publication frequency or
maximum publication interval, with an explanation consistent with update history.
Assess both conditions together; a broad latency/availability regression defeats
adoption. This practical effect size is not a confidence interval from one pair.

Quality uses the same three frozen material targets at 10/15/20 mm and 5 degrees,
for processed outputs and every scheduled opening input, with missing/lost/wrong
accepted outputs retained. Adoption requires useful correct output at 15/20 mm
without broad deterioration across the targets or benefit obtained by discarding
useful tracking. Inspect precision, terminal five seconds, continuity and native
losses together. Timed histories may differ; numerical identity is not required
here. Report the tighter 10 mm behavior and all failures as well.

The order is light baseline, light buffered, nominal buffered, nominal baseline,
once each without favorable repeats. Source timing, initialization, recipe,
models, geometry, thresholds and control stay fixed. This optimization alone is
not required to reach 150 ms. If adopted, its transition from the prototype must
preserve completed snapshots, scratch restoration and request-local ownership,
with focused tests and bounded real-CUDA comparison.

## Interruptions and preserved history

The original experiment reported full light baseline interruptions at 388/426
frames from foreign `squadbot-av-phase1` work. Their raw payloads disappeared
with the old worktree; the original session extract and Git report retain the
reports, not substitute measurements. Those prefixes do not establish a complete
comparison. The old local numbers above are historical; fresh recovered-source
gates and full exact comparisons are the authoritative current evidence.

After fresh full equivalence, RustDesk opened a CUDA context during the light
buffered memory cell. The guard stopped only this experiment after seven cycles.
`failed-concurrent-compute/rustdesk-memory-attempt-1/` retains that attempt and
source versions; its valid light baseline is reused. No other process is stopped.
The current continuation holds `/tmp/rtx4090-exclusive.lock`, requires 30 seconds
without compute contexts before each cell, and monitors foreign compute during
each process. Graphical desktop contexts are present, as in the original protocol.
Interference is retained and only incomplete work resumes after waiting.

## Decision checkpoint

The memory gate passes and the unchanged four timed runs are in progress.
Production remains at baseline pending the complete operational comparison.
No additional optimization or model/control change is authorized. General
tracking/contact qualification is separate from adoption of this implementation.

## Evidence

The historical `outputs/evidence/b1/perception/tsdf-double-buffer-20261015/` owned the candidate,
source baselines, protocol, original-fixture links, local arrays and JSON,
all logs, worker request/retained-state/query records and GPU samples.
`correctness-summary.json` and `worker-reentry-summary.json` own the completed
correctness boundary. The preceding `tsdf-snapshot-20261014/` evidence, recordings,
failed attempts and rejected implementation remain untouched. Ignored files,
including the active checklist, stay uncommitted. That original directory is now
missing; use the recovery directory above for currently available sources and
new validation, and treat original measurements above as historical reported
results. Fresh recovery local gates, full exact pairs and memory establish the
current completed boundaries; timed comparison remains in progress.
