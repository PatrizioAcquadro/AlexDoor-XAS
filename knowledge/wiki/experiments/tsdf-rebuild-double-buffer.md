# TSDF Rebuild Double Buffer

2026-10-15. Branch `PatrizioAcquadro/b1-observed-a3a4`, baseline `99fa48e`.
The one authorized successor to [[tsdf-deferred-snapshot-audit|final-only
snapshot deferral]] retains a complete CPU download after every integration.
**Recovery checkpoint: production remains at baseline. Original local success is
reported below, but its raw payloads disappeared with worktree `782a`. Recovered
sources do not inherit that validation. Fresh local and complete comparisons
are running under a coordinated exclusive GPU lease.**

## Persistent recovery

The clean primary checkout is now on `PatrizioAcquadro/b1-observed-a3a4` at
`f94be75`; `db36` is detached at `99fa48e`, and `782a/AlexDoor-XAS` is absent.
The latest three commits contain documentation only. The original candidate
survived in `/tmp/tsdf_candidate.py`, copied byte-for-byte into
`outputs/evidence/b1/perception/tsdf-double-buffer-recovery-20261015/` alongside
the original preparation/analysis scripts, independent copies of both captured
fixtures and the unchanged baseline sources. This directory belongs to the
primary checkout and has no dependency on a disposable worktree.

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
guards remain enabled. Because raw local evidence is missing, the reconstructed
24 local correctness pairs, eight real-worker pairs and preset local benefit
comparison must pass afresh before full equivalence and timed ingress.

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

## Preset local CUDA performance

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

## Conditional complete comparisons

Both local gates pass: exact correctness and disjoint ranges with at least 10%
lower median. Fresh complete fixed-input pairs are authorized but incomplete.
Full numerical identity then gates fresh timed-ingress comparisons with the
same recordings, selected recipe and frozen target scoring at 10/15/20 mm and
5 degrees. No second optimization runs after failure. Even the isolated median
rebuilds remain 415.4/339.2 ms; local savings do not establish 150 ms availability.

## Complete-comparison interruptions

Foreign `squadbot-av-phase1` CUDA work interrupted the first complete light
baseline after 388 frames. The guard terminated only this experiment's processes
and retained the attempt. After 30 continuously idle seconds, a fresh baseline
was started; foreign compute returned and stopped it after 426 frames. Both
partial packet/registration/volume/query histories exactly match the preserved
historical baseline. Neither reaches the mature target or completes a full pair;
their times are excluded from performance conclusions.

All 66 completed local CUDA cells and all 16 real-worker checks had no observed
foreign compute. Their results remain valid and are not repeated. The two
aborted full prefixes remain under `failed-concurrent-compute/attempt-1/` and
`attempt-2/`. No complete fixed-input comparison or timed-ingress run has
completed. Effective-frequency quality, publication ages, losses and 150 ms
availability therefore remain unvalidated for this candidate. Await an exclusive
GPU window; do not relax the guard, change recovery or add another variant.
A subsequent three-minute idle-window preflight observed further foreign jobs
around 6 GiB and never launched a third replay. That preflight was stopped;
no unattended process or scheduled continuation is left running.

## Decision and concrete remaining work

**Retain the active baseline while preserving the locally validated candidate.**
The single two-buffer alternative fixes the previous completed-state failure
and demonstrates substantial local CUDA savings with unchanged downloads.
It trades approximately one extra transient host snapshot pair for fewer
allocations and copies. It is not adopted into production because full numerical
and effective-frequency validation have not completed, not because a quality
regression has been measured. Adopted production savings are currently zero.

After an exclusive GPU window is available, complete the fresh 3,217-input
baseline/buffered comparison in both conditions with `run_full.py`. Only exact
full histories/volumes/queries open the unchanged four timed-ingress runs and
frozen three-target quality evaluation at 10/15/20 mm with 5 degrees, together
with service/position/rotation peaks, publication ages, replacements, native
losses and memory. Preserve the two interrupted prefixes; do not rerun the
completed local replicas. No other implementation variant is authorized.

## Evidence

The historical `outputs/evidence/b1/perception/tsdf-double-buffer-20261015/` owned the candidate,
source baselines, protocol, original-fixture links, local arrays and JSON,
all logs, worker request/retained-state/query records and GPU samples.
`correctness-summary.json` and `worker-reentry-summary.json` own the completed
correctness boundary. The preceding `tsdf-snapshot-20261014/` evidence, recordings,
failed attempts and rejected implementation remain untouched. Ignored files,
including the active checklist, stay uncommitted. That original directory is now
missing; use the recovery directory above for currently available sources and
new validation, and treat the following original measurements as historical
reported results until reproduced.
