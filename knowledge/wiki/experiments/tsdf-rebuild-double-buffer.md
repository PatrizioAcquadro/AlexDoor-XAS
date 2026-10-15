# TSDF Rebuild Double Buffer

2026-10-15. Branch `PatrizioAcquadro/b1-observed-a3a4`, baseline `99fa48e`.
The one authorized successor to [[tsdf-deferred-snapshot-audit|final-only
snapshot deferral]] retains a complete CPU download after every integration.
**Local CUDA correctness and actual worker reentry pass. Performance and full
comparisons are still pending at this checkpoint; production stays at baseline.**

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

## Preset performance and conditional follow-up

One excluded warmup per path, then three serial pairs ordered B/V, V/B, B/V
per target; every completed row is retained. One separate detailed baseline
profile is excluded. The same prior published snapshot is held by an audit
reference in both paths to verify its lifetime, including during timing; this
adds equal host residency relative to the previous experiment's local harness.
Consequently memory and total timing are compared within this new pair, not
subtracted from the historical 255.7/144.7 ms allocation/copy categories.
RSS is sampled every 5 ms during the target and process GPU usage about every
100 ms. Reconstruction and post-timer verification are excluded.

Full fresh fixed-input pairs open only if correctness passes and both target
timing ranges are disjoint with at least 10% lower median. Full numerical
identity then gates fresh timed-ingress comparisons with the same recordings,
selected recipe and frozen target scoring at 10/15/20 mm and 5 degrees.
No second optimization is authorized after failure. No 150 ms claim follows
from local rebuild savings.

## Evidence

`outputs/evidence/b1/perception/tsdf-double-buffer-20261015/` owns the candidate,
source baselines, protocol, original-fixture links, local arrays and JSON,
all logs, worker request/retained-state/query records and GPU samples.
`correctness-summary.json` and `worker-reentry-summary.json` own the completed
correctness boundary. The preceding `tsdf-snapshot-20261014/` evidence, recordings,
failed attempts and rejected implementation remain untouched. Ignored files,
including the active checklist, stay uncommitted.
