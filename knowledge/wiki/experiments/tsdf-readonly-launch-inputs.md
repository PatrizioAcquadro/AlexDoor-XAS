# TSDF Read-Only Launch Inputs

2026-10-14. Baseline `9b94978`; implementation `ae3bcec` on
`PatrizioAcquadro/b1-observed-a3a4`. The clean existing worktree was reused.
**Rejected for adoption as a rebuild-tail optimization; active code is restored to
the baseline.** The transfer reduction is exact, but the measured event/service
benefit is too small and inconsistent, and timed-ingress quality is mixed.
One candidate only, following [[p2p-sam3-cost-diagnosis|the cost diagnosis]].
Evidence, executable probes, all rows and failures are preserved locally under
`outputs/evidence/b1/perception/tsdf-readonly-20261014/` in this worktree, with
a verified preserved copy at the same relative path in
`/home/pacquadr/Desktop/AlexDoor-XAS/`.

## Change and consumer boundary

The pinned CUDA fusion kernel writes only TSDF, weight and color volumes. Seven
other arguments are read-only: dimensions, origin, intrinsics, pose, integration
parameters, packed color image and depth image. Their `astype`/`asarray` call
arguments are private temporary host buffers. Six post-call readbacks have no
consumer. The candidate changes their PyCUDA handlers from `InOut` to `In`.
Dimensions retain the original 12-byte `InOut`: PyCUDA still synchronizes the
context before reading it, inside every kernel call. This preserves input-buffer
lifetimes, same-call completion and the kernel-error boundary before
`sdf_num_integrated` or `obj.sdf` can be updated. No asynchronous deferral or new
scheduler is introduced.

`SDFBuilder.integrate_keyframe` still downloads TSDF/color after each integration
and copies them into `obj.sdf`. Registration queries consume that snapshot;
bounding-box estimation can request weights, and mesh exports can request color.
The successful rebuild loop has no consumer of its intermediate snapshots, but
those snapshots also establish the last completed state if a later integration
fails. They are retained. No host-volume construction, snapshot, filter, integration,
allocation guard or rebuild decision is removed.

At `ae3bcec`, the maintained `patch_tracking` path applies the candidate in both
setup and worker startup, before upstream imports. It validates all edits before
writing, preserves originals and rejects unexpected source. To reproduce the
candidate, use that commit in an isolated worktree and its maintained setup path
with a fresh model source root. The final branch restores both tracked code/tests
and the local upstream copy to baseline; the candidate remains recoverable from
Git. Models, BF16/TF32, SAM3, TAPIR, SuperPoint, references, draws,
candidates, thresholds, SDF calculations and scheduler remain unchanged.

## Fixed-input CUDA comparison

Fresh RTX 4090 workers replay the original 3,217 consecutive native captures per
condition, 25.016667–78.616667 s, with original saved SAM3 masks. Full chronology
reconstructs the mature keyframe/rebuild state; no unavailable checkpoint is
assumed. Python, NumPy and torch CPU/CUDA seeds are zero; the existing native
RANSAC seed and inspection frame-seeded RNG remain unchanged. Run order is
light baseline→variant, nominal variant→baseline. There is no simultaneous measured
GPU job and no renderer.

All 3,217 trace lines per condition are byte-identical, including references,
IDs, pending/promoted histories, candidates, scores, SDF decisions, losses and
published supports. Complete native output-packet array bytes also match on every
capture. Full TSDF/color/weight bytes, bounds, dimensions, origins, integration
counts and native signed query outputs match on every update and its next capture:
33 light and 19 nominal checks, including seed and consecutive-update overlaps.
Integration histories/counts and HtoD calls/bytes are identical. All subsequent
pose/decision/reference frames remain covered. `exact-equivalence.json` has zero
differences; hashes are used only to compare large diagnostic buffers.

The real CUDA regression additionally compares single/multiple launches, nonunit
weights, all three volumes and unchanged caller buffers with the pre-patch class.
It checks synchronous failure propagation and the existing expansion/memory guard.
The 25 native-patch tests cover fresh-source application, backups, idempotence,
unexpected-source refusal and existing numerical fixes. Lint passes.

Host timers add no CUDA fences or intrusive timelines. Verification readbacks and
packet/volume digests happen after the native timer and are subtracted from IPC
wall time. They are excluded from transfer counts. Memory is sampled, not an
allocation-lifetime proof: native GPU residency at 100 ms, full-provider residency
at 250 ms, torch allocation/peaks per request, and host RSS after verification.
One run per cell retains every stall; small differences are not precision claims.

Steady means seed +5 s: **2,917 frames each**, with **16/9 TSDF update frames**
and **53/21 integrations** light/nominal. The all-steady TSDF p50/p95 are zero in
both variants. `fixed-summary.json` separately retains all 3,217 frames, seed,
warmup, steady, conditional updates and nonupdates. Triplets are p50/p95/max ms.

| Measured service | Light baseline → variant | Nominal baseline → variant |
|---|---:|---:|
| Inclusive TSDF, conditional update frames | 54.4/564.7/635.3 → 54.6/562.0/653.7 | 70.2/405.0/428.4 → 70.7/403.2/428.6 |
| Complete native service, all steady frames | 112.0/145.5/735.2 → 112.7/150.7/752.9 | 97.3/129.2/548.4 → 98.2/139.5/546.7 |
| Native including IPC, all steady frames | 116.6/150.2/740.1 → 117.3/156.0/758.0 | 102.2/133.8/553.2 → 102.9/144.6/551.5 |

These native services exclude SAM3. They are not complete-provider latency.
Conditional mean TSDF saves 0.42 ms light and **costs 1.77 ms more** nominal;
all-steady native means regress 1.58/2.38 ms. The largest light TSDF event regresses
18.31 ms. No favorable retry replaces these results.

Each ordinary 600×960 launch removes six DtoH calls and **4,608,136 bytes**.
Actual steady transfer accounting:

| Quantity | Light baseline → variant | Nominal baseline → variant |
|---|---:|---:|
| DtoH calls | 477 → 159 | 189 → 63 |
| DtoH bytes | 6,523,346,372 → 6,279,115,164 | 3,727,736,004 → 3,630,965,148 |
| HtoD calls / bytes, unchanged | 383 / 938,950,808 | 156 / 841,404,432 |

The deterministic savings are 244,231,208 / 96,770,856 bytes, only 3.74/2.60% of
steady TSDF readback. Host casts/copies, GPU argument allocations and persistent
volumes remain. Sampled native driver peaks remain 2.662/6.705 GiB light/nominal;
torch peaks remain 0.954/0.983 GiB. Post-verification host RSS maxima are
2.728→2.723 / 2.778→2.800 GiB, with no supported memory improvement.

## Residual profile

At light 67.733333 s, the candidate still performs 17 integrations over a
186×304×294 volume. Inclusive TSDF 653.7 ms comprises allocation 70.7 ms, volume
readback 131.6 ms, fusion calls 41.4 ms and remaining TSDF host work 410.0 ms.
At nominal 53.083333 s, six integrations over 288×321×294 cost 428.6 ms:
111.5/74.2/13.7/229.2 ms in the same categories. These categories are additive
within those events. Fusion-call time includes host launch work; it is not pure
kernel time. The removed argument readbacks save only about 2.5/1.9 ms within
those fusion-call sums; the full events do not improve.

The restored baseline has the same dominant work at those events: TSDF
635.3/428.4 ms, comprising allocation 70.5/112.0, volume readback 130.3/73.7,
fusion calls 43.8/15.6 and remaining TSDF host work 390.7/227.1 ms light/nominal.

On all-steady candidate frames, ordinary host medians light/nominal remain:
TAPIR 36.25/36.21 ms; RANSAC 23.65/22.65; dense SDF scoring 18.24/11.53;
registration preparation/gates 10.04/9.27; SDF refinement 6.36/6.17; direct trace
finish 6.88/6.55. Quantiles of different stages are not additive. Historical
TAPIR/whole-service timing differences are not attributed to this TSDF change.

## Timed ingress

The fixed-input gate passed before operational runs started. The operational pair
uses the unchanged complete provider, real SAM3 and native P2P, the same 4,718
captures per run at 60 Hz, fresh workers/state/seeds and the original selection
request. It preserves actual retained frames, latest-slot replacements, failures,
reference histories and all three frozen evaluator targets. Truth is used only in
post-run evaluation. Run order is light baseline→variant,
nominal variant→baseline. Each run starts from fresh acquisition-time state;
actual retained histories are allowed to change under the original scheduler.

| Operational quantity | Light baseline → variant | Nominal baseline → variant |
|---|---:|---:|
| Actual seed capture, s | 29.0667 → 27.2167 | 25.2167 → 25.2167 |
| Publications / replaced inputs | 246/4472 → 245/4473 | 250/4468 → 257/4461 |
| Steady publications / effective Hz | 200/4.48 → 203/4.37 | 209/4.29 → 216/4.44 |
| Complete service p50/p95/max, ms | 208/269/449 → 218/271/437 | 218/280/436 → 213/270/945 |
| Capture-to-publication age p50/p95/max, ms | 229/293/479 → 244/294/462 | 244/309/459 → 229/293/960 |
| Native loss / nonseed outputs | 64/211 → 40/216 | 42/222 → 55/229 |
| Accepted nonseed outputs | 147 → 176 | 180 → 174 |
| Joint sampled GPU / RSS peak, GiB | 7.725/5.661 → 7.734/5.673 | 9.617/5.631 → 9.037/5.587 |

All offered inputs reconcile: publications + replacements = 4,718 in every run.
Only 17 light /46 nominal capture IDs are shared between the two published
sequences; 229/228 light and 204/211 nominal IDs occur in only baseline/variant.
All four process/selection/publication/queue resets pass, with zero memory-sampler
errors. **Zero of 828 steady publications meet 150 ms.** Startup memory spikes,
producer contention, material recovery and hardware/contact remain unqualified.

Nominal's 945 ms service stall at source 50.2 s is predominantly **SAM3 835 ms**,
with native P2P 110 ms. It is retained and is not attributed to TSDF. The paired
native-only histories are exact; these timed histories are not paired native
inputs. Light seed times and seed masks differ (2,000 pixels). Nominal has the
same seed capture and serialized seed trace, but regenerated seed masks still
differ by 185 pixels. Thus equal seed traces do not establish identical full state.
Differences before the first TSDF update and ordinary wall-clock variation confound
attribution of operational quality/throughput to this small transfer change.
One run per cell measures these outcomes, not a causal population effect.

Quality below uses the original three frozen physical targets at the original
31 s evaluator reference, transported to each actual seed using evaluator motion.
Position bounds remain 10/15/20 mm, with the unchanged 5° rotation bound. Missing,
lost and refused outputs remain unavailable. Accepted-pose errors are conditional
and are reported separately from coverage.

| Primary target, opening ≥31 s | Light baseline → variant | Nominal baseline → variant |
|---|---:|---:|
| Correct / processed at 10/15/20 mm, % | 52.13/64.93/67.30 → 78.95/78.95/79.90 | 75.61/79.02/79.51 → 58.96/69.34/72.17 |
| Correct / all 2,858 scheduled at 10/15/20 mm, % | 3.85/4.79/4.97 → 5.77/5.77/5.84 | 5.42/5.67/5.70 → 4.37/5.14/5.35 |
| Correct / accepted at 10/15/20 mm, % | 74.83/93.20/96.60 → 97.63/97.63/98.82 | 95.09/99.39/100 → 79.62/93.63/97.45 |
| Accepted position p50/p95/max, mm | 7.32/16.75/33.34 → 2.58/7.33/23.21 | 5.45/9.73/15.23 → 7.11/15.26/45.10 |
| Accepted rotation p95/max, degrees | 1.39/5.87 → 0.91/2.41 | 0.82/2.69 → 2.57/4.98 |
| Longest observed 15 mm correct-pose gap, s | 6.40 → 4.38 | 6.65 → 8.32 |
| Correct last-five-seconds outputs at 15 mm | 0/28 → 2/28 | 0/29 → 0/30 |

All three targets are retained in `operational-summary.json` and each run's
`quality.json`, including every threshold, conditional precision, errors, peaks,
gaps and terminal tails. Across the three targets, processed 10 mm correctness is
51.66–52.13→78.47–78.95% light and 74.63–75.61→54.25–61.79% nominal. Processed
15 mm correctness is 64.45–64.93→78.95–79.43% and 79.02–79.51→67.92–70.28%.
The worst nominal 15 mm gap grows to 9.35 s on target 1. These quality regressions
are preserved, without asserting a same-input numerical regression.

## Decision, failures and next discussion

Reject this candidate for the requested rebuild-tail objective. Its six removed
readbacks are demonstrably redundant and exactly preserve fixed-input behavior,
but save only 2.60–3.74% of TSDF DtoH bytes, do not reduce persistent allocation,
and do not provide a convincing event or end-to-end benefit. Light's largest
rebuild and steady native p95 regress; nominal's largest TSDF event is unchanged.
Timed light throughput regresses; nominal throughput improves with worse losses,
accuracy and a retained SAM3 stall. Those mixed, history-dependent results do not
support operational adoption or 150 ms availability. They do not prove that input-only
handlers inherently damage tracking.

The implementation remains in `ae3bcec`, reproducible through setup/patch. Final
tracked code/tests and the private upstream copy are restored to `9b94978` behavior.
The shared checkout/models and earlier evidence are untouched. No second
optimization, policy change, numerical tolerance relaxation or favorable timing
retry is attempted.

One failed probe is retained as `light-baseline.failed-packet-audit`: a diagnostic
JSON exporter rejected binary data after the seed, before any timing row was
written. Only the exporter was corrected and the fresh baseline restarted.
Every completed-run outlier, refusal, lost tail and regression remains. Ordinary
profiling uncertainty and the independent SAM3 stall are unresolved.

Discuss the residual cost before authorizing another candidate: final and
intermediate host snapshot construction, repeated full-volume downloads and
allocation/filter work dominate mature rebuilds. Any removal must explicitly
preserve partially completed failure state as well as the final same-frame query
state. Ordinary SAM3/TAPIR/CPU registration and diagnostic stalls remain separate
costs. This record proposes that discussion; it implements no additional change.
