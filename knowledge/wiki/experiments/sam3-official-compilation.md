# Official SAM3 Compilation

2026-10-15; baseline `85b5719` on `PatrizioAcquadro/b1-observed-a3a4`, initially clean.
**Reject for the current selected pipeline; maintained SAM3 eager source and the
TSDF baseline are restored.** Candidate `7d864fe` and its patch remain preserved.
Compilation works and reduces ordinary SAM3 cost, but target accuracy/availability
and native loss regress at identical inputs and native initialization. Internal
arithmetic bit equality was never an acceptance requirement.

## One intervention and preset gates

The installed SAM3 video builder exposes `compile=True`; its
[official component setup](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_video_inference.py)
is reached by the causal base-class one-frame propagation after an eager seed.
The candidate passes only that flag and makes compiler errors explicit instead
of accepting the setup's silent eager fallback. Component choices/modes remain
unchanged. No runtime upgrade or alternate compiler recipe was attempted.

Models, weights, BF16/TF32, 1008-pixel model resolution, 600×960 inputs, prompt,
thresholds, persistent identity/session, reconditioning OFF, bounded history,
TAPIR, SuperPoint and P2P are unchanged. The
[[tsdf-rebuild-double-buffer|TSDF baseline decision]] remains in force. No policy,
controller, training or contact change occurred.

Persistent primary-checkout evidence is
`outputs/evidence/b1/perception/sam3-compile-20261015/`.
`criteria.md` predates CUDA acquisition: 20-minute compatibility/cell limit,
90-minute pilot budget excluding shared-GPU waits; conditional full fixed/timed
budgets 100/10 minutes. Pilot performance requires ≥15% median and ≥10% p95 SAM3
improvement in both sparse conditions. Masks use IoU ≥0.995 on ≥99% of pairs and
minimum ≥0.98, with unchanged IDs/loss decisions and downstream assessment of
exceptions. Pose sensitivity is 1 mm / 0.1° p95; >1 percentage-point loss of correct
15/20 mm availability, persistent loss and continuity deterioration defeat adoption.
These are practical pilot effects, not statistical confidence claims.

Warmup runs 33 repeats of only the first arrived RGB in disposable causal state,
with the original prompt/thresholds. It restores Python/NumPy/Torch RNG and drops
state, timestamps and IDs before evaluation. No future image or truth is supplied.
Eager receives the same warmup. Its 121-frame light output exactly matches an
independent historical cold prefix; exact matching is an isolation check, not a
compiler arithmetic gate.

## CUDA compatibility, startup and memory

Eight valid SAM sessions cover dense 121-frame prefixes and complete saved sparse
histories of 234 light / 239 nominal calls. Both variants receive identical frame
IDs, timestamps and RGB. Sparse seeds remain 25.016667 / 26.650000 s for native P2P.
Order is light eager→compiled and nominal compiled→eager for each history.

All seven official components execute successfully. Each compiled session reaches
nine Inductor graphs during warmup and retains exactly nine throughout evaluated
inference: no new evaluated graphs, graph-limit fallback or compiler failure.
This validates this causal path and these histories, not arbitrary future shapes.

Model/process startup is about 7 s and remains separate from first inference.
First cold compiled warmup costs **179.6 s**; later cached warmups cost
**33.9–48.2 s**, versus about **5 s** eager. Evaluated initial calls and warmup
journals are separate from steady samples; cache speedups are not model speedups.
Evaluated CUDA allocation stabilizes near 3.66 GiB. SAM process device peaks are
about 5.10 GiB eager and 5.32–5.35 GiB compiled. Compiled worker RSS is about
3.1–3.6 GiB versus 2.2–2.3 GiB eager, with bounded allocator/cache high water in
these short histories. This does not qualify indefinite runtime memory.

## Paired frontend cost and masks

Post-first-ten IPC/decode-inclusive SAM3 times, **p50 / p95 / maximum ms**:

| History | Eager | Compiled | Median reduction |
|---|---:|---:|---:|
| Dense light | 109.0 / 116.1 / 902.4 | 98.6 / 113.4 / 1339.1 | 9.5% |
| Dense nominal | 111.6 / 120.6 / 944.0 | 95.0 / 101.5 / 933.3 | 14.9% |
| Sparse light | 115.5 / 121.5 / 934.3 | 93.3 / 97.0 / 959.0 | 19.3% |
| Sparse nominal | 110.7 / 117.1 / 139.1 | 92.2 / 96.6 / 99.3 | 16.7% |

Both sparse performance gates pass; p95 reductions are 20.2% / 17.5%. Peaks remain:
large light/dense stalls have no new graphs or foreign compute in the valid cells;
their internal cause is unisolated. No favorable repetition replaces them.

All SAM IDs, seed IDs and semantic loss decisions match. Dense minimum mask IoU
is 0.99883 / 0.99898. Sparse minimum is **0.92016 / 0.90259**, and only
**97.44% / 91.63%** of pairs reach 0.995. Light at 44.0333 s differs by 28,870 pixels.
The saved overlay shows a substantial region near the floor/panel boundary; a
mask difference alone is not pixel-truth evidence of worse segmentation. Actual
native output/decision checks below determine relevance. Material identity is
not qualified merely by retaining numerical SAM IDs.

## Downstream poses at fixed inputs and identical initialization

Four natural-mask native sessions retain the entire compiler-output effect,
including its seed mask. Their seed masks differ slightly (370 light pixels), so
those histories are retained separately. Before further CUDA,
`common-seed-criteria.md` declares the required identical-initialization control:
reuse both completed eager baselines and run only two compiled-mask native
sessions with the exact baseline initial mask. All later compiled masks remain
unchanged. No SAM rerun, model/precision/threshold change or favorable retry occurs.

Both controls have **exact initial source masks, external-initial masks and seed
registration traces**, matching RGB-D, original captures and seed=0 in Python,
NumPy and Torch. They process 200 / 212 native inputs per path. Truth only scores
completed results against the same three frozen physical targets. The table
uses the 180 light / 204 nominal **processed opening captures**, at **5°**; every
cell lists correct counts at **10 / 15 / 20 mm**.

| Condition / target | Eager | Compiled, common native seed |
|---|---:|---:|
| Light 0 | 147 / 152 / 153 | 140 / 148 / 148 |
| Light 1 | 149 / 152 / 153 | 139 / 147 / 148 |
| Light 2 | 147 / 153 / 153 | 140 / 148 / 148 |
| Nominal 0 | 151 / 153 / 154 | 37 / 38 / 38 |
| Nominal 1 | 147 / 153 / 153 | 37 / 38 / 38 |
| Nominal 2 | 152 / 154 / 154 | 37 / 38 / 38 |

Across all nonseed native outputs, losses are **27→30 light, 47→130 nominal**;
accepted outputs are 172→167 / 163→77. Primary correct-20-mm maximum gaps increase
**2.87→3.27 s / 3.68→36.58 s**. Among outputs accepted in both paths, target-delta
p95 is **7.29–8.78 mm, 0.684°** light and **182.7–202.5 mm, 15.31°** nominal.
All-finite deltas, including frozen lost poses, are kept separately and are not
presented as accepted-pose accuracy. Precision, wrong acceptance, all thresholds,
chronological gaps and terminal tails remain in the quality JSON.

Natural-mask histories also regress: native losses 27→34 / 47→134; correct-20-mm
counts become 146 for all light targets and 38 for all nominal targets. They are
not replaced by the common-seed control. The control establishes that the adverse
result is not solely a changed native seed. It does not establish a general SAM3
compiler defect or tracking/material qualification beyond these saved histories.

## Decision, residual cost and operational limit

The quality gate fails substantially even with identical native initialization.
**Reject this variant for the selected pipeline.** Active frontend/worker files
are restored byte-for-byte to `85b5719`; the candidate remains in Git and
`candidate.patch`. No second variant or other optimization is opened.

The saved-data pilot supports about **18–23 ms** ordinary SAM3 savings. A matched-
frame counterfactual sum of isolated SAM IPC plus the **same baseline native IPC**
gives typical sums **227.5→204.3 ms light / 217.4→199.3 ms nominal**. These exclude
geometry/binding and joint-residency effects: they are an estimate of the SAM-only
contribution, **not measured integrated pipeline latency or percentiles**. Actual
complete benefit is unvalidated; substantial residual work remains. Reaching
150 ms was not an adoption requirement.

The preset gate therefore closes the full 4,718-input fixed comparisons and then
timed ingress runs. No new complete publication-age/frequency/replacement result
is claimed, and different timed histories are not used for causal quality claims.
Prior TSDF and operational results remain intact. No qualified tracking, A3/A4,
policy or contact result follows from compiler compatibility.

Both shared locks are used with idle preflights and foreign-compute guards.
Two sparse attempts, prefixes 109 eager / 99 compiled, are preserved after foreign
CUDA appeared; only our process groups were stopped. Completed valid cells were
reused. Remaining work finished without further interference; all our workers
exited and leases were released. All sources, masks, timings, memory/counters,
initialization audits, failed prefixes, target outputs and executable harnesses
remain persistent. `decision.json`, `compiler-audit.json`, `pilot-summary.json`,
`pose-summary-commonseed.json` and `residual-cost-estimate.json` are entry points.
