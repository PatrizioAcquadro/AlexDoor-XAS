# Selected SAM3 + Point2Pose Cost Diagnosis

2026-10-14; baseline `main @6b84b04`, initially clean and the only local branch.
Diagnosis only. `configs/point2pose_selected.json`, initialization, models,
thresholds, references, candidate selection and policies are unchanged. All
recordings, baselines and failed attempts remain. Local observational scripts and
measurements are under `outputs/evidence/b1/perception/sam3-p2p-cost-20261014/`.

Follow-up: [[tsdf-readonly-launch-inputs|one read-only TSDF argument-readback candidate]]
is implemented and exactly evaluated, then rejected for adoption after small
transfer savings, inconsistent cost and mixed timed-ingress quality. Active code
is restored to this baseline; that record owns the new residual profile.

**The 150 ms target already fails on service cost.** SAM3 costs about 107 ms,
TAPIR about 44 ms, and registration/scoring/refinement adds substantial CPU work.
Rebuilds dominate dense replay extremes; operational stalls have additional
unresolved causes. Removing diagnostic logging alone cannot close the gap.

## Evidence and measurement boundaries

Reused evidence: `persistent-sam3-20261008/full-selected-{light,nominal}`
(4,718 captures each, 0–78.6167 s, seed 25.016667 s) and
`sam3-operational-20261008/{recorded,live-initial-view}`. Earlier multi-object/SAM2
profiles cannot quantify this recipe. See
[[p2p-operational-selection-and-light-peaks|the original operational experiment]].

New **RTX 4090** probes (driver 580.178.04) fill missing native stage splits using
original masks and 3,217 consecutive native captures each. No full SAM/inspection
rerun is needed. Without native checkpoints, late rebuilds require chronological
prefixes. Sparse probes separately replay actual operational capture sequences.

Nested host timers add no stage fences: original completion synchronization,
CPU model outputs and synchronous PyCUDA readback close pending device work.
Three intrusive CPU/CUDA timelines per dense condition are excluded from ordinary
statistics, leaving 2,915 steady detailed samples each. Exclusive host durations
are additive within a frame; inclusive parents, diagnostic overlays, device work
and waits overlap. Stage percentiles cannot be added into a pipeline percentile.

Paired 121-frame timer-on/off controls retain 110 nonstartup/non-timeline pairs:
median per-frame differences **+0.76/+0.87%**, paired-difference p95 8.14/7.83 ms.
The nominal unprofiled control itself pauses for 209 ms. This supports cost ranking,
not submillisecond or extreme-quantile precision. Raw timings are not corrected.

## Acquisition to publication

```text
RGB-D acquisition -> input read/snapshot -> one running request + latest slot
  -> SAM3 IPC / causal video inference / mask decode
  -> inspection geometry + frozen map + explicit seed, or existing P2P tracking
  -> P2P IPC: TAPIR -> measured depth lift -> RANSAC clusters
       -> dense SDF candidate scores -> support/fallback + SDF refinement
       -> keyframes / SuperPoint / query additions / TSDF integration or rebuild
       -> bounded renewal -> completion fence -> trace/result export
  -> binding -> result polling -> adapter/publication -> available observation
```

SAM3→P2P is serial within one acquisition; rendering and next-request work can
overlap previous-result publication. The latest slot replaces waiting frames.
Faster computation therefore changes future processed history even with identical
same-input calculations. Capture age = ingress + latest-slot wait + executor wait
plus service (including snapshot), poll wait and publication. Accounting residual
median is about 0.005 ms; snapshot must not be counted twice.

## Initialization and complete service

Steady = acquisition ≥ seed +5 s. Preserved serial distributions have 2,917
requests each, including tracing/adapter consumption but excluding ingress and
operational waiting. All timing triplets below mean **p50 / p95 / maximum ms**.

| Preserved steady component | Light | Nominal |
|---|---:|---:|
| SAM3 including IPC/decode | 106.9 / 109.7 / 136.0 | 106.6 / 109.1 / 165.4 |
| Complete request and adapter consumption | 229.2 / 261.2 / 909.6 | 213.3 / 247.2 / 1208.5 |

Before acquisition, SAM/P2P boot takes 7.36/3.03 s light and 7.68/3.25 s nominal.
First SAM inference is 1.58/1.59 s. Inspection geometry has 106 positive calls
each: **2387/4219/4525 ms** light, **2229/3913/4298 ms** nominal. Extraction,
fusion, snapshots and eligibility remain aggregated. Seed request cost is
1460/1527 ms: native 415/439, geometry 779/845, binding 150/128; one sample each.
The first five postseed seconds have 299 requests each, p50/p95 214/236 and
214/235 ms. None of these phases is conflated with steady tracking.

## Native ordinary cost and call frequency

New **isolated P2P**, 2,915 steady frames each. SAM is absent: these conditional
costs are not complete-pipeline latency. `native-summary.json` retains per-call
distributions, counts, zero-call frames and late windows.

| Host category | Light | Nominal |
|---|---:|---:|
| TAPIR tracking, including dispatch/output transfer | 44.3 / 45.9 / 191.3 | 44.5 / 47.9 / 97.7 |
| RANSAC, summed calls per frame | 23.6 / 39.9 / 85.7 | 22.7 / 31.1 / 47.4 |
| Dense SDF candidate selection, summed scores | 18.0 / 27.4 / 43.8 | 11.7 / 26.7 / 42.5 |
| Remaining registration preparation/gates/hooks | 10.4 / 21.1 / 25.6 | 9.6 / 16.9 / 22.9 |
| SDF refinement | 6.3 / 13.2 / 16.0 | 6.2 / 11.2 / 15.8 |
| Complete native service | 118.9 / 151.4 / 739.6 | 104.2 / 140.9 / 596.1 |
| Direct diagnostic total, **overlapping rows above** | 7.6 / 9.6 / 136.5 | 7.3 / 8.2 / 239.8 |

RANSAC and current SDF queries/refinement use CPU NumPy; TSDF fusion is CUDA.
Each condition has one object, but light/nominal perform **10,064/7,829 RANSAC
calls and 7,149/4,914 candidate SDF scores**. Candidate count per frame has
p50/p95/max 2/4/6 light and 2/3/5 nominal; unsuccessful RANSAC attempts also cost
time. Mean registration preparation + RANSAC + scoring + refinement is about
63/51 ms per frame. TAPIR active queries have p50/p95/max 53/81/88 and 52/70/75;
late light queries/candidates increase to medians 73/4, while TAPIR median remains
44.3 ms. Reference count alone does not explain ordinary latency.

Renewal-related call costs below exclude nested children; counts are light/nominal.
Infrequent sampling/additions have a small amortized cost; changing their cadence
changes tracking history.

| Category, calls | Light ms | Nominal ms |
|---|---:|---:|
| SuperPoint, 223/132 | 4.51 / 4.61 / 16.32 | 5.17 / 6.71 / 12.71 |
| Sampling outside SuperPoint, 223/132 | 3.73 / 4.15 / 18.10 | 3.57 / 4.38 / 8.24 |
| TAPIR query addition, 177/113 | 9.02 / 9.27 / 10.69 | 9.01 / 9.44 / 14.29 |
| Keyframe management, 2915/2915 | 0.33 / 9.66 / 27.17 | 0.33 / 1.27 / 35.25 |
| Bounded renewal, 2915/2915 | 0.38 / 2.96 / 37.47 | 0.35 / 2.67 / 18.41 |

Native unassigned host work has median
2.49/2.42 ms; it includes input/frame conversion, depth lifting, result assembly
and untimed glue. IPC minus native compute includes serialization, pipe transfer
and copies, not just PCIe: p50/p95/max 4.64/5.57/8.70 ms light and
4.67/5.65/29.45 ms nominal.

An early light timeline measures 14.7 ms CUDA kernel busy time, 7.64 ms host
launch APIs and 3.19 ms stream waits (555 calls). These overlapping, perturbed
intervals cannot be added to ordinary TAPIR time. Chrome traces distinguish
HtoD/DtoH/DtoD; `aten::to` also includes casts. The final explicit fence normally
costs about 0.03 ms; model-output transfers already synchronize internally.

## Late extremes, transfers and diagnostics

TSDF updates occur on only **16/9** steady frames, but perform **53/21** volume
integrations because rebuilds replay prior keyframes. Consequently an all-frame
TSDF p95 is zero and hides the expensive events. Every integration downloads full
TSDF and color volumes and copies them into CPU query state. Rebuilds additionally
allocate/initialize the enlarged volume and repeat those transfers/copies for
historical keyframes; this is much more than the CUDA fusion kernel alone.
Conditional on those 16/9 update frames, measured HtoD+DtoH wall cost is
**7.1/131.1/151.6 ms** light and **13.4/93.7/100.3 ms** nominal. At the largest
updates, allocation / transfers / remaining fusion calls / other TSDF host work
cost 59.7/151.6/30.7/392.3 ms light and 97.7/100.3/14.2/241.1 ms nominal.
Fusion-call timing includes host work; it is not pure kernel duration.

| Condition, source seconds | Native service ms | Inclusive TSDF update ms | History after update |
|---|---:|---:|---:|
| light 46.000 / 54.050 | 348 / 375 | 205 / 240 | 4 / 5 |
| light 62.033 / 67.733 | 683 / 740 | 541 / 634 | 15 / 17 |
| nominal 36.300 / 41.650 / 53.083 | 308 / 515 / 596 | 190 / 402 / 454 | 4 / 5 / 6 |

Preserved complete requests peak at 910/1209 ms on rebuild captures. Some isolated
native peaks are lower: residency/interleaving, pacing and machine conditions
differ across dates. Event attribution is supported; historical timing differences
are not fully explained.

Tracing compresses masks/references and exports JSON/resources; native mesh/pose/
visualization exports are disabled. Paired 111-frame trace-on/off prefixes save
median **5.94/7.44 ms**, with identical poses. Rare tracing stalls remain in the
table. Unmodified automatic GC measures 0.154 ms median /128 ms maximum nominal.
The 191 ms light TAPIR outlier lacks an exact device timeline. Allocation, I/O
and scheduling causes remain uncertain; no favorable retry replaces a result.

## Operational age, queues and loss

The preserved light run and new nominal run offer 4,718 captures each at 60 Hz,
with the same provider/selection request. Actual seeds are 25.0167/26.6500 s;
nominal seed age is 2593 ms. Scheduling can change the processed seed without
changing initialization rules. This is not an isolated lighting comparison.

| Continuous recorded operation | Light, preserved | Nominal, new |
|---|---:|---:|
| Publications / replaced inputs | 234 / 4484 | 239 / 4479 |
| Steady publications / effective Hz | 185 / 3.79 | 201 / 4.28 |
| Service p50/p95/max ms | 222 / 595 / 703 | 215 / 308 / 350 |
| Acquisition-to-publication age p50/p95/max ms | 246 / 610 / 730 | 230 / 327 / 362 |
| Latest-slot wait p95/max ms | 0.021 / 51.9 | 0.021 / 102.9 |
| Steady publications within 150 ms | 0 / 185 | 0 / 201 |

Steady p95: ingress 11.18/9.67 ms, executor wait 2.48/2.48, polling 15.22/15.87,
publication 3.56/3.67, snapshot 0.269/0.248. These are not additive percentiles.
The outer observer finds 0.047 ms median /0.559 ms maximum outside the publication
timer; deadline classification is unaffected. Its first-selection snapshot export
separately costs 212 ms and can delay subsequent ingress.

Light has 27 native losses /199 nonseed native outputs; nominal has 47 losses and
one geometric refusal /211, with 163 accepted. These differ from replacements.
Dense quality does not transfer. Exact sparse replays preserve 200/212 native
traces and 234/239 SAM masks/IDs, yet light P2P p95 falls 393→161 ms including IPC
and SAM 199→121 ms. Interleaving, residency, pacing and machine state remain
unisolated; historical memory samples do not record clocks/load.

## Comparable producer contention

One resident Isaac scene alternates stepping/rendering and pause; its images never
feed inference. Identical SAM (0–2 s) and P2P (25.0167–27.0167 s) prefixes run
light idle→active and nominal active→idle. RGB-D is validated at 600×960; active
production averages 18.2 steps/s. Each cell retains 111 post-first-ten calls,
including IPC and every outlier:

| Component / condition | Producer paused | Producer active |
|---|---:|---:|
| SAM3 light | 109 / 112 / 857 | 159 / 169 / 1140 |
| SAM3 nominal | 109 / 112 / 852 | 160 / 174 / 1134 |
| P2P light | 103 / 109 / 128 | 148 / 178 / 243 |
| P2P nominal | 120 / 508 / 668 | 214 / 243 / 253 |

SAM paired medians increase 49.7/50.5 ms with identical masks/IDs; P2P increases
47.3/80.2 ms with identical outputs. Nominal **paused** service also has unexplained
stalls. SAM outliers recur at source rows 102/81. These early prefixes, one pair
each, do not isolate pure RTX, establish tail effects or cover mature opening.
The old stationary live 3.24 Hz run is separate; its difference from recorded
opening is not attributed to rendering. Isolated costs are not a joint latency.

## Deadline and priorities to discuss

SAM3 leaves roughly 43 ms for TAPIR, registration and all waiting/publication;
TAPIR alone costs about 44 ms. Even subtracting each frame's measured tracing gives
p50/p95/min
**221/252/187 ms light** and **206/240/181 ms nominal**, with 0/2917 within 150 ms
in either condition: an arithmetic estimate, not a trace-disabled live experiment.

The first discussions should distinguish same-input equivalence from changes to
the observations/history consumed by the live scheduler:

| Priority | Candidate discussion, not implemented | Behavior boundary |
|---|---|---|
| 1, tails | TSDF allocation, redundant transfers/copies and reconstruction overhead. | Preserve volume bounds, integration order and completed same-frame volume; verify exact queries/poses/decisions. Deferral or fusion changes alter history. |
| 2, ordinary | CPU preparation/scoring/RANSAC allocations; neural dispatch/copies. | Preserve all draws, candidate order, scores, iterations and gates. The Jacobian optimization is already selected. Numerical equivalence must be checked. |
| 3 | Diagnostic serialization and IPC. | Potentially equivalent on fixed inputs; keep ordered evidence. Savings alone cannot reach 150 ms. |
| Separate quality/history proposals | Overlap SAM/P2P, reduce model work, renewal frequency or map work. | Pacing/overlap can change retained frames and contention. Resize/crop/iterations, thresholds, budgets, candidate caps, SDF bypass and stale masks require separate quality evaluation. |

No optimization is implemented or selected by this diagnosis. Material identity,
operational recovery, hardware/contact and policy qualification remain open.

Verification: `trace-equivalence.json` has zero differing rows across 3,217/3,217
dense and 200/212 sparse native traces, including reference/map histories,
candidate/SDF decisions and published supports. Operational counters reconcile
every offered input. Two pre-acquisition launcher/selection-format failures are
preserved under `.failed-*`; neither is a discarded tracking/timing attempt.
Scripts, raw rows, CUDA timelines, summaries and the prior failed experiments
remain available. Inspection substage costs, historical stall causes and mature
full-pipeline producer contention remain uncertain.
