# Selected SAM3 + Point2Pose Cost Diagnosis

## Question and method

Which stages dominate ordinary service, mature rebuild tails and publication age in the unchanged
selected pipeline?

Diagnosis from `6b84b04` separates preserved serial/coordinator accounting, new isolated native
profiles, complete operational ingress and comparable producer contention. Selected models,
initialization, references, thresholds and policy remain fixed. Overlapping host/GPU stages and
percentile distributions are not summed as independent costs.

## Result and decision

On 2,915 isolated steady frames per condition, native service p50/p95/max is 118.9/151.4/739.6 ms
light and 104.2/140.9/596.1 nominal. Ordinary TAPIR is about 44 ms median; registration
preparation/RANSAC/scoring/refinement averages about 63/51 ms. RANSAC and SDF queries/refinement run
on CPU NumPy while TSDF fusion uses CUDA.

TSDF updates occur on only 16/9 steady frames, but late light rebuild reaches about 634 ms inside
740 ms native service. Retained history, full-volume downloads and CPU snapshot construction matter
more than kernel busy time alone. IPC includes serialization and copies, not only PCIe.

| Continuous recorded operation | Light, preserved | Nominal, new |
|---|---:|---:|
| Publications / replaced inputs | 234 / 4484 | 239 / 4479 |
| Steady publications / effective Hz | 185 / 3.79 | 201 / 4.28 |
| Service p50/p95/max ms | 222 / 595 / 703 | 215 / 308 / 350 |
| Acquisition-to-publication age p50/p95/max ms | 246 / 610 / 730 | 230 / 327 / 362 |
| Latest-slot wait p95/max ms | 0.021 / 51.9 | 0.021 / 102.9 |
| Steady publications within 150 ms | 0 / 185 | 0 / 201 |

Prioritize measured rebuild/snapshot cost and ordinary SAM3/TAPIR/registration while preserving
quality at actual processed ingress. This page diagnoses costs; it implements no optimization.
Separate [[tsdf-readonly-launch-inputs|TSDF]], [[sam3-official-compilation|compiler]] and
[[p2p-ransac-point-buffers|RANSAC]] studies own their rejected adoption decisions.

## Limits and evidence

Isolated P2P excludes SAM3 and is not complete latency; diagnostic serialization overlaps measured
stages. Live renderer and producer contention are separate from timing-induced tracking histories.
No clock here establishes 150 ms useful availability, material recovery, hardware, action/contact or
policy qualification.

Evidence: `outputs/evidence/b1/perception/sam3-p2p-cost-20261014/`, native/operational summaries,
full stage/transfer profiles, CUDA timelines, contention controls and original failures. Git:
diagnosis `9b94978`.
