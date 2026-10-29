# Resumed End-to-End A1/A2 Pilot

## Question and outcome

The October 16–18, 2026 pilot tested whether the shared observed-input ACT/Diffusion
× A1/A2 stack could collect real simulated teacher data, train and execute learned
commands through ordinary admission, control and stopping. Baseline was `4ec7571`;
the scientific source/attempt history is retained through `4d21837`.

**The experiment is paused and incomplete.** No identical checkpoint completes
the unchanged task on both doors. The current project-wide pause is described in
[[status]]; this historical record grants no authority to resume.

| Frozen cohort | Complete outcomes | Other slots | Full task successes |
|---|---:|---|---:|
| selected-v10 | 24/24, across 26 physical attempts | Two prepolicy external interruptions retained separately | 0 |
| qualified-latest-v11 | 2 completed failures | 1 interrupted attempt and 21 unallocated slots | 0 among completed outcomes; remaining results unavailable |

The two panels have different frozen checkpoints and source versions. Their
outcomes cannot be pooled as identical repetitions or replaced by favorable retries.
All four policy cells execute nonzero learned commands in selected-v10. ACT A1
left/dim retains 99.54% controlled opening and valid hold but no release; it fails.
Every complete episode passes consumed-input/actual-command/native-clock and
continuing-physics-stop audits, with no physical violation. Unavailable interruption
stop/timestamp fields remain unavailable.

## Method and maintained software

The shared RGB-D recipe, measured proprioception, actual applied-action labels,
train-only statistics, fixed door/lighting panel and independent task scoring were
held explicit. Development never updates weights or normalization; final test
identities remain sealed. All four models genuinely reach 30k in the final
continuation; earlier 3k/4.4k/12k/18k states, guarded development choices, optimizer,
RNG and Diffusion EMA remain distinct saved artifacts. Loss reduction and exact
checkpoint prediction round trips establish optimization and software continuity,
not manipulation success.

The current reusable interfaces belong in [[topics/b1-policy-pilot]],
[[topics/learned-policy-stack]] and [[topics/shared-door-perception]]. The study
commissioned measured feedback, finite material/workspace support, actual controller
preview matching, pressure/hold handoff and a measured-anchor latched stop. Their
scope and remaining qualifications are documented there and in the linked component
experiments, rather than repeated as execution instructions here.

## Latest-panel failures and interruption

| Latest-v11 case | Learned commands | Controlled opening | Outcome |
|---|---:|---:|---|
| ACT A1 right/nominal | 1,757; 293 policy queries | 18.30436 degrees, 35.7087% of the 51.26025-degree reference | Contact uncertainty exceeds admitted material margin; no hold or release |
| ACT A1 left/nominal | 714 | 0 degrees | Outside the load envelope; no full hold/release |
| ACT A2 right/nominal | Final task/command audit unavailable; 144 retained control rows marked learned | Unavailable | Human interruption after the parent driver was frozen against another launch |

The first right case has 7.33372 N peak native force, no physical contact loss or
violation, and a healthy 240-sample/2 s stop with 0.24790 mm maximum travel.
Its 0.3717 mm robot-sweep residual is a different certificate from the missing
compact-log material margin. Existing whole-finite discrepancy reaches 1.973 mm;
the 3 mm allowance was retained, with no demonstrated double charge. The allocation
charges 667.922 GPU seconds and retains 0.62847 GiB. Full-horizon extrapolations are
cost scenarios, not observed successful runtimes.

The interrupted A2 source ends at 27.600 simulated seconds. Its saved 2 s stop
passes with 3.617 mm travel, 0.007361 rad rotation and final 0.0001527 rad/s speed.
It exits 130; supervisor/driver exit 1 with no restart. The allocation charges
309.442 GPU seconds, including the conservative tail. No final success or
complete command count is fabricated from the partial recording.

## Operational contact versus physical task contact

Independent same-clock contact joins cover23,743 monitored states and23,757
operational endpoints: availability99.941%, with14 terminal endpoints missing
monitor output rather than assigned synthetic values. There are13,201 operational
confirmations,13,241 windowed physicalLoaded states and11,664 complete physical
valid-contact states. Confirmed ticks without instantaneous intended native force
number31; the longest run is83.33 ms, within the100ms averaging window. Missed
instantaneous-force ticks number44; physicalLoaded without confirmation numbers40.
Confirmation delays across seven contact cases are100–116.67 ms. No confirmation
falls below windowed physicalLoaded or outside the actual finite footprint;
1,560 confirmations fail the full material/orientation/commanded physical contact
rule. These definitions distinguish operational detection/uncertainty, finite
footprint support and stricter independent task criteria. Do not report zero
false physical admissions or infer task success from contact detection alone.
Evidence is`learned-contact-metrics-selected-v10.json`, the complete24-row results
and`selected-v10-complete-panel-cost-contact-summary.json`.

## Costs, evidence and limits

At the October 18 pause, closed cumulative use is 13.827548 local GPUh,
3,503 remote GPU seconds, 120,364 upper optimizer updates (120,164 checkpointed
plus 200 possibly lost), 392.827 GiB and 11 demonstration attempts. Reservations
are zero. The then-authorized ceilings were 40 local/32 remote GPUh, 200k updates,
1 TiB and 24 demonstrations; these are historical stage policy, not current
restart authorization. The selected-v10 26-attempt cohort alone charges 3.710752
GPUh and is already included in the cumulative ledger.

Evidence remains under `outputs/evidence/b1-shared-learned-pilot-20261016/end-to-end-resume/`.
Canonical records include `continuation-state.md`, `goal-1-paused-20261018.json`,
`goal-1-pause-budget-snapshot.json`, `learned-contact-metrics-selected-v10.json`,
`selected-v10-complete-panel-cost-contact-summary.json` and
`fixed-panel-qualified-latest-v11-plan.json`. The latter freezes source `bfabece`,
all four 30k checkpoints and the original 24-slot protocol; its original first
failure is reused and counted. Reports retain all model/recipe versions, stopped
jobs, failed wrappers, exclusions, uncertainty and missing outcomes.

This study demonstrates an operational learned execution path and bounded
component evidence. It establishes neither full manipulation success, qualified
perception, unseen-door generalization nor hardware transfer. Subsequent temporal
ACT work is a separate experiment in [[act-a1-first-completion]].
