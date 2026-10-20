# Diverse bounded ACT A1 collection campaign

## Prospective declaration

Baseline `a3c7550`, branch `PatrizioAcquadro/b1-policy-pilot`, clean tracked tree.
The objective is new, varied temporal demonstrations, with no policy training,
optimizer update, learned evaluation, RL, architecture comparison or final-test access.
Prior recordings remain controls; clips and windows never count as new episodes.
The campaign supersedes the obsolete 24-attempt cap prospectively only.
[[act-a1-observations-and-time]] owns observation/action semantics;
[[act-a1-four-door-expansion]] retains the original results and opening references.

| Split / door | A acquisition | B initial opening | C push to hold | D release to settle | Complete |
|---|---:|---:|---:|---:|---:|
| Train animated-door-1-88abf40 | 6 | 6 | 6 | 6 | 2 |
| Train door-2738468b94d74c5f | 6 | 6 | 6 | 6 | 2 |
| Train psx-front-005-ee7d5c6 | 6 | 6 | 6 | 6 | 2 |
| Train psx-interior-wood-002-02e442c | 6 | 6 | 6 | 6 | 2 |
| Development modern-door-2fb8d024 | 1 targeted A/B | — | — | 1 | 2 |
| Development psx-industrial-001-4f5561b | 1 targeted A/B | — | — | 1 | 2 |

Each train family has two pose, two dynamics and two correction cells, exactly one
primary category per new physical execution. Total: 96 short train, eight complete
train and eight development episodes. No repeated deterministic replay fills a cell.
The frozen source groups and splits stay unchanged. Development cannot fit weights
or normalization. Source continuations retain ancestry and are not independent.

## Conditions and qualification order

These are candidate simulated conditions requiring physical qualification, not
validated hardware ranges. Use task-space targets and existing feasible IK/control.
First check individual factors, then the declared combinations, without a Cartesian sweep.

- Pose high/low: contact height 1.11/1.07 m (nominal 1.09 m), with unchanged
  material-relative normal/orientation. Screen reach, finite footprint and clearance.
  Complete train episodes use these two conditions per door. Initial unloaded
  approach additionally investigates 20–50 mm normal distance and 20 mm lateral
  error / 5 degree yaw; correction may converge to the same final contact.
- Dynamics light: uniform panel density scale .7 (17.5 kg), inertia scaled .7,
  unchanged center of mass, geometry and damping 4. Dynamics resistant: density
  scale 1.3 (32.5 kg), inertia scaled 1.3 and damping 6 after separate factor checks.
  These bracket the synthetic 25 kg, damping 4 model; they are assumptions, not
  identified real-door properties. Read actual composed and active PhysX values.
  Compare measured angle/speed/torque/contact response to saved nominal control.
  Contact friction remains .5 unless an interaction requires a separately declared check.
- Corrective A1/A2: physically executed lateral/yaw misalignment, or a stopped-short
  normal approach, followed by measured realignment/acquisition.
- Corrective B1/B2: insufficient normal contact, or a bounded weakening pulse during
  initial opening, followed by force-supported approach/recontact when recoverable.
- Corrective C1/C2: a weakening normal-contact pulse, or residual motion induced by
  a brief commanded push-rate increase, followed by maintained support and settling.
- Corrective D1/D2: incomplete initial withdrawal (8 mm), or a brief residual-contact
  pause, followed by ordinary complete unloading/separation and stabilization.

Disturbances use executed commands/forces, never mid-episode teleportation or
edited/spliced states. Exact pulse duration/amplitude is frozen in the per-case
protocol before each factor check; unsupported values block their dependent cells.
Injected erroneous actions and unsafe/ambiguous rows remain causal history but
are excluded from action targets; whole chunks crossing them are excluded. Corrective
commands require physical and phase validity. Hard safety stops remain latched.
Historical lost-contact failures on left train and industrial development prioritize
contact weakening/reacquisition; they never become successful complete attempts.

## Independent endpoints and history

A requires at least .5 s contiguous actual valid loaded contact, finite footprint,
material/orientation support and no violation. B additionally requires at least
1 degree actual controlled opening beyond acquisition with sustained valid contact.
C requires real late push, maintained support and at least .5 s qualified hold after
controlled settling. D requires actual unloaded separation >=10 mm and the unchanged
ordinary stabilization tail. Every short includes safe ordinary withdrawal/tail and
its independent continuing-physics stop; the stop cannot provide the expert tail.
Full episodes retain unchanged frozen opening/hold/release/stabilization criteria.
These endpoints are measured from physical records, independently of phase flags.

Phase starts must restore robot/door q/dq, targets, gains and controller state and
rebuild contact with fresh physics under the actual placement/dynamics. An existing
hold state cannot stand in for early contact or late push. Obtain real preceding
movement before critical events; unavailable history stays masked, with no borrowed
source prefix or repeated frames presented as movement. Keep source ancestry.

## Resources, admission and stopping

Verified baseline: 18.661711 conservative local GPU hours, 560.493949 GiB logical
retained evidence, 22 historical physical attempts, approximately 1.1 TiB free disk.
Campaign ceilings: 160 new attempts, 12 additional GPU workload wall-hours,
400 GiB additional retained evidence; cumulative 40 hours and 1 TiB remain binding.
Count each episode separately even in a reused simulator. Include all qualification,
failed/interrupted attempts, startup/shutdown, preparation, stop continuations and
CUDA validation. Reserve 60 s GPU and at least 12 GiB free headroom for interruptions.
Only one GPU simulation runs at a time; preserve unrelated workloads.

Initial guards: complete/source qualification 1200 s wall / 240 s simulated;
early short 600 s / 90 s; late short 600 s / 60 s, including conditioning/tail/stop.
Guards are resource terminations, never shortened success sequences. Use the
existing native RGB-D/control/feedback rates and bounded one-frame stores/batches.
Expected main payload is approximately 200–300 GiB with short states; measure the
qualification tranche before committing remaining launches. No costly favorable retry:
a replacement needs a diagnosed mechanism and recorded change, preserving denominators.
Unsupported required conditions, structural defects or exhausted budgets stop affected
launches and leave the goal unmet. Every category must show observed variation.

All 112 counted recordings must yield admissible temporal samples. Check native
alignment/clocks/feedback provenance, delivered-minus-prestate A1 labels, intervention
masks, transitions, limited actions and true tails. Check offline/online agreement
across all families and directions. Fit normalization on admitted train sources only.
Bounded CUDA forward/backward checks must be finite with unchanged weights and zero
optimizer updates. The final handoff must enumerate exact train/development sources,
ancestry, masks, outcomes and all resource costs. None of these gates is passed yet.

Local payload and ledger: `outputs/evidence/act-a1-diverse-20261019/` (ignored;
Git cannot recover it). This page owns campaign state; original evidence is preserved.

## Initial-screen numerical policy

The parked reset screen now compares both hard and soft joint bounds using exactly
one outward representable float32 bound step. Audits retain the original signed
margins and identify the rule. Focused checks cover nominal/exact-boundary positions,
one-step acceptance, two-step and milliradian rejection, NaN, and the shared hard/soft
screen. This is quantization allowance only; physical limits, planning margins,
calibration support, every-row safety and independent stop criteria are unchanged.
