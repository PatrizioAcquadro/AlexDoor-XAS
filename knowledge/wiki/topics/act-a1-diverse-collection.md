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

## Implemented producer interfaces — qualification in progress

The existing teacher now accepts declared A/B/C/D short scopes with physical
endpoint evaluation. Early shorts obtain a fresh t=0 camera reference and six
seconds of actual unloaded approach rather than borrowing an inspection/history
prefix. C restores a real source captured five seconds before push exit, then
executes a new push to the existing measured exit guards before hold (15 s
resource budget, at least 4 s genuine preceding motion). D also starts from genuine preceding push, preserving the original 3 s hold
budget and acquiring at least 4 s actual history before release. Each loaded reconstruction receives a detached start-stop check,
a complete reset and fresh conditioning before recording. Placement, contact,
controller/safety and dynamics fields must match the source; future phase durations
may differ. Source velocities, gains, targets and contact rebuilding remain explicit.

Native annotations distinguish expert (0), injected excluded (1) and corrective
(2) commands. All delivered actions remain observation history; injected rows and
all action chunks crossing them are excluded from supervision. A named short B/C
contact deviation may attempt at most one second of bounded normal recontact while
physical/material/force checks remain valid. Original complete-task loss criteria
are unchanged; any detected loss is retained as a full-task failure indicator.
No physical stop latch is cleared. Original full cycles continue to use their
unchanged teacher outcome and frozen opening requirements.

Uniform density variations scale both panel and handle mass and diagonal inertia,
preserving geometry, center of mass and principal axes. Overrides are authored in
the runtime stage before physics; active PhysX mass/inertia/damping are saved
separately. A changed configuration is not evidence of a changed physical response.
These interfaces have focused numerical tests; physical qualification and main
collection gates remain open.

First factor check: `train-1-complete-pose-high` passes unchanged complete-cycle
criteria (52.885434 degrees sustained opening versus 41.008202 required), hold,
release, ordinary tail and independent stop, without physical violations. The
recording has 7,980 usable starts; whole launch 399.136 s. It is one candidate new
complete train episode, pending the campaign-wide data/observed-coverage handoff.

A/B withdrawal, hold and ordinary tails after their early endpoints are diagnostic
collection cleanup, with explicit supervised-phase masks. They remain in the causal
recording but cannot teach early release for the unchanged full opening objective.
C/D terminal supervision is conditional on genuine late source states and the
unchanged physical exit/hold/release criteria; an arbitrary short-duration stop is
not a new hidden policy goal. Corrections retain a single opening task throughout.

## First reconstruction failure and mechanism

`train-1-D-pose-high` failed before recording. Six static joint-anchor conditioning
steps retained zero contact force while the door coasted at approximately .0104 rad/s.
The right-arm source velocity .05519 rad/s decelerated to -.00192, producing .05712
rad/s reconstruction error against the existing .05 bound. Position/tool errors
were small, and the independent stop passed. This is a failed attempt, not new
release coverage or proof of physical danger from a small empirical excursion.

The diagnosed correction continues the source's measured push/hold controller
through conditioning, carrying compression, compensation and reference forward.
It requires reconstructed contact before a loaded stop and before recording. The
original .05 rad/s reconstruction bound and all physical safety gates remain.
A replacement needs its own recorded attempt and must establish actual contact,
release and data usability; the original failure stays in the denominator.

The conditioned D replacement rebuilt contact and passed its start stop, then failed
at 2.1 s of additional hold: speed .700340 rad/s, minimum Jacobian singular value
.002106, followed by a failed continuing-physics stop (peak 4.470002 rad/s).
Its complete 127-command recording has zero expert windows. This hard failure is
retained unchanged. Adding history after an already completed hold consumed the
original release reserve and drove the arm toward singularity. No more C/D launches
use end-of-hold history extension. New D starts must be real preceding push states,
then use the same measured push exit and original 3 s hold before withdrawal.

The earlier launches also used the legacy default stop impedance. Subsequent
campaign qualification explicitly selects the existing separately validated stop
recipe from `act-a1-calibration-stop-20261019/runtime-recipe.json` (normal motion
gains unchanged). This reuses a measured profile; it does not qualify the failed
state retrospectively or widen any stop criterion. Every new loaded start still
requires its own actual-condition stop evidence.

Before the main matrix, unlaunched early pose-low A/B cells are specified as a
20 mm robot-base retreat along world X and 50 mm precontact distance, with nominal
contact height. This adds observable robot-door relationship and distance coverage.
Late pose-low C/D and complete episodes retain the declared 1.07 m contact height;
all source conditions must match. These phase-specific candidates require fresh
physical qualification and cannot replace a failed cell with nominal geometry.
