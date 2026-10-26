# Approach-to-contact corrections and guards

Status: **bounded recovery and guard qualification complete**, October 26, 2026.
Three additional expert recoveries qualify, including the widest left case. No
learned improvement is demonstrated. Work stays on `PatrizioAcquadro/b1-policy-pilot`
in the existing worktree, starting this continuation at `5b517ca`; the preceding
tranche started at `d798998`. The main checkout remains outside this work.
See [[act-a1-failure-diagnosis]] for the causal diagnosis.

## Question and fixed boundaries

Can observation-grounded expert recovery cover the recorded ACT approach/contact
neighborhood while the operational system retains physical, load, geometry and
stopping requirements?

Only the two existing training doors, saved requests, expert corrections and
independent guard diagnostics are used. There is no policy inference, optimizer
update, checkpoint comparison, new door, development payload read or final test.
Full learned completion remains **0/22**, with the original final pair **0/2**.
These finite contact subtasks omit complete opening, hold and release; their raw
full-task reports correctly remain false.

The operational 10 N stopping domain, physical 80 N limit, orientation criterion,
position calibration, speed/acceleration and sweep/stop bounds are unchanged.
The October 25 independent shoulder/wrist extension is reused without adjustment.
All native work uses the workstation RTX 4090, 240 Hz physics, 60 Hz control and
actual head RGB-D. Simulator/evaluator channels never supply policy observations
or replace observed operational safety evidence.

## Preserved preceding work

The previous tranche retains **12 qualified sources from 15 collection attempts**.
Both directions include shifted posture, early/late pauses, measured pose return,
near-contact return and early pause after genuine common inspection. Near-contact
errors include 4 mm normal/10 mm tangent translation and 4 degrees orientation.
Injected commands remain causal context and are excluded from targets.

All 12 sources pass physical/temporal recording qualification, sustained contact,
withdrawal/stabilization and independent 2 s stop plus 4 s continuation. Native
peak forces are 3.537–6.675 N. They add 4,852 windows and 3,998 positive corrective
commands, yielding the preserved **126 train plus eight development records**;
normalization remains fitted to the original 112 sources. The maintained
prospective position-support API covers these sources with the independently
qualified calibration.

The three original negatives remain: both initial posture cases miss sustained
contact, and the original early right pause reaches 0.524655 rad/s during an
insufficiently timed return. A 29-tick right contact spans only 0.467 s and does
not satisfy the unchanged 0.5 s endpoint. Their replacement interventions reserve
1.5 s for recovery and bounded contact endpoint verification, without changing
acceptance criteria. Short early sources do not invent missing history; two
common-inspection additions make their actual early corrections usable.

The prior pause-aware fixed-target reference pair qualifies physical contact and
stopping. Four earlier right clock/stimulus failures remain negative. Fourteen
independent robot-only cases qualify the upper shoulder-Z/wrist-X position support
at 10 N, including distal loads, neighboring postures, oblique loads and moving
stops. Residual/error-model coefficients and original stop bounds are retained.
The earlier right 13.763 N impact remains outside the verified stopping domain.

## Genuine-history late-left recovery

The two targeted starts replay the actual saved ACT A1 requests from common
inspection to **28.850 s** and **29.183 s**. Every preceding measured q/dq matches
the saved failure trajectory exactly. Camera, feedback, previous delivered targets,
servo targets and dynamic state belong to the new continuous native execution;
there is no posture restoration or inspection at handoff. No checkpoint is loaded.
All replayed ACT commands are context only, never expert targets.

The qualified correction brakes on the measured target for 0.4 s, realigns over
1.5 s, settles for 0.5 s and executes the existing qualified slow contact entry.
Controller bounding remains active. Supervision uses the targets actually applied,
including a first wrist target that still lies ahead of measured q because the
previous servo target cannot change instantaneously; requested zero error is not
substituted for the delivered command.

| Recovery | Positive commands | Consecutive valid contact | Native peak / observed force upper |
|---|---:|---|---:|
| Left 28.850 s | 384 | 77 ticks, 1.267 s clock span | 5.071 / 5.264 N |
| Left 29.183 s | 384 | 77 ticks, 1.267 s clock span | 5.093 / 5.289 N |
| Widest left after controlled error | 519 | 77 ticks, 1.267 s clock span | 5.254 / 5.448 N |

Each also has 1.283 s valid control-interval coverage. Qualification uses the
stricter first-to-last timestamp span, not tick count alone. All three execute
through ordinary operational guards, have zero physical violations and pass the
existing independent 2+4 s continuing-physics stop checks.

The responses address motion as well as posture. Maximum incoming joint speed
falls from **0.2009→0.0140 rad/s** at the 28.850 s start and
**0.1334→0.00635 rad/s** at 29.183 s after the 0.4 s brake. In both, shoulder-Z
velocity reverses toward the supported neighborhood and wrist-X velocity becomes
small before realignment. Sustained physical contact, force and stopping establish
an appropriate response; teacher proximity alone does not.

## Widest-left support and recovery disposition

The previous fresh-inspection activation refuses the positive finger's enclosing
swept/stopping volume with both occupied and unknown support. Its 18.23 mm query
radius is a conservative envelope, not a measured penetration. A current posture
plus fresh inspection does not reproduce the original causal observation context.
The camera views contain substantial finger/material occlusion; no hidden depth
measurement is fabricated.

The continuation therefore executes the original 384-command error prefix after
actual common inspection, retaining all intervening images, state, load feedback
and servo history. This is a **separately declared experimental prefix**: native
physical checks, observed force upper ≤10 N and the existing speed domain remain
active, but it has no claim of operational admission and contributes zero expert
targets. Every positive recovery command then uses the ordinary operational runner.
There is no state, controller or monitor reset at activation.

At the 31.600 s activation, arm q/dq match the original 6.400 s error state exactly.
Over the actual four-second history, arm q matches exactly and maximum dq difference
is 7.45e-8 rad/s. Earlier startup and tiny neck differences remain explicit; images
are actual captures, not imported or claimed pixel-identical. With this genuine
history, observed sweep support passes, initially with about 0.94 mm margin.
The fresh-inspection refusal therefore does not establish missing geometry support
for every recovery from that posture.

Motion remains a separate issue. A slow return loses robust orientation support;
a prompt direct return then exhausts the unchanged 30 mm search allowance before
contact. The qualified expert instead uses **0.25 s linear realignment**, a **1 s
retreat**, **0.6 s settling**, then the existing nominal slow reapproach. Nominal
peak joint-target speed is 0.3214 rad/s, inside the unchanged qualified motion domain.
The retreat reaches observed certain separation at 32.000 s. Only this ordinary
separation transition ends acquisition; no code directly clears its budget.
The reapproach reaches sustained valid contact at 38.983 s and completes its
declared prefix with passing force, physical and independent stop checks.

This qualifies the named recovery from the retained experimental context. The
injected error prefix itself remains inadmissible as an operational demonstration.
It does not qualify arbitrary states, unknown scenes, all observation uncertainty
or an automatic learned recovery mechanism.

## Targeted data and temporal admission

The continuation adds exactly **three train sources, 1,287 positive corrective
commands and 1,287 overlapping windows**. The union is **129 train plus eight
unchanged development records**, with all original sources and 112-source
normalization retained. No failed attempt or nonexpert/injected prefix is admitted.
The two late starts share one ACT failure trajectory; they are not independent
learned trials. The widest case is a separate controlled-error experiment.

| Saved ACT query | Closest corrective posture L2, before → after | Velocity max difference at that same source state |
|---|---:|---:|
| Left 28.850 s | 0.05523 → 0 rad | 0.09251 → 0 rad/s |
| Left 29.183 s | 0.04756 → 0 rad | 0.12997 → 0 rad/s |

These are descriptive whole-vector comparisons. Their meaning comes from the
actual incoming motion, preceding observations, braking response and independently
qualified contact outcome, not independent joint extrema or window counts.

The maintained temporal consumer runs on CUDA without a policy or optimizer.
Each first positive decision has four actual seconds of history, strictly prior
delivered-target support and valid sampled image/robot masks. Batches retain
23 five-channel 240×384 images and 241×32 measured robot values. The first target
is checked against the actual bounded applied delta. Raw originals stay unchanged;
small linked source views add review metadata and evaluator-only phase/role labels.
These annotations never enter observations. Explicit native synchronous simulation
clocks and healthy/present feedback masks are maintained by the reader; missing
optional torque remains masked. Native head-camera frames were visually inspected.

## Changing-request guard qualification

The native right control replays **all 428 saved ACT requests exactly**, including
exact preceding measured q/dq, before a separately declared continuation repeating
the final saved requests. No model runs. After three continuation commands, the
guard stops unresolved search at **3.000 s active / 3.367 s raw**, with
15.765 mm active / 16.244 mm total travel. Native force is zero, observed upper
force is 0.509 N, there are no physical violations and independent stopping passes.
This is successful guard bounding, not successful contact or learned behavior.

Ten numerical cases reuse saved motion and isolate clock semantics. Five-second
bounded changing-target pauses and settling remain supported; cumulative 0.1 mm/s
creep is charged and stopped, active search reaches its time bound, and a faster
path reaches the 30 mm bound first. Response-envelope cases just inside/outside
support retain the distinction. These are deterministic clock tests with controlled
geometry/load fixtures, not additional native physical successes. The earlier
fixed-target native pair supplies the physical positive control, complemented by
the newly executed changing-request and corrective native trajectories.

One demonstrated diagnostic defect is corrected generally: a repeated same-clock
after-check could display “pause” for an interval already counted as active.
It now reports the effective interval and provisional pause credit. The underlying
elapsed/travel accounting, 3 s/30 mm budgets, FK resolution, response envelope,
force confirmation and all geometry/stop requirements are unchanged.

## Preserved negatives and limits

There are **eight new recovery attempts: three qualified and five negative**, plus
one native right guard diagnostic and one CUDA preparation allocation. All nine
native reports and safe stops remain, without physical violations or unavailable
startups. The five recovery negatives are:

- Velocity-preserving late-left interpolation continues shoulder drift and fails
  prospective position support after four positive commands.
- The widest 1.5 s cosine return fails robust orientation after four commands.
- A 0.4 s cosine return still fails orientation after five commands.
- A 0.4 s linear direct return fixes orientation but reaches 30.059 mm active
  travel at 1.7 mm physical separation, before contact, after 174 commands.
- Reusing that initial 0.4 s alignment before retreat fails after five commands
  under different actual image uncertainty. It is not accepted as robust support.

Each changed intervention has its own preceding plan and output. The final prompt
alignment/retreat strategy changes motion, not a threshold. All older failures,
startup unavailability, original recordings and calibration evidence remain.
No result establishes learned improvement, hardware qualification or held-out
generalization; the original right impact limitation remains.

## Evidence, resources and next decision

Earlier evidence remains under
`outputs/evidence/act-a1-first-completion-20261022/approach-contact-20261025/`.
The continuation is under the sibling **`recovery-guard-20261026/`** directory.
Its executable handoff consists of:

- `broad-plus-targeted-recovery-sources.json` and the three prepared sources.
- `qualified-operational-recipe.json`, retaining every numerical operational setting
  of the independently supported calibration/search recipe, with current reviews.
- The three qualified recovery `*-plan.json` records and their command references.
- `recovery-review.json`, `widest-review.json`, `coverage-response-review.json`,
  `pipeline-review.json` and `guard-qualification.json`.
- `attempt-ledger.json` and `closeout-decision.json`, including every negative and
  the distinction between false full-task reports and qualified contact subtasks.

The October 25 baseline was 44.2550 local/31.8231 remote GPUh and 1015.89 GiB retained.
This continuation uses **0.6757 local GPUh**, zero remote allocations, zero optimizer
updates and about **8.08 GiB** additional retained evidence. Cumulative accounting
is **44.9307 local/31.8231 remote GPUh**, **1023.97 GiB retained** and **198,773 updates**
including historical failed-run uncertainty. Caps remain 48/32 GPUh, 1126.4 GiB and
200,000 updates. All allocations finish and the shared GPU lease is released.

The scoped data and operational qualification now justify considering **targeted
refinement**, requiring a separate authorized plan. Current caps leave 3.0693 local
GPUh, 0.1769 remote GPUh and 1,227 updates. No training or learned evaluation starts;
the training heartbeat remains paused, development payloads unread and final test
sealed. The next decision is a bounded refinement and matched learned verification
budget, not broader collection or a claim of solved learned manipulation.

Implementation commits: `e6bde1d`, `cd884ae`; earlier tranche: `d92fd16`, `f56d3a3`,
`9448ede`, `7c09c05`, `d222df7`.
