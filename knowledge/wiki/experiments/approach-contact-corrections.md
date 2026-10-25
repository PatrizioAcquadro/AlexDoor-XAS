# Approach-to-contact corrections and guards

Status: **partial**. Corrective data and independent position support are qualified;
the widest left recovery lacks passing combined operational admission. Baseline `d798998`, branch
`PatrizioAcquadro/b1-policy-pilot`. The main checkout is outside this work.
See [[act-a1-failure-diagnosis]] for the preceding causal diagnosis.

## Question and scope

Can observation-grounded expert recoveries cover the current ACT approach/contact
neighborhood, and can proportionate guards admit those recoveries while retaining
physical, load, geometry and stopping requirements?

Only the two existing training doors, expert collection and independent guard
qualification are used. There is no optimizer update, policy inference, checkpoint
comparison, new door, development payload read or final test. The prior full learned
result remains **0/22**, final pair **0/2**. Reference subtask contact is separate
from learned opening, hold and release.

## Corrective data

The initial train-only audit finds 114 sources, 64 with admitted approach/contact
rows and 2,703 positive corrective commands in 13 sources. Frozen normalization
remains fitted to the original 112 sources. Posture and velocity distances to
saved ACT states are descriptive coverage measures, never admission tolerances.

The qualified additions comprise 12 native demonstrations: both directions have
shifted initial posture, early pause, late pause, combined pose return, a near-contact
return, and an early pause preceded by real common inspection. Near-contact errors
include 4 mm normal/10 mm tangent displacement and 4 degrees orientation around
the observed 21 mm separation neighborhood. Returns start at measured tool state;
the expert slows before contact. Head RGB-D, measured joints/velocities, requested
and applied targets, native feedback and separate evaluator channels are retained.
Injected erroneous commands stay in causal history and are excluded from targets.

All 12 pass physical and temporal recording qualification, sustained native contact,
withdrawal/stabilization, and independent 2 s stop plus 4 s continuation. Peak native
forces are 3.537–6.675 N, inside the unchanged 10 N stop-load domain. The package
adds **4,852 training windows and 3,998 positive corrective commands**. The final
manifest retains all original sources: **126 train plus eight development records**.
It contains 76 sources with admitted approach/contact rows and 6,701 corrective
commands in 23 sources; no injected-error command is admitted.

At the nearest corrective posture, joint-vector distance improves markedly for
early ACT drift and only modestly for later left drift. The full vector is compared
at one source state; independent joint extrema are never combined into a fictional
neighborhood. Residual late-left posture/velocity mismatch remains explicit.

| ACT query | Corrective posture L2 before → after (rad) | Velocity max difference before → after (rad/s) |
|---|---|---|
| Right 27.450 s | 0.6050 → 0.0240 | 0.2801 → 0.1531 |
| Right 29.317 s | 0.0301 → 0.0294 | 0.1228 → 0.0184 |
| Right 31.600 s | 0.0286 → 0.0271 | 0.1775 → 0.0790 |
| Left 26.833 s | 0.6541 → 0.0397 | 0.2201 → 0.0468 |
| Left 28.850 s | 0.0683 → 0.0552 | 0.2735 → 0.0925 |
| Left 29.183 s | 0.0482 → 0.0476 | 0.1290 → 0.1300 |

These distances do not establish sufficient learned coverage or improvement.

The maintained temporal consumer was exercised on CUDA without constructing a
policy or optimizer. Targets require four actual seconds of history plus prior
delivered-command support. The first short early-pause sources do not admit their
early return; two additional recordings use the existing common inspection so all
90 commands of each actual 1.5 s return have sufficient history. Missing history
is never synthesized. Native head-camera contact images were inspected. Development
records are retained without reading their payloads, and normalization is not refitted.

## Demonstrated failures and expert changes

There are **15 collection attempts: 12 qualified, three negative**. Both original
shifted-posture attempts miss the unchanged sustained-contact endpoint. The right
has only 29 valid ticks spanning 0.467 s, below the required 0.5 s. The original
early right pause leaves a 61.2 mm backlog for a fixed 0.5 s recovery and reaches
0.524655 rad/s, beyond the 0.51 rad/s physical criterion. All three negative
attempts, tails and independent stops remain; none supplies positive supervision.

The separately declared intervention reserves 1.5 s for pause recovery. A short A
contact retains its force ramp and may hold at most `sustain_s + contact_force_window_s`
beyond its nominal phase until the existing endpoint is demonstrated. This repairs
expert timing and changes no acceptance threshold. Later cases use distinct plans
and outputs, with complete-run reservations made before native execution.

## Acquisition and load qualification

The opt-in active-search clock distinguishes target movement using independent FK
resolution from bounded target settling using the verified command-response envelope.
It charges completed control intervals once despite same-clock before/after checks.
The existing 3 s/30 mm budgets apply to active search; raw time and travel remain
reported. Default recipes retain the legacy clock. A fixed A1 reference requests
`held_target - current_q`: repeated zero deltas recenter on measured state and can
drift. Reference mode loads no policy checkpoint.

The declared reference pair passes guard criteria at `7c09c05`: right has a 4 s hold,
888 commands, 2.067 s native contact and 6.833 N native/7.020 N force upper; left has
a 1 s hold, 708 commands, 2.067 s contact and 5.093/5.289 N. Both independent stops
pass with zero physical violations and zero learned commands. Raw full-task reports
remain false because these finite prefixes omit complete opening/hold/release.

Four preceding right clock/stimulus trials fail qualification: charging the prior
pause at resumption, drift under zero deltas, confusing settling with FK precision,
and erasing submitted intent at a repeated callback. Their original outcomes and
safe stops remain. Earlier left controls belong to their own implementation versions.

A maintained-API audit exposes another boundary: the left near-contact correction
has 97 near-state rows outside the original wrist-X position support; other left
returns need prospective 10 N wrist support. A native reference with the original
calibration refuses at command 331 and stops safely. Physical teacher success alone
does not establish operational compatibility.

Independent robot-only tests therefore apply maintained 10 N loads at the tool center
and both calibrated distal finger faces, including neighboring postures, oblique loads,
wrist motion, upper-domain boundaries and stops initialized at wrist velocity ±0.5 rad/s.
Four shoulder cases and ten wrist cases pass their 2 s stop, 4 s loaded continuation
and 2 s removed-load checks. Frozen residual bounds have zero exceedances on motion
and stopped native samples; original joint stop bounds and native feedback checks pass.

A separate empirical calibration increases only upper shoulder-Z support from 8.5 to
12 mrad and upper wrist-X support from 4.25 to 100 mrad, the latter now qualified at
10 N. Error-model coefficients, main domains, speed/acceleration, stopping bounds,
physical 80 N limit and operational 10 N domain remain unchanged. Beyond-support and
above-10 N cases remain refused. The maintained prospective API covers all 12 saved
corrective sources near the door with this calibration. This is scoped empirical
support, not arbitrary-state or hardware qualification. The combined operational package remains unqualified for the widest correction.

The original right 13.763 N impact remains outside the verified 10 N stopping domain.
Slow expert entry supplies lower-load examples; it does not grant permission for
that impact or establish learned deceleration.

## Combined operational blocker

With the expanded calibration, the whole left reference passes the former position
refusal but stops at command 353 on `observed_contact_orientation`, during the
excluded 4-degree injected error, before the positive return. It has no native
contact or physical violation and passes its independent stop. This protects
against the deliberately erroneous prefix; it is not itself a recovery test.

A separate declared activation check starts at the measured 6.4 s error posture,
performs genuine common inspection and submits only the positive return. Its first
command is refused by `unsupported_actual_robot_sweep_and_stop`: the positive UMI
finger's swept/stopping cell has both occupied and unknown support. Zero reference
or learned commands execute, zero force/physical violations occur, and the independent
stop passes. This fresh-start observation context differs from the original full
causal history; the result does not prove every recovery from that posture impossible.

Two startup failures also remain explicit: an undeclared CLI condition and an
initializer at the wrist-Z numerical joint boundary. Their original allocations,
logs and unavailable reports remain. The declared initializer repair moves only
10 microradians inside the unchanged URDF limit; subsequent targets and acceptance
criteria do not change. Neither startup is counted as a successful native episode.

Do not promote the combined candidate recipe or declare refinement readiness.
The next bounded task must establish observed sweep/stop support at activation of
the widest left positive return, retaining causal history and existing force,
orientation and space gates. No force increase, unsupported geometry substitution,
threshold relaxation or automatic training is justified.

## Evidence, resources and next boundary

All ignored evidence is under
`outputs/evidence/act-a1-first-completion-20261022/approach-contact-20261025/`.
Exact plans and reservations precede each intervention; original plans and outcomes
remain. `broad-plus-corrections-sources-v2.json` is the final data manifest;
`pipeline-review.json`, `pipeline-review-history.json` and `coverage-final.json`
establish temporal admission and coverage. `position-support-qualification.json`
and its four linked independent reviews establish the position-support decision.
`guard-review.json` records the active-search pair and its finite-prefix scope.

Local/remote caps remain 48/32 GPUh and retained evidence 1126.4 GiB. The baseline is
42.733604 local/31.823056 remote GPUh, 996.104 GiB and 198,773 cumulative optimizer
updates including failed-run uncertainty. Final accounting is **44.2550 local/31.8231 remote GPUh**, **1015.89 GiB retained**
and **198,773 cumulative updates**. This task uses 1.5214 local GPUh, zero new
optimizer updates and zero remote allocations. All allocations finish and the
shared GPU lease is released. The new ledger contains 15 collection allocations,
13 reference-guard allocations (11 reports plus two unavailable startups), four
independent panels containing 14 load/stop cases, and two CUDA preparation allocations.
No original negative result or unavailable outcome is removed. The baseline 63 native
attempts and full learned 0/22 denominator stay historical and distinct.
`attempt-ledger.json` and `closeout-decision.json` preserve the final partial decision.
The training heartbeat stays paused; refinement needs separate authorization.

Implementation commits: `d92fd16`, `f56d3a3`, `9448ede`, `7c09c05`, `d222df7`.
