# ACT A1 failure diagnosis

Status: complete on October 25, 2026 (UTC), with actionable phase-specific evidence.
Two native attempts are retained; neither is eligible for full learned completion.
The investigation starts from `1072ebc` on `PatrizioAcquadro/b1-policy-pilot`.
The original final 61k normal-start pair remains **0/2**, and cumulative full
learned completion remains **0/22**. Reference staging never counts as learned
completion. See [[act-a1-first-completion]] for the training and final-pair record.

## Question and fixed boundaries

Identify why the trained policy fails before contact, distinguish learned behavior
from observation and execution defects, and determine whether bypassing early
approach exposes additional limitations. Task completion from the policy's own
observations remains the objective; matching a teacher is an offline diagnostic.

No optimizer update, collection expansion, new seed, architecture change,
checkpoint search or physical/operational threshold change is authorized here.
The 61k checkpoint stays selected. The 75k checkpoint is compared only offline on
the same fixed cases. The development simulation panel stays unopened and final
test stays sealed. All work is confined to the existing pilot worktree.

The diagnostic uses the qualified 240 Hz physics, 60 Hz control, four-second
causal history, 240 x 384 RGB-D and chunk 12/execute 6 recipe on the local RTX 4090,
under the existing shared GPU lease and whole-allocation accounting. Baseline
costs refresh to 42.508021 local/31.823056 remote GPUh, 198,773 cumulative optimizer
updates and 994.269 GiB retained. The original 48/32 GPUh, 200 k-update and
1126.4 GiB limits remain. Zero new updates are spent.

## Observation and action semantics

All 72 right and 40 left learned queries from the final pair reconstruct from the
actual saved control/capture/feedback streams. The first entire input, including
preprocessed images, is bit exact. Every reconstructed 61k prediction reproduces
all dispatched requests exactly, through termination. Numeric and image support
remain fully valid at every query. Delivered-target support uses the actual
pre-dispatch acquisition clocks, with strict prior-target causality; post-step
time minus dt is not substituted at floating-point boundaries.

Right request versus applied delta differs by at most 1.19e-7 rad. Left has 15
controller corrections over 1e-6 rad, all at `RIGHT_WRIST_Z`, beginning at 28.85 s
and reaching 3.650 mrad. Requested absolute targets reach 2.621640 rad while the
bounded applied target reaches 2.617990 rad. The acknowledged applied targets,
including these corrections, enter subsequent history. This is maintained
controller bounding, not an unrecorded action-semantic change. It begins well
after the first consequential trajectory divergence.

At the first decision the qualified source and learned execution share identical
numeric history, masks and timing, with slightly different rendered images.
The resulting maximum first-six-action prediction change is only 0.000623 mrad
right and 0.000820 mrad left at 61k. Initial image variation alone therefore does
not explain the later millimeter-scale departure. This comparison does not assert
image parity throughout the trajectory.

## Earliest divergence and the right acquisition timer

The first requested actions are small nonzero deltas while the corresponding
reference command is zero. Tool departure exceeds 0.1 mm at command 12 right /
command 14 left, and 1 mm at command 35 (25.7667 s) right / command 22 (25.55 s)
left. These are descriptive physical scales, not new acceptance limits. It reaches
5 mm by 27.45 s right and 26.8333 s left, before either terminal refusal.

The teacher's six-second approach intentionally reaches its nominal endpoint
over the first 70% of that interval and then settles before contact acquisition.
Waiting is therefore present in valid data. The learned right approach has an
inward and postural offset, then retains very little normal progress during and
after this pause:

| Right state | Learned | Operational reference |
|---|---:|---:|
| Physical separation at 29.3167 s | 25.737 mm | 31.705 mm |
| Finger separation lower bound at that time | -2.208 mm | +4.973 mm |
| Reported normal uncertainty, including material range | 35.819 mm | 34.592 mm |
| Continuous acquisition timer starts | 29.3167 s | 31.8167 s |
| Physical separation at 31.6 s | 23.756 mm | 27.245 mm |
| Physical separation at learned stop, 32.3333 s | 22.030 mm | 15.561 mm |
| First valid physical contact | None | 33.9333 s |

The right material normal ranges are almost identical (approximately -8 to
+16 mm); the shifted tool state and dynamic uncertainty cross the plausible
contact interval earlier. Brief earlier seeking intervals reset; 29.3167 s is
the start of the continuous interval that actually expires. The unchanged monitor
counts elapsed time whenever a loaded proposal is seeking plausible but
unconfirmed contact. It does not suspend the timer during a learned pause.

At termination, elapsed seeking is 3.0167 s and accumulated travel is 6.850 mm:
the three-second time allowance expires, while the 30 mm travel allowance does
not. The policy's pause/late normal correction and robust contact plausibility
interact. The label alone establishes neither unsafe motion nor a defective
timeout. A longer timeout would not demonstrate adequate contact acquisition or
force control, and none is changed here.

## Exact left prospective support deficit

At 29.1833 s the measured shoulder-Z position is 0.777583361 rad, inside the main
empirical upper bound 0.785471082 rad. The prospective interval includes:

| Upper travel contribution | rad |
|---|---:|
| Measured speed 0.0744554 rad/s for one control interval | 0.001240923 |
| Qualified acceleration over that interval | 0.004449999 |
| Qualified stopping travel | 0.011466930 |
| Total future upper position | 0.794741213 |
| Main upper bound plus independently qualified 10 N halo | 0.793971082 |
| Unsupported excess | **0.000770131** |

The shoulder's existing upper halo is 8.5 mrad at the required 10 N future stop
load. The current force upper bound of 0.3177 N cannot replace that prospective
load requirement. The alternative certified contact-free motion/stop margin is
-303.649 mm, so the weak-load alternative is unavailable. This negative robust
margin is not a measured penetration. Current independent self/sweep margins are
positive (6.395/0.513 mm), native contact force is zero, there is no physical
violation, and the actual independent 2+4 s stop passes.

The learned shoulder posture is 31.556 mrad above the successful reference at
the same time; the reference passes the same guards. Thus this posture/speed
combination is unnecessary for the demonstrated completion and consumes verified
support. The remaining unsupported upper excursion is small, and the records do
not prove it physically unsafe. Qualifying an extension would require independent
motion and stop evidence at the required load, rather than an epsilon adjustment
or the successful low-load stop alone. It would also not repair the earlier drift,
wrist target bounding or the independently observed right contact limitation.

## Qualified data and the fixed 61k/75k comparison

The audit reads all 114 selected train sources, with frozen 112-source
normalization and no final-test access. There are 33 sources with 3,925 admitted
expert correction commands, and zero admitted erroneous-injection commands.
The first admitted phase is inspect in 15 sources, approach in 49, hold in 26 and
push in 24. First admitted windows of a scoped hold/push source are not normal
initiation examples. There are 104 recordings at 240 Hz and ten retained historical
120 Hz recordings; all use 60 Hz control. This heterogeneity is recorded, not
identified as a cause without a matched intervention.

Measured tool waiting is defined for this analysis as translation below 1 mm/s
and rotation below 0.01 rad/s; moving exceeds 5 mm/s or 0.05 rad/s. The intermediate
band is retained. Across admitted approach commands, 5,390/22,461 are waiting and
15,571 are moving. Phase labels select diagnostic rows and never enter ACT.
There are valid initiation and correction examples; missing all such supervision
is ruled out. Their presence does not establish coverage of the policy's exact
drifted posture, delayed approach or impact state.

The fixed audit uses 392 windows across the two qualified full trajectories,
441 corpus onset/correction cases, and 112 reconstructed learned queries. The
full-cycle windows include every six-command replan through 36 s and fixed
coverage of each phase/transition. Errors below concern the first six predicted
commands, in **mrad per joint per control command**, not completion rates.
All cases are train-only diagnostic rows. Windows can overlap and cross phase
transitions; phase/motion labels describe the current decision. These are not
independent episodes, confidence estimates or held-out generalization results.

| Fixed cases | n | 61k MAE | 75k MAE |
|---|---:|---:|---:|
| Right approach | 72 | 0.1026 | 0.1039 |
| Left approach | 72 | 0.2039 | 0.2675 |
| Right contact | 41 | 0.0947 | 0.1080 |
| Left contact | 41 | 0.1397 | 0.1287 |
| Right push | 30 | 0.1346 | 0.1178 |
| Left push | 30 | 0.1620 | 0.1228 |
| Right physically waiting windows | 56 | 0.0550 | 0.0549 |
| Left physically waiting windows | 53 | 0.0664 | 0.0583 |
| Right physically moving windows | 88 | 0.1092 | 0.1152 |
| Left physically moving windows | 92 | 0.2039 | 0.2418 |
| Corpus normal-onset cases | 192 | 0.1221 | 0.1902 |
| Corpus expert-correction cases | 99 | 0.4132 | 0.3651 |

On the qualified approach states, local recorded-Jacobian projections give mean
61k tool action errors of 0.0754 mm/command right and 0.1076 mm/command left.
These are linearized local action errors, not measured integration or success.
The small per-command error is compatible with consequential accumulated drift.
The 75k file improves some phases/corrections and worsens normal onset/approach;
it is not a uniform repair and is never selected for a favorable native retry.
The old 256 fixed-window aggregate imitation monitor remains useful for training
monitoring but cannot establish phase competence or closed-loop success.

Offline sensitivity probes hold the current numeric state fixed while replacing
past robot contents with that current row, flattening dynamic image history while
preserving the static image, or zeroing images. At 61k, flattening robot history
raises paired full-cycle MAE from 0.1079→0.1369 mrad right and 0.1612→0.2251 left.
Zeroing images raises it to 0.1855/0.1861. Visual content and proprioceptive history
therefore influence predictions and help this fixed diagnostic. Flattening only
dynamic visual history changes MAE to 0.1117 right and 0.1555 left: its benefit is
weak and inconsistent on these cases. These synthetic probes do not prove that
either modality supplies adequate corrective feedback during physical execution,
and do not justify an architecture change.

## Preregistered native panel

Before launch, reserve all three possible complete allocations, including
recording and shutdown: 3×1,865 GPU seconds and 16 GiB. Only two cases are mandatory:
right after 384 fixed reference commands at 31.6 s, and left after 648 at 36 s.
A third right post-contact case at 36 s is conditional on the near-acquisition
case never achieving valid learned physical contact. No normal start is repeated.
Each case allows at most 600 learned commands, with the same 61k checkpoint.

Prefixes execute from normal inspection under live admission/readiness/guards.
The four-second history is built from this execution's images, measured state,
feedback and acknowledged targets; no source image or state is imported. Source
commands end permanently at handoff. Each mixed diagnostic is explicitly
ineligible for full learned completion. Unsupported starts remain unavailable
phase evidence, and physical/independent-stop failure blocks subsequent cases.

| Case | Actual reference prefix | Learned commands | Measured learned outcome | Termination |
|---|---:|---:|---|---|
| Right near acquisition, 31.6 s | 384 | 107 | One valid physical-contact control interval at 33.3833 s; peak native force 13.763 N | Force upper 14.199 N exceeds the unchanged 10 N verified stop-load domain |
| Left post contact, 36 s | 648 | 600 | All 600 commands retain valid physical contact for 10 s; +10.471 deg from 1.214 deg at handoff; learned peak force 1.252 N; zero contact losses | Declared 600-command horizon at 46 s |

Every staging request and measured joint position matches the qualified source
exactly. Actual handoff clocks and the 241-row native histories reconstruct
exactly, with all masks/timing and strictly prior applied-target support. Dynamic
image history spans 27.6–31.6 s right and 32–36 s left and uses only live captures.
There is no teacher action after either handoff. Learned request/applied deltas
agree within 1.20e-7 rad in both diagnostics, without controller corrections.

The right native contact interval is an independent evaluator result; the observed
monitor is still seeking and has not completed its 100 ms contact confirmation.
The load bound is driven by an actual 13.763 N native impact, rather than an
uncertainty-only refusal. Immediately preceding measured tool speeds are about
15 mm/s, versus 4.6–8.3 mm/s at the same clocks in the successful reference. This
supports insufficient entry deceleration as a specific hypothesis; changed
posture/contact geometry also differs, so speed alone is not causally isolated.
The near-acquisition start removes most early approach error and reaches the
surface before the timer; it exposes a distinct operational load limitation.
The unchanged physical 80 N limit is not violated.

The left run demonstrates learned early-push contact maintenance after a valid
start. Its whole mixed-recording peak 5.095 N includes the reference prefix; the
learned-only peak is 1.252 N. Whole-recording controlled opening 11.279 deg is not
credited entirely to the policy: the measured valid-contact increment after
handoff is 10.471 deg. Later opening, hold, release and stabilization remain
unvalidated. The apparent 0.452 mm terminal tool/material separation is distinct
from native mesh contact gaps; native contact and positive observed load remain
valid throughout the measured phase.

The third right case is **not allocated** because its preregistered no-valid-
physical-contact condition is false. It is not replaced with another checkpoint
or a normal-start retry. Both attempted recordings and independent 2+4 s stops
pass, with zero physical violations. Both wrappers return 1 because neither mixed
run completes the full task; neither allocation times out or has a runtime crash.
The left `time_budget` is the intentional finite diagnostic horizon.


## Evidence and limits

Campaign evidence is under
`outputs/evidence/act-a1-first-completion-20261022/diagnosis-20261025/`:
`scope.json`, `offline-plan.json`, `numeric-audit.json`, `supplement.json`,
`offline-audit.json`, `offline-summary.json`, `data-scope-audit.json`,
`offline-predictions.npz` and `native-panel-plan.json`.
Original recordings, failures, all fixed predictions, interventions and resource
records remain in place. Campaign-specific audit scripts remain ignored evidence;
no production source, trained weights or dataset is modified. The two offline
bootstrap errors (identity type and optional Jacobian schema) and their charges
are preserved separately; neither launches a simulator or optimizer.

## Subsequent data audit

The October 26 [[act-a1-complete-demonstration-audit|complete-demonstration audit]]
checks the actual full-phase sources, native continuity, current execution proof
and sampler exposure. It finds coverage/compatibility gaps but does not identify
them as the cause of these failures. It also compares saved earlier 52.295°
opening/hold evidence with these nominal failures without a new historical
evaluation. All results, checkpoint selection and denominators here are preserved.

## Confirmed diagnosis and smallest corrective task

The first consequential failure is learned approach drift under valid causal
observations, preceding both guards. The right timer is consumed by early robust
contact plausibility and a learned pause/late approach correction. A qualified
near-acquisition start then exposes an impact outside the verified operational
stop-load domain. The left normal start instead consumes shoulder-Z future
motion/stop support; its loaded-start diagnostic demonstrates that early-push
contact maintenance is available for 10 s once that initiation bottleneck is
bypassed. There is no common evidence of broken A1 delivery or a policy that
cannot use either visual content or numeric history.

The evidence does not identify architecture capacity, insufficient total training,
the ten historical 120 Hz sources, loss weighting, or missing learner-state
corrections as the unique cause. Nor does it qualify longer acquisition time,
extra shoulder halo, or stronger-load motion. Fixed phase errors and sensitivity
probes diagnose competence and dependence, not full learned success.

The smallest proposed next task is one bounded, train-only correction
qualification for **approach→contact**, reusing the current sources first. Verify
that observation-grounded corrective targets cover the measured left postural
drift before wrist bounding and the right contact-entry/deceleration neighborhood.
Require left prospective support under the original 10 N domain and right
observed contact confirmation without exceeding it. Preserve waiting and moving
examples separately and score normal/tangential tool motion in physical units.
If existing data lack a safe correction at these states, qualify only those
missing correction cases with genuine history before authorizing a narrow
refinement. Do not begin with broader collection, more base training, a new
architecture, threshold relaxation or a lucky checkpoint search.

This is a proposal, not an executed data or training change. A low-load extension
alone is insufficient for the left future 10 N requirement. Any separate support
qualification must establish the required position/speed/load and independent
stop domain; the observed 0.770 mrad deficit must not become an arbitrary epsilon.

Final cumulative accounting is **42.733604 local/31.823056 remote GPUh**,
**198,773 optimizer updates**, and approximately **996.10 GiB retained**. This
investigation adds 0.225583 local GPUh, zero remote GPUh, zero updates and two
native attempts. There are now 63 preserved native attempts: 22 full learned
attempts (0 successes), 14 teacher attempts and 27 diagnostics. The four historical
unavailable reports remain explicit; both new reports are complete. The GPU lease
is released. The original final pair stays 0/2, full learned completion stays 0/22,
development remains gated and final test remains sealed. The paused training
heartbeat is unchanged.
