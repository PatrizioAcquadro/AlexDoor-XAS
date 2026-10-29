# Diverse bounded ACT A1 collection campaign

**Status: complete — all 112 original cells qualified; zero missing.** The campaign
contains 96 short training episodes, eight complete training episodes and eight
development episodes. Every counted source passes its declared physical scope,
measured variation and causal temporal/CUDA gates. Collection is closed. Original
and resumed failures, raw traces and prior usable scopes remain preserved.
No policy training, optimizer update, RL, architecture comparison, learned evaluation
or final-test access occurred. This is simulation-data qualification, not global
stability, hardware validation or learned task success.

The later [[../experiments/act-a1-complete-demonstration-audit|complete-source audit]]
distinguishes original declared-scope qualification from uninterrupted full-task
continuity and current observed execution. Seven of this campaign's eight complete
training sources have native pre-release loading losses; existing masks exclude
their invalid commands and crossing chunks. Their valid local supervision and
original outcomes remain. No active selection or collection result is rewritten.

This is a closed historical campaign. Its declarations and budgets below describe
that experiment; [[status]] owns current authority and the project-wide pause.

## Frozen matrix and conditions

Original baseline `a3c7550`, resumed at `6ae4c59`, branch
`PatrizioAcquadro/b1-policy-pilot`; both inspections found a clean tracked tree.
The original campaign inventory, recordings, failures and resource baselines continue.
[[act-a1-observations-and-time]] owns observation/action semantics;
[[act-a1-four-door-expansion]] retains the prior controls and opening references.

| Split / door | A acquisition | B initial opening | C push to hold | D release to settle | Complete |
|---|---:|---:|---:|---:|---:|
| Train animated-door-1-88abf40 |6|6|6|6|2|
| Train door-2738468b94d74c5f |6|6|6|6|2|
| Train psx-front-005-ee7d5c6 |6|6|6|6|2|
| Train psx-interior-wood-002-02e442c |6|6|6|6|2|
| Development modern-door-2fb8d024 |—|1|—|1|2|
| Development psx-industrial-001-4f5561b |—|1|—|1|2|

Each training family has two pose, two dynamics and two correction conditions:
96 short training episodes, eight complete training episodes and eight development
episodes. Every cell needs a distinct physical recording; extra windows, seeds,
replays and extracted states cannot fill cells. Continuations retain source ancestry
and frozen door/source-group splits. Development never fits weights or normalization.
Exact case declarations, endpoints and aliases are in the preserved
[protocol](../../../outputs/evidence/act-a1-diverse-20261019/protocol.json).

- Pose: high/low contact height 1.11/1.07 m around nominal 1.09 m for late/complete
  episodes, unchanged material-relative normal/orientation. Early low A/B instead
  use 20 mm robot-base retreat and 50 mm precontact distance; high A/B use high contact.
  Admit measured task-space differences after reach, footprint and clearance checks.
- Dynamics: light density/inertia scale .7 (17.5 kg) with damping 4; resistant scale 1.3
  (32.5 kg) with damping 6. COM, axes, geometry and contact friction .5 are unchanged.
  These are synthetic assumptions, not identified hardware parameters. Read composed
  and active PhysX values; qualify separate factors before dependent combinations.
- Corrections: A lateral 20 mm/yaw 5 degree or 15 mm stopped-short normal approach;
  B insufficient approach or bounded contact weakening; C weakening or a brief
  commanded 1.5-times push-rate pulse; D incomplete 8 mm withdrawal or a1 s loaded pause.
  Actual residuals and delivered actions determine the demonstrated scope.

## Physical endpoints, labels and history

A needs at least .5 s contiguous actual loaded contact with valid material/orientation,
finite footprint and no violation. B adds at least 1 degree controlled opening after
acquisition. C needs genuine late push and a current .5 s qualified supported hold,
including the unchanged 1 degree/s door-speed and 2.5 mm tool-travel bounds. D needs
actual unloaded separation>=10 mm and the complete ordinary stabilization tail.
Complete episodes retain each door's original opening, hold, release and tail criteria.
Task outcomes, rather than matching one teacher trajectory, decide success.

Every short physically executes ordinary withdrawal/stabilization and an independent
continuing-physics stop. A data audit found that older C masks supervised cleanup.
All 14 affected admissions passed a data-only revision: original raw/prepared files
and physical outcomes stay intact, fresh masks supervise only push/hold, and
separately charged bounded CUDA validation passed before readmission, with zero
physical recollections and unchanged weights. New C recordings use the same
task-only mask. Recorded phase/tail presence and physical
completion remain required; excluded cleanup transitions need no expert-label chunks. A/B post-endpoint hold/release/tail and C post-hold release/tail are excluded diagnostic
cleanup so short supervision does not teach premature release in a full task.
B's cleanup still executes 3 s with unchanged loss/physical guards; it need not satisfy
an additional complete-task final-hold predicate. The retained front-door low B
producer refusal is reviewed through an explicit metadata-only view of its original
streams/UUID: its task endpoint, safety, tail and stop passed;178/180 cleanup hold
rows were valid. The failed producer outcome remains preserved in
[B-cleanup-scope-review](../../../outputs/evidence/act-a1-diverse-20261019/B-cleanup-scope-review.json).
No in-episode safety stop is cleared or converted to success.

Native action roles are expert 0, injected/excluded 1 and corrective 2. Erroneous
commands and consequences remain causal history; supervision chunks crossing them
are excluded. Corrective labels require actual physical validity. Labels use delivered
minus measured-prestate A1 actions, preserving requested/applied distinctions and
clipping. Simulator truth, teacher phases, intervention roles and outcomes never
enter policy observations. Torque observations retain simulated projected joint-output
provenance; they are neither hardware-verified torque nor direct contact force.

Loaded starts restore robot/door q/dq, targets, gains, controller state and material
contact through fresh physics. Both checked and recording resets pass independently.
Saved states retain source-row ancestry; no source images/history cross a reset.
Acquire 4 s temporal history through actual preceding motion. Never extend an already
completed hold or repeat frames to fill it. C holds execute at least 3 s and continue
only until the original current-window settling predicate passes, bounded at 6 s.
This is controlled settling, not temporal-buffer filling.

## Demonstrated runtime cause and correction

The original six physical attempts yielded three usable episodes and three D-start
failures, leaving 109 missing cells. All original outcomes remain preserved. A fixed
joint-target restoration lost moving contact but stopped safely; extending an already
completed hold exceeded ordinary-motion speed and then destabilized stopping; an
earlier push start passed reconstruction but failed its stop. Old stop profiles were
applied at different states and do not isolate gain effects.

Matched native-state diagnostics identify an explicit-PD/physics discretization pole:
the measured alternating shoulder-Z pole is-1.01898527 per 120 Hz step, versus-1.01896264
predicted from actual local mass plus configured armature; mode cosine .99999978.
Targets stay fixed and native normal force is zero as oscillation grows to the
configured 4.47 rad/s shoulder-Z velocity ceiling. The ceiling is a symptom bound.
Gravity feedforward, target handling, gains, inertia, armature .01 and TGS solver
settings were inspected. No invented inertia/damping/friction or velocity clipping
is used to make the failure disappear.

Only physics integration changes to 240 Hz. Control remains 60 Hz, RGB-D10 Hz, history 4 s,
images 240×384, chunk 12 and execution 6, with visual feature reuse disabled. PD gains,
physical model, velocity ceiling, stop targets and all physical/task criteria stay
unchanged. Extra native samples are not extra policy observations.

| Matched condition |120 Hz|240 Hz|
|---|---:|---:|
| Failed push-state stop: peak speed at 2 s |.32121 rad/s|.00021569 rad/s|
| Same stop: delayed continuation |4.47 rad/s|stable through 6 s|
| Ordinary hold replay: motion peak |.70034 rad/s|.47379 rad/s|
| Hold-replay stop and continuation |4.47 rad/s|.000791 to .00007636 rad/s|

The checked/recording restoration mismatch is independently repaired: both
`prepare_probe` calls explicitly pass `continue_controller=qualify_start_stop`.
A focused regression executes both paths, including targets, gains and controller
state; merely changing the argument default would not test the failure. Runtime
qualification includes both opening directions, fresh push/hold/release/tail, nominal
and cone 10 N loaded/contact-free calibration controls, the original 2 s stop and a
separate 4 s diagnostic continuation. Earlier usable 120 Hz recordings keep their
original demonstrated scope; no continuation is fabricated for them. The scoped
[runtime decision](../../../outputs/evidence/act-a1-diverse-20261019/runtime-qualification-decision.json)
and [all runtime cases](../../../outputs/evidence/act-a1-diverse-20261019/runtime-diagnostic-attempts.json)
include intentional failures and the failed integration wrapper.

Other observed defects were isolated separately from this numerical pole:

- Temporal ingress capacity disagreed with pruning at 4.03 s. The failed prefix stays
  unusable. Capacity now covers the unchanged carry-in/history horizon; six-second
  native 240 Hz regression verifies bounded storage and exact offline/online agreement.
- Ordinary separating contact used a stale geometric estimate. Measured next-control
  gap catch-up qualifies the left complete task, preserving the failed candidates.
- `max(gap catch-up, pressure admittance)` starved pressure beyond first touch in
  resistant full opening. Summing these distinct increments, with the same gain,
 6 mm/s rate bound, offset bounds and unloading priority, passes the matched source
  at 51.53 degrees versus 41.01 required; all 31 rows beyond the former 36 degree loss
  retain valid contact. [Pressure comparison](../../../outputs/evidence/act-a1-diverse-20261019/pressure-composition-qualification.json).
- Recovered C hold used a .1 s load window containing a prior impact while current
  native force was zero. Only that achieved-contact hold now uses the preceding
  native-interval mean for pressure feedback; the original window still qualifies
  loading. Matched left pre-hold motion is exact for 359 rows: failed .317 s hold becomes
  a valid3s hold plus tail and2+4s stops. [C comparison](../../../outputs/evidence/act-a1-diverse-20261019/C-hold-feedback-qualification.json).

## Measured recovery mechanisms and retained negative results

A transfers the full disturbance to eligible expert correction without an excluded
falling taper. A measured-origin cosine return physically corrects it. The original
5.22 mm residual example stays admitted for that honest scope in
`handoff.retained_usable_scope_examples`. Its original cell is now filled once by the
full-transfer follow-up:19.82 mm/5.01 degrees at first eligible supervision,100% plateau
retention,119 injected actions excluded, zero masked corrective actions; residual
falls to 3.28 mm at .5 s and .092 mm at 1 s before acquisition. Both original and new recordings
are retained. [Full-transfer qualification](../../../outputs/evidence/act-a1-diverse-20261019/A1-full-transfer-qualification.json).
Other A targets retain their measured 15 mm or 20 mm task-space errors, not claimed errors
that the perturbation taper already removed.

B1 corrects an actual shortfall before initial loading: approximately 3.18 mm reference
error and 10 mm material gap in the first prototype. It is not loaded-contact recovery.
B2 uses a material-relative 1.5 mm weakening target and suppresses independent positive
gap catch-up during the position return to avoid two competing corrections; unloading
remains active. Two failed right attempts and the left overlap failure are retained.
Qualified first-used native gaps include 1.241 mm right and 1.415 mm left, all 29 injected
actions excluded and no corrective labels masked. Native gap, material separation
and adaptive-reference tracking error are distinct measurements.

C1's .25 s return fits the .8 s weakening pulse and unchanged 1 s bounded recontact window;
its measured .933 mm return implies 5.86 mm/s peak, within the existing 6 mm/s catch-up rate.
Earlier return/near-reach failures remain. Achieved support needs the original .5 s
contiguous valid-contact window before a single bounded force-hold handoff; no state,
velocity or gain is reset. Right first-used native gap is .695 mm; the corrected left
case has .490 mm gap and 1.835 mm reference error,47 excluded actions and zero masked
corrective labels. After reference handoff, native contact and task hold establish
recovery; distance to the old reference is not a fixed-goal success metric.

C2's original clock multiplier was capped by the ordinary lead limit and did not
execute the declared pulse. The corrected teacher accumulates the extra angular
increment and physically commands it. A truncated 52/59 pulse and a complete 59-command
pulse missing the final current-window hold are preserved failures. The qualified
right case settles at 3.2 s,59 actions excluded,41 usable corrective actions. Actual
speed rises from .447 to 1.611 degrees/s; input factor 1.5 is not an assertion that the
physical speed ratio equals 1.5. The recorded extra request .006693347 rad matches the
sum of delivered pulse increments. Other C1/C2 starts retain individual gates.

Phase-state retreat ancestry is independently corrected. A left C2 completed its
pulse/correction/hold but withdrew only 7.05 mm versus 10 mm required: its earlier start
had inherited a retreat from the source's later full hold. Preparation now chooses
an actually achieved pose at least 5 degrees before the selected start, preserving
current q/dq, targets, gains and controller state. The matched comparison retains
739 control rows and all native feedback/efforts exactly through recovery, hold and
the first 3 s withdrawal; corrected final separation is 45.79 mm. Full tail,2+4 s stops,
data and measured recovery pass. Original states and failure remain preserved.
[Retreat qualification](../../../outputs/evidence/act-a1-diverse-20261019/preceding-retreat-qualification.json).

Front C1 at 46.26 degrees reveals insufficient physical reserve for perturbation
momentum and settling. Its predictive guard enters hold before sustained recovery,
then forbidden contact occurs. A hold-phase handoff trial instead reaches .528 rad/s
near Jacobian sigma .00136. Its stop decays to .000222 rad/s at 2 s and .0000969 rad/s at 6 s,
but fails forbidden contact. Matched actual-profile saved-target replays reproduce
all 460 commands,1840 native incoming q/dq/target/effort rows and the failure exactly.
Removing only door collision after the latched stop makes 2+4 s stop checks pass;
retaining contact fails. This isolates contact at an already unsafe geometry, not
a growing PD mode. Contact-off results never qualify that loaded posture or data.
An accidentally copied older-profile pair also remains preserved and charged.
[Contact comparison](../../../outputs/evidence/act-a1-diverse-20261019/front-stop-contact-causal-comparison.json).

Bounded same-state C controller variants were tested and rejected: removing opening
lead loses contact at .267 s; continuous goal capture, earlier native force feedback
and a .1 s gap horizon each reach the unchanged speed criterion before a supported
hold completes. Two contact-stop failures also halt dependent work until examined.
Native gap kicks/preload are diagnostic evidence, not an established general cure.
All trial code, raw failures and charges remain in history; active code returns to
the physically qualified push-handoff/native-feedback controller.

Explicitly declared earlier C1 native starts retain the same 1 mm disturbance,
correction, endpoints and criteria: front 40.31 degrees (84% of its 47.77-degree
full-task minimum), wood 45.42 degrees (91% of 50.15). Both pass full recovery,3 s hold,
ordinary release/stabilization,2+4 s stops and data/diversity checks. Front first-used
reference error is 1.001 mm, native gap .685 mm;47 injected actions are excluded and no
corrective labels are masked. Peak actual speed is .167 rad/s and minimum hold sigma
.0810. This qualifies those late-push scopes; old near-boundary starts stay
unsupported. No disturbance, mass, damping, velocity ceiling or criterion changes.
[C1 reserve decision](../../../outputs/evidence/act-a1-diverse-20261019/C1-reach-reserve-qualification.json).

Front C2 requires a separate supported-state transition. The 40.31-degree start
enters hold before sustained recovery and exceeds ordinary speed near the workspace
boundary. An explicitly declared 38.50-degree native start executes all 59 pulse
commands and one handoff, but fails after 5.13 s holding; its independent 2+4 s stop
remains numerically stable. Wood 43.14-degree C2 passes the original qualified
controller. All source postures and original outcomes remain explicit.

Matched front comparisons rejected force-window filtering, fixed-angle holding,
fixed/aligned orientation, transported pressure goals, acquisition-target changes
and a softer return. Failures include contact loss, unsettled loading, ordinary
speed and forbidden stop contact; decaying native velocity distinguishes these
from the original growing PD mode. All unqualified force/frame implementations
were removed from active code; raw traces, commits and charges remain preserved.
Full failure reasons belong in the linked attempt review, not a replacement success.

The successful front correction carries the actually supported contact/controller
state once at pulse exit, after a real .5 s safe contact window. It then executes
fresh corrective push support before ordinary hold, with no extra opening lead,
physics reset or velocity reset. Keeping the existing .1 N acquisition target through
hold avoids the failed .05 N case's short .433 s loaded window; the original .020 N/.5 s
qualification remains unchanged. The exact qualified ProbeSetup/source is retained;
a full-setup change correctly refused by source-fidelity screening remains charged.
The actual teacher-phase force target is recorded as evaluator diagnostics.

The full pulse and first corrective residual match the earlier failure exactly for
312 control/1249 native rows. The lower-load comparison is exact through 369
control/1477 native rows, until ordinary hold begins. Actual speed rises from .404
to 1.639 degrees/s; it remains 1.546 degrees/s at first eligible correction and falls
to .434 degrees/s in the final qualified window. Extra requested angle is .006572 rad.
All 59 injected commands remain excluded; no corrective labels are masked. The
supported state's reference error is .00051 mm, so this case claims physical pace
recovery, not correction of the old stale 1.48 mm reference. The nominal task passes
4.45 s hold, ordinary release/stabilization and 2+4 s stopping; peak joint speed .181 rad/s.

Separate excluded qualification controls investigate delayed failure. A deliberately
interrupted loaded prefix matches all 636 control/2545 native rows before nominal
release, then holds for 5.983 s, beyond the former 5.13 s failure. Its original incomplete
producer outcome remains failed, with zero eligible commands; independent 2+4 s stop
passes and continuation peak is .0001874 rad/s. A nearby 39-degree start completes 6 s
hold, release/stabilization and 2+4 s stopping, peak ordinary joint speed .252 rad/s and
continuation peak .0003237 rad/s. Stop targets stay fixed; final loaded windows pass
the original contact, speed and tool-travel bounds. Neither control fills a target
cell or normalization set. These observations qualify only the measured correction
scope; other doors/directions retain their own independent qualifications.
[C2 causal and extended qualification](../../../outputs/evidence/act-a1-diverse-20261019/C2-corrective-control-qualification.json).

D1 targets 8 mm separation in the moving material frame, preserving the incomplete
withdrawal until corrective supervision; the right first-used separation is 7.33 mm
with 20.85 mm reference error. It exceeds 10 mm actual separation within .25 s;59 erroneous
actions stay excluded. D2 physically follows supported contact during its 1 s excluded
pause. Fixed-world targets would let panel motion cancel that disturbance.

The old D2 maximum-opening posture fails at .533 rad/s; removing inherited opening
lead still fails at .522, and adding 1.5 s linear reach reserve still fails at .530.
Native shoulder-Y velocities have the same sign within intervals, with 60 Hz tracking
lag at Jacobian sigma .00136. This is a reach/phase-planning failure, distinct from
the alternating PD pole; its independent fixed-target stops remain stable.
The supported task-goal correction uses the same saved start and 84 physically exact
push rows, then ends pushing once the original full-task opening goal and genuine
4.4 s pre-release context are met. The original 3 s hold,1 s pause,.5 s corrective blend,
release, stabilization and 2+4 s stops all still execute. Held opening 50.53 degrees
exceeds 41.01 required; peak speed .498 rad/s, first-supervised reference error 19.07 mm,
real loaded native gap 1.26 micrometres,59 excluded actions and zero masked corrections.
[Task-goal D2 comparison](../../../outputs/evidence/act-a1-diverse-20261019/D2-task-goal-qualification.json).
The former maximum-opening pause posture remains unsupported. Other D2 candidates
start from actual supported windows meeting their per-door task goal and require
fresh reconstruction and full per-case qualification.

## Diversity and full-task scope

All 32 training pose cells and eight complete training cells pass their individual
physical, measured-diversity and data gates. Observed late/full height changes are
approximately+20/-20 mm. Early retreat cases retain measured 20 mm base/precontact
differences. Identical converged contact alone would not establish diversity.

| Train door | High / low held opening | Original minimum |
|---|---:|---:|
| animated-door-1 |52.89/49.96 degrees|41.01 degrees|
| door-2738468 |53.54/50.95 degrees|42.03 degrees|
| PSX front |52.86/49.84 degrees|47.77 degrees|
| PSX wood |55.95/52.84 degrees|50.15 degrees|

All 32 training dynamics cells and all 16 light/resistant family pairs pass.
The 16 early A/B cells also retain independent factor controls. Separate mass-only controls are physically safe
but small effects remain negative coverage evidence. Three nominal pre-pulse B
prefixes replace redundant auxiliary launches, never additional coverage cells.
The right resistant A at .1 N fails diversity and remains retained. One prospectively
matched nominal/resistant 1 N acquisition comparison passes .546 versus .705 degree
opening response (23% difference); the negligible force change is not claimed as
evidence. This explicit excitation change remains within the separately checked
loaded scope. Other actual task/physical criteria stay unchanged.

Dynamics admission requires at least 10% difference and a material .1 degree angle,
.1 degree/s speed,.02 N native mean force or .02Ns impulse. These are coverage criteria,
not relaxed safety bounds. Late targets need qualified complete same-dynamics
sources, fresh resets and actual native-force/kinematic corroboration. The mean
of per-control force maxima cannot alone establish force variation across 120/240 Hz.
The [direct light/resistant audit](../../../outputs/evidence/act-a1-diverse-20261019/dynamics-pair-diversity-audit.json)
reports all 16 planned pairs, posture/context differences and availability. Conditional
response diversity is not an isolated local mass estimate. Resistant source push
budgets allow 225 s (150 s×6/4 damping),300 s total simulation, unchanged 1200 s wall guard,
speeds, task criteria and observations per physical second.

The first development low-height full episode is a retained task failure:
43.2708 degrees sustained versus 43.5244 required, with successful ordinary tail and
2+4 s stopping. Its terminal hold angle 43.5841 already crosses the goal, but the
current .5 s valid-support minimum does not; unloaded release angle cannot count.
Actual hold/release Jacobian sigma remains>=.0396, with no physical violation.
A scoped candidate retains the original 3 s minimum and continues the same supported
hold only until the unchanged opening-goal window passes, bounded at 6 s. All push
planning, physical/model/stop criteria and temporal/action rates stay fixed. This
is incomplete task-goal execution, not extension of a completed hold for history.
The matched comparison passes: all 5679 control rows and native q/dq, targets and
computed/applied efforts through the original 3 s hold are exactly equal. At3.417 s
the valid window reaches 43.5294 degrees versus 43.5244 required. Ordinary tail,
2+4 s stopping, actual-data/CUDA and measured low-height variation all pass; the
original failure remains. The same rule is declared before launching remaining
complete sources/development targets; each retains its own physical/data gates.
[Qualification](../../../outputs/evidence/act-a1-diverse-20261019/development-task-goal-hold-qualification.json).

All eight declared development cells now pass physical, measured-variation and
causal data/CUDA gates. Both industrial full variants retain their 37.60-degree
opening requirement: high/low hold 43.45/40.38 degrees and measured height changes
+19.99/-20.02 mm. Both B1/D1 correction pairs retain actual residuals and complete
ordinary tails. Development sources remain excluded from normalization fitting;
weights and the final test remain untouched.

## Data admission, budgets and handoff

Each admitted source passes native integrity, clocks, causal joins, actual A1 labels,
requested/applied actions, intervention masks, phase transitions, true tails and
both-direction offline/online phase observations. Bounded CUDA forward/backward
checks consume actual prepared data with exactly unchanged initialized weights and
zero optimizer updates. The original six-training-source normalization is frozen
for compatibility; no development or new-source fit occurs. This is pipeline
validation, not policy learning or learned evaluation.

New recordings additionally retain native q/dq, delivered joint targets, velocity
targets, pre-clipping computed PD-plus-feedforward effort, clipped command effort
and gravity feedforward as evaluator diagnostics, excluded from policy inputs.
Projected solver torque remains a separate simulated observation. Native samples
are neither clipped nor smoothed for validation. The inspection prefix and teacher
probe trace have distinct clocks: the development join fix compares the proper
probe suffix while auditing every prefix q/dq/target row. Its first failed validation
allocation remains charged; physical recording is reused, not recollected.

A later resistant-D validation refuses one initial-push observation NPZ because
its ZIP CRC metadata disagrees with the compressed payload. The actual stored
image payload matches the causal raw-data reconstruction byte for byte; the other
fields and four phase snapshots are intact. A separate reviewed copy changes only
the local/central CRC fields, preserving all compressed payloads and the original
file. Strict causal/CUDA validation then passes without recollecting physics. Both
failed CUDA allocations remain charged. The metadata-corruption origin remains
undetermined; no observation value, native trace or original outcome is changed.
[Container review](../../../outputs/evidence/act-a1-diverse-20261019/train-3-D-dynamics-resistant/snapshot-crc-review.json).

Original baseline:18.661711 conservative GPU hours,560.493949 GiB logical evidence
and 22 historical physical attempts. The user raised only the campaign attempt
ceiling from 160 to 220 total on October 20; all existing costs remain. Other caps:
12 additional charged GPU wall-hours,400 GiB additional retained evidence,40 h/1TiB
cumulative. Count diagnostics, failures, qualification, startup/shutdown, stopping
continuations and CUDA validation. Reserve 60 s workload and 12 GiB disk headroom;
one simulation at a time, preserve unrelated workloads. Resource guards never
shorten a success sequence. No unchanged favorable retry is permitted.

Physical limits, empirical calibration bounds, reset fidelity and numerical tolerance
remain separate. Parked reset screening permits one outward float 32 step only,
records original signed margins and rejects real/two-step excursions and NaN.
Small empirical mismatches do not establish danger; growing oscillations require
causal repair. No physical threshold, velocity ceiling, stop duration or outcome is
changed to pass a failed case.

Ignored evidence is preserved under `outputs/evidence/act-a1-diverse-20261019/`;
Git cannot recover those payloads. These records are canonical for current counts:

- [All 112 qualified cells](../../../outputs/evidence/act-a1-diverse-20261019/coverage.json).
- [Every attempt and original failure](../../../outputs/evidence/act-a1-diverse-20261019/attempts.json).
- [Admitted train/development, retained scope, masks and ancestry](../../../outputs/evidence/act-a1-diverse-20261019/handoff.json).
- [Inventory and pipeline audit](../../../outputs/evidence/act-a1-diverse-20261019/admitted-inventory-audit.json).
- [Checked/recording restoration comparisons](../../../outputs/evidence/act-a1-diverse-20261019/restoration-path-comparison.json).
- [Resource totals and original baselines](../../../outputs/evidence/act-a1-diverse-20261019/resource-summary.json).
- [Launch halt/qualification state](../../../outputs/evidence/act-a1-diverse-20261019/campaign-stop.json).
- [Local commit inventory](../../../outputs/evidence/act-a1-diverse-20261019/local-commits.json).

Completion requires every original gate: A all 112 varied physical scopes; B admissible
temporal data for all counted recordings; C complete pipeline/CUDA checks with no
weight change and complete resource accounting. A partial inventory or one repaired
runtime does not complete the goal. Continue independent qualified scopes while a
local mechanism is diagnosed; shared unstable stopping halts dependent collection.

All original gates A/B/C are satisfied. The completed inventory contains 113,721
training windows and 23,705 development windows with unchanged weights, frozen
normalization/splits and zero optimizer updates. Fifty admitted loaded starts pass
checked/recording reconstruction comparison;48 have direct controller/gain readback.
Two early sources lack that readback but retain bitwise matching measured reset
feedback paths and original physical/data acceptance; no measurement is invented.

Final workload totals are 194/220 campaign attempts,56 retained failed/refused or
intentional negative checks, and two separately charged failed CUDA allocations.
Additional charged GPU workload is 9.410130 h of 12 h; cumulative upper allocation is
28.071841 h of 40 h. Additional logical evidence is approximately 349.97 GiB of 400 GiB,
cumulative 910.46 GiB of 1024 GiB; the resource record owns exact as-of byte counts.
No allocation is pending. The incomplete loaded-stop diagnostic remains in the
failure denominator, and no safety failure was cleared. The reviewed CRC-only
container derivative and its unknown corruption origin remain documented above.
No unresolved blocker remains for the 112 recorded scopes; excluded near-boundary
starts, global stability, hardware and policy learning remain unqualified.

- [Completion proof](../../../outputs/evidence/act-a1-diverse-20261019/completion-proof.json).
- [Full report and all local commits](../../../outputs/evidence/act-a1-diverse-20261019/REPORT.md).
- [All failed/refused attempts and original outcomes](../../../outputs/evidence/act-a1-diverse-20261019/failed-attempts-review.md).
- [Measured correction magnitudes and unavailable fields](../../../outputs/evidence/act-a1-diverse-20261019/correction-scope-summary.json).
