# Diverse bounded ACT A1 collection campaign

**Status: scoped collection active; goal unmet.** Current fully qualified counts,
remaining cells and any temporarily held data revisions are in the canonical
coverage and handoff records linked below. The14 earlier C stores have completed data-only cleanup-mask revalidation.
The objective remains all112 physically qualified, meaningfully varied, temporally
valid target cells. No policy training, optimizer update, RL, architecture comparison,
learned evaluation or final-test access is part of this campaign.

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

- Pose: high/low contact height1.11/1.07m around nominal1.09m for late/complete
  episodes, unchanged material-relative normal/orientation. Early low A/B instead
  use20mm robot-base retreat and50mm precontact distance; high A/B use high contact.
  Admit measured task-space differences after reach, footprint and clearance checks.
- Dynamics: light density/inertia scale.7 (17.5kg) with damping4; resistant scale1.3
  (32.5kg) with damping6. COM, axes, geometry and contact friction.5 are unchanged.
  These are synthetic assumptions, not identified hardware parameters. Read composed
  and active PhysX values; qualify separate factors before dependent combinations.
- Corrections: A lateral20mm/yaw5degree or15mm stopped-short normal approach;
  B insufficient approach or bounded contact weakening; C weakening or a brief
  commanded1.5-times push-rate pulse; D incomplete8mm withdrawal or a1s loaded pause.
  Actual residuals and delivered actions determine the demonstrated scope.

## Physical endpoints, labels and history

A needs at least.5s contiguous actual loaded contact with valid material/orientation,
finite footprint and no violation. B adds at least1degree controlled opening after
acquisition. C needs genuine late push and a current.5s qualified supported hold,
including the unchanged1degree/s door-speed and2.5mm tool-travel bounds. D needs
actual unloaded separation>=10mm and the complete ordinary stabilization tail.
Complete episodes retain each door's original opening, hold, release and tail criteria.
Task outcomes, rather than matching one teacher trajectory, decide success.

Every short physically executes ordinary withdrawal/stabilization and an independent
continuing-physics stop. A data audit found that older C masks supervised cleanup.
All14 affected admissions passed a data-only revision: original raw/prepared files
and physical outcomes stay intact, fresh masks supervise only push/hold, and
separately charged bounded CUDA validation passed before readmission, with zero
physical recollections and unchanged weights. New C recordings use the same
task-only mask. Recorded phase/tail presence and physical
completion remain required; excluded cleanup transitions need no expert-label chunks. A/B post-endpoint hold/release/tail and C post-hold release/tail are excluded diagnostic
cleanup so short supervision does not teach premature release in a full task.
B's cleanup still executes3s with unchanged loss/physical guards; it need not satisfy
an additional complete-task final-hold predicate. The retained front-door low B
producer refusal is reviewed through an explicit metadata-only view of its original
streams/UUID: its task endpoint, safety, tail and stop passed;178/180 cleanup hold
rows were valid. The failed producer outcome remains preserved in
[B-cleanup-scope-review](../../../outputs/evidence/act-a1-diverse-20261019/B-cleanup-scope-review.json).
No in-episode safety stop is cleared or converted to success.

Native action roles are expert0, injected/excluded1 and corrective2. Erroneous
commands and consequences remain causal history; supervision chunks crossing them
are excluded. Corrective labels require actual physical validity. Labels use delivered
minus measured-prestate A1 actions, preserving requested/applied distinctions and
clipping. Simulator truth, teacher phases, intervention roles and outcomes never
enter policy observations. Torque observations retain simulated projected joint-output
provenance; they are neither hardware-verified torque nor direct contact force.

Loaded starts restore robot/door q/dq, targets, gains, controller state and material
contact through fresh physics. Both checked and recording resets pass independently.
Saved states retain source-row ancestry; no source images/history cross a reset.
Acquire4s temporal history through actual preceding motion. Never extend an already
completed hold or repeat frames to fill it. C holds execute at least3s and continue
only until the original current-window settling predicate passes, bounded at6s.
This is controlled settling, not temporal-buffer filling.

## Demonstrated runtime cause and correction

The original six physical attempts yielded three usable episodes and three D-start
failures, leaving109 missing cells. All original outcomes remain preserved. A fixed
joint-target restoration lost moving contact but stopped safely; extending an already
completed hold exceeded ordinary-motion speed and then destabilized stopping; an
earlier push start passed reconstruction but failed its stop. Old stop profiles were
applied at different states and do not isolate gain effects.

Matched native-state diagnostics identify an explicit-PD/physics discretization pole:
the measured alternating shoulder-Z pole is-1.01898527 per120Hz step, versus-1.01896264
predicted from actual local mass plus configured armature; mode cosine.99999978.
Targets stay fixed and native normal force is zero as oscillation grows to the
configured4.47rad/s shoulder-Z velocity ceiling. The ceiling is a symptom bound.
Gravity feedforward, target handling, gains, inertia, armature.01 and TGS solver
settings were inspected. No invented inertia/damping/friction or velocity clipping
is used to make the failure disappear.

Only physics integration changes to240Hz. Control remains60Hz, RGB-D10Hz, history4s,
images240×384, chunk12 and execution6, with visual feature reuse disabled. PD gains,
physical model, velocity ceiling, stop targets and all physical/task criteria stay
unchanged. Extra native samples are not extra policy observations.

| Matched condition |120Hz|240Hz|
|---|---:|---:|
| Failed push-state stop: peak speed at2s |.32121rad/s|.00021569rad/s|
| Same stop: delayed continuation |4.47rad/s|stable through6s|
| Ordinary hold replay: motion peak |.70034rad/s|.47379rad/s|
| Hold-replay stop and continuation |4.47rad/s|.000791 to.00007636rad/s|

The checked/recording restoration mismatch is independently repaired: both
`prepare_probe` calls explicitly pass `continue_controller=qualify_start_stop`.
A focused regression executes both paths, including targets, gains and controller
state; merely changing the argument default would not test the failure. Runtime
qualification includes both opening directions, fresh push/hold/release/tail, nominal
and cone10N loaded/contact-free calibration controls, the original2s stop and a
separate4s diagnostic continuation. Earlier usable120Hz recordings keep their
original demonstrated scope; no continuation is fabricated for them. The scoped
[runtime decision](../../../outputs/evidence/act-a1-diverse-20261019/runtime-qualification-decision.json)
and [all runtime cases](../../../outputs/evidence/act-a1-diverse-20261019/runtime-diagnostic-attempts.json)
include intentional failures and the failed integration wrapper.

Other observed defects were isolated separately from this numerical pole:

- Temporal ingress capacity disagreed with pruning at4.03s. The failed prefix stays
  unusable. Capacity now covers the unchanged carry-in/history horizon; six-second
  native240Hz regression verifies bounded storage and exact offline/online agreement.
- Ordinary separating contact used a stale geometric estimate. Measured next-control
  gap catch-up qualifies the left complete task, preserving the failed candidates.
- `max(gap catch-up, pressure admittance)` starved pressure beyond first touch in
  resistant full opening. Summing these distinct increments, with the same gain,
 6mm/s rate bound, offset bounds and unloading priority, passes the matched source
  at51.53degrees versus41.01 required; all31 rows beyond the former36degree loss
  retain valid contact. [Pressure comparison](../../../outputs/evidence/act-a1-diverse-20261019/pressure-composition-qualification.json).
- Recovered C hold used a.1s load window containing a prior impact while current
  native force was zero. Only that achieved-contact hold now uses the preceding
  native-interval mean for pressure feedback; the original window still qualifies
  loading. Matched left pre-hold motion is exact for359 rows: failed.317s hold becomes
  a valid3s hold plus tail and2+4s stops. [C comparison](../../../outputs/evidence/act-a1-diverse-20261019/C-hold-feedback-qualification.json).

## Measured recovery mechanisms and retained negative results

A transfers the full disturbance to eligible expert correction without an excluded
falling taper. A measured-origin cosine return physically corrects it. The original
5.22mm residual example stays admitted for that honest scope in
`handoff.retained_usable_scope_examples`. Its original cell is now filled once by the
full-transfer follow-up:19.82mm/5.01degrees at first eligible supervision,100% plateau
retention,119 injected actions excluded, zero masked corrective actions; residual
falls to3.28mm at.5s and.092mm at1s before acquisition. Both original and new recordings
are retained. [Full-transfer qualification](../../../outputs/evidence/act-a1-diverse-20261019/A1-full-transfer-qualification.json).
Other A targets retain their measured15mm or20mm task-space errors, not claimed errors
that the perturbation taper already removed.

B1 corrects an actual shortfall before initial loading: approximately3.18mm reference
error and10mm material gap in the first prototype. It is not loaded-contact recovery.
B2 uses a material-relative1.5mm weakening target and suppresses independent positive
gap catch-up during the position return to avoid two competing corrections; unloading
remains active. Two failed right attempts and the left overlap failure are retained.
Qualified first-used native gaps include1.241mm right and1.415mm left, all29 injected
actions excluded and no corrective labels masked. Native gap, material separation
and adaptive-reference tracking error are distinct measurements.

C1's.25s return fits the.8s weakening pulse and unchanged1s bounded recontact window;
its measured.933mm return implies5.86mm/s peak, within the existing6mm/s catch-up rate.
Earlier return/near-reach failures remain. Achieved support needs the original.5s
contiguous valid-contact window before a single bounded force-hold handoff; no state,
velocity or gain is reset. Right first-used native gap is.695mm; the corrected left
case has.490mm gap and1.835mm reference error,47 excluded actions and zero masked
corrective labels. After reference handoff, native contact and task hold establish
recovery; distance to the old reference is not a fixed-goal success metric.

C2's original clock multiplier was capped by the ordinary lead limit and did not
execute the declared pulse. The corrected teacher accumulates the extra angular
increment and physically commands it. A truncated52/59 pulse and a complete59-command
pulse missing the final current-window hold are preserved failures. The qualified
right case settles at3.2s,59 actions excluded,41 usable corrective actions. Actual
speed rises from.447 to1.611degrees/s; input factor1.5 is not an assertion that the
physical speed ratio equals1.5. The recorded extra request.006693347rad matches the
sum of delivered pulse increments. Other C1/C2 starts retain individual gates.

Phase-state retreat ancestry is independently corrected. A left C2 completed its
pulse/correction/hold but withdrew only7.05mm versus10mm required: its earlier start
had inherited a retreat from the source's later full hold. Preparation now chooses
an actually achieved pose at least5degrees before the selected start, preserving
current q/dq, targets, gains and controller state. The matched comparison retains
739 control rows and all native feedback/efforts exactly through recovery, hold and
the first3s withdrawal; corrected final separation is45.79mm. Full tail,2+4s stops,
data and measured recovery pass. Original states and failure remain preserved.
[Retreat qualification](../../../outputs/evidence/act-a1-diverse-20261019/preceding-retreat-qualification.json).

Front C1 at46.26degrees reveals insufficient physical reserve for perturbation
momentum and settling. Its predictive guard enters hold before sustained recovery,
then forbidden contact occurs. A hold-phase handoff trial instead reaches.528rad/s
near Jacobian sigma.00136. Its stop decays to.000222rad/s at2s and.0000969rad/s at6s,
but fails forbidden contact. Matched actual-profile saved-target replays reproduce
all460 commands,1840 native incoming q/dq/target/effort rows and the failure exactly.
Removing only door collision after the latched stop makes2+4s stop checks pass;
retaining contact fails. This isolates contact at an already unsafe geometry, not
a growing PD mode. Contact-off results never qualify that loaded posture or data.
An accidentally copied older-profile pair also remains preserved and charged.
[Contact comparison](../../../outputs/evidence/act-a1-diverse-20261019/front-stop-contact-causal-comparison.json).

Bounded same-state C controller variants were tested and rejected: removing opening
lead loses contact at.267s; continuous goal capture, earlier native force feedback
and a.1s gap horizon each reach the unchanged speed criterion before a supported
hold completes. Two contact-stop failures also halt dependent work until examined.
Native gap kicks/preload are diagnostic evidence, not an established general cure.
All trial code, raw failures and charges remain in history; active code returns to
the physically qualified push-handoff/native-feedback controller.

Explicitly declared earlier C1 native starts retain the same1mm disturbance,
correction, endpoints and criteria: front40.31degrees (84% of its47.77-degree
full-task minimum), wood45.42degrees (91% of50.15). Both pass full recovery,3s hold,
ordinary release/stabilization,2+4s stops and data/diversity checks. Front first-used
reference error is1.001mm, native gap.685mm;47 injected actions are excluded and no
corrective labels are masked. Peak actual speed is.167rad/s and minimum hold sigma
.0810. This qualifies those late-push scopes; old near-boundary starts stay
unsupported. No disturbance, mass, damping, velocity ceiling or criterion changes.
[C1 reserve decision](../../../outputs/evidence/act-a1-diverse-20261019/C1-reach-reserve-qualification.json).

The front C2 at40.31degrees also enters hold before its sustained push handoff,
then fails ordinary speed near the workspace boundary; its independent2+4s stop
passes without a growing native mode. Earlier front38.50-degree and wood43.14-degree
native late-push candidates are declared explicitly with source ancestry and the
same initial Jacobian reserve. The earlier front attempt executes the full59-action
pulse and one sustained handoff, but still reaches the speed criterion after5.13s
hold; its2+4s stop is stable. Mean native load is.640N versus.05N target. On the
preserved observations, short-interval feedback produces nine positive geometric
catch-up ticks versus two with the existing.1s window. This arithmetic does not
predict physical behavior. One matched C2 prototype tests only the.1s force window
after handoff; C1 feedback, source, model, low-level PD, full pulse, endpoints and
all criteria are unchanged. That comparison fails the ordinary speed criterion
at4.15s hold and forbidden contact during stopping, despite numerical velocity
decay through6s; it is rejected and prior native feedback restored. A subsequent
matched C2 prototype holds the achieved handoff angle during ordinary contact
holding instead of advancing that angular goal with the moving door. Normal
force/tangent support, all guards, full pulse and separate independent stop remain
unchanged. Qualification is pending. No failed attempt is replaced silently or
converted to a success.

D1 targets8mm separation in the moving material frame, preserving the incomplete
withdrawal until corrective supervision; the right first-used separation is7.33mm
with20.85mm reference error. It exceeds10mm actual separation within.25s;59 erroneous
actions stay excluded. D2 physically follows supported contact during its1s excluded
pause. Fixed-world targets would let panel motion cancel that disturbance.

The old D2 maximum-opening posture fails at.533rad/s; removing inherited opening
lead still fails at.522, and adding1.5s linear reach reserve still fails at.530.
Native shoulder-Y velocities have the same sign within intervals, with60Hz tracking
lag at Jacobian sigma.00136. This is a reach/phase-planning failure, distinct from
the alternating PD pole; its independent fixed-target stops remain stable.
The supported task-goal correction uses the same saved start and84 physically exact
push rows, then ends pushing once the original full-task opening goal and genuine
4.4s pre-release context are met. The original3s hold,1s pause,.5s corrective blend,
release, stabilization and2+4s stops all still execute. Held opening50.53degrees
exceeds41.01 required; peak speed.498rad/s, first-supervised reference error19.07mm,
real loaded native gap1.26micrometres,59 excluded actions and zero masked corrections.
[Task-goal D2 comparison](../../../outputs/evidence/act-a1-diverse-20261019/D2-task-goal-qualification.json).
The former maximum-opening pause posture remains unsupported. Other D2 candidates
start from actual supported windows meeting their per-door task goal and require
fresh reconstruction and full per-case qualification.

## Diversity and full-task scope

All32 training pose cells and eight complete training cells pass their individual
physical, measured-diversity and data gates. Observed late/full height changes are
approximately+20/-20mm. Early retreat cases retain measured20mm base/precontact
differences. Identical converged contact alone would not establish diversity.

| Train door | High / low held opening | Original minimum |
|---|---:|---:|
| animated-door-1 |52.89/49.96degrees|41.01degrees|
| door-2738468 |53.54/50.95degrees|42.03degrees|
| PSX front |52.86/49.84degrees|47.77degrees|
| PSX wood |55.95/52.84degrees|50.15degrees|

All16 early A/B dynamics cells pass. Separate mass-only controls are physically safe
but small effects remain negative coverage evidence. Three nominal pre-pulse B
prefixes replace redundant auxiliary launches, never additional coverage cells.
The right resistant A at.1N fails diversity and remains retained. One prospectively
matched nominal/resistant1N acquisition comparison passes.546 versus.705degree
opening response (23% difference); the negligible force change is not claimed as
evidence. This explicit excitation change remains within the separately checked
loaded scope. Other actual task/physical criteria stay unchanged.

Dynamics admission requires at least10% difference and a material.1degree angle,
.1degree/s speed,.02N native mean force or.02Ns impulse. These are coverage criteria,
not relaxed safety bounds. Late targets need qualified complete same-dynamics
sources, fresh resets and actual native-force/kinematic corroboration. The mean
of per-control force maxima cannot alone establish force variation across120/240Hz.
The [direct light/resistant audit](../../../outputs/evidence/act-a1-diverse-20261019/dynamics-pair-diversity-audit.json)
reports all16 planned pairs, posture/context differences and availability. Conditional
response diversity is not an isolated local mass estimate. Resistant source push
budgets allow225s (150s×6/4 damping),300s total simulation, unchanged1200s wall guard,
speeds, task criteria and observations per physical second.

The first development low-height full episode is a retained task failure:
43.2708degrees sustained versus43.5244 required, with successful ordinary tail and
2+4s stopping. Its terminal hold angle43.5841 already crosses the goal, but the
current.5s valid-support minimum does not; unloaded release angle cannot count.
Actual hold/release Jacobian sigma remains>=.0396, with no physical violation.
A scoped candidate retains the original3s minimum and continues the same supported
hold only until the unchanged opening-goal window passes, bounded at6s. All push
planning, physical/model/stop criteria and temporal/action rates stay fixed. This
is incomplete task-goal execution, not extension of a completed hold for history.
The matched comparison passes: all5679 control rows and native q/dq, targets and
computed/applied efforts through the original3s hold are exactly equal. At3.417s
the valid window reaches43.5294degrees versus43.5244 required. Ordinary tail,
2+4s stopping, actual-data/CUDA and measured low-height variation all pass; the
original failure remains. The same rule is declared before launching remaining
complete sources/development targets; each retains its own physical/data gates.
[Qualification](../../../outputs/evidence/act-a1-diverse-20261019/development-task-goal-hold-qualification.json).

All eight declared development cells now pass physical, measured-variation and
causal data/CUDA gates. Both industrial full variants retain their37.60-degree
opening requirement: high/low hold43.45/40.38degrees and measured height changes
+19.99/-20.02mm. Both B1/D1 correction pairs retain actual residuals and complete
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

Original baseline:18.661711 conservative GPU hours,560.493949GiB logical evidence
and22 historical physical attempts. The user raised only the campaign attempt
ceiling from160 to220 total on October20; all existing costs remain. Other caps:
12 additional charged GPU wall-hours,400GiB additional retained evidence,40h/1TiB
cumulative. Count diagnostics, failures, qualification, startup/shutdown, stopping
continuations and CUDA validation. Reserve60s workload and12GiB disk headroom;
one simulation at a time, preserve unrelated workloads. Resource guards never
shorten a success sequence. No unchanged favorable retry is permitted.

Physical limits, empirical calibration bounds, reset fidelity and numerical tolerance
remain separate. Parked reset screening permits one outward float32 step only,
records original signed margins and rejects real/two-step excursions and NaN.
Small empirical mismatches do not establish danger; growing oscillations require
causal repair. No physical threshold, velocity ceiling, stop duration or outcome is
changed to pass a failed case.

Ignored evidence is preserved under `outputs/evidence/act-a1-diverse-20261019/`;
Git cannot recover those payloads. These records are canonical for current counts:

- [All112 cells and remaining gaps](../../../outputs/evidence/act-a1-diverse-20261019/coverage.json).
- [Every attempt and original failure](../../../outputs/evidence/act-a1-diverse-20261019/attempts.json).
- [Admitted train/development, retained scope, masks and ancestry](../../../outputs/evidence/act-a1-diverse-20261019/handoff.json).
- [Inventory and pipeline audit](../../../outputs/evidence/act-a1-diverse-20261019/admitted-inventory-audit.json).
- [Checked/recording restoration comparisons](../../../outputs/evidence/act-a1-diverse-20261019/restoration-path-comparison.json).
- [Resource totals and original baselines](../../../outputs/evidence/act-a1-diverse-20261019/resource-summary.json).
- [Launch halt/qualification state](../../../outputs/evidence/act-a1-diverse-20261019/campaign-stop.json).
- [Local commit inventory](../../../outputs/evidence/act-a1-diverse-20261019/local-commits.json).

Completion requires every original gate: A all112 varied physical scopes; B admissible
temporal data for all counted recordings; C complete pipeline/CUDA checks with no
weight change and complete resource accounting. A partial inventory or one repaired
runtime does not complete the goal. Continue independent qualified scopes while a
local mechanism is diagnosed; shared unstable stopping halts dependent collection.

The fixed-angle prototype decelerates1.28→1.06degrees/s with no physical-speed
violation, but loses ordinary-hold contact after.283s; stopping passes. This exposes
premature C2 recovery completion: contact-only support had cleared the corrective
state while excess pace remained. The next prototype keeps the captured goal in
corrective settling with the existing1s bounded recontact envelope. It requires a
fresh.5s valid low-speed contact window before starting the original3–6s ordinary
hold, whose.25s loss guard remains unchanged. Forward-opening forecasts do not
advance a captured stationary corrective goal; all native physical/IK, contact,
footprint and independent stopping criteria remain active. Qualification is pending.

Corrective settling with the fully fixed pose decelerates safely but fails sustained
loading before ordinary hold starts; its complete cleanup and2+4s stop pass.
Orientation error grows while the panel coasts, normal support reaches its existing
offset bound, and loading is intermittent. The next matched prototype changes only
contact orientation to track the panel, retaining the captured pressure-position
reference, normal/tangential gains and all correction/hold/physical/stop criteria.
No earlier failed run is admitted. Qualification remains pending.

Orientation alignment removes the angular mismatch but still fails sustained
contact when the normal-support offset reaches7.5mm relative to the old captured
origin. Speed decays safely and2+4s stopping passes. The next prototype carries the
desired world pressure point, re-expresses it in the current material frame without
moving it, and applies the unchanged normal/tangential increments and bounds.
Panel orientation remains aligned. This avoids charging panel-frame motion to a
pressure offset or feeding that motion directly into the desired world point.
A rotation/translation regression verifies world-position preservation. Actual
corrective settling, ordinary hold and independent stopping remain required.

Transported goals prevent offset windup and decelerate safely, but sustained
contact still fails: mean native force is.013N, below the unchanged.02N loaded
criterion. Corrective settling had prematurely selected the.05N hold target while
reacquisition remained pending. The next matched prototype restores the existing
.1N push/recontact target until ordinary hold actually begins, where.05N remains.
No new force/gain/bound value or admission criterion is introduced.

Restoring acquisition pressure does not establish sustained load. Saved desired
normal increments match controller arithmetic within3.5e-17m. A matched comparison
now uses the existing.1s force-feedback window with the transported world goal and
original.1/.05N phase targets. C1 and raw240Hz forces/velocities remain unchanged.
The earlier filtered trial used the rejected moving world goal, so it does not
qualify this different control model. Physical qualification remains pending.

The filtered transported-goal trial remains physically within speed limits but
exhausts15s with intermittent load and no ordinary hold. It remains failed. The
next matched prototype restores native feedback and fixes only gripper orientation
at the captured handoff while retaining transported world-pressure coordinates.
Earlier fixed-orientation results had a different, saturated fixed-origin offset.
All native evidence, force targets, contact/orientation bounds and stops remain.

The transported-point/fixed-orientation trial also fails contact and passes
stopping. All unqualified force-hold variants are rejected and removed from active
code, retaining their commits, raw traces and costs. The physically qualified
right/left C2 controller is restored. Wood C2 is independently checked under that
controller. A matched front prototype changes only the supervised return from.5
to.75s: its measured1.48mm residual implies3.10mm/s cosine peak, within the existing
6mm/s bound and with.25s margin inside the unchanged1s recontact envelope. The full
pulse remains applied until eligible correction; all physical/task/stop gates stay.

Wood C2 passes under the original qualified controller, bringing coverage to111.
The front.75s return misses contact before the unchanged deadline; stopping passes.
The preserved front pulse ends with.5s valid loaded contact, but the nominal
pressure goal is1.48mm behind the actual contact pose. A single declared prototype
carries that supported state into ordinary force holding immediately after the
full59-action pulse. This does not rewind the door: the first eligible actual rate
residual remains and must physically decay into the original hold window. Stale
reference distance is diagnostic, not a claimed corrected error. A focused test
requires the executed pulse and real contiguous safe support before one handoff.

Immediate pulse-to-hold transfer loses contact after.283s and passes stopping.
It also lacks an admissible expert push-to-hold chunk because the boundary follows
the excluded pulse directly. The next prototype physically performs post-pulse
corrective support with the original.1N acquisition target and1s recontact envelope,
without opening lead. After.5s fresh expert contact it enters the unchanged.05N
ordinary hold. No injected row is unmasked and no phase/transition gate is relaxed.

Post-pulse expert support reaches ordinary hold safely for6s: final speed-window
peak.190degrees/s, minimum hold sigma.0765 and no physical violation. Its longest
valid loaded window is.433s versus.5s required; occasional.016–.020N means miss the
unchanged.020N threshold. The next matched case maintains the already used.1N
acquisition target into hold instead of halving it to.05N. No criterion, gain,
model or control bound changes; this preserves actual positive-load margin.
