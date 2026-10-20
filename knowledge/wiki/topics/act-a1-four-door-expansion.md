# ACT A1 four-door expansion

The first bounded collection is complete on 19 October 2026: two new nominal
recordings and one qualified left hold/release continuation. Training and learned
evaluation remain outside this task. Measured results are below. Retain current
physical outcomes, all raw attempts and frozen source groups. The usable software/data interfaces
are in [[act-a1-observations-and-time]]. This replaces the earlier proposed
36/44-trajectory sweep with coverage-driven stages and current measured costs.
It does not predict improved learning or learned task success.

## Geometry and first complete episodes

The four identities below are all in the frozen **train** inventory. Actual
composed USD colliders, contact surfaces, hinges and inertia were read, not just
filenames. At fraction .295 / height 1.09 m all nine queried points in a 40×40 mm
contact neighborhood have leaf support. This establishes a local surface, not
robot clearance, contact qualification or an entire reachable trajectory.
[Geometry, saved margins and cost audit](../../../outputs/evidence/act-a1-consumption-20261019/geometry.json).

| Door / frozen group | Relevant geometry and modeled dynamics | Learning coverage / proposed variation | Teacher/controller and acceptance | Original planning estimate (measured totals below) |
|---|---|---|---|---|
| Retain right `animated-door-1-88abf40` / `animated-doors` | Width/height/thickness .839/1.946/.029 m; contact X/Y −.0146/−.1806 m; hinge-to-contact radius .249 m; panel Izz 1.469 kg m² | Thin-leaf baseline; existing nominal plus qualified hold-start C. Use existing acquisition, push-entry and release windows now. Later P or O variation only for a demonstrated missing cell. | Existing shared setup; frozen required opening 41.008°; retain qualified release and ordinary tail. No nominal recollection. | Zero for existing records/views. Later full cycle approximately 9.1 GiB raw+prepared and 6.2–7.3 min whole launch. |
| Retain left `door-2738468b94d74c5f` / `djmaesen` | .945/2.038/.050 m; X/Y −.0243/+.1952 m; radius .288 m; Izz 1.866 | Wider opposite motion; left approach/limit and release coverage. A separately qualified left hold-start now adds release coverage without another nominal push. | Same controller, fraction/height; required opening 42.029°. Full left path reaches a joint boundary; no symmetric physical pose range is assumed safe. | Existing views zero. A C-length continuation would be about .55 GiB raw+prepared / .6 min; allow qualification/conditioning overhead separately. |
| Added right `psx-front-005-ee7d5c6` / `psx-residential` | .816/1.969/.092 m; collidable glazing inset plus outer leaf; X/Y +.0021/−.1703 m; radius .246 m; Izz 1.406 | Approximately 3× thickness and a 16.7 mm different push surface versus current right, with similar lever length. Tests geometry/appearance dependence without changing opening direction. Nominal complete; any later P/O condition requires a demonstrated coverage gap. | Asset-specific `PreparedDoor.contact_pose`/handedness, unchanged fraction/height and shared control. Fixed external benchmark 59.713° (80%=47.771°); current-setup reach/contact/stop applicability passed before collection. No target refit. | First complete attempt: approximately 9–10 GiB and 6–8 min; qualification overhead/failures additional. |
| Added left `psx-interior-wood-002-02e442c` / `psx-residential` | .757/1.809/.086 m; different frame/leaf fit; X/Y −.0423/+.1552 m; radius .239 m; Izz 1.210 | Approximately 20% narrower than current left, 17% shorter contact radius and 35% lower modeled Izz. Tests approach/release on a different lever/surface rather than recoloring the wide door. Nominal complete; any later P/O condition requires a demonstrated coverage gap. | Same controller; correct asset geometry/sign, no mirrored arm logic. Fixed external benchmark 62.683° (80%=50.146°); current-setup applicability passed before collection. | Approximately 9–10 GiB and 6–8 min per full attempt, plus qualification/failures. |

All four USDs use the same synthetic 25 kg panel, angular damping 4 and material
friction .5; their geometry-derived inertias differ. These are modeling
assumptions, not identified real-door parameters. Keep mass, damping, friction,
controller gains, force/speed/stop requirements and contact thresholds fixed.
There is no justified nonzero dynamics-randomization range in this task. The
right `animated-door-2` has effectively the same dimensions as door 1 and adds
less geometric coverage; it is not selected merely for a different filename.
The two added PSX doors share one frozen group and do not add two independent
families. The four-door proposal covers three train groups.

Use `configs/b1_shared_pilot_setup.json`: floor (−.437, .482327, 0), contact
fraction .295 / height 1.09 m, 6 s approach, 3 s acquisition, 1°/s push,
.3° lead, 3 mm compression, existing qualified hold, 6 s release, .4 s real
nominal pre-roll and 2–6 s ordinary stabilization. Geometry changes parameterize
the existing teacher; they do not justify changing the controller or its limits.
Full acceptance retains opening >=80% of the prospectively frozen reference,
required valid hold, actual withdrawal/release, ordinary stabilization, original
contact/footprint/material/speed/force checks and independent healthy stop.
The 80 N hard limit, .5 rad/s command limit with existing numerical allowance,
tracking bounds and independent operational-admission requirements remain.
Data acceptance separately requires integrity/terminal support, safe reviewed
commands, causal labels/masks, all required phases/transitions and genuine tail
support. Partial candidates have a separate explicit admission, not a relaxed
full-demonstration gate.

## Bounded variation candidates

These are **screening ranges for later qualification**, not measured safe
operating ranges. Use one factor at a time first; do not form their Cartesian
product or schedule equal phase proportions.

| Factor | Concrete candidate range | Evidence, parameterization and limit |
|---|---|---|
| P: initial right arm | `RIGHT_SHOULDER_Y` nominal ±.05 rad; all other initial joints unchanged | Actual URDF CUDA FK gives ±21.8 mm tool displacement and about 2.86° rotation, with .216 rad minimum URDF joint margin at these initial points. This changes observable initial approach geometry. Runtime ready/approach margins are different: right minimum .06295 rad; left approaches a limit within numerical tolerance. Recheck runtime soft limits, self/door clearance, approach, visibility and stop before collecting either endpoint. Do not offset loaded trajectories by these angles. |
| R: robot–door relationship | Separate candidate translations ΔY ∈ [−.010,+.010] m, or retreat ΔX ∈ [−.010,0] m; yaw and Z fixed | Translate robot and pedestal together via `floor_pose`. New nominal contact Ys remain inside the current [−.1806,+.1952] m endpoints. A 10 mm lateral move keeps coarse initial lateral travel below .198 m versus the old .290 m setup. This is only a geometric screen. Left has no spare joint-boundary margin and recorded AABB clearance lower bound is zero; there is **no qualified nonzero translation range** yet. Start with nominal and retain only candidates passing the same whole-path checks. |
| O: observations | Declared dome-light scale .9–1.1, with nominal 1.0; camera mount/calibration and neck recipe fixed | Already implemented and used by the pilot. Pair bright right with dim left to break the older direction/light association; these do not count as independent geometry. No invented sensor noise, timestamp delay, depth replacement or occlusion duration is added. Actual missing observations retain existing masks/freshness refusal. |

The saved right/left nominal traces reach minimum Jacobian singular values about
.037/.037 near hold/release; right/left peak actual speeds are .320/.284 rad/s.
The left approach clips a position boundary; right learned evaluation reached
hold but refused before release, and the prior left learned attempt refused
before contact. Earlier tiny positive self/sweep margins do not establish
millimeter-scale pose slack. These findings prioritize approach/contact and
hold/release coverage and forbid presenting a nominal pass as a validated
placement neighborhood. A changed setup's failure remains in the denominator;
no unchanged favorable retry or empirical threshold adjustment follows.

## Targeted starts alongside complete episodes

The existing two complete nominal episodes and right C are retained. Both new
nominal geometries and the existing-left hold continuation now pass the physical
and usability gates. Inspect this combined coverage before selecting another cell. Targeted starts
supplement full episodes;
they do not replace the complete-task validation or count as new independent
source motions. The existing right C and new **left H hold→release→stabilize**
are physically qualified in their measured nominal source states. Other starts
below are proposed interfaces for their dedicated
qualification/collection tasks, not implemented reset variants.

| Start / priority | Physical state and reconstruction checks | Genuine history and intended observable endpoint | Added coverage / status |
|---|---|---|---|
| Qualified hold starts: right C and left H | Same-group qualified hold source; restore full q/dq, door q/dq, delivered targets, impedance, lead/compression/compensation. Reuse existing state tolerances, conditioning, contact/footprint/material checks and independent stop proof. New door/side needs its own qualification. | Existing method uses six real conditioning ticks, fresh local t=0 reference and 1 s real guarded pre-roll, including sustained .5 s qualified hold; then real release and ordinary tail. Masks describe unavailable earlier history. Endpoint: measured unloaded withdrawal >=10 mm and existing final .5 s arm/door/tool stability checks. | Directly adds left or new-geometry release/stabilization without another ~100 s push. Right C and left H are usable; other source geometries remain unqualified. |
| Unloaded precontact, dedicated future task | Closed unlatched door, known 30 mm precontact tool separation, zero acquired contact, supported q/dq/targets and collision-free arm. Capture/reconstruction and unloaded-to-contact stopping must be qualified; do not use a post-refusal state as an expert start. | Acquire a fresh local reference and actual measured pre-roll; .2 s model readiness is not contact qualification. Record approach→acquisition and first push actions with their true preceding history. Endpoint: independently verified contact/load/footprint/material support through the existing acquisition window and a safe first admitted push interval. RGB-D and measured q/dq/tau/targets must show the approach/response; torque alone is not contact force certification. | Intended to diversify contact approach under P/R on each geometry. Success predicate and reset still physically unqualified; a proposed `subtask_success` cannot admit data until `subtask_qualified` is established. |
| Established-contact before push, dedicated future task | Restore a genuinely qualified source contact, full robot/door velocities and controller state, preserving force/material support during conditioning. No assumption that a hold snapshot reproduces early contact. | Fresh images and actual guarded measured history; begin eligible labels only when the existing readiness and contact conditions hold. Continue through contact→push; endpoint is sustained qualified contact plus real opening response under the unchanged monitor, not a teacher phase flag. | Separates successful acquisition from transition into opening; do this only if full-source/event views leave a coverage gap. No generic early-contact reset or new success threshold is implemented. |

Each new loaded start needs independent contact/stop qualification before any
policy execution. Producers use the existing explicit `startup(observer, io)`
interface with fresh generation and local reference, never nominal closed-door
parking, fabricated times or borrowed source history. Additional late-push,
recovery and post-release start curricula remain separate tasks. Their likely
needs are real pre-deviation/history support, trustworthy correction labels and
independent stopping/recontact checks; this task adds no recovery infrastructure.

## Available supervision and sampling

Rows below are **reviewed command rows / usable 12-step window starts**. Phase
labels remain diagnostic/sampling annotations outside ACT. Original right's 14
ambiguous commands stay excluded; crossing chunks account for its 37 fewer
starts. Left H retains 18 diagnostic commands and four reviewed early commands
without an admissible chunk across the internal invalid interval. No hole is
replaced with artificial tail padding.

| Phase | Original right | Original left | Right C | Added right | Added left | Left H |
|---|---:|---:|---:|---:|---:|---:|
| Inspect ready pre-roll | 12 / 12 | 12 / 12 | 0 / 0 | 12 / 12 | 12 / 12 | 0 / 0 |
| Approach | 360 / 360 | 360 / 360 | 0 / 0 | 360 / 360 | 360 / 360 | 0 / 0 |
| Contact | 180 / 180 | 180 / 180 | 0 / 0 | 180 / 180 | 180 / 180 | 0 / 0 |
| Push | 6362 / 6325 | 6784 / 6784 | 0 / 0 | 6679 / 6679 | 8305 / 8305 | 0 / 0 |
| Hold | 180 / 180 | 180 / 180 | 48 / 48 | 180 / 180 | 180 / 180 | 42 / 38 |
| Release | 360 / 360 | 360 / 360 | 360 / 360 | 360 / 360 | 360 / 360 | 360 / 360 |
| Stabilize | 120 / 120 | 120 / 120 | 120 / 120 | 120 / 120 | 120 / 120 | 120 / 120 |
| Total | 7574 / 7537 | 7996 / 7996 | 528 / 528 | 7891 / 7891 | 9517 / 9517 | 522 / 518 |

Push is **28,130/34,028 = 82.67%** of reviewed commands. There are four nominal
source motions across three families and two hold derivatives. Acquisition and
push-entry each have four nominal sources; release/stabilization have all six
records but still four nominal ancestries. Every present phase transition retains
11 valid crossing starts. `phase_focused_rows` selects 25 views per nominal and
13 per continuation: 126 overlapping source-window views, not demonstrations.
The shared initialized checkpoint consumes all 126 without updates.

[Current source, phase, transition and CUDA audit](../../../outputs/evidence/act-a1-four-door-20261019/verification.json).
The [original three-source audit](../../../outputs/evidence/act-a1-consumption-20261019/sampling-quality.json)
remains the pre-expansion baseline. Original right A1 retains 7,337 **partial
candidates** only after separate explicit admission/unique derivative identity;
it remains excluded from the combined handoff, with its original outcome and
missing inspect→approach coverage unchanged.

Use `episode.batches`/`mixed_temporal_batch` for an explicit event pass alongside
the complete background source windows. Preserve crossing labels, actual history
and true-tail masks. Report available/selected rows, source ancestry and background
exposure. Long push runs must not set transition exposure merely by uniform row
count. No equal-phase quota or learned sampling benefit is selected; later
scheduling must respond to missing independent events and development diagnostics.

## Evaluation, costs and remaining decisions

Keep every render, crop, reset and reviewed partial derivative in its source's
frozen split. The added PSX sources remain **train** together. Existing
`psx-industrial-001-4f5561b` is **left development**; its dim/bright variants are
one motion, not two independent development episodes. Plan independently reset
motion/condition evaluation on that left development identity and frozen right
development `modern-door-2fb8d024`, subject to their own setup qualification.
Their data never fit normalization or weights. Do not turn a trajectory on one
of the four train doors into frozen development, and do not move groups or open
the final test. Later report per-door/handedness/phase and physical task outcomes,
with all failures and uncertainty at source/group level.

The original complete nominal payloads are 6.52/6.81 GiB raw plus 2.61/2.73 GiB
prepared. Whole launches took 374.6/438.6 s; case reports exclude some wrapper
startup/exit and must not replace those totals. The prospective estimate was **18–20 GiB
and 12–16 min for the two essential new nominal attempts**, before qualification,
failed attempts, traces and development collection; measured new costs are below. At the current uncompressed
rates each additional recorded second costs about .043 GiB raw + .017 GiB prepared;
a C-length 9 s continuation is about .55 GiB. Source-row phase views add no image
storage. New precontact/contact subtasks should be priced from their eventual
qualified real duration; a speculative duration is not a launch allocation.

Before any later launch, declare concrete cases and reconcile remaining attempt,
GPU and storage ledgers including failures. Existing per-case ceilings (1200 s
wall / 210 s simulated / 12 GiB RSS / 20 GiB total GPU) provide a conservative
starting envelope, not new authorization. Stop the current case on physical or
resource violation; retain its outcome and diagnose before a changed attempt.
Do not fill an arbitrary demonstration target.

Remaining choices are physical initial-posture/placement coverage, independent
development motions, and dedicated precontact/contact-to-push reconstruction
interfaces. Four nominal source motions and two hold derivatives do not establish
those variations or ACT task completion. The smallest next tranche is specified
below; broad recovery, RL, training and learned evaluation remain separate work.

## First bounded execution: fixed requirement applicability

The September PSX references use floor Y=.38 rather than the current .482327 m
and a 3 s release rather than 6 s; they also predate current hold support and
ordinary stabilization. Their prepared qualification flags do not establish
current setup applicability. Keep their 80% requirements fixed at 47.770638°
right / 50.146446° left, without treating the old maxima as current achievable
maxima. Before scored collection, one dedicated current-setup cycle per door
must pass that requirement and the unchanged physical/release/tail/independent
stop gates. This establishes applicability of an externally fixed benchmark; it
does not publish a new maximum reference or replace the corpus paired-reference
procedure. A failure blocks scored collection on that door.

`pilot_physics.py qualify` performs this non-supervising check with the shared
setup and inspection/pre-roll. `collect` and `qualify` first check prescribed
footprint support, obstructions and pedestal intersection, then retain all
existing every-row reach/contact/speed/force checks. Detached independent stop
verification remains separate. The left existing-source continuation qualifies
reconstruction and sustained loaded pre-roll within its single physical launch,
then records release and ordinary stabilization; it is not a closed-door opening.

The prospective protocol and resource reconciliation are in
`outputs/evidence/act-a1-four-door-20261019/declaration.md` and `protocol.json`.
Five launches were prospectively planned, with one diagnosed-correction reserve; maximum six
launches, two GPU workload hours and 100 GiB additional evidence, within project
limits. Per-case 1,200 wall s / 210 sim s / 12 GiB RSS / 20 GiB total GPU ceilings
remain. No later posture/placement tranche is launched here.

## Measured first expansion and usable handoff

The separate fixed-requirement checks pass at 51.281° right / 54.371° left,
including valid hold, ordinary release/stabilization and detached independent
stop. Fixed collection requirements stay 47.770638° / 50.146446°; no requirement
or corpus maximum reference is refitted. Scored collection is separate.

| New source | Physical result | Usable 12-step windows |
|---|---|---:|
| New right PSX nominal | 51.281° complete opening; hold/release/2 s ordinary tail/stop pass | 7,891 |
| New left PSX nominal | 54.371° complete opening; hold/release/2 s ordinary tail/stop pass | 9,517 |
| Existing-left hold/release continuation | Reconstruction, fresh guarded loaded hold, release/2 s ordinary tail/stop pass; not closed-door task success | 518 |

All three recordings pass native integrity, real terminal support, scoped phase
coverage and reviewed command/window admission. Raw RGB-D, requested/delivered
commands, native measured simulated torque, clocks, masks and prior ambiguous
samples remain. Left H uses the exact qualified original-left source, six real
conditioning ticks, fresh generation/local t=0 reference and genuine one-second
pre-roll. Its successful continuation never becomes full closed-door opening.
Eighteen H commands remain diagnostic; four otherwise reviewed short-prefix
commands cannot form a valid 12-action chunk across the internal invalid interval.
All 518 usable starts retain real history and terminal padding.
New-left requests are limited/clipped on 64 hold and 216 release rows (maximum
.036347 rad request/target difference); labels use actual delivery minus measured
pre-state. None of the three sources has a refused command.

The combined six-record train handoff contains 33,987 complete eligible
windows, alongside 126 overlapping phase-focused source views. It represents
four independent nominal motions across three train families and two hold
reconstruction derivatives. One initialized CUDA checkpoint/shared train-only
normalization consumes nominal and intermediate records; masks, causal joins,
actual labels and genuine tail padding pass with unchanged weights and zero
optimizer updates. Focused event anchors do not replace background/crossing
windows or count as independent demonstrations.

Five physical launches, no failed/refused/interrupted attempt or retry, consume
1837.91 s including startup/exit; CUDA consumption adds
54.44 s. Additional retained evidence is 21.79 GiB,
within the six-launch/two-hour/100-GiB allocation and remaining project budgets.
The 10.056 N acquisition peak is retained below the unchanged 80 N hard limit;
no empirical load envelope is expanded. Zero AABB clearance lower bound is not
positive placement slack. Measured healthy post-cycle stops do not establish
global loaded emergency-stop coverage or learned operational admission.

[Complete outcomes, costs and recommendation](../../../outputs/evidence/act-a1-four-door-20261019/report.md),
[explicit source handoff](../../../outputs/evidence/act-a1-four-door-20261019/handoff.json)
and [scoped quality/CUDA audit](../../../outputs/evidence/act-a1-four-door-20261019/verification.json).
The prior three-source handoff and every historical failed/interrupted outcome
remain intact. No ACT learned task result is added; the eventual criterion is
physical completion from its own observations, not teacher trajectory similarity.

## Smallest next physical tranche — proposed, not launched

Qualify one initial-posture variation on the existing wide left
`door-2738468b94d74c5f`: initial `RIGHT_SHOULDER_Y` −.05 rad, with all other initial
joints, robot-door placement and nominal lighting fixed. This is a candidate,
not a measured safe range. It adds a distinct approach/acquisition event where
prior learned execution refused before contact and the path approaches a joint
boundary; all four nominal motions currently share one initial posture/placement.
The two hold derivatives add no independent acquisition events.

Check initial runtime joint limits, clearance/visibility and stopping support
before a full attempt. Keep the loaded targets unchanged. Physical success is
existing-left opening ≥42.028823°, valid hold, actual release, ≥2 s ordinary
stabilization and independent healthy stop; data require native integrity,
reviewed windows and approach/acquisition/crossing/tail support. Retain any
failure without unchanged favorable retry. A lighting-only cell does not fill
this physical gap. Nonzero placement ranges and precontact/contact-to-push resets
still need their dedicated physical qualification; no subsequent tranche is
launched in this task.
