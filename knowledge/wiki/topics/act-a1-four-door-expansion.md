# ACT A1 four-door expansion proposal

Review-ready plan as of 19 October 2026; **no new collection, simulation or
training is authorized or performed here**. Retain current physical outcomes,
all raw attempts and frozen source groups. The usable software/data interfaces
are in [[act-a1-observations-and-time]]. This replaces the earlier proposed
36/44-trajectory sweep with coverage-driven stages and current measured costs.
It does not predict improved learning or learned task success.

## Geometry and next complete episodes

The four identities below are all in the frozen **train** inventory. Actual
composed USD colliders, contact surfaces, hinges and inertia were read, not just
filenames. At fraction .295 / height 1.09 m all nine queried points in a 40×40 mm
contact neighborhood have leaf support. This establishes a local surface, not
robot clearance, contact qualification or an entire reachable trajectory.
[Geometry, saved margins and cost audit](../../../outputs/evidence/act-a1-consumption-20261019/geometry.json).

| Door / frozen group | Relevant geometry and modeled dynamics | Learning coverage / proposed variation | Teacher/controller and acceptance | Estimated additional cost |
|---|---|---|---|---|
| Retain right `animated-door-1-88abf40` / `animated-doors` | Width/height/thickness .839/1.946/.029 m; contact X/Y −.0146/−.1806 m; hinge-to-contact radius .249 m; panel Izz 1.469 kg m² | Thin-leaf baseline; existing nominal plus qualified hold-start C. Use existing acquisition, push-entry and release windows now. Later P or O variation only for a demonstrated missing cell. | Existing shared setup; frozen required opening 41.008°; retain qualified release and ordinary tail. No nominal recollection. | Zero for existing records/views. Later full cycle approximately 9.1 GiB raw+prepared and 6.2–7.3 min whole launch. |
| Retain left `door-2738468b94d74c5f` / `djmaesen` | .945/2.038/.050 m; X/Y −.0243/+.1952 m; radius .288 m; Izz 1.866 | Wider opposite motion; left approach/limit and release coverage. Prioritize a separately qualified left hold-start after source review, rather than more long push rows. | Same controller, fraction/height; required opening 42.029°. Full left path reaches a joint boundary; no symmetric physical pose range is assumed safe. | Existing views zero. A C-length continuation would be about .55 GiB raw+prepared / .6 min; allow qualification/conditioning overhead separately. |
| Add right `psx-front-005-ee7d5c6` / `psx-residential` | .816/1.969/.092 m; collidable glazing inset plus outer leaf; X/Y +.0021/−.1703 m; radius .246 m; Izz 1.406 | Approximately 3× thickness and a 16.7 mm different push surface versus current right, with similar lever length. Tests geometry/appearance dependence without changing opening direction. First nominal; then O=bright or P, separately if missing. | Asset-specific `PreparedDoor.contact_pose`/handedness, unchanged fraction/height and shared control. Propose freezing existing corpus reference 59.713° (80%=47.771°) before launch; shared-setup reach/contact/stop support must first pass. Never lower the target from the attempt's outcome. | First complete attempt: approximately 9–10 GiB and 6–8 min; qualification overhead/failures additional. |
| Add left `psx-interior-wood-002-02e442c` / `psx-residential` | .757/1.809/.086 m; different frame/leaf fit; X/Y −.0423/+.1552 m; radius .239 m; Izz 1.210 | Approximately 20% narrower than current left, 17% shorter contact radius and 35% lower modeled Izz. Tests approach/release on a different lever/surface rather than recoloring the wide door. First nominal; then O=dim or P, separately if missing. | Same controller; correct asset geometry/sign, no mirrored arm logic. Propose freezing corpus reference 62.683° (80%=50.146°); qualify current setup before admitting data. | Approximately 9–10 GiB and 6–8 min per full attempt, plus qualification/failures. |

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

Start with the existing two complete nominal episodes and C; the next essential
complete cells are the two new doors at nominal setup/lighting. Inspect their
coverage before selecting another cell. Targeted starts supplement full episodes;
they do not replace the complete-task validation or count as new independent
source motions. Only the existing **right hold→release→stabilize C** is physically
qualified now. Other starts below are proposed interfaces for their dedicated
qualification/collection tasks, not implemented reset variants.

| Start / priority | Physical state and reconstruction checks | Genuine history and intended observable endpoint | Added coverage / status |
|---|---|---|---|
| Qualified hold start, current right C; next candidate left hold | Same-group qualified hold source; restore full q/dq, door q/dq, delivered targets, impedance, lead/compression/compensation. Reuse existing state tolerances, conditioning, contact/footprint/material checks and independent stop proof. New door/side needs its own qualification. | Existing method uses six real conditioning ticks, fresh local t=0 reference and 1 s real guarded pre-roll, including sustained .5 s qualified hold; then real release and ordinary tail. Masks describe unavailable earlier history. Endpoint: measured unloaded withdrawal >=10 mm and existing final .5 s arm/door/tool stability checks. | Directly adds left or new-geometry release/stabilization without another ~100 s push. Existing C usable now; new side/source unqualified. |
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

Rows below are **admitted command rows / usable 12-step window starts**. Phase
labels are diagnostic/sampling annotations outside ACT. Right's 14 ambiguous real
commands remain excluded; crossing chunks also account for its 37 fewer window
starts. Full raw phase counts and source references are retained in the
[actual sampling audit](../../../outputs/evidence/act-a1-consumption-20261019/sampling-quality.json).

| Phase | Right nominal | Left nominal | C derivative |
|---|---:|---:|---:|
| Inspect ready pre-roll | 12 / 12 | 12 / 12 | 0 |
| Approach | 360 / 360 | 360 / 360 | 0 |
| Contact | 180 / 180 | 180 / 180 | 0 |
| Push | 6362 / 6325 | 6784 / 6784 | 0 |
| Hold | 180 / 180 | 180 / 180 | 48 / 48 |
| Release | 360 / 360 | 360 / 360 | 360 / 360 |
| Stabilize | 120 / 120 | 120 / 120 | 120 / 120 |
| Total | 7574 / 7537 | 7996 / 7996 | 528 / 528 |

Push is **13,146/16,098 = 81.66%** of admitted commands. There are two nominal
source motions and one reset derivative; acquisition and push-entry each have
two source episodes. Release/stabilization have those two plus C, still only two
source ancestries. There are 11 valid crossing starts per present phase transition
in each admitted recording. `phase_focused_rows` currently selects six valid
positions around each of contact acquisition, contact→push and hold→release,
plus seven for stabilization including its true final target: 25 right, 25 left,
13 C, 63 unique per-source row views overall. Original right adds 7,337 **partial
candidates** only after a separate explicit admission/unique derivative identity;
its inspect→approach coverage stays missing and its original outcome stays intact.

Use the maintained source-row views with `episode.batches`/`mixed_temporal_batch`
to make an explicit event pass alongside background source windows. Preserve all
crossing labels, actual previous support and true-tail masks. Report each event's
available/selected rows, source ancestry and background exposure. Long push runs
must not set transition exposure merely through uniform row count. No equal-phase
quota is selected: later scheduling should respond to missing independent events
and development diagnostics; repeated views increase exposure, not evidence size.
The initialized checkpoint already consumed all 63 focused windows without
updates. No learned benefit is inferred.

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

Measured complete nominal payloads are 6.52/6.81 GiB raw plus 2.61/2.73 GiB
prepared. Whole launches took 374.6/438.6 s; case reports exclude some wrapper
startup/exit and must not replace those totals. Plan approximately **18–20 GiB
and 12–16 min for the two essential new nominal attempts**, before qualification,
failed attempts, traces and development collection. At the current uncompressed
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

Remaining high-level choices for review: adopt the two proposed added identities
and frozen new-door opening references; prioritize P versus O after nominal
coverage; decide whether R qualification is worth its left-side margin risk;
choose a first new hold-start source before considering unqualified early-contact
subtasks; allocate independent development and failed-attempt budgets. Broad
recovery, further start phases and later sampling/training schedules remain
separate work. Usable now: shared-checkpoint consumption, safe reviewed source
windows, phase views, the existing C continuation and explicit subtask reporting.
New doors, posture/placement variants and new short-start success criteria still
need physical qualification/collection.
