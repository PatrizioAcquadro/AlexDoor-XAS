# Historical Approach / Current Controller Compatibility

Baseline `e391ffd`, branch `PatrizioAcquadro/b1-policy-pilot`, October 15, 2026.
Follow-up to [[b1-pilot-approach-release|the residual diagnosis]]. V2 and uniform
sampling remain selected; the phase-balanced variant remains evidence of its
release/hold/push tradeoff. No policy, sampler, memory, controller, SAM3 or P2P
change, training or learned rollout is part of this comparison.

## Protocol fixed before execution

Persistent local source, protocol and evidence:
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1-approach-controller-20261015/`.
Original recordings/checkpoints and failures stay at their original paths.
Ignored payloads are not recoverable from Git.

Audit both complete train approaches (animated-door right and door-2738468 left),
360 commands / 6 seconds each, without development/test selection. Verify world
root/tool conventions, SI units, rotation matrices/XYZW quaternions, pre-command
observation timestamps, 60 Hz command and 120 Hz feedback joins. Distinguish absolute
Cartesian reference, local differential-IK proposal, applied bounded joint target,
and poststep measured motion. Reuse the original independent label/FK audit and
check recorded trace joins; never apply old A2 deltas to a different state.

Use the current benchmark controller/configuration frozen at baseline, with the
same floor, fixed base, URDF/tool and physical limits. Initialize each independent
case from the recorded pre-approach q/dq, neck and previous applied targets via
public simulator state APIs. Do not replay the 25-second inspection prefix: it
supplies the saved initial state. Cameras disabled, fixtures remote, no corpus
door or injected force; abort on any robot contact. This isolates simulation-time
trajectory compatibility and does not validate camera scheduling or hardware.

For each train trajectory execute exactly two independent cases:

- Full original Cartesian reference and exact 60 Hz timing, then two seconds
  holding its final reference and two seconds stopped with physics active.
- The unchanged prefix through maximum saved reference translation speed,
  then a moving measured-anchor stop and two seconds of active physics.

Select the peak before execution, using saved reference increments only. Do not
retime, smooth, improve, relabel or repeat a completed result. Record q/dq/ddq,
Cartesian errors, full Jacobian, local IK proposal, target-step scaling and joint
limit margins, applied targets/efforts, solver feedback and both physics samples.
Also run the existing bounded pose-IK solver on CUDA as an evaluator-only static
screen, seeded from historical targets and replay states. A local solve failure
does not prove global unreachability; a successful solve does not prove dynamic
tracking or a safe continuous path. Never send its solutions to the controller.

Use the original load calibration unchanged. Report native joint-output feedback
availability separately from residual-estimator availability, including every
out-of-envelope rejection. No qualification flag is promoted.

### Criteria and declared limits

These are practical local engineering checks, not contact/hardware admission:

- Tracking: original 10 mm / 5 degree approach tolerance at p95 and endpoint.
  Report mean/median/p95/max, pre/poststep, historical and current values.
- Hold: final half-second joint-target error <=0.02 rad and Cartesian error
  <=10 mm / 5 degrees. Report the complete two-second transient.
- Stops: <=10 mm / 3 degree maximum travel from the measured stop anchor;
  final peak joint speed <=0.01 rad/s after two seconds, unchanged anchor/targets.
- Feedback: all native substep samples healthy, finite and correctly timed.
  Original residual calibration failures remain failures; no bound refitting.
- Report actual speed against the unchanged 0.5 rad/s target-step budget
  (excursions above 0.51 rad/s flagged with 0.01 numerical allowance), physical
  joint limits, effort clipping and IK conditioning. Target limits are not
  measured-speed guarantees. A static support screen uses <=1 mm /0.1 degree;
  these numerical screening values do not replace trajectory criteria.

Require both existing `/tmp/rtx4090-exclusive.lock` and
`/tmp/alexdoor-xas-gpu.lock` and the existing audio-sensor GPU lease, plus 180 continuously compute-idle seconds, checked
at 30-second intervals with graphics/compute occupancy. Monitor foreign work;
preserve interference, stop only this diagnostic if necessary, and resume only
incomplete cases. Never stop other processes or substitute CPU simulation.

Reuse [[b1-loaded-hold-and-stop|local 0/4 N hold/stop and latch/reset evidence]]
without repeating it. New trajectory holds/stops extend only the unloaded tested
poses. Neither historical contact success nor an unloaded replay qualifies current
push/hold/release. Full required 6.0D feedback/fault/domain/contact checks and
independent local geometry/6.0E still precede 6.0F contact.

## Saved-data audit

Both approaches start at raw row 1,500 / source time 25 s and contain 360 commands.
Command timestamps equal pre-execution observation times; 720 feedback samples
cover each interval at 120 Hz. Teacher trace time labels the relative command
start, while its tool pose/error fields describe that command's poststep state
(observation i+1). These different meanings are checked explicitly, not aligned
by equal-looking trace timestamps. Double-precision URDF/tool FK matches the independent
poststep teacher trace within 0.8 micrometres / 0.0001 degrees. Reference and target
units/frames agree; recorded torque is joint-output feedback, distinct from effort
commands. A2 is the reference-to-current-state residual, not a reference increment.

| Historical quantity | Train right | Train left |
|---|---:|---:|
| Reference increment, mean | 0.440 mm | 0.878 mm |
| Prior-target tracking residual, mean | 7.310 mm | 25.012 mm |
| Poststep Cartesian error, p95 | 15.236 mm | 66.130 mm |
| Reference versus FK of bounded joint target, p95 | 0.412 mm | 44.933 mm |
| Joint target versus measured joints, p95 max-joint error | 0.04246 rad | 0.12698 rad |
| Applied target within 1 mrad of a joint limit | 0/360 | 241/360 |
| Actual peak joint speed, both physics samples | 0.15017 rad/s | 0.50487 rad/s |

The left reference already has substantial error before actuator tracking, while
the right target FK closely follows its Cartesian reference. Norms of these
components do not add; neither whole residual is attributed to impedance.
The saved pre-command Jacobian, observed joints, reference and previous target
reconstruct the original applied differential-IK target within 1.7 microradians.
The right approach has zero target-step scaling or position clamps. The left has
84/360 scaled steps (maximum scale 6.47) and 241/360 position clamps. Its historical
target-step speed reaches the unchanged 0.5 rad/s budget. The mixed-unit Jacobian
condition number peaks at 11.54 right / 21.09 left; this descriptive value is not
a calibrated singularity/admission threshold. Native observation feedback is
available on all 361 approach observations per trajectory. Historical actuator
clipping is unknown: stored effort targets are gravity feedforward, not the
explicit actuator's total commanded/applied torque.

The current URDF matches the recordings' existing robot-asset identity; no robot
geometry change is introduced. The double-precision FK check is numerical analysis,
not CPU simulation or a new dynamic qualification.

Both references reach peak translation speed after 2.1 s, fixing both moving-stop
prefixes at 126 ticks before CUDA execution. Historical approach panel force is
zero; no historical full-path collision qualification is inferred.

## Current-controller results

One accepted Isaac/PhysX CUDA run on RTX 4090 completes all four independent
cases: 1,692 control ticks / 3,384 physics samples, 28.2 simulated seconds.
All three shared leases were acquired after the external work finished. Seven
30-second preflight observations span 180.64 s with no compute/unknown graphics,
0–5% utilization and about 21.7 GiB free. Accepted monitoring detects no foreign
work. No physical case is repeated and no control setting is changed.

`verification.json` checks exact original Cartesian references, source tick order,
60/120 Hz timing, initial arm positions/previous targets, recorded asset identity,
resolved controller gains and native feedback semantics. Source/config/tests/scripts
are unchanged from `e391ffd`. Initial q/dq are written using the public SDK state
APIs for the nine recorded arm/neck joints; other joints use the common setup.
Initial velocity has no separately saved prestep readback, so whole-articulation
historical state equality is not claimed. The executed source
is preserved in the accepted attempt, distinct from an unexecuted logging addition.
All source versions and queue-supervisor corrections remain local evidence.

### Full six-second approaches

Translation/rotation below use the **60 Hz poststep endpoint**, matching the old
recording comparison. Each cell is mean / median / p95 / maximum. Both 120 Hz
samples and all native columns remain in the complete analysis.

| Trajectory / regime | Cartesian translation, mm | Cartesian rotation, deg |
|---|---|---|
| Train right, historical | 7.290 / 7.621 / 15.236 / 15.317 | 0.480 / 0.498 / 1.009 / 1.014 |
| Train right, current | 0.358 / 0.374 / 0.673 / 0.679 | 0.106 / 0.100 / 0.216 / 0.218 |
| Train left, historical | 24.943 / 19.404 / 66.130 / 67.363 | 4.016 / 0.777 / 13.780 / 14.010 |
| Train left, current | 6.994 / 2.316 / 28.783 / 29.796 | 2.325 / 1.333 / 7.873 / 8.064 |

The right trajectory passes the declared local tracking criteria. The left fails
both p95 criteria (10 mm / 5 deg) and the actual-speed reporting guard; its rotation
median also worsens despite improved mean/tail. Endpoint-only success is insufficient:
final approach errors are 0.0703 mm / 0.0158 deg right and 0.0131 mm / 0.0022 deg left.
The p95 comparison criterion is prespecified here; it does not retroactively replace
the historical teacher's task/endpoint qualification rules.

| Current approach quantity | Train right | Train left |
|---|---:|---:|
| Reference versus FK of bounded target, p95 | 0.00956 mm / 0.00064 deg | 30.068 mm / 7.715 deg |
| Target versus measured joints, endpoint p95 max-joint error | 0.004234 rad | 0.044017 rad |
| Target versus measured joints, both-substep p95 max-joint error | 0.004473 rad | 0.047956 rad |
| Joint error maximum across both substeps | 0.004624 rad | 0.047975 rad |
| Steps scaled by the target-rate limit | 0/360 | 100/360 |
| Local IK proposals clipped by position limits | 0/360 | 129/360, RIGHT_WRIST_Z |
| Maximum target-step speed | 0.15007 rad/s | 0.500003 rad/s (float32 roundoff) |
| Actual peak joint speed | 0.15359 rad/s | 0.52682 rad/s, RIGHT_WRIST_X |
| Physics samples above 0.51 rad/s | 0/720 | 91/720 |
| Maximum actual tool linear speed | 0.05690 m/s | 0.13126 m/s |
| Maximum actual joint acceleration | 1.956 rad/s² | 9.325 rad/s² |
| Actuator clipping | 0/720 | 0/720 |
| Largest applied effort / physical limit | 12.59% | 11.68% |

The left's whole-vector step scale reaches 8.95. The smallest measured joint-limit
margin is -9.54e-7 rad, within simulator roundoff; neither path exceeds a physical
position limit by 1e-4 rad. The unchanged 0.5 rad/s **target-step** budget is not a
measured velocity guarantee. The left actual peak occurs at 2.725 s, later than
the prespecified moving-stop point; no stop bound for that later state is claimed.

The existing bounded static IK solver finds support for all 360 poses on each
trajectory from **both** fixed seed sets, within the 1 mm / 0.1 deg screen. This
rules out treating the observed local failures as proof of globally unreachable
goals. Independent static solutions do not establish a continuous, rate-feasible
joint path or safe swept motion. Its solutions never enter control.

### Endpoint holding and continuing-physics stops

Both extra unloaded endpoint holds last two seconds. Last-half-second maximum
joint-target errors are 0.000266 / 0.0000679 rad (right/left); final Cartesian errors
are 0.0605 / 0.0126 mm and 0.0136 / 0.00254 deg. These pass the declared local hold
criteria. They are free-space endpoint holds, not the task's loaded hold phase.

All stops latch measured joints without advancing time, then advance physics
for 240 samples / two seconds with exactly unchanged targets.

| Stop point | Max travel, mm | Max rotation, deg | Speed at stop, rad/s | Settling below 0.01 rad/s | Final peak joint speed, rad/s |
|---|---:|---:|---:|---:|---:|
| Right, moving at 2.1 s | 2.573 | 0.284 | 0.1481 | 0.400 s | 0.000152 |
| Left, moving at 2.1 s | 4.616 | 0.813 | 0.3885 | 0.533 s | 0.000124 |
| Right, after endpoint hold | 0.00704 | 0.00101 | 0.000968 | Already below threshold | 0.0000507 |
| Left, after endpoint hold | 0.00203 | 0.00047 | 0.001271 | Already below threshold | 0.000152 |

All four stops pass the prespecified 10 mm / 3 deg / final-speed criteria. Moving
stops return within 0.0032 mm of their measured anchors at two seconds. The very
small final/settled values reflect ideal native simulation, not hardware precision.
These four observations do not bound other trajectory points, the later left peak,
external loads, fault latency or frictional unloading. Reuse the separate ready-pose
0/4 N/latch/reset evidence; no such trial is repeated.

### Feedback and timing limits

All 3,384 native joint-output feedback samples are healthy, finite, immediate on
the simulation clock and correctly spaced. This is distinct from the residual
estimator: only 108/3,384 samples are inside the original calibration domain
(ready joints +/-0.08 rad, 0.25 rad/s and 5 rad/s²);
3,276 refuse with `outside_verified_load_envelope`. All endpoint holds and stops
are unavailable to that estimator. Keep those refusals in the denominator.

Zero-load residuals exceed an original joint bound on 392 samples, all outside
the calibration domain; none of the 108 available samples exceeds a bound.
The old ready-pose calibration is neither refitted nor enlarged from these data.
Native telemetry availability does not establish calibrated load admission.
Across all cases there is zero actuator clipping and zero robot contact; peak
applied effort fraction is 12.60%. No external load is injected.

Instrumented full-approach wall-cycle p95 is 16.03 / 16.67 ms right/left, with a
retained 262.11 ms first-cycle outlier. The physics/control clocks remain exactly
120/60 Hz. Cameras and learned/perception inference are absent, so this does not
qualify camera-enabled wall-clock scheduling, feedback timeouts or hardware.
Native PhysX/model feedback is the declared error model; physical friction, noise,
mount/FK uncertainty and actual contact/unloading remain unmeasured.

## Attribution, data reuse and decision

- **Controller/servo:** tracking is substantially better under the current regime,
  particularly on the right. The old A2 residuals are not intended Cartesian
  increments. Historical versus current recorded regimes are compared; no isolated
  gain ablation is performed, and no blanket impedance attribution is made.
- **Local kinematics and limits:** the left still has large error already in its
  bounded joint targets, with wrist-Z clamping and whole-vector rate scaling,
  despite static support and large effort headroom. Remaining error is not explained
  by torque saturation or purely actuator lag. IK/state feedback and servo response
  interact, so the scalar error norms are not additive causal contributions.
- **Reference:** the given left time-parameterized Cartesian path does not satisfy
  the declared tracking criteria with this frozen local controller. Accurate final
  goals/static IK do not establish dynamic path compatibility. Neither longer
  policy memory nor retiming alone is demonstrated to repair it.

Retain original RGB-D/proprioception, timestamps, historical commands/actions,
checkpoints and failed criteria for offline data/normalization/prediction audits,
old-regime supervision and reference/kinematic diagnosis. Their schemas still load.
The right Cartesian reference is reusable as a tested local unloaded probe; the
left retains useful static/endpoint support and explicit transient failures.

**Do not relabel old observations with replay commands or inferred dynamics.**
New-controller matched learning demonstrations require newly recorded observations,
actions and actual physical outcomes under one frozen controller/reference regime.
If approach references change, record those approaches again. Historical contact,
push, loaded hold and release results do not transfer and require new matched
physical demonstrations after their prerequisites; no such collection is performed
or admitted by this unloaded replay. All qualification flags remain false.

**One next direction:** construct and validate a continuous, time-parameterized
approach reference that respects the frozen controller's joint/rate limits, starting
with the failing left trajectory. Keep controller and policy fixed. This is a
future bounded reference-planning check, not an implemented fix; do not assume a
simple slowdown resolves the wrist limit. Defer memory/sampling/training changes
and matched recollection until reference execution is demonstrated.

## Remaining pre-contact boundary

Required **6.0D** work still includes independent dynamic/peak residual uncertainty
and a supported domain covering actual trajectory q/dq/ddq, saturation/model/friction
limits, feedback/control timing gaps and fault/latched stops with continuing physics.
Stop/sweep margins must cover the admitted states, including later fast/near-limit
motion; ready-pose calibration and the two moving-stop points are insufficient.
Finite contact assumptions and conservative unloading/stop bounds need support;
actual contact/load-removal behavior remains unqualified by free-space evidence.

Required **6.0E** work still includes supported observed local ownership/footprints,
continuous sweep/clearance and response evidence, independent surface/trajectory
scoring under both profiles, and chronological operational-v1 replay on both train
pilots/conditions with one recipe. Preserve all missing/provisional/failed states
in denominators and meet the unchanged coverage/accepted-precision and 1 cm/5 deg
gates. Required robot checks plus independent local geometry/6.0E precede bounded
6.0F interaction; final push/hold/release qualification remains separate.

## Evidence and validation

`audit.json`, `historical-kinematics.json` and input arrays retain saved-data checks.
`accepted/` links to the four original completed case directories, each with raw
`trace.npz`, `ticks.npz`, `case.json`, applicable `static-ik.npz` and `analysis.json`.
`comparison.json`, `additional-summary.json` and `verification.json` retain all
criteria, counts, unavailable estimates and checks. The accepted attempt includes
resolved runtime/controller/calibration, exact executed sources, full stdout and GPU
monitoring. Queue corrections and the unexecuted readback addition are preserved;
no physical failure or completed case is erased/repeated. Numerical assertions,
source syntax/lint and wiki link/whitespace checks complement the measured result.
Only canonical documentation is committed; ignored source/data remain local.
