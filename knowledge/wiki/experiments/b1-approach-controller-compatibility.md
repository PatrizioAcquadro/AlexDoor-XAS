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
`/tmp/alexdoor-xas-gpu.lock`, plus 180 continuously compute-idle seconds, checked
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
cover each interval at 120 Hz. Double-precision URDF/tool FK matches the independent
poststep teacher trace within 0.4 micrometres / 0.0001 degrees. Reference and target
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
Both references reach peak translation speed after 2.1 s, fixing both moving-stop
prefixes at 126 ticks before CUDA execution. Historical approach panel force is
zero; no historical full-path collision qualification is inferred.

## Current-controller results

Queued CUDA comparison. No compatibility or qualification conclusion is claimed
before execution.
