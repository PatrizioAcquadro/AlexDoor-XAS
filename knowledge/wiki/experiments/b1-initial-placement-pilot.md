# Bounded Initial Placement Study

Baseline `4e5a27f`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
After [[b1-joint-limit-aware-ik|one rejected IK cycle]], the user authorizes one
train/development placement audit, at most one common candidate and four Isaac
cases within the original three local GPU-hours. The goal remains incomplete.

The three existing selected episodes share initial tool Y = -0.095034 m. Their
closed-door precontact endpoints span Y [-0.180583, +0.195168] m. One analytical
candidate moves the ready hand to that interval midpoint, reducing maximum lateral
travel from 0.290203 to 0.187876 m. Translate robot and mounted pedestal together
by +0.102327 m: floor (-0.437, 0.482327, 0). Keep X/yaw, ready joints, neck,
controller, physical limits, task endpoints and all acceptance rules unchanged.
This is one derived point, not a validated placement range or global IK optimum.

Prospective references reuse the existing six-second/70%-movement cubic/Slerp
teacher formula with the translated start. Historical observations remain intact.
Teacher geometry is used only for setup qualification, never hidden policy or
operational-admission input. Development informs setup, never training statistics.

CUDA endpoint and ideal-servo continuation screens precede any Isaac case. A
numerical pass cannot establish physical tracking, contact, camera visibility or
load support. If all three audited episodes pass the fixed screen, execute only
two complete train approaches/endpoint holds/stops and their two moving stops.
Base pose must remain stationary at both physics substeps; no door/contact or
learned command is admitted by these robot-only cases. Original tracking, speed,
hold, stop and load-calibration rules remain fixed. No alternative candidate or
completed-case retry is authorized.

## Measured result and bounded adoption

The CUDA ideal-servo screen passes on all three existing train/development
approaches. It predicts left p95 1.814 mm /0.698 deg while still clipping 92 targets;
clipping remains a diagnostic, not a new rejection criterion. Right/development
p95 positions are 0.0063/0.0051 mm. These are numerical screening results only.

Exactly four new CUDA physical cases complete without interference, at 60 Hz
control/120 Hz physics. Independent FK agrees with measured tool poses below
0.8 micrometers /1.9e-6 rad. Every native feedback sample is available; no effort
clipping or unexpected robot contact occurs. Actual robot base is stationary at
both substeps and the pedestal shares the selected common floor transform.

| New unloaded approach | Right train | Left train |
|---|---:|---:|
| Poststep position p95 / maximum, mm | 2.186 /2.210 | 5.070 /5.279 |
| Poststep rotation p95 / maximum, degrees | 0.763 /0.768 | 1.701 /1.719 |
| Actual peak joint speed, rad/s | 0.319 | 0.277 |
| Target-rate-scaled / position-clipped steps | 0 /0 | 0 /178 |
| Moving-stop travel / rotation | 3.434 mm /0.370 deg | 3.668 mm /0.603 deg |
| Moving-stop final speed after two seconds, rad/s | 0.000155 | 0.000305 |
| Original approach/hold/stop/speed criteria | Pass | Pass |

Both endpoint holds and continuing-physics endpoint stops pass. The left measured
minimum joint margin is -4.77e-7 rad, within the unchanged 1e-4 rad numerical
allowance; no joint limit is widened. Position clipping remains present, but the
complete resulting physical path passes the existing tolerances. Do not confuse
the ideal-servo prediction with measured dynamic tracking.

The four cases total 28.2 simulated seconds and 26.20 case wall seconds including
instrumentation/static checks, excluding GPU waiting, startup/teardown and final
trace serialization. Only 128/3,384 residual estimates are inside the unchanged
old calibration domain. Complete native feedback does not qualify the missing
load envelope, known-load error or contact stopping.

Adopt `configs/b1_shared_pilot_setup.json` **for the bounded pilot's unloaded
approach only**. It differs from the preserved corpus setup solely in floor Y;
the historical corpus setup and prior qualifications remain unchanged. The
`pilot_physics.py --setup <path>` option explicitly selects a prospective setup and
records its values/source in runtime metadata; omission keeps the corpus setup.
No new episode, training, learned actuation or final dataset is created here.

This one-candidate/four-case study is closed successfully for its declared scope.
The later [[b1-pilot-contact-prerequisites|complete-teacher prerequisite study]]
uses three of the twelve available cases and fails all three; the current setup
and unchanged teacher do not yet support a valid pilot task. Its failure rule
stops that allocation with nine cases unspent.
Door/frame clearance, camera visibility and a nonzero placement-variation range
remain unverified; one common nominal point cannot establish those claims.

## Preserved evidence

`outputs/evidence/b1-shared-learned-pilot-20261016/placement-cycle/` retains user
authorization, previous TODO, raw-input audit, candidate derivation, frozen protocol,
exact scripts, three numerical continuations, all four physical traces, telemetry,
comparison, independent verification and `outcome.json`. The initial read-only
metadata-path error and preflight-only entrypoint/criteria review are retained;
neither adds a physical/model outcome or causes a completed-case retry.

Verify existing traces without simulation from the requested worktree:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p outputs/evidence/b1-shared-learned-pilot-20261016/placement-cycle/verify.py --repo "$PWD"
```

The verifier writes its local summary. Do not rerun the completed physical study
under its exhausted four-case allocation. New collection must explicitly select
the prospective setup only after its remaining qualification succeeds.
