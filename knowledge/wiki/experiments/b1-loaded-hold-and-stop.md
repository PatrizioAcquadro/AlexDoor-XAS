# B1 Loaded Hold and Measured-Anchor Stop

October 15, 2026. Baseline `6002fcc`; controller correction `8d1df08` on
`PatrizioAcquadro/b1-policy-pilot`. This is bounded 6.0D simulation evidence.
No door contact, policy change, training, perception, geometry or TSDF work occurred.

## Question and causal diagnosis

The [[b1-pilot-offline-and-robot-dynamics|earlier dynamic diagnosis]] retained a
106.845 mm loaded stop failure. This follow-up separates fixed-target holding
from stop target reassignment, with independent resets and the same 4 N reference.
Fixed targets already permit 56.621 mm / 9.375 degrees of deflection. Resetting
those targets to measured positions removes the spring effort opposing the load;
a further 111.472 mm / 20.349 degrees occurs over the next two seconds.

The arm uses explicit `IdealPDActuator`, not implicit PhysX drives. Solver drive
gains are zero; the resolved actuator gains carry the control law. The old
terminal stiffness is 2 N m/rad, versus reflected load torque around 0.22 N m in
the earlier trajectory. Large terminal deflection is consistent with that low
stiffness. Proximal joints also contribute. Gravity remains enabled and is
compensated from the current robot model at every physics substep. Neither missing
gravity compensation nor effort clipping explains the tested failure: all baseline
commands stay below 10.0% of physical effort limits, with zero clipping samples.

The correction uses a task-local right-arm impedance budget and an environment
command latch; exact gains and stop semantics are canonical in
[[../topics/purdue-b1-robot-and-contact|the robot/controller contract]]. It retains
the explicit actuator, original limits, measured stop anchor and gravity model.
It does not preserve the previous unachieved goal, integrate tracking error,
freeze a disturbance torque or use the injected reference in control. The former
spring compression is cleared at stop. Support then arises from bounded motion
relative to the new anchor, with damping. This verifies free-space support/load
removal; frictional contact unloading remains untested.

## Prespecified protocol

Evidence is outside the managed worktree at
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1-load-stop-20261015/`.
`WORKLOG.md` states conditions and criteria before baseline/candidate execution.
Each complete run has 1,440 control ticks / 2,880 physics samples, 24 s simulated
across resets, on RTX 4090 CUDA. Physics is 120 Hz, control 60 Hz. Cameras are
disabled for robot-only isolation; no corpus door is spawned, commissioning
fixtures move to X=3 m and any active robot contact aborts the test.

A fixed ready target is held for two seconds unloaded, three seconds under
4 N along negative world X, then two seconds after removal. Independent loaded
and unloaded moving-stop cases each settle for two seconds, preload for two,
execute half a second of the earlier 80%-amplitude cosine excursion, stop without
an extra tick, hold physics for two seconds, then remove the force for two more.
Both substeps contribute to motion, feedback, velocity, acceleration and torque
measurements. Force application point/moment are evaluator-only references.

Before verification the local criteria are fixed: steady joint tracking error
at most 0.02 rad; hold and post-stop travel at most 10 mm / 3 degrees; final
post-stop joint speed at most 0.01 rad/s at two seconds; return within 2 mm of the
stop anchor two seconds after removal; persistent command refusal until reset.
These are local engineering criteria, not contact safety or hardware tolerances.

The previous `dynamics/calibration.json` is reused unchanged: initial joints
+/-0.08 rad, 0.25 rad/s, 5 rad/s², sign -1 and the saved residual bounds. No new
calibration, fitting on verification, widened envelope or threshold retry occurs.
Native deterministic PhysX/model feedback is the error model; hardware noise,
friction uncertainty and sensor precision are unmeasured. All qualification flags
remain false, including `verified` and `stop_verified`.

## Results

| Quantity | Baseline | Corrected |
|---|---:|---:|
| Loaded hold, maximum tool displacement | 56.621 mm | 5.178 mm |
| Loaded hold, maximum rotation | 9.375 deg | 0.665 deg |
| Loaded hold, last 0.5 s maximum joint error | 0.074879 rad | 0.004389 rad |
| Loaded moving interval, maximum joint error | 0.077636 rad | 0.008648 rad |
| Loaded stop, maximum travel / rotation over 2 s | 111.472 mm / 20.349 deg | 5.976 mm / 0.800 deg |
| Loaded stop, final travel / peak joint speed at 2 s | 111.472 mm / 0.007772 rad/s | 4.126 mm / 0.000339 rad/s |
| Loaded stop, time after which all joint speeds stay below 0.01 rad/s | 1.892 s | 0.433 s |
| After load removal, final distance from stop anchor | 0.0247 mm | 0.00282 mm |
| Unloaded moving stop, maximum travel / rotation | 1.558 mm / 0.241 deg | 1.214 mm / 0.175 deg |
| Maximum actual speed across complete trace | 0.666881 rad/s | 0.070877 rad/s |
| Maximum actual acceleration across complete trace | 19.785 rad/s² | 3.356 rad/s² |
| Available residual estimates | 2,568 / 2,880 | 2,880 / 2,880 |
| Evaluator-corrected residual outside original bounds | 41 samples | 0 samples |
| Actuator clipping | 0 samples | 0 samples |

Corrected force-error maximum is 0.04142 N over the complete sample set, and the
upper bound covers the reference on every sample. The largest corrected joint
residual is 0.01359 N m, below the unchanged saved bounds. Estimation availability
is not calibrated admission: no flags are promoted. Baseline unavailable samples
and corrected-residual failures stay in the denominator. Actual stop motion is
not instantaneous: the corrected loaded peak speed is 0.07088 rad/s, and the
0.433 s settling observation must be included in any future stop margin.

Corrected actuator-command maxima in canonical right-arm order are
`[5.9705, 16.1609, 6.3640, 5.5341, 0.1375, 0.1442, 0.7186] N m`.
The largest effort-limit fraction is 10.06%; commanded and applied efforts match
without clipping. Solver joint-output feedback remains separately recorded.
In the final half-second after load removal, non-gravity applied effort is at
most 0.000103 N m. At stop, the supplementary check directly verifies measured
position targets and gravity-only feedforward. No old spring preload is retained.

A separate CUDA check exercises commands while physics continues under one second
of 4 N and one second after removal: 72 A1/pose/A3/neck/Gym/IO command attempts
refuse before time advances; 12 repeated stops preserve the anchor. Explicit reset
clears the latch and accepts a new A1 command. A2 absolute-pose 4 mm cosine cycles
reach maximum error 5.240 mm / 0.705 deg under 4 N, then 0.265 mm / 0.124 deg after
removal. This exercises the common pose path; no A3/A4 geometry or policy task is
validated. The targeted portable regressions pass (74 tests).

## Evidence, failures and reproduction

`baseline/` and `corrected/` each preserve full traces, report and analysis.
Every executed source is preserved separately; `baseline-purdue.py` retains the
old stop. `latch-reset-a2-complete/` holds the supplemental protocol/source/trace.
The original baseline report's `applied_approx` field and implicit-effort description
are an instrumentation naming mistake: these arrays actually contain explicit
`applied_torque`. The analyzer handles that legacy field without changing raw data.
`corrected/report.json` captures resolved actuator gains and correct semantics.

`latch-reset-a2/` and its log retain a supplemental start interrupted only in this
agent's process after detecting concurrent SquadBot CUDA work. It supplies no
valid result. The first guarded retry is refused before simulator startup and its
log is retained. The completed retry waits for a free GPU window. No other job is
stopped. `verify_loaded_stop.py` now acquires the shared local GPU lock and checks
active compute processes before starting Isaac; external tools may not share this
lock, so the occupancy check remains necessary.

Use Isaac Lab Python to run `scripts/verify_loaded_stop.py --output <new-path>
--calibration <saved-calibration.json>`, then `scripts/analyze_loaded_stop.py
<new-path>`. Set this checkout's `src` on `PYTHONPATH` when invoking scripts from
another directory, or pass `--repo`. The analyzer performs no fitting and retains
all samples/rejections. Ignored evidence is local and cannot be recovered from Git.

## Verified regime and remaining 6.0D boundary

Verified: this fixed-base ready-pose neighborhood, small cosine/A1 and 4 mm A2
motions, 0/4 N negative-world-X tool loading, two-second post-stop windows, ideal
native simulation feedback and explicit same-process reset. No general workspace,
load-direction, torque saturation, near-singular posture or long-duration bound
is established. The gain change preserves command-step limits but does not turn
those limits into measured velocity guarantees outside the tested regime.

Before 6.0D completion, independently qualify dynamic/peak residual uncertainty,
active saturation and model/friction error, feedback/control timing gaps and fault
stops with continuing physics, ill-conditioned postures, supported operating bounds
and contact/unloading response. Camera-enabled scheduling and hardware behavior
remain separate. Required 6.0D checks and independent local geometry/6.0E must pass
before 6.0F; none is bypassed by this robot-only success.

Existing demonstrations/checkpoints and the previous synthetic qualifications
are preserved under their original controller. Feature/action schemas and loading
remain compatible, so offline diagnosis remains usable. Their physical tracking,
contact and replay results do not transfer to the new impedance. Before comparative
closed-loop learning, deliberately revalidate or recollect matched demonstrations
under one frozen controller; do not mix controller regimes silently. No checkpoint
is retrained, relabeled or newly qualified here.
