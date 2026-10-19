# ACT A1 calibration and stopping correction

October 19, 2026. Baseline `8b6b10c`, implementation `2bea7df`, branch
`PatrizioAcquadro/b1-policy-pilot`. Local evidence and the explicit corrected
recipe: `outputs/evidence/act-a1-calibration-stop-20261019/`. Ignored payloads are
not recoverable from Git. The complete local report describes every attempt.

## Question and acceptance

The [[b1-act-a1-two-door-diagnostic|original 0/2 diagnostic]] stopped at a
10.014-microradian wrist calibration exit and a 12.2-microradian future shoulder
exit. Neither established a physical violation. The latter's negative margin
was a future uncertainty certificate, not measured penetration. Prior 10 N halo
stop failures prevented unconditional position widening.

The [[../topics/purdue-b1-robot-and-contact|current robot/load/stop contract]]
uses one decision rule: empirical side-specific support at the *required future
stop load*, or independently observed contact-free motion and stopping. Current
weak force does not qualify stronger future contact. Lower wrist Z and upper
shoulder Z have independently checked 8.5 mrad coverage, one complete control
interval at the existing 0.51 rad/s bound; other old halo sides remain 1 N only.
No q clipping, teacher command substitution, trajectory-distance criterion or
privileged contact input is added. Main acceleration/combined-domain guards and
all physical/task criteria remain unchanged.

The actual swept robot volume already contains control motion, acceleration and
stop travel. Remove duplicate tracking displacement and duplicate target-rate
expenditure; retain target translation and angular growth through the full 2 s.
The old negative contact-free certificate need not turn positive: these two
regions now have separately measured loaded-stop support. Weak unsupported
regions still refuse.

## Physical evidence and retained failures

The original 12 robot-only cases cover right/left operating postures, nominal,
boundary and perturbation conditions, and 0/10 N loads. The perturbed 10 N stops
fail the original physical travel criterion: **23.849 mm right / 16.626 mm left**.
Keep both failures and all original traces. A shoulder-only stop gain leaves
large wrist/proximal compliance.

One explicit stop-only impedance profile, fixed before testing, passes 12 further
cases including the full 8.5 mrad boundary. Maximum stop travel becomes
**8.627 mm right / 5.301 mm left**, within the original 10 mm; the 3 degree,
actual-speed, final-speed, effort and two-second checks remain unchanged. Normal
motion gains are identical; reset restores them. The legacy profile remains the
default; only the separate recipe requests the new profile.

All first-matrix samples, including physical failures, calibrate the residual.
One intermediate changed-controller residual misses by 6.59 micro-N m. Transfer
independent reference-point uncertainty once as `10 N * 12.882 micrometers`;
physical tool FK checks cover 6,720 saved poses with maximum 0.851 micrometer
position error. Only three base residual bounds increase; dynamic coefficients
and weak scalar model do not change. Directional stop boxes cover the changed
response direction; existing radial, joint and rotation bounds are retained.

Four independent frozen-parameter cases add intermediate load-cone angles and
larger neighboring-joint perturbations. All physical, model and continuous
radial/directional volume checks pass, including 2 s maintained 10 N load plus
2 s removal. Minimum directional reserve is 0.231 mm. No fitting on these cases.
All 28,560 robot-only native rows are retained, including out-of-domain states.

Two corrected corpus-door contact probes pass: pre-stop loads 0.664/1.328 N,
stop travel 0.064/0.005 mm, no forbidden contact. They use source states only for
diagnostic initialization, six fixed target ticks and an actual two-second stop;
they do not count as learned execution, release or full-task success. An initial
right target-width failure and left stale collision-kinematics initialization
remain invalid attempts. The latter produces an artificial collision and unsafe
speed; it is preserved explicitly. One deterministic harness correction per door
checks the restored articulation before physics, then succeeds. No favorable
physical-outcome retry or threshold change occurs.

32 physical attempts: 30 interpretable outcomes (28 passes, two original physical
failures) plus two invalid contact-harness attempts. The corrected controller has
18 valid physical passes among 20 total attempts, including those two invalid
attempts. Original diagnostic/panel denominators remain unchanged.

## Closeout and next step

Saved decisions now support both terminal estimates and their complete future
joint intervals at required 10 N. Unverified sides, excessive loads/positions,
speed/acceleration and missing free-space support still refuse. Saved-trace
recalculation is not changed-trajectory physical evidence. Six essential new
regressions and affected consumers pass: 160 numerical tests; lint and whitespace
checks pass.

Total charged allocation **277.741 s (4.63 min)**, including all seven startups,
shutdowns and conservative tails; cumulative **16.777459/40 h**. New evidence is
about 6 MiB, below the 2 GiB tranche; prior retained evidence is 423.988/1024 GiB.
No owned compute workload remains. Old ledgers, recordings, checkpoints, failures
and paused panels are unchanged; this task's `summary.json` is the accounting
head. No training, collection, encoder, memory, action-space or RL work occurred.

This simulation correction is sufficient to proceed to the separately scoped
data/observation pipeline. Use the new local `runtime-recipe.json` together with
its load calibration and selected stop profile. Evidence is finite sampled
operating-neighborhood/load support, not a global all-speed/workspace proof,
hardware qualification or learned task completion. Full ACT opening/hold/release
and post-release stabilization remain unvalidated. Any later operating-domain
expansion needs its own proportional loaded-motion/two-second-stop check.
