# ACT A1 operational tolerance investigation

Baseline: `PatrizioAcquadro/b1-policy-pilot` at `1756fc9`, 2026-10-18.
The user authorized a bounded diagnostic of load-position and right-door material
margin refusals. Other combinations, training and full panels remain deferred.
All earlier failures and interrupted attempts retain their original labels.
Evidence is `outputs/evidence/operational-tolerances-20261018/`.

## Frozen inputs and question

ACT A1 remains the latest 30,000-update checkpoint with SHA256
`d0ecc8a8a061ed1575d433c118f1e78824bdb19ae611f8b4c531c74f7b6d4d60`.
No weight, normalization, task-success threshold or physical limit changes.
A recorded reference command sequence is diagnostic input only; current RGB-D,
proprioception and robot dynamics still supply every operational decision.
Evaluator geometry and scoring never feed admission, contact or control.

Saved selected-v10 position exits include 1.11 microradians at shoulder Z,
35–152 microradians at wrist X and 279 microradians at gripper Y. These are
exits from an empirical calibration box, distinct from physical joint limits.
The earlier calibration already included 20 mrad position padding; another
unconditional allowance would need independent error and stopping evidence.

## Material-margin correction

`MaterialCover.query` returns tangent-plane distance from the two finite finger
footprints to observed material boundaries. The monitor previously spent the
complete 3 mm normal visible/collision discrepancy against this lateral margin.
With a separately admitted whole-robot collision/stop proof, it now spends
`normal_model_bound_m * sin(min(geometry_angle + pose_angle, pi/2))` laterally.
The preserved `contact-normal-finite-model-development-audit.json` measures
normal model offset/protrusion up to 1.973 mm across finite footprints; the 3 mm
normal bound is retained. It is not reduced to make a failed episode pass.
The complete 3 mm remains in the normal separation interval and existing robot
collision/interface checks. Without independent collision evidence, the original
isotropic expenditure remains. Shared geometry/pose errors are not charged twice.

Missing material-margin evidence has a different refusal from an insufficient
margin. Receipts record each finger's measured clearance, empirical/model/robot/
response/temporal/stop terms, total uncertainty and resulting material margin.
Contact decisions record material identity, actual margin, normal discrepancy,
projection angle, required tangent allowance and numerical tolerance (zero).
The independent robot sweep residual is reported separately.

The preserved right reference replays 7,230 recorded states: 7,224 available,
six initial missing material identities, 6,390 contact confirmations and maximum
force upper bound 9.9939709 N. There is no first operational refusal. These are
saved-data results, not fresh execution or learned success.

## Empirical position support

Calibration supports explicit lower/upper per-joint position allowances. Defaults
remain zero; q, Jacobian, torque and the fitted error model are never clipped.
A supplied halo must have an independently supported force upper domain. Speed,
acceleration, physical joint limits, force and task criteria remain unchanged.
Future motion/stop leaving the main position envelope additionally needs observed
contact-free support; a weak current measurement cannot qualify a stronger load.
Refusals retain measured quantities, original bounds, allowances and excesses.

The prospective candidate is one native sample of travel at the unchanged
0.51 rad/s speed and 120 Hz physics rate: 4.25 mrad. Only the affected sides of
shoulder Z, elbow Y, wrist X and gripper Y are candidates. It is a finite support
experiment, not an assertion that sensor error is 4.25 mrad or that all extrema
are physical walls.

The first probe stops after trying to resume a latched stop; its safe continuing
stop and interruption remain. The corrected strong-load probe retains 21,988
native measurements. At 10 N, shoulder/gripper stops travel 23.223/31.439 mm,
exceeding the unchanged 10 mm requirement; gripper model error also exceeds the
original residual bound in 11 position/speed-eligible samples. This does not
justify an unconditional or strong-load position extension. A separate weak-load probe passes the existing 1 N scope without refitting any
coefficient: 13,492 position/speed/acceleration-eligible native states and 11,861
weak-load states. Minimum joint residual margin is 0.0004985 N m; minimum scalar
normal-error margin is 0.004975 N. Maximum measured tracking response is
11.514 mm / 0.027763 rad, below unchanged 13 mm / 0.04 rad response limits.
All four known-1 N stops pass, with tool travel 0.904–3.209 mm. Continuous 14-shape
radial and directional stop boxes, held-joint travel and acceleration also pass;
minimum directional residual is 0.2555 mm.

The new recipe is `runtime-recipe.json`, with `weak-position-calibration.json`.
Only lower shoulder Z/elbow Y and upper wrist X/gripper Y receive 4.25 mrad,
only while force upper <=1 N. A future main-domain exit still requires independent
contact-free motion/stop support. Larger, opposite-side, strong-load and combined
unsupported acceleration cases remain blocked.

Replaying the twelve retained ACT A1 terminal native samples estimates eleven,
including all seven actual position excursions. The remaining 5.7434 N case
still refuses its 12.241 rad/s2 main-acceleration excess. This proves estimation
support, not new actuation or historical task success. The complete saved right
reference has zero proposed motion/stop envelopes requiring the new position guard;
there the weak-position recipe and original position recipe behave identically.
The retained latest30k ACT A1 right/nominal prefix also introduces zero new
future-position guard refusals. A separate replay of 4,899 independently measured
8/10 N states outside the main position box rejects every state; none gains an
estimate from the new weak tolerance. These labels select validation cases only
and never enter the estimator. `position-halo-validation-v2.json` additionally
checks the first stop interval from its native anchor, with the same passing
radial/directional margins and unchanged calibration bytes.

## Reference execution and limits

`pilot_physics.py rollout --reference-recording PATH` loads the fixed ACT A1
checkpoint for compatibility and substitutes explicit recorded A1 deltas through
`MatchedReplay` and the same `B1Runner`, preview, motion, load and contact checks.
Commands are marked non-learned. Independent scoring reports
`reference_cycle_success` separately and cannot claim learned task success.
The first startup rejects a recording-column indexing error before commands.
Its stop fails the speed criterion because the environment had not yet been reset
to the configured robot state; that unfavorable result remains. Runtime now resets
immediately after construction, before fallible checkpoint/reference setup.
The corrected recording-column path passes an offline 7,217-command preflight.
The fresh `reference-cycle-2` completes all 7,217 admitted reference commands.
Independent scoring gives 51.0242 degrees controlled opening (99.5394% of the
unchanged 51.2603-degree reference), valid hold and commanded release, zero
contact losses and no physical violations. All 6,149 push states and all 180
hold states are physically valid. Peak native force is 7.8637 N; maximum load
upper bound is 8.0535 N. The two-second stop retains 240 healthy native samples,
maximum 0.02134 mm tool travel and final speed 0.0001525 rad/s.

The recording, actual-command and control/native clock audit passes. There are
zero learned commands and no prediction queries; `task_success` remains false,
while `reference_cycle_success` is true. The minimum actual material margin is
25.133 mm, leaving 24.758 mm after projected normal-model error. All native q
and every proposed motion/stop envelope stay in the original main position box;
the new weak-position recipe therefore makes exactly the same decisions on this
executed path. No second full reference simulation is needed for inactive bounds.

A single prospectively frozen ACT A1 right/nominal diagnostic uses source
`618e153`, the new runtime recipe and at most 1,900 learned commands. This extends
past the earlier 1,757-command/54.4833 s contact-margin refusal without running a
new panel. Checkpoint, seed, setup and full task criteria remain fixed; its
short diagnostic horizon cannot establish a complete learned task result.
Final review identifies a missing combined-domain guard before the first policy
command: a new position halo could otherwise combine with the independent weak
acceleration extension. The guard now refuses that unsupported combination, with
a focused regression. The first short-window attempt is explicitly interrupted
during inspection and preserved; the corrected single complete outcome remains
pending. The reference result is unaffected: no state leaves its main position
or acceleration domain.

No physical robot result, broader calibration qualification, policy success or
unseen-door claim follows from these diagnostics. See
[[../topics/b1-policy-pilot|pilot interfaces]] and
[[b1-end-to-end-pilot-resume|unchanged historical outcomes]].
