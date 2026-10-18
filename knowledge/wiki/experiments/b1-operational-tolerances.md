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
a focused regression. The first short-window attempt is explicitly interrupted during inspection at
25.0 s: zero learned commands, incomplete recording retained and a healthy
two-second continuing-physics stop. The corrected complete outcome is retained
separately as `act-a1-window-2`. The reference result is unaffected: no state leaves its main position
or acceleration domain.

No physical robot result, broader calibration qualification, policy success or
unseen-door claim follows from these diagnostics. See
[[../topics/b1-policy-pilot|pilot interfaces]] and
[[b1-end-to-end-pilot-resume|unchanged historical outcomes]].


## Fixed ACT A1 short-window outcome

The corrected run uses source `6ab5b80`, the unchanged latest30k checkpoint,
original seed/setup and the separately validated weak-position recipe. It
executes 1,631 learned commands and stops at 52.3833 s with
`insufficient_local_action_margin`. Controlled opening is 17.2595 degrees;
there is no hold/release or learned task success. Peak native force is 7.2611 N,
with zero contact losses and no physical violations. The measured two-second
stop passes: 0.2422 mm maximum tool travel and 0.0001525 rad/s final speed.
Recording completeness, consumed-command/control/native clocks and unchanged
checkpoint audits pass. No native q uses the position halo in this live window.

The first captured input is not bitwise identical to the prior episode: mean
feature difference is 0.001594 and maximum first-prediction difference is
0.06289 mrad; backend settings and checkpoint are identical. The earlier 18.3044-
degree failure is not relabeled, nor is an overall performance gain claimed.

A same-trace comparison isolates the changed contact rule. The old full-3 mm
lateral charge would refuse four independently physically valid control states
(six repeated before/after checks). Its first refusal would occur at 51.3833 s:
actual material margin 0.9768 mm, corrected projected requirement 0.4577 mm,
remaining margin 0.5190 mm. The corrected rule permits 60 subsequent commands;
all 61 endpoint states from that boundary through the actual stop are physically
valid and have no violations. This is a conditional gate comparison on identical
actual observations, not a new learned-policy success.

At the terminal state the completed interval still has a valid material receipt:
1.8597 mm margin versus 0.4241 mm additional projected normal error. The next
proposal is different: minimum finite-finger clearance 141.5858 mm versus
142.7915 mm total declared uncertainty, leaving **-1.2058 mm**. Its terms include
83.6183 mm material-pose position uncertainty, 13 mm robot response, 13 mm
material response and 20 mm stopping allowance; full angular, calibration and
temporal terms remain in the JSON. The independent whole-robot sweep still has
0.02575 mm residual. These are distinct certificates. This refusal reports
insufficient evidence for the next finite footprint, not observed physical loss
or a missing receipt. Actual current contact validity cannot justify shrinking
an uncertain future bound after seeing the failure.

No further GPU retry, training or panel is run. Next work, if authorized, is an
independent calibration of lateral material-pose/response uncertainty on saved
observations before proposing any tighter bound or longer ACT A1 execution.
The present stop is retained until that evidence exists. Strong-load position
extension, broader configurations, hardware and real-time qualification remain
unvalidated. All original panels, failures, interruptions and frozen checkpoints
remain unchanged; new diagnostics occupy their own evidence directory.


## Goal 1 follow-up: primitive uncertainty and finite footprints

The new 2026-10-18 authorization limits completion to justified uncertainty and
explicit decisions on known small ACT A1 position excursions. Learned task
success is not required by this narrower completion rule. All original panel
outcomes and the earlier pause remain historical evidence. Follow-up artifacts
are separate under `outputs/evidence/operational-tolerances-20261018/goal1-closure/`.

One GPU replay of the **524 original ACT A1 captures**, without physics, policy
commands or training, reproduces the terminal 83.6183 mm frame-origin position
bound exactly. Its components are:

| Primitive | Translation | Rotation propagated to frame origin | Total |
|---|---:|---:|---:|
| Native registration | 5.7454 mm | 62.9408 mm | 68.6863 mm |
| Camera/FK calibration | 0.0129 mm | 0.0087 mm | 0.0216 mm |
| 83.33 ms acquisition age | 8.3333 mm | 6.5770 mm | 14.9103 mm |

All twelve measured native inliers remain. Maximum pair residual is 4.7454 mm,
including 4.7374 mm tangential and 2.1713 mm normal residual. The unchanged 1 mm
point-error assumption is added to that residual. Initial/current finite
plane-edge basis bounds are 0.035570/0.084123 rad; their sum remains 0.119693 rad.
No empirical maximum replaces these analytic bounds, and no residual is deleted
as a favorable outlier. Their predominantly tangential error cannot be removed
using the normal-plane contact correction.

The 83.6183 mm number belongs to the selected material anchor, **608.279 mm away
from the actual tool**. Registration uncertainty rotates about the measured
correspondence centroid, not that arbitrary anchor. The former admission mixed
an anchor-position bound with only the finger radius for subsequent rotation.
Simply adding the missing anchor-to-tool lever would instead spend correlated
rotation twice along a broken path. `LocalPatchState.pose_uncertainty_sources`
now retains every primitive's translation, angle, original frame lever and world
rotation origin. Admission transports each primitive directly to the actual
proposed footprints. At this endpoint the registration lever including the finger
is 243.367 mm, versus 526.165 mm to the anchor before adding any finger radius.
Missing primitive detail retains a conservative offset-footprint fallback;
incomplete or inconsistent supplied detail refuses. Geometry rotation also covers
the actual offset footprint, and robot tracking rotation retains its tool origin.

Two additional accounting defects are corrected. Both `MaterialCover` and
`footprint_support` already erode the declared 4 mm geometry tolerance. The
producer now declares that prepaid amount, credited only against material error
and only with an independent positive robot sweep proof. Invalid/excessive credit
refuses; holes and measured coverage remain unchanged. Also, the old 13 mm /
0.04 rad material-response allowance copied the separately measured **robot**
response. Material motion over the 1/60 s proposal now uses the existing
0.1 m/s and 0.15 rad/s envelopes: 1.667 mm and 0.0025 rad. Acquisition aging and
future material response are charged once in their respective terms. Robot
13 mm / 0.04 rad response and the 20 mm stopping reserve remain unchanged.

`terminal-uncertainty-audit.json` checks the resulting material-point calculation
against the existing tracker point budget. The current-point total is 90.145 mm.
The exact next endpoint was not serialized, so the offline comparison inflates
all levers by the unchanged maximum 20 mm proposal travel: uncertainty remains
at most **93.831 mm**, leaving at least **47.755 mm** against the preserved
141.586 mm proposal clearance. The original -1.206 mm arithmetic refusal is
therefore unnecessary under the corrected accounting. This is a conservative
term comparison on saved data; no post-terminal action or new runtime receipt
was executed. The full 3 mm normal-model bound and independent robot collision,
force, stop and task requirements are unchanged.

Independent evaluator joins happen only after observed estimates. The latest
recording has 273 available pose samples and one unavailable initialization
sample: maximum position/orientation errors are 10.566 mm / 0.001293 rad, with
zero errors outside the original complete position/angular bounds. Terminal
acquisition error is 4.506 mm, almost entirely tangent. Two independently retained
teacher recordings contribute 40 right and 43 left fixed three-second samples;
all are available and covered, with maximum position errors 14.765/20.060 mm and
orientation errors 0.001287/0.049556 rad. These finite checks support this pilot
domain, not hardware or global statistical qualification.

The latest material motion peaks at 0.01435 m/s and 0.01839 rad/s, below the
unchanged rate envelopes. Measured next-endpoint robot tracking peaks at
2.035 mm / 0.01333 rad, below the retained robot-response allowances. The first
analysis serialization failure and invalid native-angle handedness join remain
preserved and explicitly excluded from valid conclusions; neither changed any
operational decision or original outcome. No ACT A1 learned retry is launched.


## Goal 1 follow-up: coherent position rule and closure

The same prospective rule remains **one native sample of travel** at the unchanged
0.51 rad/s and 120 Hz: 4.25 mrad, only on independently tested joint sides,
force upper <=1 N and original main acceleration. It is empirical model support,
not sensor roundoff or a physical joint-limit extension. No per-failure epsilon
is added. All other sides remain zero and no combined position/acceleration
extension is admitted.

Existing weak recordings cover no upper-shoulder-Z halo. One additional robot-only
GPU probe therefore uses the retained latest30k left posture and the original
weak-load perturbation/force/yaw schedule. It preserves all **5,346 native
measurements**, including out-of-domain samples: 1,455 are position/speed/main-
acceleration eligible, 1,215 satisfy the operational weak-load cap and 698 of
those use the halo. Measured shoulder excursion reaches 7.856 mrad; adoption
stays at 4.25 mrad. Unchanged joint/scalar residual margins remain positive,
minimum 0.000533 N m / 0.005079 N. Robot response is 1.899 mm / 0.026681 rad,
below the original 13 mm / 0.04 rad allowances. The independent known-1 N stop
travels 0.948 mm, with 1.048 mm radial and 0.256 mm directional reserve; held
joints, acceleration and continuing-physics stop checks pass unchanged.

The separate adopted recipe is `goal1-closure/runtime-recipe.json`, with
`goal1-closure/weak-position-calibration.json`; original recipes are untouched.
It adds upper shoulder Z to the previously supported sides without refitting
any error coefficient. Review includes 14 retained ACT A1 outcomes and eleven
terminal domain-exit samples: all ten observed position excursions now obtain
an estimate. Latest30k left shoulder-Z excesses of 7.331 and 20.683 microradians
have force uppers 0.06150 and 0.06132 N and are accepted for estimation. The
remaining 5.743 N acceleration-only case still refuses; all 4,899 retained 8/10 N
out-of-position states remain refused.

Estimation support is distinct from command permission. On the retained latest
left trace, the existing prospective guard would already refuse at 30.9333 s:
its motion/stop envelope can leave the main position domain and its independent
contact-free stop margin is negative. All 370 such saved proposals remain
unsupported. A weak current load cannot prove a future stronger loaded stop;
accepting the tiny measured position excess does not bypass that requirement
or turn this historical failed task into success.

**Goal 1 closes under the latest user-defined completion criterion:** uncertainty
is traced and corrected, and every known small ACT A1 position excursion has
an evidence-based estimation decision plus the unchanged future-motion guard.
Goal 2 has not started and may be addressed separately. This closure does not
complete or restart the original paused learned-success panels, qualify stronger
loads or hardware, or demonstrate post-correction learned task success.
`goal1-closure/completion-report.json` records the retained checks and incremental
resource accounting; no training, panel, other combination or post-terminal
learned-policy attempt is run. Numerical regression and affected-consumer checks
pass; failed analysis/test attempts remain separately preserved.
