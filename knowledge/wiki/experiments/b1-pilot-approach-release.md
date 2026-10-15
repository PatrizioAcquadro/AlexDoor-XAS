# ACT A2 Approach and Release Residuals

Baseline: `6337eb8`, branch `PatrizioAcquadro/b1-policy-pilot`.
Follow-up to [[experiments/b1-pilot-preprocessing|fixed-unit preprocessing]].
Persistent local evidence and executable audit/trial source:
`/home/pacquadr/Desktop/AlexDoor-XAS/outputs/evidence/b1-policy-residuals-20261015/`.
The original episodes, v1/v2 predictions and initial/final checkpoints remain in
their original locations. Ignored evidence is not recoverable from Git.

## Diagnosis before intervention

The audit uses saved predictions and numerical feature/command/trace arrays only;
it performs no model forward pass, training or simulation. All three complete
episodes are included. No demonstration, controller, SAM3 or Point2Pose change is
made. The two train identities are pooled before computing quantiles, while
individual-identity results remain available. Correlated action windows are
descriptive samples, not independent trials.

### Alignment and actual minibatches

Raw command timestamps equal observation timestamps before execution, exactly.
Prepared timestamps, all 18 proprioceptive columns, original action labels and the
saved prediction targets/phase joins agree exactly. Each raw episode has N+1
observations for N commands, including the 1,500-tick inspection prefix.
The original independent FK reconstruction audit is retained: maximum A2 position
reconstruction error is zero and rotation error is below 1.1e-14 rad.
This follow-up checks its joins and verifies raw command tracking errors against
the independent poststep teacher trace; it does not claim a new FK replay.
Annotations at observation t describe the preceding transition. They disagree
with command phase at five boundaries in each recording; scoring and sampling
correctly use command phase, so this is not a training-label shift.

The exact original sampler is reconstructed with seed `17*10000+epoch`, first
32 batches of 32, for all three epochs. Actions and padding are checked against
the actual maintained normalized batch factory in every one of the 96 batches.

| Anchor phase | Available train ticks | Draws | Unique ticks seen | Batches without phase |
|---|---:|---:|---:|---:|
| approach | 720 | 313 | 268 | 1 |
| contact | 360 | 136 | 116 | 21 |
| push | 5,252 | 2,299 | 1,983 | 0 |
| hold | 360 | 159 | 136 | 20 |
| release | 360 | 165 | 139 | 16 |

Push receives 74.8% of anchors; release receives 5.37%. Valid loss targets retain
the same imbalance, rather than omitting release through padding. Only five
sampled chunks cross hold into release, with **none in the first epoch**.
All phases are present; this is limited exposure and objective weighting, not
an empty release loader. It motivates one controlled sampling test, but does not
establish that balancing will repair the errors.

### Errors are not only development generalization

For the v2 consumed prefix, cells below are **mean / median / p95**.
Translation is in mm; rotation uses relative SO(3) angle in degrees.

| Split | Phase | Translation mm | Rotation deg |
|---|---|---|---|
| train pooled | approach | 15.351 / 8.240 / 60.551 | 2.431 / 0.813 / 12.955 |
| train pooled | release | 13.489 / 11.073 / 29.972 | 1.822 / 1.689 / 3.482 |
| development | approach | 20.759 / 19.570 / 41.532 | 1.635 / 1.234 / 4.039 |
| development | release | 14.938 / 12.091 / 33.341 | 1.682 / 1.613 / 2.854 |

Individual train release means are 9.147 / 17.831 mm and 1.451 / 2.192 deg.
Train zero release means are 10.231 mm / 1.016 deg; train median gives
11.562 mm / 1.027 deg. Development zero gives 9.091 mm / 0.912 deg and train
median 9.958 mm / 0.944 deg. V2 therefore fails to fit release even on train,
and its error exceeds simple controls. V1 also has high train release errors.

Errors extend well beyond phase boundaries. After excluding the first and last
0.2 s of release, v2 means are 9.155 / 18.758 mm on train and 15.783 mm on
development. For approach interiors they are 6.469 / 25.842 / 21.950 mm.
The first release half has larger translation errors than the second on every
identity. This is not solely a boundary-attribution or discarded-chunk effect.
`physical-errors.json` retains all phase/position intersections, transition bins,
full horizon, consumed positions 0–5 and discarded positions 6–11, with padding
excluded. `pooled-errors.json` pools sample arrays before taking quantiles.

### Can the inputs distinguish hold from release?

The teacher holds for 3 s, blending its opening lead over the same interval.
Release first withdraws along the fingers, then returns toward an earlier
achieved pose selected from its trajectory. It also maintains accumulated
tangential compensation during contact/push/hold, and removes it at release.
The policy receives one static image, three causal pooled acquisitions at 10 Hz
(0.2 s acquisition span, with sample age below 0.1 s), and current q/dq.
It receives no explicit timer, trajectory memory or compensation accumulator.
These facts motivate an information question; they do not prove insufficiency.

There are **no exact full-input duplicates** across hold/release within any
episode. All 360 hold/release full input vectors per identity are distinct.
The visual block repeats frequently between acquisitions, while q/dq change.
Full fixed-unit nearest-neighbor matching is retained for every release tick,
along with separate RGB, metric depth, coverage, q/dq and target-chunk distances.

The immediately adjacent hold/release inputs have identical visual blocks on
all three identities. Maximum q changes are 0.264 / 0.0677 / 0.0521 deg, and
maximum dq changes are 0.00773 / 0.000826 / 0.000689 rad/s. Yet current command
differences are 7.670 / 5.470 / 7.398 mm and 1.661 / 0.599 / 0.986 deg.
A declared resolution probe (RGB RMS <= 1/255, depth RMS <= 1 mm, q max <=
0.1 deg, dq max <= 0.01 rad/s, identical coverage) finds 0 / 1 / 1 cross-phase
pairs, covering only 0 / 1 / 1 of the 180 release ticks per identity.
Those thresholds are descriptive, not calibrated observation noise.

Thus a local command discontinuity occurs near nearly unchanged observations,
but the recordings do **not** demonstrate widespread irreducible hold/release
aliasing. At 0.1–2 s separation, proprioception and visual/depth history differ.
No timed phase input, oracle phase, global-clock shortcut or longer-history
variant is justified by this audit alone. A memory/observer contract would need
its own causal discrimination evidence before changing policy inputs.

### Historical compensation and command semantics

A2 is `goal_t - measured_tool_t` plus relative rotation, **not** the difference
between successive goals. The exact translation decomposition is:
`A2_t = (goal_t-goal_(t-1)) + (goal_(t-1)-measured_tool_t)`.
The latter term includes historical tracking residuals and cannot be interpreted
as intended physical motion or as an independent causal effect.

| Quantity, mean mm | Train first | Train second | Development |
|---|---:|---:|---:|
| approach target increment | 0.440 | 0.878 | 1.028 |
| approach prior-target residual | 7.310 | 25.012 | 20.225 |
| release target increment | 0.259 | 0.246 | 0.255 |
| release prior-target residual | 10.355 | 9.924 | 8.969 |
| hold tangential compensation magnitude | 7.490 | 5.241 | 6.656 |

Tangential compensation is zero in approach and release, so it cannot directly
explain their whole-interval errors. Removing 7.273 / 4.081 / 6.700 mm of hold
compensation coincides with target jumps of 7.575 / 5.452 / 7.372 mm at release;
the release reference is also reset to the measured pose, so this is not a
single isolated compensation effect. The first release action becomes about
0.023 mm, while the previous hold action is 7.659 / 5.454 / 7.388 mm.
The audit subtracts the recorded compensation only as a diagnostic calculation;
no target is relabeled and privileged trace fields never enter policy inputs.
Command increments, prior tracking residuals, compensation and new-controller
validity remain separate questions.

## One prespecified intervention

Only anchor sampling changes. Each 32-example batch contains six examples from
each of five phases plus two extras rotating evenly over phases. Each phase is
independently shuffled using seed `17*10000+epoch`; no tick repeats within an
epoch. Across 96 batches this gives 614–615 anchors per phase, including 614
release anchors and 36 hold-to-release crossings. Unique release ticks rise
139→323/360. The complete order and valid/padded target counts are saved before
training. Teacher phases select supervision rows only, never input features.

Reference: reuse the completed, accepted v2 CUDA cell, without repeating it.
The candidate must match its initial model tensors and complete normalization
exactly. Model, v2 inputs, source actions, seed 17, optimizer, learning rate,
12-step horizon, batch 32 and 3×32 updates are unchanged. Checkpoints save each
completed epoch for operational resume. Use the final checkpoint and perform no
development evaluation during training. No extra seed, budget, model or input
variant is allowed.

Before CUDA, `trial-protocol.json` freezes the following decision margins:

- Pooled train **and** development release translation mean, median and p95
  must each improve by at least 1 mm against v2.
- Release rotation mean, median and p95 may not regress by more than 0.1 deg.
- Approach/push/hold prefix medians and p95 may not regress by more than 1 mm
  or 0.1 deg on train or development. The same guard applies to development
  overall median/p95.

These are practical offline comparison margins, not physical admission or
statistical-significance thresholds. Report every result regardless of passing.
Keep zero and componentwise global train median; report all chunk positions and
phases. The controls are computed from the same unchanged train actions.
Both historical shared GPU leases and 180 continuously compute-idle seconds are
required; retain interference and resume only incomplete work.

## Trial state

The diagnosis and trial protocol are complete. One balanced CUDA cell is queued
under the existing shared GPU protocol. Results and adoption decision are pending.
V2 remains the explicit experimental recipe. No physical-policy improvement or
new-controller validity has been demonstrated by this diagnosis.
