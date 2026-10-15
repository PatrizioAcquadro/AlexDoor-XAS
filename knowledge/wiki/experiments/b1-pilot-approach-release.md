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

Anchor coverage is not target coverage. Overlapping chunks supervise 352/360
release target ticks (97.8%), at 5.389 presentations per available tick versus
5.262 for push. Thus release is not generally missing from supervision or sampled
less frequently per recorded tick; its much shorter duration gives it low total
weight in the time-uniform objective. `exposure-detail.json` distinguishes anchors,
unique supervised targets and presentation counts. Reweighting phases tests that
objective allocation, without establishing an absolute sample-sufficiency limit.

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

This persists on queries that actually appeared as original training anchors:
102 consumed release predictions from those queries have v2 mean/median/p95
13.616 / 12.212 / 28.763 mm and 1.720 / 1.717 / 2.658 deg. High train release
error is not explained merely by scoring query anchors absent from the short run.
The same fixed query groups are retained for the candidate comparison; they do
not select its checkpoint or change the declared criteria.

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

## Completed single CUDA comparison

Exactly one candidate completed all 96 CUDA updates on RTX 4090 after both
shared leases and 180 continuously idle compute seconds. Initial model state and
complete normalization match the preserved v2 reference exactly. No development
validation is supplied to the trainer, and final.pt is used. Accepted monitoring
contains no foreign compute or new graphics process. The original checkpoint
hash remains unchanged. No controller, observer, model, action schema or SAM3–P2P
source is modified; the scoped sampling/trial source stays with ignored evidence.

The 3,072 sampled chunks and 96 updates are unchanged. More terminal-release
anchors increase padding from 52 to 256 positions: valid loss targets are
36,812→36,608 (0.55% fewer), with the same padding mask semantics. This is a
sampling consequence, not extra optimization budget. Release target coverage rises
352→360/360 and mean presentations per available tick 5.389→20.392, while push
coverage falls 5,237→4,089/5,252 and mean presentations 5.262→1.451. The fixed-budget
reallocation gives a concrete mechanism for the push tradeoff. Final training L1 is not
directly comparable across the differently weighted sample distributions.

All table cells are **mean / median / p95**, in mm or degrees. Phase attribution
uses the target action, including chunks crossing phase boundaries. The consumed
prefix contains every action once: 7,052 train and 3,103 development samples.
Full and discarded-suffix results, every phase-position intersection, native
column errors, zero and train median are retained in `trial-errors-pooled.json`,
`trial-errors-episodes.json`, each cell's `physical-errors.json`, and readable
`trial-tables.md`. No discarded position is called executed.

### Pooled train: consumed prefix

| Phase | V2 translation mm | Balanced translation mm | V2 rotation deg | Balanced rotation deg |
|---|---|---|---|---|
| all | 3.123 / 0.952 / 14.100 | 3.580 / 1.632 / 12.383 | 0.514 / 0.195 / 1.593 | 0.613 / 0.336 / 1.463 |
| approach | 15.351 / 8.240 / 60.551 | 15.233 / 7.850 / 60.190 | 2.431 / 0.813 / 12.955 | 2.197 / 0.498 / 12.918 |
| contact | 1.929 / 2.128 / 3.146 | 2.171 / 2.505 / 3.362 | 0.423 / 0.376 / 0.633 | 0.124 / 0.120 / 0.199 |
| push | 0.912 / 0.846 / 1.716 | 1.636 / 1.385 / 3.685 | 0.183 / 0.175 / 0.277 | 0.389 / 0.342 / 0.785 |
| hold | 1.751 / 1.463 / 3.562 | 1.776 / 1.841 / 2.314 | 0.294 / 0.292 / 0.533 | 0.260 / 0.263 / 0.369 |
| release | 13.489 / 11.073 / 29.972 | 11.861 / 10.473 / 26.482 | 1.822 / 1.689 / 3.481 | 1.569 / 1.554 / 3.027 |

### Development: consumed prefix

| Phase | V2 translation mm | Balanced translation mm | V2 rotation deg | Balanced rotation deg |
|---|---|---|---|---|
| all | 6.443 / 4.014 / 29.707 | 6.801 / 4.658 / 28.013 | 0.900 / 0.768 / 2.085 | 0.550 / 0.415 / 1.830 |
| approach | 20.759 / 19.570 / 41.532 | 20.772 / 19.641 / 41.430 | 1.635 / 1.234 / 4.039 | 1.318 / 0.925 / 3.794 |
| contact | 3.863 / 3.858 / 4.645 | 4.056 / 4.076 / 4.855 | 0.705 / 0.714 / 0.863 | 0.269 / 0.262 / 0.444 |
| push | 4.006 / 3.710 / 5.301 | 4.553 / 4.621 / 5.226 | 0.789 / 0.770 / 1.046 | 0.382 / 0.391 / 0.566 |
| hold | 1.726 / 1.764 / 1.940 | 4.666 / 4.641 / 5.088 | 0.200 / 0.196 / 0.260 | 0.555 / 0.562 / 0.614 |
| release | 14.938 / 12.091 / 33.341 | 11.253 / 8.102 / 29.221 | 1.682 / 1.613 / 2.854 | 1.347 / 1.303 / 2.545 |

### Zero and train-median controls

Mean / median / p95 on the consumed prefix; all other phases and positions also
remain in the complete tables and JSON. No statistic is fitted on development.

| Split | Phase | Predictor | Translation mm | Rotation deg |
|---|---|---|---|---|
| train | all | zero | 6.861 / 5.376 / 15.250 | 0.934 / 0.694 / 1.509 |
| train | all | train_median | 6.133 / 4.735 / 13.584 | 0.941 / 0.700 / 1.482 |
| train | approach | zero | 16.660 / 11.849 / 63.227 | 2.248 / 0.606 / 13.093 |
| train | approach | train_median | 15.862 / 9.868 / 61.857 | 2.320 / 0.657 / 13.199 |
| train | release | zero | 10.231 / 9.049 / 24.106 | 1.016 / 0.774 / 2.451 |
| train | release | train_median | 11.562 / 10.302 / 25.781 | 1.027 / 0.765 / 2.478 |
| development | all | zero | 7.396 / 4.980 / 27.208 | 0.638 / 0.501 / 1.670 |
| development | all | train_median | 6.385 / 4.061 / 27.178 | 0.660 / 0.512 / 1.710 |
| development | approach | zero | 21.132 / 21.455 / 43.603 | 1.366 / 0.991 / 3.828 |
| development | approach | train_median | 20.315 / 19.693 / 41.780 | 1.445 / 1.061 / 3.922 |
| development | release | zero | 9.091 / 5.195 / 26.450 | 0.912 / 0.850 / 2.054 |
| development | release | train_median | 9.958 / 5.927 / 27.279 | 0.944 / 0.869 / 2.075 |

### Pooled train: chunk position

Mean / median / p95; positions **0–5** are consumed, **6–11** discarded.

| Position | V2 translation mm | Balanced translation mm | V2 rotation deg | Balanced rotation deg |
|---:|---|---|---|---|
| 0 | 3.170 / 0.996 / 14.043 | 3.616 / 1.683 / 12.917 | 0.522 / 0.200 / 1.599 | 0.609 / 0.331 / 1.458 |
| 1 | 3.159 / 0.985 / 14.021 | 3.621 / 1.727 / 12.559 | 0.508 / 0.181 / 1.582 | 0.613 / 0.334 / 1.440 |
| 2 | 3.122 / 0.975 / 13.716 | 3.561 / 1.592 / 12.263 | 0.510 / 0.189 / 1.608 | 0.618 / 0.343 / 1.440 |
| 3 | 3.108 / 0.964 / 13.937 | 3.547 / 1.544 / 11.982 | 0.513 / 0.196 / 1.560 | 0.617 / 0.342 / 1.453 |
| 4 | 3.095 / 0.938 / 14.084 | 3.560 / 1.617 / 12.076 | 0.517 / 0.199 / 1.570 | 0.613 / 0.336 / 1.467 |
| 5 | 3.082 / 0.911 / 14.117 | 3.577 / 1.658 / 12.642 | 0.516 / 0.197 / 1.545 | 0.611 / 0.332 / 1.462 |
| 6 | 3.099 / 0.931 / 13.967 | 3.596 / 1.717 / 12.930 | 0.511 / 0.189 / 1.602 | 0.612 / 0.332 / 1.463 |
| 7 | 3.131 / 0.982 / 14.015 | 3.607 / 1.700 / 12.652 | 0.507 / 0.182 / 1.594 | 0.611 / 0.330 / 1.447 |
| 8 | 3.151 / 0.999 / 13.773 | 3.599 / 1.641 / 12.359 | 0.511 / 0.181 / 1.615 | 0.609 / 0.328 / 1.440 |
| 9 | 3.142 / 1.004 / 13.901 | 3.579 / 1.658 / 11.990 | 0.515 / 0.185 / 1.576 | 0.608 / 0.330 / 1.453 |
| 10 | 3.148 / 0.996 / 14.198 | 3.616 / 1.771 / 12.185 | 0.518 / 0.193 / 1.603 | 0.606 / 0.328 / 1.475 |
| 11 | 3.201 / 1.031 / 14.271 | 3.720 / 2.011 / 12.765 | 0.519 / 0.191 / 1.571 | 0.608 / 0.329 / 1.471 |

### Development: chunk position

Mean / median / p95; positions **0–5** are consumed, **6–11** discarded.

| Position | V2 translation mm | Balanced translation mm | V2 rotation deg | Balanced rotation deg |
|---:|---|---|---|---|
| 0 | 6.442 / 4.014 / 29.399 | 6.843 / 4.706 / 27.563 | 0.918 / 0.787 / 2.057 | 0.567 / 0.434 / 1.792 |
| 1 | 6.593 / 4.164 / 29.628 | 6.968 / 4.853 / 27.982 | 0.897 / 0.759 / 2.085 | 0.544 / 0.404 / 1.829 |
| 2 | 6.525 / 4.075 / 29.322 | 6.842 / 4.711 / 28.023 | 0.884 / 0.743 / 2.120 | 0.532 / 0.394 / 1.841 |
| 3 | 6.406 / 3.919 / 28.844 | 6.704 / 4.568 / 28.043 | 0.885 / 0.751 / 2.097 | 0.537 / 0.402 / 1.828 |
| 4 | 6.350 / 3.891 / 29.036 | 6.700 / 4.560 / 27.648 | 0.900 / 0.770 / 2.075 | 0.553 / 0.421 / 1.807 |
| 5 | 6.342 / 3.897 / 29.349 | 6.747 / 4.595 / 27.591 | 0.917 / 0.783 / 2.046 | 0.565 / 0.432 / 1.778 |
| 6 | 6.352 / 3.891 / 29.484 | 6.806 / 4.657 / 27.662 | 0.925 / 0.788 / 2.051 | 0.561 / 0.427 / 1.783 |
| 7 | 6.426 / 3.963 / 29.530 | 6.855 / 4.716 / 28.002 | 0.928 / 0.788 / 2.078 | 0.565 / 0.430 / 1.814 |
| 8 | 6.456 / 3.987 / 29.165 | 6.822 / 4.676 / 28.053 | 0.928 / 0.791 / 2.121 | 0.574 / 0.441 / 1.818 |
| 9 | 6.423 / 3.915 / 28.783 | 6.763 / 4.623 / 28.050 | 0.933 / 0.801 / 2.129 | 0.580 / 0.449 / 1.824 |
| 10 | 6.449 / 3.982 / 29.099 | 6.868 / 4.734 / 27.617 | 0.939 / 0.809 / 2.086 | 0.588 / 0.457 / 1.836 |
| 11 | 6.524 / 4.132 / 29.470 | 7.016 / 4.891 / 27.633 | 0.945 / 0.810 / 2.068 | 0.589 / 0.457 / 1.801 |

## Interpretation and decision

The intervention demonstrates **sensitivity to phase weighting**, supporting
limited release exposure as one contributing cause. Development release mean,
median and p95 translation improve 14.938→11.253, 12.091→8.102 and 33.341→29.221 mm;
rotation improves 1.682→1.347, 1.613→1.303 and 2.854→2.545 deg.
Pooled train release mean/p95 improve, but median improves by only 0.600 mm,
below the declared 1 mm margin. Improvement is not uniform across train identities:
the first train release mean/median worsen 9.147→9.638 and 8.427→9.532 mm,
while the second improves 17.831→14.084 and 17.690→13.828 mm.

The complete decision fails **8 of 40** declared metric checks. Besides the train
release median margin, train push p95 worsens 1.716→3.685 mm and 0.277→0.785 deg,
and its rotation median worsens 0.175→0.342 deg. Development hold median/p95
worsen 1.764→4.641 / 1.940→5.088 mm and 0.196→0.562 / 0.260→0.614 deg.
Development overall p95 improves to 28.013 mm / 1.830 deg, but mean translation
worsens 6.443→6.801 mm. Better release and tails do not establish overall dominance.

Approach translation remains essentially unchanged, including the large second
train tail. On development, release remains worse than zero and train median in
all six reported mean/median/p95 physical error metrics. The saved data also carry
large prior-controller tracking residuals, a timer-dependent reference and a
teacher-selected historical retreat pose. The present comparison cannot separate
their individual effects or prove that a longer memory would solve them. It also
does not prove that a larger optimizer budget would resolve the remaining errors.

**Retain `rgbd-grid-fixed-units-v2` as the opt-in experimental recipe with its
existing uniform sampler. Do not adopt the balanced sampler.** Preserve the
balanced initial/final/resume checkpoints, failed criteria and every original
artifact. No second variant, added budget, new model or input is launched.
Two numerical audit construction failures (unequal terminal windows and float32
subtraction in an algebraic check) are preserved separately; both were corrected
before the single CUDA trial, without changing scoring, decision margins or data.

## Concrete next steps and controller boundary

The previously proposed three-second causal-memory audit is unexecuted and
**deferred**. The
[[b1-approach-controller-compatibility|historical Cartesian reference/current-controller comparison]]
now verifies saved conventions and right unloaded tracking, but the left timed path
still fails despite static IK support. The single next direction is a feasible
continuous/timed approach reference under the frozen controller, starting on the
left. No history extension, timer, sampler, budget or controller variant is selected.

Offline command fit does not demonstrate task success, safe motion or controller
transfer. Before any policy/contact comparison with the newer controller:

- Preserve the verified observation–command semantics and the measured left-path
  incompatibility. Historical compensations remain old-regime targets. New-controller
  matched observations/actions require new recording after feasible references and
  the physical prerequisites; no old observations are relabeled from replay.
- Complete the outstanding 6.0D load/uncertainty/fault/moving-stop/contact envelope,
  calibration-domain requirements, observed local sweep/response/contact evidence
  and independent 6.0E admission. Existing robot-only ready-pose 0/4 N results
  remain local evidence; see [[experiments/b1-loaded-hold-and-stop|controller limits]].
- Keep separate train fitting, one-identity development prediction, final matched
  data, sealed-test generalization and physical policy qualification. The latter
  capabilities remain unvalidated. No new demonstration or contact rollout occurs
  in this follow-up.
