# B1 Perception — Initial Training Diagnosis

## Scope and outcome

Run `outputs/b1/perception/run-01` used the frozen 38 train/12 development
engineering episodes and DINOv2 feature caches, with code at `522f1ba`.
It stopped for development stagnation after 28 epochs, 21,896 optimizer steps
and 1,483.94 seconds (24.7 minutes), within the one-hour budget. The best
checkpoint is epoch 18; none of the six development doors passes the fixed
engineering gates. Subphase 6.0 remains open; no dynamic validation was attempted.
The recipe and gates are in [[topics/shared-door-perception|Shared Door Perception]].

The subsequent diagnosis used CUDA inference on both saved checkpoints and
read-only gradient inspection. It did not update weights or start another
training run. Local evidence is in `outputs/b1/perception/run-01/analysis/`:
`diagnosis.json`, `annotation-audit.json` and `curves.png`; the original run's
`summary.json` and `metrics.jsonl` retain the per-door selection evidence.

## Measured behavior

The following are **pooled manipulation-frame diagnostics**, not replacements
for the frozen per-door gates:

| Checkpoint / split | Position p95 | Orientation p95 | Confidence-valid coverage |
|---|---:|---:|---:|
| Best / train | 4.58 cm | 4.45 degrees | 0% |
| Best / development | 9.01 cm | 13.93 degrees | 0% |
| Last / train | 5.91 cm | 5.37 degrees | 0% |
| Last / development | 14.93 cm | 11.29 degrees | 0% |

At the best checkpoint, per-door development position p95 ranges from 5.10 to
9.96 cm, and orientation p95 from 9.44 to 18.42 degrees. Development's mean
per-door position p95 improved from 15.46 cm at epoch 1 to 7.78 cm at epoch 18,
then ended at 14.24 cm. This is learning followed by unstable/non-improving
behavior, not evidence that simply extending the same recipe will meet the gates.

Only 16.59% of pooled train manipulation samples satisfy both geometric
thresholds at the best checkpoint; development has zero such samples. Development
confidence has median 0.144 and maximum 0.168. Lowering the confidence threshold
would admit inaccurate estimates rather than solve the geometric error.

## Evidence for the next correction

The initial objective adds Smooth L1 errors in meters, rotation/angle losses and
confidence BCE with no task-scale normalization. On 64 evenly spaced train
windows from the best checkpoint, operational-position loss was 0.000108 and
confidence BCE was 0.318793. The confidence gradient norm at the shared fusion
weights was 0.034399, versus 0.001519 for operational position, about 22.6 times
larger. These measurements establish an imbalance on the diagnostic batch;
they do not establish that reweighting alone will solve development generalization.

Confidence targets depend on whether the current geometric prediction meets the
1-cm/5-degree thresholds. Their positive frequency changes while geometry learns,
so total training loss is not a fixed geometric-accuracy measure. Logging each
loss component and train geometric errors is needed for the next recipe.
Balancing competing task gradients is an established multitask concern; see
[GradNorm](https://arxiv.org/abs/1711.02257), without committing this project to
that particular algorithm or a new dependency.

There is also a geometric generalization challenge. Train door heights span
1.809–2.038 m; modern and void-frame development doors are 2.157 and 2.283 m.
Industrial development hinge X coordinates are 0.090–0.099 m, beyond the train
maximum of 0.052 m. The current regressor must infer these differences from
observations; repeating the same training trajectories under more lights does
not add those geometric cases. This does not authorize changing the frozen split,
physical setup or injecting asset identity/truth into inference.

The annotation audit reconstructs all 31,654 cached operational points within
3.7e-8 m and rotation elements within 9.3e-8. Three source/cache samples per
episode match timestamps and all geometric targets exactly. No sign or cache
alignment mismatch was found by these checks; they do not prove every target is
visually observable or rule out architectural limits.

## Proposed next experiment — not implemented or launched

1. Normalize the principal geometric objectives to meaningful physical scales,
   log individual losses and train geometric metrics, and prevent confidence
   training from dominating shared geometric features. Keep the acceptance gates
   unchanged; consider separate confidence calibration after geometry improves.
2. Run a small train-only fitting diagnostic on one door per handedness. Failure
   to fit this subset would trigger a representation/alignment investigation
   before collecting more data or extending compute. This is a trainability
   diagnostic, not evidence of generalization or Subphase completion.
3. If the diagnostic passes, run a separate bounded full-corpus experiment using
   the existing recordings/features. Preserve run-01 and compare the same per-door
   development metrics. Give a revised recipe a new run directory.
4. If train precision becomes adequate but development remains poor, investigate
   geometric representation and targeted observational variety. Consider a longer
   run only when the measured development trend supports more optimization.

No additional collection, architecture replacement, gaze controller, gate change,
sealed-test evaluation or overnight training is justified by this first run alone.
