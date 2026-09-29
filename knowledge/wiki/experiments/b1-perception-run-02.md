# B1 Perception — Corrected Full-Corpus Run

## Outcome and stopping

User-launched `outputs/b1/perception/run-02` used `articulated-state-v3`, code
`1bd633e` (checkout `34c2660`), fresh initialization and the existing 38 train/12
development episodes. CUDA training completed normally after 15 epochs,
11,730 optimizer steps and 816.29 seconds (13.60 minutes). The one-hour limit was
a maximum; the actual stop reason is `development_stagnation`, not an exception,
out-of-memory failure or time-budget expiry.

The best development checkpoint is epoch 1. Its selection score is 27.492,
versus 40.877 at epoch 15; lower is better. There are 14 consecutive evaluations
without improvement over the first reference. Patience is 10 evaluations,
minimum relative improvement is 0.5%, and minimum training duration is 15 epochs.
The minimum-epoch rule delays stopping until epoch 15 even though patience was
already exhausted. Replaying the stopping logic reproduces that decision exactly.

## Read-only GPU evaluation

Both checkpoints were reloaded on the RTX 4090 and evaluated on all 25,006 train
and 6,498 development causal windows. Development results reproduce the saved
checkpoint metrics exactly. Model tensors and checkpoint file hashes remain
unchanged. The diagnosis did not train, collect episodes or evaluate sealed test
doors. Scripts and JSON evidence are in the run directory: `diagnose.py`,
`diagnosis.json`, `probe.py`, `probes.json` and `target-support.json`.

The ranges below are **per-door p95 over manipulation frames**, not pooled p95:

| Checkpoint / split | Contact position p95 | Contact orientation p95 | Doors passing all gates |
|---|---:|---:|---:|
| Best, epoch 1 / train | 1.43–3.27 cm | 2.32–6.25 degrees | 0/19 |
| Last, epoch 15 / train | 0.40–1.45 cm | 0.88–1.77 degrees | 5/19 |
| Best, epoch 1 / development | 7.88–15.43 cm | 13.30–18.32 degrees | 0/6 |
| Last, epoch 15 / development | 8.34–32.09 cm | 6.72–18.72 degrees | 0/6 |

The mean per-door development contact-position p95 worsens from 11.60 to
18.75 cm. Epoch-average geometric training loss falls from 26.671 to 0.526;
total loss falls from 26.917 to 0.964. Full train fitting improves substantially
but remains incomplete: only five doors pass, and 77.29% of pooled manipulation
frames satisfy all state tolerances. Development has zero such frames at both
saved checkpoints. The same original 1-cm/5-degree contact gates fail independently
of the added complete-state checks.

## Generalization and confidence failures

This is a measured train/development gap with worsening development performance,
consistent with overfitting. Passing the
[[experiments/b1-perception-fit-check-02|two-door fitting check]] established
trainability on those examples, not generalization to the held-out families.
The present results do not establish that longer training or a larger patience
would fix the gap.

Existing targets expose missing geometric coverage: train heights are
1.809–2.038 m, while modern and void-frame development doors are 2.157 and
2.283 m. Industrial hinge X positions are 0.090–0.099 m, above the train maximum
0.052 m; industrial local-contact X is approximately 0–0.010 m, versus entirely
negative train values. The four train right doors also have a narrow width range
of 0.816–0.839 m, compared with modern's 0.994 m.

Fixed probes use 32 evenly spaced manipulation windows per development episode,
in both nominal/light conditions. At the last checkpoint, modern's median
predicted height is about 1.955 m instead of 2.157 m; void-frame's is about
1.822 m instead of 2.283 m. Modern's predicted hinge Y has the wrong sign on
25% of these probes, and industrial-004's on 12.5%. These are sample diagnostics,
not full-trajectory sign-error rates. They show geometry and side errors beyond
a small contact offset; they do not isolate whether representation, visual
observability, shortcut learning or data variety is the dominant cause.

Confidence is also unreliable. Best-checkpoint confidence coverage is zero;
last-checkpoint confidence coverage is 100% on every train and development door.
Yet last-checkpoint development accuracy under all state tolerances is zero,
so accepted-state precision is zero. Confidence in the fixed development probes
is 0.831–0.904, despite wrong geometry. Detaching confidence gradients protects
geometry learning; it does not guarantee confidence calibration on unfamiliar
doors. The offline gates correctly reject this model.

## Next decision

Keep both checkpoints as diagnostic evidence; neither is qualified for dynamic
use. Do not merely extend this run or lower the acceptance threshold. Before
another full trial, distinguish residual full-train fitting error from the much
larger generalization failure, test dependence on RGB/depth versus proprioceptive
trajectory cues, and evaluate how to calibrate rejection of inaccurate states.
Any representation or targeted-data change needs a bounded comparison; the
current diagnosis alone does not select a proven correction. Preserve the frozen
split and sealed test partition. Subphase 6.0 and dynamic validation remain open.
