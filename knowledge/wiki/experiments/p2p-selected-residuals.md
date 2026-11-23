# Residuals of the Selected Renewal-Filtered Trajectory

## Question and method

What causes the remaining selected-trajectory position/orientation errors, and does the evidence
justify one new observable correction?

Saved-data diagnosis `9aefffa` retains all 2,858 rows per condition and stable reference IDs.
Integer-truncated pixel lifting and the native rigid inverse reconstruct birth/current points;
reference birth bias, promotion changes and current correspondence are compared as vectors.
Evaluator rigid motion/hull membership remains an explanatory proxy, not runtime material truth.

## Result and decision

Light final-five-second moving-born inliers show:

| Moving-born tail inliers | Median / p95 / maximum norm (mm) |
|---|---:|
| Bias already in reference at birth | 10.98 / 19.34 / 21.43 |
| Reference change during promotion | 3.46 / 11.68 / 13.03 |
| Total current reference bias | 13.63 / 24.03 / 25.79 |
| Current correspondence displacement | 3.59 / 8.95 / 19.34 |

Frozen local fits reduce persistent bias when evaluator-corrected birth coordinates replace stored
references, but still leave a 20.22 mm peak; removing promotion changes alone leaves 24.16 mm. These
counterfactuals do not establish a deployable correction. Nominal’s 39.7 s position cluster
predominantly follows fixed frame support; excluding it by oracle membership leaves only one pair.

Full orientation deterioration is real on matched accepted rows. SDF usually improves orientation,
while the 38.8833 s nominal angular maximum validly falls back from a more accurate but three-inlier
SDF pose to the supported cluster. Neither globally disabling SDF nor weakening five-inlier support
is justified.

**Retain bounded SAM3 OFF plus the renewal depth gate.** No additional inference control or
correction replay follows; reference, correspondence, rotation and target-lever effects remain
coupled.

## Limits and evidence

Light rejected-reference identities/counts remain missing; admitted points cannot prove how much
useful spread was removed. Approximate hull labels and birth-transport markers are not independent
material annotations. Scalar residual medians do not add as a causal budget. The evaluator’s
original convention mismatch remains recorded without inference retry.

Evidence: `outputs/evidence/b1/perception/p2p-sam3-residual-02/`, `summary.json`,
`source-audit.json`, same-ID image pairs and complete stage timelines. Preserve
`p2p-sam3-followup-01/hull_evaluator.py`, prepared collision geometry and
`p2p-performance-01/full-target-evaluation.json`.
