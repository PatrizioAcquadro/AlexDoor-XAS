# Outputs

Generated results remain local and ignored by Git. Selected pretrained weights
and required resources live in `models/perception/`; reusable RGB-D recordings,
calibration and dataset products live in `datasets/`.

`b1/perception/evidence/` retains the completed perception experiments' protocols,
per-door results, metrics, diagnoses, aligned component masks and essential images.
Run names are preserved inside this directory. `cleanup.json` maps moved paths,
records preserved episodes/model digests and lists explicitly removed payloads.
Original report contents are unchanged; embedded old paths describe their original
execution and can be resolved using that mapping. The corrected aligned scores
supersede the original offset component scores.

Discarded model weights, feature caches, completed experiment scripts/builds and
five interrupted recordings were removed. Ignored payloads are not recoverable
from Git; retained evidence does not promise complete executable reproduction.
No qualified perception replacement or policy result exists. See the
[perception findings](../knowledge/wiki/experiments/b1-perception-findings.md).

`b1/perception/geometric-*` contains the current frozen-model prototype's CUDA
smokes, separate pilot attempts and causal full-state replay results. Protocols
record the common recipe/source identity and authorized episode selection.
`summary.json` requires complete campaign membership before offline eligibility;
per-door/condition reports retain missing states in coverage and distinguish finite
rejected errors from accepted accuracy. Masks and observed geometry diagnostics
are not qualified full-state results. Dynamic tests require all offline doors to pass.

Qualification evidence and runtime verification normally remain under
`~/.cache/alexdoor-xas/`; prepared door resources remain under `assets/doors/b1/`.
The cleanup's bounded checks use `b1/perception/evidence/cleanup-validation/`.
