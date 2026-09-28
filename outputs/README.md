# Outputs

Local B1 perception preparation uses `b1/perception/` for the pretrained backbone,
readiness reports and later estimator runs. Generated payloads stay outside Git.
No estimator training has been started by preparation. Commands and the required
pretraining handoff are in the
[perception contract](../knowledge/wiki/topics/shared-door-perception.md).
Final policy learning/evaluation remains
[Phase 7 work](../knowledge/wiki/implementation_phases/phase-7-training-and-generalization-evaluation.md).

B0 D0–D4 scenes, runs and W&B residues were removed. Historical conclusions remain
in the wiki. Verification reports, temporary scenes and intake work belong under
`~/.cache/alexdoor-xas/`; reusable episodes belong in `datasets/` and promoted
doors in `assets/doors/b1/`.
