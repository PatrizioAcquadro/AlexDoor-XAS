# Approach-to-contact corrections and guards

Status: qualification in progress from `d798998` on `PatrizioAcquadro/b1-policy-pilot`.
Only expert approach/contact corrections and their operational guards are in scope.
No training, policy inference, checkpoint comparison, new door, development panel
or final test is authorized. The prior learned result remains 0/22.
See [[act-a1-failure-diagnosis]] for the completed diagnosis, which is not repeated.

## Plan and acceptance

Reuse the 114 train sources, retaining their original outcomes and frozen
112-source normalization. The focused numeric coverage audit finds 64 sources
with admitted approach/contact rows and 2,703 corrective commands in 13 sources;
zero injected-error commands are admitted. Nearby postures alone do not establish
joint posture/velocity coverage of ACT's later states.

Eight preregistered short A collections cover both current training directions:
shifted initial shoulder posture, an early pause, a late pause, and combined
normal/tangent/orientation error with a measured-origin return. Precontact ranges
25–35 mm, contact intervals 3–4 seconds, with the already supported slow-tail
entry. Pauses and injected offsets remain causal history, excluded from positive
supervision; corrective commands begin from the actual measured state. These
are expert subtask demonstrations, not learned completion or a replacement for
the broad corpus. Existing settling and physical endpoint criteria remain.

Require sustained native contact, ordinary withdrawal/stabilization, no physical
violation, native force within the unchanged 10 N stop domain, independent 2+4 s
stop and maintained temporal recording qualification. Every attempted case and
original outcome remains; no favorable retry or retrospective relabeling.

Three additional slots reserve independent guard/load qualification. A left
position extension needs motion and maintained 10 N stopping evidence, not an
unloaded stop. The physical 80 N limit and operational 10 N scope are distinct.
No threshold is increased to make a recorded attempt pass.

The complete reservation is 11 × 1,265 GPU seconds (3.866 h) and 48 GiB, including
startup, execution and shutdown. Baseline costs are 42.733604 local/31.823056
remote GPUh, 996.104 GiB retained and 198,773 cumulative updates. New updates and
remote allocations remain zero. The 48/32 GPUh and 1126.4 GiB caps remain.

Evidence: `outputs/evidence/act-a1-first-completion-20261022/approach-contact-20261025/`.
`scope.json`, `coverage.json` and `native-plan.json` hold exact cases and reserves.
Refinement readiness is pending physical and temporal qualification.
