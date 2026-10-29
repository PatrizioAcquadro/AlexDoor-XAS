# TSDF Read-Only Launch Inputs

## Question and method

Can avoiding readback of six read-only CUDA arguments reduce TSDF rebuild tails without altering
tracking or failure behavior?

Candidate `ae3bcec` against `9b94978` changes only argument-transfer handling. Both complete
3,217-frame fixed-input CUDA histories compare volumes, queries, poses and decisions; continuous
timed ingress is scored separately. Original models, references, native ordering and thresholds stay
fixed.

## Result and decision

Both fixed histories match exactly. Transfer volume decreases, but transfer savings are small,
rebuild/service benefit is inconsistent and timed-ingress quality is mixed. An unrelated SAM3 stall
and changed processed histories cannot be credited to TSDF transfer optimization.

**Reject adoption as a rebuild-tail optimization; restore the baseline.** Residual cost is dominated
by full-volume downloads, snapshot construction and allocation/filter work. Any future removal must
preserve partially completed failure state and same-frame queries.

## Limits and evidence

Every completed-run refusal, outlier, lost tail and regression remains. An exporter-only failure was
repaired without discarding its log. These comparisons do not qualify 150 ms publication, material
identity, hardware or contact; exact transfer equivalence is distinct from timed operational
benefit.

Evidence: `outputs/evidence/b1/perception/tsdf-readonly-20261014/`, with executable probes,
fixed/timed summaries, residual profiles and original failed runs. Git: implementation `ae3bcec`,
rejection `1f9cb20`. The separate [[tsdf-deferred-snapshot-audit|snapshot audit]] tests the next
hypothesis.
