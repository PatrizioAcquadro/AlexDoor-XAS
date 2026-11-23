# Official SAM3 Compilation

## Question and method

Does the official SAM3 compiler improve the selected pipeline while preserving downstream quality?

Candidate `7d864fe` uses official `compile=True` against eager baseline `85b5719`. Test compiler
compatibility, sparse frontend costs/masks and downstream poses with identical arrived inputs and
native initialization. Models, selected recipe and five-inlier/4 mm gates remain fixed; internal bit
equality was not an acceptance requirement.

## Result and decision

All seven official components compile with stable graphs. Sparse SAM3 time improves approximately
19.3% light / 16.7% nominal, but masks and downstream target accuracy/availability and native loss
regress even with shared native initialization.

**Reject adoption and restore eager SAM3 and the TSDF baseline.** The pilot quality gate closes the
planned complete fixed-input and timed-ingress comparisons. Compiler compatibility is established;
no new complete publication-age or frequency result is claimed.

## Limits and evidence

Foreign-compute prefixes (109 eager / 99 compiled frames) are retained; only incomplete cells were
restarted. Full models, original masks, quality regressions and initialization audits survive. No
hardware, freshness, A3/A4 or contact qualification follows from compiler success.

Evidence: `outputs/evidence/b1/perception/sam3-compile-20261015/`; `compiler-audit.json`,
`pilot-summary.json`, `pose-summary-commonseed.json`, `residual-cost-estimate.json`, `decision.json`
and saved candidate patch. Git: candidate `7d864fe`, decision `fa36656`.
