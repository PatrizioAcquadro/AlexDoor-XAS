# Temporal Evidence Before P2P Reference Promotion

## Question and method

Can independent observed body motion distinguish new moving-object references from static-world
points before native promotion?

One chronological saved-data criterion uses confirmed-before-birth body pairs and background RGB-D
camera fits, excluding pending points, native published pose and evaluator truth. The unchanged
three-good-observation rule and 15-acquisition TTL provide a 0.25 s deadline. A residual budget with
4 mm endpoint terms and measured pixel/fit errors gives about 6.26 mm tolerance; hypotheses must
separate by more than twice that value.

## Result and decision

**Reject activation under the existing waiting/observation budget.** All 637 light / 412 nominal
births remain indeterminate: zero body promotions, zero confidently static exclusions and zero
informative decisions. Maximum measured separation is 4.242/5.921 mm; even evaluator-only exact
motion reaches only 6.801/6.833 mm, below the approximately 11.4 mm tolerance floor. Repairing
camera fitting alone cannot pass this fixed-budget criterion.

Withholding renewals removes all useful moving support: median eligible points fall 40→1 light /
32→2 nominal; 68.37%/62.60% of the 2,858 opening captures have fewer than five pairs, and both
final-five-second pools are empty. This is a projection with saved native retirement, not a modified
tracking trajectory. Expiry occurs at the first capture beyond the deadline (266.667 ms here).

The diagnostic implementation/tests are retired; no runtime option or CUDA correction was activated.
Keep the selected pipeline and its original target results.

## Limits and evidence

Attempts 1/2 retain evaluator failures and native-count mismatches; the final evaluator repairs them
without changing parameters or retrying inference. Offline added-compute p95 is 37.2/27.8 ms per
capture, excluding acquisition/export. This conditional failure does not rule out every temporal
method and establishes no material, freshness or loaded-control qualification.

Evidence: `outputs/evidence/b1/perception/p2p-temporal-renewal-20261009/`; final `attempt-3/`,
`analysis.json`, `support-and-cost.json`, stationary/observability controls and original target
records. Baseline `9cb0a17`; rejection `6b84b04`; historical diagnostic source `24a2aa7`.
