# Temporal Evidence Before P2P Reference Promotion

## Decision and boundary

**Discard this correction under the existing waiting and observation budgets.**
The saved-data prerequisite fails for both light and nominal: none of the new
references can distinguish observed body motion from a world-static hypothesis.
Withholding them also removes every useful moving reference renewal. The criterion
and chronological evaluator are implemented as diagnostic software; no runtime
option is connected or activated, and no new CUDA trajectory attempt is justified.
The selected persistent pipeline remains unchanged.

Baseline checkout is `main` at `9cb0a17`, with no pre-existing tracked changes.
The scientific baseline is `persistent-sam3-20261008/full-selected-{light,nominal}`:
SAM3 runs from inspection, P2P starts at 25.016667 s, and each saved native trajectory
contains 3,217 captures through 78.616667 s. The historical or hybrid 31 s replay is
not a new baseline. This follows the single conditional correction proposed in
[[p2p-operational-selection-and-light-peaks|the latest controlled-replay report]].

## One criterion and its rationale

The candidate has two predicted current camera-frame positions. The body prediction
comes from an independent rigid fit between measured same-ID points already
confirmed before its birth; the entire new batch and all native pending points
are excluded. The static prediction comes from short-interval RGB-D background
correspondences outside the unchanged masks. Classical forward/backward image
matching estimates camera motion; it neither adds P2P queries nor replaces TAPIR,
SuperPoint or SAM3. The decision functions receive no camera extrinsics, panel
truth, ownership classes, stored candidate coordinates or published object pose.
This exclusion matters because native tentative fallback can use unpromoted points.

Both fits need observable consensus and broad spatial support. The body fit retains
five pairs/4 mm and the existing >8 mm second-principal-RMS reliability scale;
at least 60% consensus avoids treating a minority fit as reliable. The camera fit
requires 20 pairs and >50 mm second RMS, with at most 256 image features and a
one-pixel forward/backward discrepancy. These camera measurement parameters are
fixed once, without a peak-conditioned sweep. The observed background can contain
unmodelled motion: failure to establish its consensus returns insufficient evidence.

The temporal residual budget is
`tau = sqrt(2 * (4 mm)^2 + pixel_size^2 + body_fit_RMS^2 + camera_fit_RMS^2)`.
The 4 mm term uses the maintained observation acceptance scale; the two measured
endpoints, integer depth lift and fit residuals must all be accounted for. This is
a conservative engineering tolerance, not a calibrated Gaussian confidence bound.
Measured supported fit RMS medians are about 2.0 mm for body and 1.6 mm for camera;
the resulting temporal tolerance medians are 6.26 mm in both conditions. Precision
does not justify a submillimetre discriminant for these observations.

The hypotheses must separate by more than `2*tau`. A candidate passes only if its
body residual is within `tau` and its static residual is larger by more than `tau`.
The reverse yields static evidence. Unreliable reference motion, small motion,
missing measured depth or neither hypothesis yields **insufficient**. Stationary
points are not automatically accepted as body points. No extra observation-count
requirement, mask refinement, SDF check or cluster/fallback change is introduced.

The evaluator mirrors the existing keyed native pending/promotion lifecycle rather
than adding an operational queue. Native three-good-observation eligibility is
unchanged. Its 15-acquisition pending TTL supplies a 0.25 s wall deadline from
timestamps observed during inspection. The first acquisition beyond that deadline
expires the point, at 0.266667 s here, regardless of gate failures; another frame
cannot make an expired point fresh. Pending state and anchor captures are released,
with no indefinite wait or fallback acceptance. The selected `f2m` frontend supplies
no relative pose, so the native pose-stability gate passes; reconstructing a tighter
consecutive-pose gate would be an unintended second intervention.

Because the prerequisite fails, this lifecycle is evaluated only against saved
pending candidates. A live promotion hook or query retirement change is not
implemented. There is consequently no new runtime queue, budget consumption or
permanent renewal blockage in the maintained pipeline.

## Chronological prerequisite results

All captures, new-point births, eligibility and decisions are processed in order.
No future point observation enters a decision. All 636/412 saved native promotion
checks reproduce their native good counts exactly. Existing confirmed-before-birth
support and saved TAPIR/mask histories are held fixed for this preliminary screen;
screened support below is a projection, not a replayed trajectory. Truth and hull
membership annotate completed decisions only.

| Quantity | Light | Nominal |
|---|---:|---:|
| New points admitted by the saved sampler | 637 | 412 |
| Saved native promotions | 636 | 412 |
| Body-supported experimental promotions | **0** | **0** |
| Confidently static exclusions | **0** | **0** |
| Indeterminate moving-only births | 205 | 266 |
| Indeterminate fixed-only births | 431 | 141 |
| Indeterminate unknown births | 1 | 5 |
| Moving-only points used as measured baseline inliers and withheld | **205** | **264** |
| Maximum observed hypothesis separation, where both fits are supported | 4.242 mm | 5.921 mm |
| Informative observed decisions within deadline | **0** | **0** |
| Exact-pose available-motion maximum, evaluation only | 6.801 mm | 6.833 mm |
| Exact-pose informative decisions with the same tolerance floor | **0** | **0** |

All 431/141 frame births would therefore be withheld, but so would all 205/266
moving births. Withholding is not evidence-backed frame rejection: every result
is indeterminate. There is no promotion-delay distribution because nothing passes.
Saved native promotion delay is 50 ms median/p95 in both conditions, with nominal
maximum 200 ms. Experimental expiry is 266.667 ms for every candidate.

The camera fit lacks consensus on 7,853/8,268 light and 4,771/5,335 nominal
nonexpired candidate decisions. Both motions are supported on only 415/564, all
indistinguishable. Thus this background observer is also a substantial limitation.
Separately, evaluator-only exact poses show that available motion within the same
deadline is always below even the noise-free-fit separation requirement: at least
about 11.4 mm versus at most 6.84 mm. This audit cannot promote points or rescue the
criterion; it explains why merely repairing camera estimation would not make this
fixed-budget criterion pass. These conclusions are conditional on the stated
tolerance model, not proof that every possible temporal method is impossible.

The saved stationary control at 25.016667–25.266667 s has negligible true leaf
motion and 30/30 insufficient leave-one-out decisions in each condition. These
seed points are probes of the criterion, not changed seed admissions. Essential
tests supply a genuinely informative 20 mm body/static separation with a moving
camera and verify both outcomes. The saved openings contain **no** sufficiently
informative renewal interval under this deadline. Five-second bins cover the full
trajectory, with separate 72.066667/78.15 s diagnostic windows and both final tails;
the criterion was not fitted to those peaks.

## Support and measured diagnostic overhead

Support is the measured visible confirmed pool at <0.3 tracking uncertainty,
before cluster selection/SDF; it is not the published final inlier set. The frozen
projection retains original seed IDs and any criterion-promoted renewals, of which
there are none. Saved native retirement is retained, so this is not a causal
prediction of a modified tracker or its losses.

| Full 31–78.616667 s opening, 2,858 captures | Light | Nominal |
|---|---:|---:|
| Baseline / screened median eligible point count | 40 / **1** | 32 / **2** |
| Baseline / screened median second principal RMS | 280.8 / **0 mm** | 242.1 / **0 mm** |
| Screened captures with fewer than five pairs | **1,954 (68.37%)** | **1,789 (62.60%)** |
| Final 5 s / final 1 s screened pool | **0 throughout 301 / 61 captures** | **0 throughout 301 / 61 captures** |

There is no demonstrated geometric benefit to offset this depletion. Available
spatial support fails in both conditions, not only at known light peaks.

The diagnostic peaks at 25/27 pending points and 9/6 retained RGB-D captures,
62.21/41.47 MB of anchor RGB-D arrays. These omit interpreter/matching workspace,
shared body dictionaries and baseline P2P state. Added offline compute summed per
processed capture is p95/max 37.2/87.2 ms light and 27.8/50.8 ms nominal. On captures
with a check it is 46.9/87.2 and 35.1/50.8 ms p95/max. Fits are shared within a birth
batch. This is standalone diagnostic compute, excluding acquisition/read/export,
not measured added CUDA-session latency or memory. No latency optimization ran.

The preserved baseline target records still contain all three physical targets at
10/15/20 mm and 5 degrees, availability, precision, wrong acceptances, losses,
refusals, distributions, peaks and tails. No new trajectory exists to score against
them: these values are not relabelled as results of the rejected filter. Conditional
step 3 is intentionally not entered. Roughly four-Hz operational tracking, freshness,
material identity, hardware, A3/A4 and contact remain unqualified.

## Evidence and validation

Evidence is preserved at `outputs/evidence/b1/perception/p2p-temporal-renewal-20261009/`:
`protocol.json`, final `attempt-3/` decisions/observation traces/support/summary,
`analysis.json`, observability/stationary controls, baseline target records,
`support-and-cost.json` and visually inspected `support-and-observability.png`.
The plot shows that widening the wait merely to obtain more frames is not the
tested correction. Reproduction uses the maintained diagnostic module with a fresh
output directory; saved `analyze.py`/`plot.py` reproduce evaluator annotations.

`attempt-1` retains the partial evaluator failure on nonfinite depth in a spatial
summary. `attempt-2` retains complete preliminary counts with 75/4 native-good-count
mismatches from an incorrectly reconstructed stability gate. `attempt-3` repairs
both evaluator issues, with zero mismatches and unchanged temporal parameters.
These are read-only evaluator repairs, not CUDA retries or threshold variants.
The original baselines, masks, recordings and all prior evidence remain intact.

Essential tests cover moving-camera compensation, body/static outcomes, stationary
and small-motion insufficiency, missing motion, elapsed-time expiry, robust
independent fitting, degenerate support, nonfinite depth summaries and measured
background RGB-D camera estimation. Selected renewal/trace/sampling regressions and
lint pass. No model comparison, training, contact or full CUDA attempt runs.

The next step remains performance and quality at the actual processing frequency;
do not automatically launch another quality-filter variant.
