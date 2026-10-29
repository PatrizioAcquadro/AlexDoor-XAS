# Point2Pose RANSAC Point Buffers

## Question and method

Can precomputing immutable point pools reduce registration cost without changing RANSAC decisions or
complete-provider quality?

Candidate `dd2f0ae` against `fa36656` preserves all 100 four-point draws, weighted SVD, degeneracy
checks, first-wins ties, five inliers/4 mm and refit rejection. Paired CPU checks cover 296 saved
comparisons and 32 boundary/error cases; full CUDA checks cover both 3,217-input histories. Four
continuous timed runs then retain all 4,718 arrivals per run.

## Result and decision

All paired fields, decisions, traces and final NumPy RNG states match exactly. Local median RANSAC
reductions are 14.62% light / 15.18% nominal, saving 4.202/2.896 ms. Fixed-native service saves
about 3–4 ms; all three target counts and losses are unchanged.

Timed provider service/age does not improve consistently. Light target-0 correct 20 mm counts fall
142→42 of 2,858 opening captures, with accepted precision 97.26%→39.62%; native losses are 68→48,
showing that fewer losses do not establish useful support. Nominal correct counts improve slightly,
while publication-age p95 remains about 294 ms. No steady output meets 150 ms among 819 outputs.
Changed initialization and sparse histories prevent attributing every quality difference to
arithmetic.

**Reject default adoption.** Maintained/native registration is restored byte-for-byte; SAM3 eager
and TSDF baseline remain selected. Local equivalence and savings do not establish complete-provider
benefit.

## Limits and evidence

Three interfered prefixes (392/314 light, 2,798 nominal frames) remain separate. Valid fixed/timed
runs consume 27.15/6.09 minutes; interrupted active spans add 7.54 minutes. No favorable repeat or
threshold change was used. Sampling does not establish indefinite memory stability, hardware,
material identity or contact qualification.

Evidence: `outputs/evidence/b1/perception/ransac-buffers-20261015/`, especially `correctness.json`,
`fixed-summary.json`, `operational-summary.json`, `compact-summary.json`, `decision.json` and saved
candidate sources. Git: `dd2f0ae`, `b5d0f69`, final `24a2aa7`.
