# SAM3 Selection, Operational Timing and Integrated Light Peaks

## Question and method

What quality and age does the selected provider deliver with continuous ingress, and what explains
dense light peaks?

Selection `402b832` and measurement `6c3f872` retain observed-only same-capture references and
immutable capture times. Offer all 4,718 recording arrivals at 60 Hz, then run a separate 65 s
parked-arm stationary live diagnostic with one renderer. Score processed outputs and
skipped/all-scheduled denominators separately; a later saved-mask bridge changes one controlled
historical boundary.

## Result and decision

Complete timing/resource accounting distinguishes recording from concurrent live:

| Measurement | Continuous recording | Concurrent stationary live |
|---|---:|---:|
| Inputs / publications / replaced | 4718 / 234 / 4484 | 1331 / 141 / 1190 |
| Offered acquisition rate Hz | 60.00 | 20.53 |
| Seed acquisition / publication s | 25.0167 / 26.9596 | 26.2800 / 28.9614 |
| Steady publications, defined seed capture +5 s | 185 | 109 |
| Effective steady publication rate Hz | 3.79 | 3.24 |
| Complete service p50/p95/max ms | 222/595/703 | 291/324/354 |
| Observation age at publication p50/p95/max ms | 246/610/730 | 315/346/375 |
| SAM3 p50/p95 ms | 108/199 | 136/144 |
| P2P including IPC/export p50/p95 ms | 113/393 | 154/186 |
| Publication p50/p95 ms | 2.36/3.56 | 1.15/2.08 |
| Latest-slot wait p95/max ms | 0.021/51.9 | 0.042/61.8 |
| Poll wait p50/p95 ms | 9.14/15.22 | 14.94/33.77 |
| Geometry maximum during inspection s | 2.910 | 5.608 |
| Simultaneous sampled GPU / RSS peak GiB | 7.727 / 5.631 | 12.128 / 11.937 |
| Steady publications within 150 ms | 0/185 | 0/109 |

No result qualifies 150 ms. Stationary live has 117 accepted processed outputs within 10 mm/5
degrees relative to its own seed; that is not absolute initialization or opening quality. The first
live attempt had no eligible reference; the repaired initial-view run saved operating data but
failed post-run reset reporting. It remains a failed diagnostic command, and was not repeated to
improve its label.

Dense saved-light peaks involve reference birth/promotion bias and narrow or fixed/moving support.
The controlled saved-mask bridge retains 12 refusals and a new 74.55 mm accepted peak: masks and P2P
initialization/history both matter, with no unique isolated cure. The proposed
[[p2p-temporal-renewal-admission|temporal renewal prerequisite]] subsequently fails before CUDA
activation.

## Limits and evidence

Sampled post-ready GPU/RSS peaks do not certify startup spikes. Relative stationary support, native
return and finite rejected outputs do not establish original-material recovery, physical occlusion,
hardware/contact or A3/A4 qualification. Dense acquisition-time bridge quality is separate from
approximately four-Hz sparse operational tracking.

Evidence: `outputs/evidence/b1/perception/sam3-operational-20261008/` and the preserved integrated
dense light attempt; selection requests, every input/publication/replacement, quality/failure/reset
records, peak audits and controlled bridge outputs remain. Git: `402b832`, `6c3f872`, `fe41e1f`,
`9cb0a17`.
