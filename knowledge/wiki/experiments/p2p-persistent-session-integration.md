# Persistent SAM3 Inspection and Tracking

## Question and method

Does one persistent SAM3 session support inspection-to-tracking integration without weakening
initialization or transferring historical quality claims?

Implementation `472f6af`, explicit math `95b831a` and publication timing `a12fd4b` share the
selected recipe across scan/live/replay/offline. Separate SAM2/GroundingDINO workers are retired;
SAM3 and P2P remain isolated. Historical 33-row CUDA prefixes first match masks, poses, decisions
and traces exactly. Independent attempts then process all 4,718 captures from reset, initializing
P2P at 25.016667 s with 3,217 native outputs.

## Result and decision

Inspection history and seed differ from the historical 31 s replay. Full scored opening results
retain all 2,858 captures and the same three physical targets; slashes denote 10/15/20 mm with 5
degrees:

| Condition | Target | New availability % | New precision % | Saved availability % | Saved precision % | New position p95/max mm | Saved p95/max mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0 | 87.58/97.55/98.81 | 87.82/97.82/99.09 | 85.83/99.79/99.93 | 85.86/99.82/99.96 | 11.814/45.213 | 12.163/23.822 |
| light | 1 | 86.53/97.55/98.74 | 86.77/97.82/99.02 | 87.12/99.90/99.93 | 87.15/99.93/99.96 | 12.147/44.999 | 11.985/24.444 |
| light | 2 | 87.54/97.38/98.78 | 87.79/97.65/99.05 | 83.80/99.58/99.93 | 83.83/99.61/99.96 | 11.652/46.617 | 12.385/23.727 |
| nominal | 0 | 98.25/99.79/100.00 | 98.25/99.79/100.00 | 99.13/99.72/99.86 | 99.19/99.79/99.93 | 8.524/19.557 | 7.435/31.475 |
| nominal | 1 | 98.64/99.83/100.00 | 98.64/99.83/100.00 | 99.13/99.69/99.86 | 99.19/99.75/99.93 | 8.246/18.468 | 7.083/30.199 |
| nominal | 2 | 97.62/99.76/99.97 | 97.62/99.76/99.97 | 98.74/99.65/99.83 | 98.81/99.72/99.89 | 8.834/20.933 | 7.985/32.814 |

There are zero native losses; light retains eight nonseed geometric-spread refusals and nominal
none. Light target-0 position maximum grows to 45.213 mm while nominal falls to 19.557 mm. Different
initialization/history prevents calling this numerical equivalence or a uniform quality improvement.

Serial request p95 is about 260/246 ms with zero timely requests. **Maintain the shared lifecycle,
while leaving qualification open.** The first ambiguous light attempt and its unprocessed suffix
remain; its legacy interrupt label is clarified as an agent diagnostic stop, not a human request.

## Limits and evidence

The new seed precedes the scored opening; historical seed-denominator differences are not
improvement. Independent ownership/material recovery, absolute initialization, hardware/FK/floor
uncertainty, 150 ms publication and action/load/stop qualification remain open. Lifecycle validation
does not restart the separate policy pilot.

Evidence: `outputs/evidence/b1/perception/persistent-sam3-20261008/`, historical/final prefix
checks, original and selected complete attempts, `comparison.json`, `history-audit.json`,
masks/traces and resource records. Operational follow-up:
[[p2p-operational-selection-and-light-peaks]].
