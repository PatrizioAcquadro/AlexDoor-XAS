# P2P to A3/A4: Initialization, Hinge and Runtime Cost

## Question and method

Which A3/A4 prerequisites are actually available from observed closed initialization, and what
blocks later use?

Saved analysis `4c2aa27` uses all 2,858 original rows per light/nominal partial-only opening,
31–78.6167 s, retaining five/six candidates. Independent seed images and evaluator annotations
distinguish the task’s closed/unlatched protocol from image evidence. A separate bounded CUDA
profile measures existing computation without changing tracking.

## Result and decision

The observed closed normal gives static yaw errors 0.05093/0.04776 degrees. It suffices for the
algebraic rotation of A3 free vectors, but neither the seed point nor that rotation supplies a
physical hinge, contact point or complete operational reference.

| Primary hinge diagnostic | Light | Nominal |
|---|---:|---:|
| First numerical fit, acquisition time | 51.3667 s | 39.5500 s |
| Delay from 31 s seed | 20.3667 s | 8.5500 s |
| Estimated opening magnitude at first fit | 23.40 degrees | 6.33 degrees |
| First origin error measured by evaluator | 34.08 mm | 49.18 mm |
| First calculated position bound | 392.30 mm | 1001.72 mm |
| Fits / scheduled rows | 1636 / 2858 | 2345 / 2858 |
| Origin error p95 over available fits | 31.23 mm | 26.23 mm |
| Calculated position bound p95 / minimum | 600.17 / 171.06 mm | 921.86 / 212.25 mm |
| Relative angle error p95 over available fits | 0.438 degrees | 0.422 degrees |
| Fits with calculated origin bound at most 1 cm | 0 | 0 |

The first numerical hinge is delayed and its origin uncertainty is not qualified. Better point error
cannot replace a complete uncertainty bound. Registration/SDF is the measured computational hotspot;
a vectorized Jacobian is the separately evaluated successor in
[[p2p-observed-geometry-and-sdf-jacobian]].

Keep the shared [[../topics/pre-a4-initialization-and-hinge-budget|pre-A4 protocol]] independent of
action representation. Bounded articulation before a physical hinge exists would require a
separately admitted Cartesian path, contact/load feedback, continuous sweeps and stops; none is
implemented here.

## Limits and evidence

Partial-only accepted precision remains 85.75/86.94% and point p95 11.89/12.30 mm. Ideal depth,
vertical-axis assumptions and original 63.7209-degree motion do not establish other handedness,
inclined axes, hardware calibration, original-material recovery, loaded control or 150 ms freshness.

Evidence: `outputs/evidence/b1/perception/p2p-a3a4-diagnosis-01/`, saved seed/context images,
geometry/hinge audits and CUDA profiles. Git: diagnosis from `4c2aa27`, geometry follow-up baseline
`e3dc046`.
