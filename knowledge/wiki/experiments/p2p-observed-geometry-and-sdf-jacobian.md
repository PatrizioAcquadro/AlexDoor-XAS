# Observed Geometry and SDF Jacobian: Separate Interventions

## Question and method

Can explicit observed reference/angle support improve geometric diagnosis, and can vectorized
Jacobian assembly reduce cost independently?

From `e3dc046`, geometry recomputation uses both saved complete openings (31,438 candidate rows),
preserving tracker output. A separate CUDA comparison alternates scalar/vector refinements on 482
original frames and 2,640 paired refinements with identical SDF inputs, decisions and RNG.

## Result and decision

Freeze the observed initial reference; separate every accepted image’s current angle from
displacement-selected hinge samples. Propagate initial/current uncertainty and vertical-model
residuals without inventing floor or calibration bounds.

| Primary candidate | Old light | New light | Old nominal | New nominal |
|---|---:|---:|---:|---:|
| First numerical hinge, source seconds | 51.3667 | 40.4833 | 39.5500 | 39.9667 |
| Delay from seed, seconds | 20.3667 | 9.4833 | 8.5500 | 8.9667 |
| First origin error, mm | 34.08 | 8.36 | 49.18 | 26.77 |
| Available hinge rows / 2,858 | 1,636 | 2,289 | 2,345 | 2,320 |
| Origin error p95 on each available set, mm | 31.23 | 3.23 | 26.23 | 18.97 |
| Calculated/conditional bound p95, mm | 600.17 | 1,819.15 | 921.86 | 1,736.54 |
| Bound within 1 cm, rows | 0 | 0 | 0 | 0 |

Angle p95 on common hinge rows improves about 0.438→0.0435 degrees light / 0.421→0.0428 nominal. No
origin bound reaches 1 cm; zero angles are operationally fresh at publication within 150 ms.

The maintained Jacobian replaces per-point `[I,-skew(x)]` assembly with direct gradients/cross
products. Median complete refinement cost falls 15.370→4.592 ms light / 15.573→4.735 nominal.
Jacobians, query/cost histories, poses, support, decisions and RNG match exactly in the tested CUDA
probes. **Retain the vectorized assembly and explicit geometry support semantics.**

## Limits and evidence

Local assembly gains are distinct from complete service; instrumented pipeline subtraction is an
estimate, not measured operational latency. A failed nvcc-path startup is retained. Better
diagnostic geometry and cheaper refinement do not qualify the initial physical hinge, hardware
uncertainty, A3/A4 execution or contact.

Evidence: `outputs/evidence/b1/perception/p2p-geometry-jacobian-01/`, scalar source, retained
arrays, numerical geometry reports and paired CUDA checks. Original observations and tracker
controls remain unchanged.
