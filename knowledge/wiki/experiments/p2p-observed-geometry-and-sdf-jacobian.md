# Observed Geometry and SDF Jacobian: Separate Interventions

Implemented from clean `main @e3dc046` on 2026-10-06. Only `main` exists and is
retained. The comparison reference is bounded P2P with **graph off, partial batch
on, rollback off, five inliers at 4 mm**. Models, candidates, original 60 Hz
schedule, observations and tracking decisions are fixed. A3/A4 and operational
admission are unchanged. No acquisition, diagnostic movement, contact, training
or SAM3 comparison is performed. All earlier complete and failed evidence remains.

Local ignored scripts, complete candidate rows, reports and failed evaluator
attempts are preserved in `outputs/b1/perception/p2p-geometry-jacobian-01/`.
The two interventions are evaluated independently; numerical geometry uses the
original saved P2P output and does not rerun or modify tracking.

## Observed geometry

`StaticReference` freezes the observed initial pose, horizontal normal/+Z rotation
and its actual `FieldSupport`. Closed initialization is a task-protocol assertion,
not inferred from image identity, truth or zero angle. The normal runtime leaves
`closed_by_protocol=False`; the saved-data evaluation states the already established
closed protocol explicitly. Neither case publishes an operational reference.
`closed_reference_validated` remains false.

`MotionObservation` retains acquisition, support and availability times and the
initial-world centroid of the actual measured registration inliers. Hinge fitting
uses the original displacement selection and 32-sample history. Current angle
uses every current accepted pose independently, so it no longer inherits the time
of the last displacement-selected sample. Loss makes angle unavailable; an old
static reference cannot refresh it.

The vertical model uses the closest yaw rotation and measures its residual
rotation against the combined current/initial angular envelope. At least three
informative samples above `max(5 degrees, 2 * combined angular error)` and two-thirds
vertical consensus are required. Incompatible samples remain in history with their
support times; no known offending sample is singled out or deleted. Every retained
compatible motion must also explain a common pivot within its positional envelope.
Persistent inclined motion and contradictory translations remain explicit refusals.
Vertical direction is a protocol assumption, not an estimated axis direction claim.

The fit is uncertainty-weighted. Its envelope uses the actual correspondence
centroid rather than assuming the zone seed is the registration rotation center,
includes the initial surface error and the measured discrepancy from the vertical
model, and propagates angular error at the unknown hinge. If `B0` is the absolute
pseudoinverse propagation at the fitted lever and `beta` propagates angular error
at the uncertain origin, the finite conditional bound is `B0 / (1 - beta)` only
when `beta < 1`. Shared reference errors are not averaged down. Otherwise the bound
is absent with `unbounded_rotation_feedback`, even if a numerical pivot exists.
Floor uncertainty affects Z under the imposed vertical model; missing floor and
camera/FK calibration leave the **complete** position bound absent. Conditional
bounds are diagnostic envelopes, not calibrated confidence intervals.

Both original light/nominal episodes retain all 2,858 rows, 31–78.6167 s, and all
five/six candidates: 31,438 candidate rows. The evaluator checks timestamps and
recomputes the original measured registration bounds to 1e-12 before constructing
new geometry. Only subsequent scoring reads annotations. Root/ownership admission
is not reconstructed. The recorded motion ends at 63.7209 degrees.

| Primary candidate | Old light | New light | Old nominal | New nominal |
|---|---:|---:|---:|---:|
| First numerical hinge, source seconds | 51.3667 | 40.4833 | 39.5500 | 39.9667 |
| Delay from seed, seconds | 20.3667 | 9.4833 | 8.5500 | 8.9667 |
| First origin error, mm | 34.08 | 8.36 | 49.18 | 26.77 |
| Available hinge rows / 2,858 | 1,636 | 2,289 | 2,345 | 2,320 |
| Origin error p95 on each available set, mm | 31.23 | 3.23 | 26.23 | 18.97 |
| Calculated/conditional bound p95, mm | 600.17 | 1,819.15 | 921.86 | 1,736.54 |
| Bound within 1 cm, rows | 0 | 0 | 0 | 0 |

The available sets differ. On the exact common hinge rows, light origin p95 improves
31.23 to 2.60 mm (1,636 rows); nominal improves 19.92 to 18.97 mm (2,320 rows).
Light gains 653 available rows and loses none; nominal loses the first 25 rows
because initial-reference uncertainty raises the informative-motion requirement.
Its first appearance is 0.4167 s later. Improved point errors do not justify
substituting truth errors for the larger audited bounds. No finite conditional
primary bound is violated, but this does not calibrate it or qualify hardware.

The frozen observed static rotation remains within 0.05093/0.04776 degrees.
Current angle is supported on 2,857/2,858 rows per condition, including before a
hinge exists. Its all-supported p95 error is 0.04288/0.04252 degrees; the old
last-motion-sample angle p95 over available hinges was 0.438/0.422 degrees.
These denominators differ. Source age is now zero on accepted images, while the
saved request-time publication age p95 remains 783.88/974.58 ms. **Zero angles are
operationally fresh at publication under 150 ms.** This separation fixes timestamp
semantics without claiming a runtime improvement.

Insufficient evidence remains visible: light secondary candidate 2 has 14 numerical
fits with unbounded rotation propagation; nominal candidate 2 has 228 such fits
and 62 vertical-consensus refusals. Nominal candidates 4/5 have no accepted support
and no hinge. Complete calibration/floor bounds and operational references remain
absent for every candidate. The normal provider is still local and invalid.

Keep the explicit reference, independent current-angle support, uncertainty-aware
vertical comparison and audited bounds. These improve the diagnostic estimate
without granting control. Remaining initial A4 bootstrap limitation: the stationary
seed has no supported physical hinge, and the later fitted origin has no bound
within 1 cm or complete calibration budget. A Cartesian pre-articulation/contact
admission path with observed identity, finite contact geometry, load/feedback,
sweep and stopping bounds must be specified before any initial robot motion.

## SDF Jacobian

Separate CUDA comparison and results are pending. The authorized change is only
Jacobian assembly, preserving all point selection, eight refinement iterations,
random draw order and decisions. It must not be called a solution to 150 ms.
