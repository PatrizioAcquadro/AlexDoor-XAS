# Observed Geometry and SDF Jacobian: Separate Interventions

Implemented from clean `main @e3dc046` on 2026-10-06. Only `main` exists and is
retained. The comparison reference is bounded P2P with **graph off, partial batch
on, rollback off, five inliers at 4 mm**. Models, candidates, original 60 Hz
schedule, observations and tracking decisions are fixed. A3/A4 and operational
admission are unchanged. No acquisition, diagnostic movement, contact, training
or SAM3 comparison is performed. All earlier complete and failed evidence remains.

Local ignored scripts, complete candidate rows, reports and failed evaluator
attempts are preserved in `outputs/evidence/b1/perception/p2p-geometry-jacobian-01/`.
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
Its source/support time is 31 s; initial-mask availability retains the saved
semantic inference delay, 372.32/397.38 ms. Static support has no age-only expiry,
but this does not supply a physical hinge at initialization.
Current angle is supported on 2,857/2,858 rows per condition, including before a
hinge exists. Its all-supported p95 error is 0.04288/0.04252 degrees; the old
last-motion-sample angle p95 over available hinges was 0.438/0.422 degrees.
These denominators differ. On the exact common hinge rows, angle p95 improves
0.43787 to 0.04353 degrees light and 0.42083 to 0.04278 degrees nominal.
The common-row conditional-bound p95 is 494.66 versus 600.17 mm light and
1,736.54 versus 915.50 mm nominal. The large new all-fit light p95 includes the
newly available weakly excited early motions; it is not a common-row regression.
Source age is now zero on accepted images, while the
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

`point2pose_patches.patch_tracking` replaces only the per-point Python construction
of `[I, -skew(x)]` with `J[:, :3] = g` and `J[:, 3:] = cross(x, g)`. This is
NumPy assembly inside the CUDA P2P workload, not a new CUDA SDF kernel. Sampling,
point order, trimming, weights, normal-equation reductions/solve, damping, clips,
line search, up to eight iterations, early exits and final gates are unchanged.
The pinned installer preserves the original source and checks the patch idempotently.

Fresh RTX 4090 comparisons use every original row from 31–35 s: 241 rows per
condition, all five/six candidates, 482 frames total. There are 1,200/1,440 paired
scalar/vector refinements on identical current SDF volumes and arrays, alternating
order with CUDA synchronization. The stateless scalar result is checked before
publishing the vector result; it cannot change tracker/map state or random draws.
Actual retained Jacobian sizes are 987–1,359 light and 977–1,290 nominal, and both
comparisons exercise all eight iterations. Separate numerical regressions include
30- and 1,500-point cases.

| Measured median cost | Scalar light | Vector light | Scalar nominal | Vector nominal |
|---|---:|---:|---:|---:|
| Jacobian assembly on retained inputs, ms | 4.248 | 0.0268 | 4.551 | 0.0271 |
| Complete `_refine_pose_with_sdf` call on CUDA SDF, ms | 15.370 | 4.592 | 15.573 | 4.735 |

Jacobian microtiming includes allocation; it uses 60 retained observed arrays and
10 alternating repeats per condition. Its approximately 159/168-fold gain is
local to assembly. The actual CUDA SDF refinement gains are **3.35/3.29-fold**
by the medians above. Matched per-frame refinement saving is about
63.44/96.91 ms at the median.
No timing gain is assumed for SAM2, TAPIR, RANSAC, SDF scoring or other stages.

All measured Jacobians, refinement outputs and query/cost trajectories are
**exactly identical** in these CUDA probes. Iterations, support/inlier histories,
accepted/full/half-step decisions and debug histories match; NumPy RNG is unchanged
in both runs, and nominal additionally checks CPU/CUDA Torch RNG. All 2,651 candidate
rows preserve saved published poses, `lost`, inliers, cluster selection, support,
jump and fallback decisions exactly. Initial masks also match. Scalar original
source, retained inputs, frame accounting and decision checks are preserved.

The allowed comparison tolerances are 1e-12 for float64 Jacobians, 1e-12 absolute /
1e-10 relative for cost reductions, and 1e-7 absolute for native float32 poses,
well below micrometer and operational angle gates at door scale. Decisions receive no
tolerance; they must match. Actual differences are zero, so no threshold margin
is borrowed to obtain equivalence. This is targeted equivalence evidence; complete
late map volumes, all 2,858-frame inference windows and physical occlusions are
not rerun or newly qualified.

Paired pipeline estimates obtained by subtracting the duplicate refinement have
p95 647 to 562 ms light and 792 to 665 ms nominal. These are **estimates from
instrumented direct calls**, not uninstrumented end-to-end latency or a measured
IPC/capture/queue/control result. They remain far above 150 ms. There is no observed
precision or decision regression in the tested Jacobian cases; total operational
freshness remains unqualified. Keep the vectorized assembly and retain the scalar
snapshot as a reproducible reference.

Validation: focused numerical/adapter/offline/native/renewal checks pass (64 tests,
one existing GPU-storage test skipped by sandbox visibility); documentation/static
checks pass (19 tests). Dedicated real CUDA Jacobian probes above complete outside
the sandbox without CPU fallback. Ruff and whitespace/link checks pass. The first
CUDA launch lacked nvcc in PATH during first-volume initialization; that failed
startup is retained, and only the launcher environment was corrected.

Geometry and Jacobian effects are separately attributable. The better relative
geometry and cheaper SDF refinement do not remove A4's initial physical-hinge
bootstrap, complete uncertainty, contact/admission or freshness obstacles.
