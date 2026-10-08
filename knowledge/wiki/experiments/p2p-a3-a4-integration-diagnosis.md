# P2P to A3/A4: Closed Initialization, Hinge Bootstrap and Runtime Cost

Bounded diagnosis on 2026-10-06 from clean `main @4c2aa27`. Only `main` exists,
so it is retained. The reference is the independently tested **bounded P2P,
global graph off, partial reference batch on, own-seed rollback off, five inliers
at 4 mm**. This is a diagnostic reference, not a new operational default.
A3/A4, tracker behavior, models, masks, thresholds and candidate selection are
unchanged. No acquisition, training, contact, SAM3 comparison or optimization is
performed. Earlier complete and failed evidence remains preserved.

Saved analysis uses both original `animated-door-1-88abf40` light/nominal episodes
and all 2,858 original 60 Hz rows, 31–78.6167 s, from
`outputs/evidence/b1/perception/point2pose-isolated-improvements-01/partial-batch/`.
Automatic candidate-0 remains primary; five/six candidates are retained.
All 31,438 candidate rows are accounted for. Truth and exact teacher targets
are used only by evaluators. The recorded opening ends at 63.7209 degrees.

New ignored evidence, reproducible scripts and logs are in
`outputs/evidence/b1/perception/p2p-a3a4-diagnosis-01/`. They are local payloads, not
recoverable from Git. See [[point2pose-isolated-improvements|the preserved partial-only comparison]]
and [[point2pose-consumer-impact-and-failure-windows|consumer metric boundaries]].

## 1. Closed seed and the static A3 orientation

Targeted review uses two original 31 s seed images, their same-pixel detail views,
and nominal 25/40 s context images. The red measured seed lies on the leaf face
in both recordings, away from the handle/frame. This verifies the primary local
seed in these recordings; it does not certify every SAM2 pixel or query.
The regenerated mask expands onto relief, handle and edge regions; only 5/30
light and 6/30 nominal initial queries are inside the original planar component.
Component membership and whole-leaf ownership are different assertions.

Evaluator annotations independently establish the closed initial condition:
signed angle is `1.46e-17 rad` at the seed and remains numerically zero through
33.2167 s. The first sample exceeding 0.01 degree is at 33.2833 s; this is an
evaluation descriptor, not a changed tracker gate. Closed/unlatched is supplied
by the task protocol; these RGB-D images alone do not measure latch state.

| Primary initialization quantity | Light | Nominal |
|---|---:|---:|
| Seed row / frame / time | 1860 / 1868 / 31 s | 1860 / 1868 / 31 s |
| Measured seed pixel, x/y | 244 / 315 | 70 / 318 |
| Distance to evaluator front plane | 0.144 mm | 0.044 mm |
| Closed static yaw from observed normal | +0.05093 degrees | +0.04776 degrees |
| Static rotation error against evaluator closed frame | 0.05093 degrees | 0.04776 degrees |
| Original full local contact-frame rotation error | 0.07209 degrees | 0.06929 degrees |

The evaluator leaf width is 839.12 mm. The seeds are respectively 97.50/9.51 mm
from the free edge. Nominal's proximity to the edge matters for a possible
finite finger footprint, not for a free-vector coordinate conversion. A tracking
seed is neither a selected contact point nor a hinge origin. Hardware
calibration, full material ownership and contact support remain unqualified.

For this vertical-hinge protocol, a precise static convention is available from
the **observed closed normal**, without locating the hinge:

1. Set `z = calibrated world up`. Normalize the horizontal projection of the
   measured closed normal as `x`, preserving its original sign away from the
   initial camera toward the observed surface. Set `y = z cross x`.
2. Freeze `R_WD = [x, y, z]` at the closed observation. The measured rotations
   above are evaluator checks; identity/zero yaw must never be inserted from truth.
   Never replace this static rotation with P2P's moving zone orientation.
3. Use right-handed signed angle about +Z. Positive angle rotates +X toward +Y;
   this recorded right-hinged opening has negative angle. Do not make both hands
   positive or infer push direction from image-left/image-right. A reversed viewing
   side requires an explicit reference convention, not silent per-frame sign flips.

`frame_delta_to_world` rotates translation and axis-angle free vectors by
`R_WD`; its origin does not affect the primitive A3 conversion. For a 10 mm vector,
these observed orientation discrepancies bound the rotation-only discrepancy by
about 0.0089/0.0083 mm. This is an algebraic sensitivity, not an executed command.
The vertical model is a protocol assumption; these runs do not test inclined axes
or the other handedness.

The **controller contract remains stricter** than that conversion. Operational
`validate_reference`/`admit_action` requires supported hinge hypotheses, identity,
angle, panel/contact geometry and appropriate freshness. The current provider
returns `DoorEstimate(valid=False, reason="point2pose_local_diagnostic", local=...)`;
it does not publish a complete operational reference. Thus the seed passes this
bounded initialization diagnosis but A3 control/contact is not admitted.

## 2. A hinge before A4 needs it

A4 needs a supported closed hinge frame and signed angle at admission of its
**first approach segment**, even when its requested hinge delta is zero, because
its target is panel-local. Waiting until A4's push to discover the hinge creates
a bootstrap dependency. The present numerical axis is diagnostic only and always
reports `closed_reference_validated=False`; no actual A4 predictions/admissions
exist in these saved runs.

The original static scan contains alternative **panel-edge** hypotheses with
`support=None` and `panel_edge_is_not_physical_axis`. Its physical arc fits accept
none. Static RGB-D can in principle locate an exposed pin/knuckle axis with
measured support; a leaf plane, straight seam or free edge alone cannot. The
available saved static observations establish closed orientation and local surface
geometry, but do not supply a supported physical hinge. Camera motion alone does
not resolve a hidden kinematic pivot from a rigid static leaf.

The current `fit_motion_axis` is reconstructed causally, unchanged: accepted poses,
sample insertion only after translation exceeds its registration position bound,
32 retained motion samples, at least three rotations above both 5 degrees and
twice their rotation bound, vertical-axis tilt within 5 degrees, rank/conditioning
and residual-envelope checks, known floor intersection. Root consistency and
ownership-dependent operational publication are not reconstructed; the times
below are **first numerical fits**, not first usable A4 geometry.

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

Nominal becomes numerically eligible as soon as three informative samples exist.
Light already has three at 39.6167 s, but one inferred axis tilts 7.6508 degrees.
The unchanged fit rejects the entire window when any informative axis exceeds
5 degrees. Such evidence remains in the displacement-selected 32-sample window:
705 accepted rows are refused for tilt before the first fit at 51.3667 s.
The maximum retained tilt there is 3.7829 degrees. This explains the additional
light delay; it is not missing depth or insufficient rotational excitation alone.
No rejection threshold is widened and no offending sample is removed.

Acquisition time is separate from computation: summing the **saved** request wall
times through first fit gives 813.77/357.81 s light/nominal, excluding worker boot.
These sums are sequential offline processing time, not live time-to-usable-hinge.
Operational usable-hinge time remains absent because no admitted reference exists.

### Calculated uncertainty, measured error and command effect

The observed fit solves `(I-R) h = t` with floor Z fixed. Its bound is
`sensitivity * envelope`, where sensitivity is the absolute pseudoinverse row-sum
norm and envelope is the largest `p + 2 L sin(a/2)` over selected motions.
`L` includes the full floor-origin-to-seed lever (about 1.43/1.51 m), and `a` is
the measured-correspondence rotation envelope, not a truth error. At first fit,
light/nominal sensitivity is 5.65/14.73 and envelope is 69.39/68.02 mm.
Small rotation scale amplifies position uncertainty even though condition numbers
are close to one. Residual measures self-consistency, not physical axis accuracy.

This is a calculated diagnostic envelope, **not a calibrated confidence interval**,
measured error or a complete hardware uncertainty budget. Initial-reference and
floor/calibration uncertainties are not explicitly propagated by this fit.
The zero axis-direction error would be imposed by the vertical model and this
recording's vertical truth; it is not an estimated-axis accuracy result.
Large bounds cannot simply be replaced by smaller evaluator errors.

At fixed panel-local coordinates, evaluating an exact saved teacher target under
the fitted origin/relative angle and observed frozen closed rotation gives position
error p95 32.06/27.34 mm and maximum 36.67/51.42 mm. These are conditional effects
of hypothetical geometry substitution, not actual A4 actions or truth-fed commands.

For an arc **anchored at the same physical current target**, the origin-only effect
instead is `||(I-Rz(phi)) e_h|| = 2 |sin(phi/2)| ||e_h_xy||`.
At a 5-degree increment its measured p95 is 2.72/2.29 mm, whereas propagation of
the calculated origin bound gives p95 52.36/80.42 mm. Angle, normal, contact,
robot and stopping uncertainty are additional. Anchoring can cancel a static
origin translation at admission; it cannot establish contact safety or justify
altering A4's coordinates/semantics.

The fitted relative angle uses the **last displacement-selected sample**, which
can precede the current accepted image. Its age exceeds 150 ms on 566/1636 light
and 800/2345 nominal fit rows, with maxima 0.800/0.783 s. `PanelTracking.consume`
currently attaches the latest candidate support time to this diagnostic record.
An eventual provider must preserve the angle's actual supporting sample time;
static axis support and dynamic angle freshness have different requirements.
This diagnostic timestamp issue is not an observed controller failure: no such
reference is consumed by A3/A4 today.

### Proposed initial motion, not executed

First retain the observed closed orientation/zero reference and a local leaf
patch. Try static axis observation only when a physical pin/knuckle is visible
with metric support. Otherwise a separate **pre-A4 Cartesian diagnostic** is
needed, followed by axis estimation and only then normal A4 admission.

A bounded hypothesis is slow monotonic initial leaf motion sufficient to obtain
at least three independent supported rotations above `max(5 degrees, 2 a)`;
6–8 degrees is a candidate nominal observation range, **not a validated command
or guaranteed light solution**. Stop at the specified excursion/time/uncertainty
bound if fit evidence is insufficient. Do not extend toward 23 degrees merely
because the saved light fit first succeeds there. A reversible fixture/external
leaf motion can isolate perception first; robot-induced motion would require
separately qualified local contact, feedback/load, sweep and stopping support.

Current operational validators require a supported hinge even for A2, so such a
probe needs an explicit pre-articulation admission path. Do not fabricate a hinge,
relax validators, use the near-edge visual seed as a contact target or encode a
guessed A4 arc. No probe or that new path is implemented in this diagnosis.

## 3. CUDA runtime cost

Fresh RTX 4090 probes process every original row from 31–35 s, 241 rows per
condition, retaining all five/six objects and the exact saved source masks.
A separate 11-frame light prefix measures renewal at rows 1866/1868 and TSDF
integration at 1869. Total new inference is 493 rows; no favorable retry.
Ground truth and commands are never read by the profiling driver.
SAM2, BootsTAPIR and SuperPoint are verified on `cuda:0`.

All candidate poses match the preserved partial-only output **exactly** across
each measured prefix, and initial masks match exactly. Instrumentation observes
existing methods; it does not change decisions, RANSAC draws or state. Worker
startup is 4.04/4.08 s, separate from frame latency. Uninstrumented native wall
time p50/p95 is 479/545 ms light and 563/641 ms nominal (217 frames each).
These are direct native calls with diagnostic traces, not complete live IPC,
capture/queue/adapter/control latency.

The table uses medians of 20 synchronized non-cProfile frame samples per
condition. Costs are exclusive, so nested SDF/register timings are not counted
twice; medians do not sum to a median total. Registration uses CPU NumPy work
alongside CUDA SDF/model work. cProfile frames are preserved separately and
excluded from these latency estimates.

| Existing block, synchronized wall median | Light, ms | Nominal, ms |
|---|---:|---:|
| SAM2 | 75.09 | 79.06 |
| BootsTAPIR tracking | 93.34 | 90.40 |
| RANSAC/SVD over all objects | 100.67 | 126.15 |
| SDF hypothesis scoring | 40.86 | 39.12 |
| SDF pose refinement | 100.42 | 148.67 |
| Other registration, including dense lifting and observational copies | 47.13 | 45.38 |
| Candidate/keyframe maintenance plus bounded renewal | 2.63 | 2.80 |
| Diagnostic capture/export | 26.97 | 28.28 |

Registration/SDF together has mean 285.60/363.16 ms, about 58/62 percent of
instrumented native time. Primary registration accounts for only 19.7/14.8
percent of that total; secondary candidates consume the remainder. This is
measured cost attribution, not permission to prune alternatives or a measured
single-candidate speedup. SAM2/TAPIR are shared passes with work dependent on
object/query count; the two recordings do not isolate a candidate-count effect.

The renewal microprobe measures SuperPoint 4.60–5.50 ms and new TAPIR query
features 7.84–7.96 ms per event, plus 2.11–2.12 ms sampling and 2.53–3.68 ms
candidate management. Its one incremental TSDF integration costs 8.12 ms.
Large late volume rebuilds are outside these early probes; retained full runs
do not establish their component cost. Prefix queries peak at 228/220, below
the full-run 599/720 maxima, so mature tracker/model/map cost is not qualified.

The principal reducible area is repeated **registration/SDF**, rather than
diagnostic serialization or ordinary candidate maintenance. cProfile identifies
the per-point Python SDF Jacobian construction (allocating an identity/skew
matrix per retained point) and repeated RANSAC/SVD calls as concrete hotspots.
The smallest future performance comparison should vectorize only that Jacobian
assembly on identical observed arrays, preserving point counts, eight iterations,
draw order, models and gates, then verify pose/support equivalence. No such
optimization is implemented; no speedup is claimed.

Result pickle round-trip is sub-millisecond in these direct probes for roughly
2.9/3.5 MB outputs; it is not a measured pipe/queue transfer. The preserved full
partial-only requests have p95 783.88/974.58 ms; diagnostic-export p95 is
45.45/58.27 ms. Subtracting **each frame's** reported export duration gives native
p95 732.27/908.47 ms and minimum 393.56/424.87 ms. This arithmetic is not an
export-disabled replay, but it rules out export as the entire deadline gap here.
None of the 2,857 non-seed frames per condition meets 150 ms even under that
subtraction. Shared models alone exceed 150 ms in 19/20 timed samples per condition.

The unchanged 150 ms requirement remains a capture-to-available freshness gate.
Further validation must include mature 120-reference states, map-growth/rebuild
events, selected-candidate quality/ambiguity, real IPC/copy/queue/adapter timing,
initialization and camera motion, followed by live publication without copied
support times. Lower mean throughput or removal of logs cannot certify it.

## Verified findings, open problems and the minimum next intervention

1. **Initialize the diagnostic reference before extending control.** Primary seeds
   are on the closed leaf in these recordings and a signed static A3 rotation can
   be measured/frozen. Next make that observed initialization and sample-time
   ownership explicit in a small pre-A4 state specification; leave A3/A4 and their
   admission checks unchanged. Whole masks, other handedness/view sides, hardware
   calibration and finite contact footprints remain open.
2. **Resolve the hinge bootstrap before A4 admission.** Static edge hypotheses are
   unsupported; current motion fits are delayed, one tilt sample can veto a long
   window, and no calculated position bound reaches 1 cm. First audit uncertainty
   propagation (initial frame, floor/vertical assumptions and lever origins) on
   these saved samples. Then specify one bounded 6–8-degree perception probe with
   an explicit pre-articulation path; do not execute it until its needed contact,
   feedback, sweep and stop evidence exists. It is not yet a qualified remedy for
   light or for the origin bound.
3. **Reduce one measured computational hotspot after that contract is reviewable.**
   Registration/SDF is the main cost; a frozen-array Jacobian vectorization is a
   small justified future experiment. Neural time, late map state and complete
   live freshness still need evidence. Do not start a broad optimization campaign.

The partial-only tracker still has accepted precision 85.75/86.94 percent and
accepted point p95 11.89/12.30 mm on the preserved complete openings. Passing
closed-seed diagnosis does not repair this later quality limitation or establish
original-material recovery. No tracking, control, contact or release qualification
is claimed by this report.
