# Observed Contact Certificate Study

Baseline `a9bf503`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The user reopens at most three of the nine residual prerequisite cases for one
contact-certificate study using the existing frontend and unchanged 0.1 mm
physical rule. No placement, IK, controller, training or threshold variant is run.
The study is complete; **no operational contact certificate is qualified**.

The [[b1-end-to-end-pilot-resume|resumed A1/A2 pilot]] corrects the earlier
interpretation: the physical 0.1 mm score is not an operational sensor requirement.
The measurements below remain preserved component evidence. They neither require
a 0.1 mm depth detector nor make full A3/A4 initialization a pilot prerequisite.
The new operational monitor requires separate calibration and validation.

## Physical property and observed candidate

The existing monitor adds pose/robot uncertainty to the absolute distance between
the nominal tool center and the observed material plane. This differs from the
physical fingertip separation property. On the three previously saved teacher
traces, even exact local tangent geometry rejects valid contact at 353/362 right,
218/241 left and 74/239 development ticks after a 2 micrometer numerical guard.
Maximum nominal-center distances are 0.558 /0.953 /0.776 mm while the actual
solver contact remains within 0.1 mm. This is an evaluator-only counterexample,
not an operational source of geometry or contact truth. A smaller generic pose
bound alone cannot repair this semantic mismatch.

Probe one current-image component: unchanged selected `Sam3Worker` masks and
`geometry.surfaces` planes, queried at proprioceptive tool and known robot finger
locations. Require unique local support, at least 120 measured depth pixels
within a 50 mm tangent radius and normal alignment within the existing 5 degree
orientation tolerance. Keep the 4 mm plane-fitting recipe unchanged. Measure
signed fingertip-to-plane separation, including both distal geometries. Fit
residuals alone never certify calibration, hidden material, load or uncertainty.
No native Point2Pose quality/performance experiment or parallel frontend is added;
this reuses maintained frontend/geometry components, not a qualified provider.

The initial 72-frame historical dry run uses the old placement. Its support
counter incorrectly counts the reduced fitting cloud; its availability counts
are superseded, with original code/results retained. Before current-image
inference, count the actual dense support mask instead, preserving all thresholds
and the plane model. Do not attribute those earlier counting refusals to frontend
quality. A pre-outcome protocol review also separates the historical camera domain
from the current-placement measurement; both protocol versions are retained.

## Fixed current-placement acquisition and results

Run exactly three camera-enabled current-teacher diagnostics at the accepted
common floor pose, one each on the two selected train identities and existing
development identity, with unchanged 6100/6100/6200 seeds. Render every control
tick, retain fixed-time sparse RGB-D/calibration/proprioceptive packets, all robot
commands, both native feedback substeps and complete teacher/contact traces.
Evaluator annotations are separate files and never enter the component estimator.
The neck starts at the existing inspection's final target; these are diagnostic
captures, not complete demonstration recordings or inspection qualification.

| Current identity | Valid sampled contact states | Available components | Finger distance error p95 /max, mm | Direct nominal-query support |
|---|---:|---:|---:|---:|
| `animated-door-1-88abf40` | 6 | 5 | 1.110 /1.115 | 0 |
| `door-2738468b94d74c5f` | 5 | 0 | Unavailable | 0 |
| `psx-industrial-001-4f5561b` | 5 | 5 | 8.123 /8.129 | 0 |

Errors compare the observed component against an independent annotated local
collision tangent plane at the actual robot query. They are sparse descriptive
measurements, not complete temporal availability, a confidence bound or an
operational contact oracle. Unsupported/ambiguous/occluded queries remain in the
records. No measured component is promoted to an admitted physical separation
certificate. The existing >=4 mm pose budget and all qualification flags remain.

All three teacher results, numeric native arrays and numeric command arrays are
identical to their earlier camera-disabled cases: right `invalid_hold`, left/dev
`lost_contact`, valid sustain 46.091 /35.480 /32.625 degrees, no valid hold and all
released. The previous failures remain. The additional captures do not establish
a successful teacher, learned execution or a different task denominator.

Independent FK agrees within 1.2 micrometers /1.8e-6 rad; camera FK within
0.24 micrometers /6.7e-7 rad. Retain 41 observed frames, 6,876 native samples and
57.30 simulated seconds. Root endpoints are checked at 60 Hz; feedback is 120 Hz.
Two seconds of continuing physics follow release; this does not qualify a moving
loaded stop or the residual load model's enlarged contact domain.

## Decision, resources and exact missing interface

This study uses three additional physical cases: **six of twelve prerequisite
cases are now used, six remain unspent**. No new demonstrations, optimizer steps,
learned commands, pose/IK/controller/threshold changes or sealed-test access occur.
Close this bounded candidate without substituting evaluator truth, a pose flag or
an unmeasured error bound. This rejects this component route, not every possible
observed contact method. The [[b1-shared-learned-pilot|end-to-end goal remains incomplete]].

The missing maintained-provider/monitor interface is current-query physical
separation support tied to the actual finger/material identity, with measured
error, calibration, occlusion, timestamps and material-response bounds, plus
independently qualified robot-only load/stop consistency. It cannot be replaced
by the current generic pose distance or by per-door collision truth. Resolving
that measured producer is the next indispensable perception/contact dependency;
the [[b1-pilot-contact-prerequisites|shared teacher load/reference failure]] also
remains before recollection or learning. A3/A4 first-command geometry/bootstrap
and matched operational recording dependencies stay open.

The three acquisitions take 146.95 case wall seconds and
168.11 outer workload seconds. Historical/current component
workloads plus acquisition reserve 265.05 workload wall seconds
(0.0736 one-GPU hours), excluding idle preflights; this is an allocation
upper estimate, not GPU kernel time. No foreign GPU work is recorded during these
workloads. Local evidence uses approximately 258.7 MiB. Simulation and
wall clocks stay separate; no real-time claim is made. The original three-GPU-hour
ceiling and [[b1-shared-learned-pilot|campaign cost proposal]] stay in effect.

Ignored local `outputs/evidence/b1-shared-learned-pilot-20261016/contact-certificate/`
retains protocols/review, the support-count defect and fix, exact scripts, all
attempts/logs/monitors, raw packets, separate evaluator annotations, per-query
errors, semantic/physical checks, resource accounting and `outcome.json`.
`evaluate_physical.py` regenerates the numerical report from retained inputs;
it does not start Isaac, model inference or a new physical case.
