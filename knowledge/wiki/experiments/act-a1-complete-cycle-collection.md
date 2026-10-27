# ACT A1 complete-cycle collection

The approved campaign starts from `1550291` on
`PatrizioAcquadro/b1-policy-pilot` in the existing `23f1` worktree. Its objective
is useful autonomous manipulation from allowed RGB-D and proprioception.
Teacher similarity is not the success criterion. This campaign collects and
prepares data; training, learned evaluation, development collection and final-test
access are excluded.

## Current result: five new full-cycle sources qualified

**7/36 cells are independently qualified: C15/D1-N and C14/D2-N reuse, plus
new D1-H+, D2-V, D1-DL, D1-V and D1-DR.** The new source acquires continuous valid native contact, reaches
51.86 degrees against the unchanged 41.01-degree requirement, holds for 3 s,
intentionally releases and stabilizes for 2 s. Independent saved native-force,
gap and stop-stream reconstruction reproduces the passing continuity and 2+4 s
stops. Native peak normal force is 7.63 N; the separate observed operational upper
estimate peaks at 7.94 N within its 10 N domain. These are distinct from the
80 N physical hard limit.

The measured hold-tool height changes by 20.02 mm from C15. The same robot,
240/60 Hz clocks, active mass/inertia/damping, placement and lighting are verified.
The temporal consumer admits **4,190 windows**, including all six manipulation
phases, with actual applied-minus-measured A1 targets, four-second history,
23 x 5 x 240 x 384 images, 241 x 32 robot state and chunk 12 / execute 6.
One independent recording supplies these overlapping views. No recovery or
learned success is thereby established; the historical learned result stays 0/22.

D2-V independently passes uninterrupted native contact, 52.54 degrees against
42.03 required, 3 s hold, release, 2 s stabilization and both stops. Actual lighting
scale is 1.1; initial RGB changes while camera pose, physics, dynamics and controls
retain the agreed comparison. Six real phase images pass visual review. CUDA
preparation and the actual consumer admit 8,126 windows from this one recording;
overlapping windows are not independent demonstrations. A completed receipt was reconciled with D2-V's stale running
matrix state before admission; raw evidence and original outcomes were unchanged.

D1-DL independently verifies active panel/handle mass and inertia at 0.7 of the
nominal values, hinge damping 4, continuous contact and 50.28 degrees of controlled
opening against 41.01 required. All 180 hold intervals remain valid and loaded;
minimum hold opening is 47.86 degrees. Intentional release, 120 stabilization
intervals and both independent stops pass. Native peak is 6.75 N; the separate
observed operational upper estimate is 6.95 N. Six actual phase images pass review.
Its allocation costs 920.259 s and explicitly returns VLA ownership after cleanup.

The first actual cached-frame pipeline retains 937 CUDA captures inside the raw
file. CPU-only metadata preparation reuses those exact values through a relative
external link and admits 4,107 windows with unchanged masks, shapes and exact A1
deltas. The prepared numeric store is about 3.9 MB and requires its 2.67 GB raw
source; no duplicate image payload or additional GPU allocation is created. This source raises the then-current new-source window count to 16,423.

The first Blackwell full D1-V source passes independently: 50.16 degrees against
41.01 required, continuous contact, 3 s hold, release, 2 s stabilization and both
stops. Actual lighting is 0.9; initial RGB mean changes by -3.95/255 while camera
pose and active dynamics retain their agreed bounds/settings. Native peak is
6.72 N. All six phase images and the CUDA-prepared temporal consumer pass,
adding 4,111 windows. Its native allocation costs 2,267.593 s and explicitly
returns VLA slot ownership; local CUDA preparation costs another 32.350 s.
Four new independent recordings now provide **20,534 windows**.

D1-DR independently verifies panel/handle mass and inertia1.3 of nominal, hinge
damping6, continuous contact and50.43 degrees against41.01 required. Full hold,
intentional release, stabilization and both stops pass. Native peak is6.92 N.
Six actual phase images and4,879 temporal windows pass, reusing the already CUDA
frames through CPU metadata preparation. Its native allocation costs1,058.234 s;
no additional GPU preparation is needed. Five new recordings now supply **25,413 windows**.

Eleven completed new matrix collection attempts retain six negatives and five
qualified full cycles; all eleven pass both independent stops. D1-H- terminates for
contact loss or uncertainty after 1,791 guarded commands, with zero positive targets.

The fixed D3 bundle completes all six preregistered cases in 602.026 charged
seconds: old/new RGB crossed with 10 Hz/reset-1, 10 Hz/reset-2 and 60 Hz/reset-2.
Every case selects one tracked candidate yet reproduces unavailable local material
support. At the terminal capture, the native pose keeps 63–64 of 65 sampled
reference points depth-consistent; the local correction keeps zero. Retained
first-failure packets show 3,996–3,998 of the original 4,000 material points still
visible before correction, with median depth residual below 0.06 mm. Their full
corrected clouds have zero supporting points, so increasing root sampling is not
a remedy. RGB, reset count and tested update cadence do not explain this refusal.
The original missing-selection failure remains preserved and distinct.

The production local surface query returns only the first dominant measured plane.
On relief geometry, that plane can differ from the selected material. The corrected adapter expands to all current measured components only when the dominant query
fails, retaining conditioning, identity, ambiguity and missing-support gates.
Regression scenarios cover nominal geometry, relief on either side and absent
material. The correction is installed and committed as `8aadab9` only after D1-DR terminates
and returns its GPU. Production geometry/session regressions pass. One distinctly
named full `D3-N-material-query` is queued behind a separate VLA interval; native
qualification remains pending. No physical threshold or uncertainty bound is relaxed.

Completed cost is **3.0899 aggregate GPUh** (2.1564 local / 0.9335 Blackwell)
across 25 allocations. Retention is31.32 GiB including replicas. Exact accounting
is in `resources.json`; waiting is uncharged. All failed allocations and original
source outcomes remain preserved. Training, normalization and historical selections
are unchanged.

The first preparation rejects invalid CLI arguments. The corrected CUDA image
preparation omits eligibility, so its reader safely finds zero usable rows. Both
failures and the diagnostic store remain preserved. Independent physical review
then permits a separate metadata-only reviewed copy with the maintained
`teacher_eligibility` mask. D2-V uses one corrected CUDA preparation. Future native
sources can optionally retain the exact CUDA frames already computed during
capture and reuse them through CPU-only metadata preparation; see
[[../topics/act-a1-observations-and-time|temporal preparation]]. The option defaults
off; the completed D1-DL source verifies native integration. It changes no
controller, timing, thresholds or teacher endpoint.

Nine inactive original HDF5 payloads, **188.3648 GiB**, now reside in **159.7574 GiB**
of restoration-verified persistent archives. Active dataset links resolve. Final
Data Depot quota use is **80.2% of the shared 2.0 TB limit**; temporary restoration
copies are removed. The initial physical storage shortage is resolved.

Persistent GPU ownership and complete-episode handoffs govern local collection.
The protected D4 follow-up retains its unsupported observed finger-sweep refusal
after 112 guarded approach commands. D1-H+'s original 192-command weak-position
refusal remains negative; its changed follow-up uses the independently qualified
lower shoulder-Z recipe, with frozen physical, residual and stop bounds. D3's original
selection refusal remains unresolved; the fixed bundle instead reproduces a local
material-query failure in every case. Its identical depth/calibration/robot inputs
and all six outcomes remain preserved. These diagnoses do not prevent unrelated cells. See the
[resume handoff](../../../outputs/evidence/act-a1-complete-cycle-20261026/RESUME.md).

## Agreed coverage and present evidence

There are **36 prospective cells: 28 ordinary complete cycles and eight complete
recoveries**, on four existing train doors, using the same right arm. This is an
initial coverage design, not a factorial experiment or sufficient-data claim.
The [[act-a1-complete-demonstration-audit|source audit]] remains authoritative for
historical outcomes: two executable references, five qualification candidates,
and eight discontinuous originals with useful masked segments. Missing
qualification is not a demonstrated defect. No historical source or selection is
retroactively changed.

| Door | Identity / opening direction | N | H+ | H- | P | DL | DR | V | RV | RP |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | animated-door-1-88abf40 / right | Reuse C15 | Qualified new | C07 candidate | Planned | Planned | Planned | 0.9 | Planned | Planned |
| D2 | door-2738468b94d74c5f / left | Reuse C14 | Planned | Planned | Planned | Planned | Planned | 1.1 | Planned | Planned |
| D3 | psx-front-005-ee7d5c6 / right | Unqualified; selection refusal | Planned | Planned | Planned | Planned | Planned | 1.1 | Planned | Planned |
| D4 | psx-interior-wood-002-02e442c / left | Unqualified; observed sweep refusal | Planned | Planned | Planned | Planned | Planned | 0.9 | Planned | Planned |

The machine-readable [matrix](../../../outputs/evidence/act-a1-complete-cycle-20261026/matrix.json)
records planned and actual conditions, source/reference links, attempts, outcomes,
admitted scopes and gaps. Actual fields remain empty until measured. C14/C15 are
reused only within their saved qualified domains; they do not prove a replay of
the latest implementation. New attempt outcomes remain separate from the C03/C04 historical candidate dispositions.

Exact candidate setups are in the campaign's `setups/` directory. Nominal contact
height is 1.09 m; H+/H- use 1.11/1.07 m. P translates the fixed robot +20 mm in
world X and +25 mm in Y with +4 degrees yaw on D1/D2; D3/D4 use -25 mm in Y and
-4 degrees yaw. Translation magnitude is 32.0 mm. Shoulder-Y changes by -0.05 rad
and elbow-Y by +0.04/-0.04 rad respectively. These are proposed physical
variations, not measured reachability or new safety thresholds. Screen reset,
visibility, feasible approach and actual posture/path differences before counting
P; retain a failed meaningful variation rather than silently reducing it.

DL uses coherent panel/handle mass and inertia scale 0.7 with hinge damping 4;
DR uses 1.3 and 6. Read active PhysX mass, inertia and actuator damping rather than
accepting configuration alone. V changes dome illumination only, with the scales
in the table; recorded pixels must establish the visual difference. All retain
240 Hz physics, 60 Hz control, actual head RGB-D and the maintained four-second,
240 x 384 temporal contract with chunk 12 / execute 6 compatibility.

RV uses an actual guarded precontact pose/velocity disturbance followed by a
0.4 s measured-origin brake, 1.5 s realignment and full continuation. Its declared
pose offset is 15 mm laterally and 0.04 rad in rotation, with signs balanced across
the two door pairs, during approach seconds 3.2-3.8. Measure the resulting q/dq;
the desired offset alone does not establish the recovery condition. This builds
on [[approach-contact-corrections|the late-left braking mechanism]], without
claiming identical saved failure history on other doors.

RP uses the documented misplaced late pause, during approach seconds 4.1-5.3,
then a 1.5 s measured-origin return. This is the agreed pause alternative, not a
claim of an 18 mm measured shortfall. Recovery setups reserve a 7.5 s approach,
the nominal 30 mm precontact and 3 s entry. Every prefix is actual execution through ordinary
operational guards. Injected commands remain causal history and are excluded from
targets. No restoration, fabricated history, deliberately lost established
contact, or bypassed error-prefix admission qualifies a cell.

## Collection integration and qualification

`pilot_physics.py collect --guarded-teacher --operational-recipe ...` retains the
ordinary teacher and recorder. A pure preview of its bounded pose controller
produces the proposed A1 endpoint. The maintained observed runtime checks load,
contact, robot sweep, stopping support and predicted response before the command
is dispatched. IO independently compares the actual controller preview with that
admitted endpoint and checks measured response afterward. Evaluator annotations
are never supplied to this runtime. Actual requests and applied actions remain
in the native recording.

The optional full-cycle correction reuses `CollectionCorrection` for an approach
error and continues through ordinary push, hold, release and stabilization. Its
brake holds the first measured joint anchor before the existing smooth return;
this prevents pose-IK nullspace motion from being mistaken for arm braking.
The original short-task behavior defaults to zero braking and is preserved.
A recovery positive requires an executed error, actual correction, resolved
pending correction and full-cycle success. Existing role/quality masks exclude
injection and crossing target chunks while retaining sensor/controller context.

A full positive must acquire contact, open under uninterrupted valid contact,
hold, intentionally release, and stabilize ordinarily for at least two seconds,
extending under the existing settling rules. Existing independent 2+4 s stops
and current operational admission remain required. The 10 N operational stopping
domain and 80 N physical limit are unchanged. Native force peaks, filtered loads
and operational upper estimates remain separate. An unloaded terminal stop does
not qualify earlier loaded motion.

Establish anchors, then height/placement/posture, dynamics/lighting and complete
recoveries. Diagnose failures before any changed follow-up. An individual cell
failure need not block unrelated cells; shared instability or failed independent
stopping blocks dependent execution. No unchanged retry-until-success.

## Approved resources and storage

The human approved **up to 12 additional aggregate GPU wall-hours, 200 GiB additional
retained evidence and reversible archival to existing persistent project storage**.
The native-allocation ceiling is 60, including qualification/diagnostics. A fresh
resource baseline preserves 44.930708 local / 31.823056 remote GPUh, 198,773
cumulative historical optimizer updates and approximately 1023.967 GiB retained.
There are zero new optimizer updates. The later host authorization permits the
local RTX4090 and Blackwell, alternating with VLA-LEGO and SquadBot-AV. All hosts,
diagnostics and preparation share the original 12 GPUh / 60 allocation ceiling;
all retained replicas share 200 GiB. No new allowance or historical refund is
created by moving hosts.

Initial physical free space was about 3.49 GiB, distinct from the old 102.43 GiB
logical allowance. The consumer inventory protects current train/development
source paths, HDF5 external/VDS dependencies, complete-source evidence, operational
recipes and open process files. Nine inactive historical HDF5 payloads total
188.365 GiB. Each is streamed losslessly to the existing project Data Depot,
fully restored there, and checked for identical bytes, size, mode and nanosecond
mtime before its local file is removed. Local restore pointers and per-lot
receipts retain exact commands and identities. No active payload is selected.

The destination is
`gilbreth-purdue:/depot/euge/data/pacquadr/alexdoor-xas/act-a1-complete-cycle-20261026/inactive-archives`.
The live shared `euge` quota snapshot was 72.4% of 2.0 TB used. Purdue documents
[Data Depot as non-purged](https://docs.rcac.purdue.edu/userguides/depot/overview/).
Recheck quota/capacity during the batch. Temporary restoration copies are removed
only after verification. Scratch is not a sole archive.

Saved complete observed references consumed about 913 s on D1 and 1,557 s on D2,
including allocation tails. Their simple 34-attempt extrapolation is 11.66 GPUh
before extra qualifications, recovery overhead or slower doors; it is not a
promise that the approved 12 h suffices. Existing compressed anchor raw/prepared
pairs occupy 2.49-4.82 GiB, suggesting roughly 85-164 GiB for 34 analogous sources
before failures and diagnostics. Runtime and compression must be measured on new
attempts. Each launcher reserves a complete attempt, safe termination and storage,
preserves the initial disk headroom and retains the legacy GPU leases. The local
launcher additionally holds `/tmp/codex-local-gpu0.lock` for the entire allocation
and process-group cleanup. Under that lock it requires its named owner, active or
reserved state, exact device UUID and completed prior-owner handoff in
`/tmp/codex-local-gpu0.reservation.json`. Queued successors and restarts consult
the same persistent owner. An empty GPU or elapsed estimate never transfers
ownership. Only explicit release after cleanup changes the owner. The first
45-minute window was released early after the D4 negative completed. Foreign
workloads are preserved, and waiting is not charged as GPU allocation.

Blackwell uses its existing shared global workload lock and exclusive renderer
slots, with a short separate reservation-update mutex. The unrelated
`train_debug.py` process remains protected. AlexDoor has staged its source, four train-door assets, exact static
calibration-ray subsets and private perception model/runtime copies in an
isolated path. The external Alex package commit and robot URDF match locally. The supported
Isaac wrapper supplies Torch 2.10/CUDA 12.8 with `sm_120`. The first
12.096-second allocation fails before robot creation because the converter uses
an unwritable shared `/tmp/IsaacLab`; native stop evidence is unavailable there.
The optional `pilot_physics.py --urdf-cache-dir` routes robot and pedestal
conversion to private directories. Its distinct 507.275-second follow-up verifies
this route, the actual robot/head sensor, 240/60 Hz clocks, GPU model initialization
and both independent stops. The first Point2Pose geometry request then fails
because `nvcc` is absent. Model initialization does not establish usable kernels
or host qualification; no guarded approach command or positive source is admitted.

A private NVIDIA CUDA 12.8 compiler/header subset now compiles the actual
Point2Pose integration kernel for `sm_120` on CPU. Only the isolated child's
`CUDA_HOME` changes; shared runtimes, drivers, models and gates remain unchanged.
The newly named compiler-fixed native probe completes after the agreed actual
VLA188 cleanup and slot-1 handoff. Actual SAM3/Point2Pose GPU kernels, one ordinary
guarded approach interval, 240/60 Hz clocks and both independent stops pass. The
head-camera frame also passes visual review. Its controlled interruption is
expected: it supplies zero matrix cells or positive targets. The entire allocation
costs 322.276 s and explicitly returns VLA ownership after process-group cleanup.

The fixed six-case robot-only 10 N transfer bundle now passes on Blackwell.
Its nominal, near-boundary and perturbed measured postures cover both qualified
door directions. Every moving/stopped sample and all 18 stop continuations pass
the unchanged residual, position, speed, acceleration, feedback and joint-stop
bounds. No calibration is refit or widened. Its charged allocation is 251.431 s,
with an explicit VLA return after cleanup. This qualifies the original recipe's
measured transfer domain; the newer local shoulder extension is not transferred
by this bundle. The first full Blackwell D1-V starts only after verified actual
VLA201 completion and driver exit, with the full 2,555 s reserve inside a 2,700 s
window. That first full source subsequently passes independent native, visual and temporal
review. It qualifies this measured source, not arbitrary future conditions. Owned GPU
usage and shared-device headroom preserve other workloads; no third native lane
is authorized. A second local 4090 remains unavailable until endpoint/runtime
verification.

## Prospective data handoff

The separate [prospective selection](../../../outputs/evidence/act-a1-complete-cycle-20261026/prospective-selection.json)
keeps qualified full cycles, unresolved candidates, masked segments and existing
partial sources distinct. The [phase views](../../../outputs/evidence/act-a1-complete-cycle-20261026/phase-views.json)
reference acquisition, contact-to-push, push-to-hold, hold-to-release and
stabilization indices in existing recordings. They duplicate no payload and count
no overlapping window as an independent demonstration. Historical manifests and
the original 112-source normalization stay unchanged.

A later sampler proposal is uniform across doors, then 50% ordinary qualified full
cycles, 25% qualified complete recoveries and 25% qualified segments. Within a
complete source, mix 40% uniform eligible timeline with 60% across the five
critical views. Report resulting per-source/phase exposure and actual draws;
view overlap is an alternate route to the same source, not extra demonstrations.
This proposal is not implemented or approved training. Empty/unqualified buckets
must be reported before adoption rather than populated by unresolved sources.

Completion requires all 36 cells to qualify and pass the temporal consumer.
Otherwise report the exact partial denominator, retained failures, resource use,
remaining cells and demonstrated blocker. Reference and teacher success remain
separate from the historical learned result of 0/22.

## Native integration diagnosis

The first D3-N allocation stopped during common inspection, before arm approach.
The new cached `PolicyObservation` used the inherited recorder's `observation`
method name, so the next raw-image capture raised `TypeError`. Its five executed
inspection commands, incomplete original recording, report and 105.361 charged
GPU seconds remain in `runs/D3-N` and `allocations/D3-N-allocation.json`. There
are zero expert targets. Both independent 2 s and 4 s stops pass. This is an
implementation failure, not a physical rejection of D3.

The correction renames the cache to `latest_observation`; a regression exercises
repeated native state updates while preserving the inherited capture method.
The diagnosed follow-up has the distinct identity `D3-N-recorder-fix`. No
threshold or condition is changed to convert the original outcome. The corrected path subsequently completes inspection; D4 admits one real
approach interval. Contact, opening and recovery integration remain unvalidated.

## Independently measured position support

The first six robot-only cases test the existing 4.25 mrad lower shoulder-Z halo
at 10 N. All 18 specified stops and frozen residual checks pass, but measured
loaded positions reach only 1.97 mrad beyond the original core; the declared
coverage gate fails. This remains an unqualified candidate, not a physical failure.
A numeric screen of the complete historical H+ path predicts about 7.23 mrad of
lower-Z extension under the current generic stop envelope. That 120 Hz source
screen is planning evidence, not current native qualification.

One distinctly named six-case bundle therefore declares a 12 mrad lower-Z
candidate at 10 N and compensates the measured fixture servo displacement in its
commanded starts. All nominal, boundary, neighboring and oblique-force cases
complete. Every moving and stopped native sample satisfies the frozen residual
model; all 18 existing stop checks, joint-travel, speed/acceleration and feedback
gates pass. Measured loaded shoulder-Z reaches 0.066143 rad, covering the declared
lower endpoint 0.066863 rad. No target is clipped or relabeled to achieve coverage.

The separate `operational-recipe-shoulder-qualified.json` changes only this
empirical lower-Z width (4.25 to 12 mrad) and its qualified force cap (1 to 10 N).
Core q bounds, residual coefficients, physics, force/speed limits and stop bounds
are unchanged. The original recipe and first coverage failure remain intact.
This is a local empirical position-support extension, not arbitrary-pose/hardware
qualification or a new demonstration. Both qualification bundles stay out of
imitation targets, matrix cells and recovery counts. Exact plans, complete native
samples and independent reviews remain in the campaign directory.

The D3 dtype-only replay also selects at 25 s, so float32 alone does not reproduce
the original native refusal in that replay. The full native calibration/event
path remains unresolved. A prepared controlled native diagnostic records the
actual permitted calibration, every observer clock/state and exact model-packet
digests without changing inputs or control. It is not yet executed or qualified.

## Preserved attempts and demonstrated boundary

| Execution | Outcome | Charged GPU seconds | Positive scope |
|---|---|---:|---|
| D3-N | Recorder method-name collision before approach; repaired with regression evidence | 105.4 | None |
| D3-N-recorder-fix | Complete actual inspection, then `missing_explicit_contact_selection` before approach command 1 | 372.1 | None |
| D4-N | One applied guarded approach command; total-device memory reached 20,571 MiB and the existing 20 GiB resource monitor requested a safe stop | 330.7 | None |
| D4-N-reserved-window | Protected window; 112 guarded approach commands, then unsupported observed finger sweep; both independent stops pass | 226.9 | None |
| D1-H+ | 192 guarded approach commands, then weak-position future-stop refusal; both independent stops pass | 240.4 | None |
| Saved D3 float32 diagnostic | Changes only camera matrix dtype from the prior saved replay; still selects a patch at 25 s | 100.4 | Diagnostic only |
| Saved D3 inspection diagnostic | One observed-only replay of the retained 10 Hz images selects a patch at 25 s | 119.6 | Diagnostic only; not a physical episode or qualification |

The D4 monitor signals only the owned collector. The unrelated process is neither
stopped nor edited. This is a resource interruption, not a demonstrated physical
failure of the door or robot. The named GPU leases do not prevent a foreign task
that does not participate in the same coordination from restarting. Repeating
unchanged native attempts under that contention would waste the campaign. The
subsequent persistent-owner protocol prevents the queued VLA successor from
starting during the first AlexDoor allocation; it does not qualify the D4 motion.

The saved D3 diagnostic uses actual images and measured robot states, unchanged
geometry settings and the original synchronous publication convention. It reads
no evaluator payload or door CAD, produces no targets and is not the full 60 Hz
live observer/event path. Its available patch does not retroactively qualify the
failed native inspection. Diagnose the event/selection discrepancy before any
changed D3 follow-up; do not select evaluator truth or promote a favorable replay.

All five new raw recordings remain incomplete, with zero admitted positive
windows. They remain actual causal context and negative evidence. Existing source
views retain **12,234 eligible windows across two qualified complete sources**,
including 22 crossing starts for each named contact/push, push/hold and hold/release
boundary. These are two physical traversals per boundary, not independent window
trials. Acquisition views contain 1,080 starts and stabilization views 240. No
optimizer has drawn from the prospective selection.

Use `closeout.json`, `new-source-dispositions.json`, `archive-closeout.json` and
`resources.json` in the campaign directory for exact identities, denominators,
restoration receipts, current accounting and remaining resource allowance. Every
historical charge is retained; no compression or archival savings are refunded.
