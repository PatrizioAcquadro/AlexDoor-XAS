# ACT A1 complete-cycle collection

The approved campaign starts from `1550291` on
`PatrizioAcquadro/b1-policy-pilot` in the existing `23f1` worktree. Its objective
is useful autonomous manipulation from allowed RGB-D and proprioception.
Teacher similarity is not the success criterion. This campaign collects and
prepares data; training, learned evaluation, development collection and final-test
access are excluded.

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
| D1 | animated-door-1-88abf40 / right | Reuse C15 | Planned | C07 candidate | Planned | Planned | Planned | 0.9 | Planned | Planned |
| D2 | door-2738468b94d74c5f / left | Reuse C14 | Planned | Planned | Planned | Planned | Planned | 1.1 | Planned | Planned |
| D3 | psx-front-005-ee7d5c6 / right | C04 candidate | Planned | Planned | Planned | Planned | Planned | 1.1 | Planned | Planned |
| D4 | psx-interior-wood-002-02e442c / left | C03 candidate | Planned | Planned | Planned | Planned | Planned | 0.9 | Planned | Planned |

The machine-readable [matrix](../../../outputs/evidence/act-a1-complete-cycle-20261026/matrix.json)
records planned and actual conditions, source/reference links, attempts, outcomes,
admitted scopes and gaps. Actual fields remain empty until measured. C14/C15 are
reused only within their saved qualified domains; they do not prove a replay of
the latest implementation. There are no new native results at initial preparation.

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

The human approved **up to 12 additional local GPU wall-hours, 200 GiB additional
retained evidence and reversible archival to existing persistent project storage**.
The native-allocation ceiling is 60, including qualification/diagnostics. A fresh
resource baseline preserves 44.930708 local / 31.823056 remote GPUh, 198,773
cumulative historical optimizer updates and approximately 1023.967 GiB retained.
There are zero new optimizer updates. The campaign separately measures its
incremental use; archival does not refund historical charges.

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
preserves the initial disk headroom and takes both existing GPU leases. Foreign
GPU workloads are preserved and queue waiting is not charged as GPU allocation.

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
threshold or condition is changed to convert the original outcome. Native
validation of the corrected integration remains pending.
