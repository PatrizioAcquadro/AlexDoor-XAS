# ACT A1 Complete-Cycle Collection

## Question and current outcome

Can four existing train doors supply 36 independently qualified full-cycle and
real-history recovery conditions under the maintained controller, observed
admission and fixed scientific criteria? **The campaign is incomplete and paused
by the human on October 27, 2026: 10/36 qualified conditions, comprising eight
ordinary cycles and two complete recoveries.** There are 29 full attempts: eight
positive and 21 negative. Two qualified reused references plus eight new recordings
fill the ten conditions. The new sources yield 45,556 overlapping windows, not
45,556 independent demonstrations. Zero new optimizer updates; learned outcomes
remain 0/22. Development collection, training and final test are excluded.

The [[act-a1-complete-cycle-pause-diagnosis|pause diagnosis]] owns the current
same-door comparisons, negative mechanisms and sole proposed discriminator.
No collector, successor or allocation remained at closeout; the heartbeat is
PAUSED. No new cycle, component test, retry or GPU job is authorized.

The retained negative full attempts comprise eight position/load-support refusals,
five perception/material/sweep refusals, two contact-load failures and one each of
prescribed geometric intersection, opening-progress exhaustion, contact-search
exhaustion, unresolved runtime stall, historical GPU contention and repaired
recorder failure. These are outcome mechanisms, not independent causal counts.
Passing components and stops never add a full cell. The last H- bundle passes six
cases/18 stops but misses simultaneous loaded gripper coverage by 3.252 mrad;
qualification and calibration adoption are rejected.

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
| D1 | animated-door-1-88abf40 / right | Reuse C15 | Qualified new | Unqualified; contact loss | Unqualified; local margin | Qualified new | Qualified new | Qualified new;0.9 | Unqualified; inspection timeout | Qualified new complete recovery |
| D2 | door-2738468b94d74c5f / left | Reuse C14 | Unqualified; outside full elbow support | Unqualified; latest component coverage rejected | Unqualified; outside full gripper support | Qualified new | Three negatives; acquisition-stroke block | Qualified new;1.1 | Qualified new complete recovery | Unqualified; contact-load refusal during hold |
| D3 | psx-front-005-ee7d5c6 / right | Unqualified; material uncertainty/sweep | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted; planned light 1.1 | Unexecuted | Unexecuted |
| D4 | psx-interior-wood-002-02e442c / left | Unqualified; observed sweep refusal | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted; planned light 0.9 | Unexecuted | Unexecuted |

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

These qualification rules describe the retained experiment; they do not authorize
resumption. Shared instability or failed independent stopping blocks dependent
execution. No unchanged retry-until-success is permitted.



## Resources, archival and closeout

The latest historical authorization removes cumulative GPU-hour, allocation-count
and retained-data administrative ceilings. It does not remove physical safety,
scientific admission, actual disk-capacity checks or finite per-job watchdogs.
The subsequent human pause supersedes execution authority. Earlier 12 GPUh,
60-launch and 200 GiB values remain historical, not reinstated active caps.

The recorded precampaign baseline is 44.930708 local/31.823056 remote GPUh,
198,773 historical updates and approximately 1023.967 GiB retained. Campaign
closeout separately charges **8.5416 aggregate GPUh across 61 closed allocations**.
At pause, 79.74 GiB is retained locally and 441.5 GiB is physically free. The
remaining 20 ordinary/six recovery conditions project 106.5–137.2 GiB additional,
or 192.2 GiB at legitimate physics budgets, before future failures/diagnostics.
These dated scenarios are neither guarantees nor restart permission.

Original ownership rules retain the shared local lease, named owner, exact device
UUID and complete process-group cleanup. Idle hardware or elapsed estimates do
not transfer ownership. Final authority restricts new acquisition to the local
RTX 4090/storage; earlier Blackwell compatibility attempts remain historical.
No current remote job or replica is authorized. Default watchdogs are 2400 wall/
240 physics seconds; the reviewed D2-DR 180 s push instead has 2500 wall/250 physics
seconds. Existing stall/termination/cleanup reserves and physical criteria remain.
These are retained case declarations, not permission to run them.

Nine inactive historical HDF5 payloads totaling 188.365 GiB were losslessly archived
to `gilbreth-purdue:/depot/euge/data/pacquadr/alexdoor-xas/act-a1-complete-cycle-20261026/inactive-archives`.
Each was restored and checked for identical bytes, size, mode and nanosecond mtime
before local removal. Adjacent `.archive.json` pointers preserve overwrite-protected
restore commands. Active train/development inputs, external/VDS dependencies,
checkpoints, failures and selections were protected. Git cannot recover ignored
payloads. The present cleanup moves or deletes none of them.

Earlier Blackwell preparation failures, missing native stop evidence, repaired
private converter/compiler paths, the initial recorder naming collision, failed
component coverage and controlled prefixes all retain their original outcome and
charges. They are not additional positive cells or new hardware qualification.
Ordinary attempt chronology remains available at Git revision `4d21837`.

## Evidence

The canonical directory is `outputs/evidence/act-a1-complete-cycle-20261026/`.
`matrix.json` owns cell membership and original attempts; `campaign-state.json`,
`authorization.json`, `closeout.json` and `RESUME.md` record the pause.
`resources.json`, `storage-forecast.json`, allocation reviews and archive receipts
own actual expenditure, capacity and recoverability. `pause-diagnosis-cpu-evidence.json`
retains CPU comparisons of all 29 original full attempts. Exact configurations,
independent stop/contact reviews and the source dispositions remain beside them.
The [[act-a1-complete-demonstration-audit|source audit]] and
[[approach-contact-corrections|corrective qualification]] remain distinct evidence.

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
