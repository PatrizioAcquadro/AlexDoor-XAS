# Selected-Candidate Registration with Preserved Alternatives

One diagnostic schedule is implemented from `main @da804ad`, following
[[../topics/pre-a4-initialization-and-hinge-budget|the shared pre-A4 protocol and uncertainty audit]].
The reference retains graph off, partial batch on, rollback off, five inliers,
4 mm and the vectorized SDF Jacobian. A3/A4 and admission are unchanged.
No acquisition, robot/leaf movement, contact or training is performed.

## Fixed intervention and lifecycle

`selected_registration_only=True` is an opt-in offline diagnostic. The normal
worker/provider defaults are unchanged. It requires graph-off frame-to-map and
`diagnostic_only=True`. Select the original first eligible automatic observed
candidate (native object 0), with original masks, order, seeds and geometry.
Selection is not leaf identity or contact admission. No truth, asset identity or
teacher command crosses the worker's observation whitelist.

Register the selected candidate every processed native frame. Register alternatives every
nine frames (150 ms at original 60 Hz). If selected native support fails,
expand to all alternatives in that same frame and continue on the next frame;
recovery retains an already expanded frame. There is no oracle switch, merging
of object maps, identity transfer or model reset. These are observed native gates,
not truth-driven recovery triggers. The audit period is an acquisition cadence,
not a claim that an alternative's publication meets the 150 ms freshness limit.
The counter uses processed native frames: nine frames equal 150 ms of source
time only in this dense 60 Hz replay. A latest-waiting queue that drops captures
would lengthen that source interval and change sampling, pending expiry and
tracker/map history. A live source-time audit schedule and its resampled quality
would need separate validation; neither is tested as another variant here.

Every original image still enters shared SAM2, BootsTAPIR and the track table.
Keep all initial maps/TSDFs, queries, historical IDs and original bounded 120-point
renewal. A deferred candidate follows native insufficient-registration handling:
its pose is frozen and unavailable, its current pairs are empty, and map growth,
keyframe sampling/promotion and retirement are suppressed. Explicit
`registration_evaluated=False` distinguishes planned deferral from measured native
failure. No copied pose gains a new support time. Alternatives remain observable
at audits and during expansion, but their dense pose availability is deliberately
reduced. They cannot be described as unchanged backup estimators.

Skipping alternative registration and its unsupported map updates changes the
shared NumPy random stream, candidate/reference history, pending expiry/promotion,
mask filtering and later sampling. The selected estimate may therefore differ on
later frames even though its own gates and registration implementation are fixed.
No independent RNG intervention is combined with the schedule. Deferred native
loss streaks cannot activate a relaxation in this reference: the existing
`pose_jump_guard_relax_when_lost` defaults false. Frame-to-frame/global-graph modes
are refused because their history semantics have not been assessed.

## Comparison protocol

Ignored drivers, protocols, original-frame/candidate rows, native traces, actual
CUDA storage inventories and reports remain under
`outputs/b1/perception/p2p-selected-registration-01/`. Original baselines and
failed trials elsewhere are preserved. The targeted comparison starts at the
original 31 s initialization and processes **every original frame through 41 s**:
601 frames per attempt, both light and nominal, fresh processes and one attempt
per condition/system. It covers static initialization, initial motion and the
previous first numerical hinge times, not just a reset at a favorable event.

Both systems use the current vectorized implementation, identical automatic
initialization and actual IPC. The all-registration reference is also checked
against the preserved partial-only poses/decisions. The pilot progression rule
is fixed before inference: no more than two percentage points lower correct
availability/accepted precision, no more than 150 ms extra maximum correct-pose
gap, and at least 20% median request saving in both conditions. This is an
exploratory progression rule, not a physical qualification threshold or a
statistical confidence claim. If promising, evaluate both complete original
31–78.6167 s sequences from the original seed, retaining tails and all failures.

Correct availability remains accepted integration and relative seed-zone position
within 1 cm/full rotation within 5 degrees. Exclude seed from error/acceptance
counts but retain the complete scheduled denominator. Native loss, inaccurate
acceptance, deliberate deferral, unavailable integration and original-material
recovery are distinct. Compare paired gains/losses, longest gaps, the terminal
window, every candidate's audit availability, reference/cohort/keyframe/TSDF
history, measured CUDA storage and resident peaks, and complete request timing.
Acquisition-time tracking quality is separate from publication age.

## Complete CUDA results

The four 601-frame pilots complete without failure: each system has 600 correct
nonseed primary poses, no inaccurate acceptance and no native primary loss.
Median request saving is 37.66% light / 36.01% nominal, meeting the fixed
progression rule. Four fresh full attempts then complete all **11,432 scheduled
frames**, with all 62,876 candidate rows retained, no inference retry or process
failure. Both all-registration vectorized replays reproduce every preserved
partial-only pose exactly (maximum matrix difference zero), including native
loss/integration decisions. Initial masks/maps/IDs, source times, models and
native settings match; actual CUDA query/feature/causal storages respect the
120-reference cap and graph state stays empty.

The following primary errors/precision exclude the seed; correct availability
uses all 2,858 scheduled frames. All four systems accept 2,857 nonseed poses.

| Condition / schedule | Correct / wrong accepted | Accepted precision | Correct availability | Position p95 (mm) | Rotation p95 (degrees) | Longest correct-pose gap (s) |
|---|---:|---:|---:|---:|---:|---:|
| Light / all | 2,450 / 407 | 85.75% | 85.72% | 11.89 | 0.835 | 0.250 |
| Light / selected | 2,752 / 105 | 96.32% | 96.29% | 9.05 | 0.349 | 0.067 |
| Nominal / all | 2,484 / 373 | 86.94% | 86.91% | 12.30 | 0.476 | 0.317 |
| Nominal / selected | 1,785 / 1,072 | 62.48% | 62.46% | 13.70 | 0.329 | 1.683 |

Paired primary gains/losses are 370/68 light (net +302) and 136/835 nominal
(net -699). In the final 518 frames from 70 s, correct poses change 350→445
light, but **325→30 nominal**. The nominal position maximum falls 22.92→21.37 mm
while wrong acceptance grows by 699; a lower maximum alone would hide the
regression. Preserved full timing/error curves are in
`outputs/b1/perception/p2p-selected-registration-01/full-comparison.png`.

## Alternatives, recovery and reference history

No primary native loss or return occurs in any full attempt. In each selected run
there are 2,540 selected-only frames, 317 periodic audits and the initialization;
no primary-loss expansion is exercised. Its same/next-frame expansion is unit
checked, but actual primary recovery is **unvalidated**. Native support does not
detect the nominal accuracy loss, so it cannot trigger expansion for that tail.

Each alternative has 2,540 deliberate deferrals and only 317 nonseed registration
audits. At those exact same 317 times, correct alternative poses change as follows:

| Condition | Candidate 1 | Candidate 2 | Candidate 3 | Candidate 4 | Candidate 5 |
|---|---:|---:|---:|---:|---:|
| Light, all→selected | 84→36 | 35→37 | 124→204 | 167→186 | Absent |
| Nominal, all→selected | 142→68 | 38→49 | 179→204 | 0→0 | 0→0 |

Evaluated native returns for alternatives change 30/92/0/0→5/6/6/7 light and
100/50/0/25/52→3/6/11/4/5 nominal. Deliberate deferral is excluded from these
returns; neither these counts nor numerical consistency establishes material
reacquisition. The evaluator-only original-seed reference check finds correct
poses with at least five measured, rigid-motion-compatible original inliers
449→417 light and 376→417 nominal. All four final frames have zero such original
inliers. This is a necessary correspondence consistency check, not ownership
proof; renewed references remain essential to the late trajectory.

All-candidate historical-point/keyframe peaks change 2,831/112→1,378/52 light
and 2,939/114→1,359/46 nominal. Final primary active/historical/keyframe counts
change 120/556/16→111/674/28 light and 120/668/22→117/625/22 nominal. Thus
preserving candidate maps and neural query history does not preserve their
registration/reference history. The one scheduling intervention changes shared
random sampling and later histories; these runs do not separately attribute
quality changes to RNG, mask filtering, registration or map growth.

## Memory, complete observation latency and queue limits

Measured latency includes copying the saved sensor snapshot, actual IPC inference
and the real `PanelTracking.consume` geometry adapter before evaluator truth is
read. It excludes live capture/rendering, runtime waiting, action admission and
control. Seed semantic initialization/startup is separate; result export is
recorded separately rather than included as operational service work. These are
complete measured observation-processing costs, not a qualified robot loop.

| Condition / schedule | Observation p50 / p95 (ms) | Process GPU peak (GiB) | Causal-state peak (GiB) | Retained-mask peak (GiB) | TSDF peak (GiB) |
|---|---:|---:|---:|---:|---:|
| Light / all | 623 / 736 | 8.41 | 1.097 | 1.202 | 0.602 |
| Light / selected | 427 / 578 | 7.85 | 1.099 | 0.665 | 0.488 |
| Nominal / all | 652 / 892 | 10.41 | 1.318 | 1.687 | 1.400 |
| Nominal / selected | 498 / 619 | 9.69 | 1.317 | 0.837 | 0.973 |

Median request saving is 31.43%/23.59%; complete observation savings are similar.
Total query peaks remain 599→600 light and 720→719 nominal: neural causal memory
is effectively unchanged. Fewer retained keyframe masks and map growth explain
the observed memory reduction. The light pilot had higher selected GPU memory,
so a short prefix alone did not predict the complete memory result. Every
nonseed observation in all four full runs takes more than 150 ms: **zero fresh
publications even before adding live waiting/capture/control**.

A hypothetical FIFO fed at 60 Hz ends with 1,727→1,171 s backlog light and
1,916→1,361 s nominal. The actual engine instead has one in-flight request and
one latest waiting capture. A service-time-only model of that queue processes
74→106 light and 68→94 nominal captures out of 2,858; the rest are overwritten,
and zero modeled publications meet 150 ms. This is not an executed live replay:
dropped observations change tracker sampling, causal state, native-frame audit
cadence and expiry/map history. Dense replay accuracy cannot be transferred to
that sparse schedule, nor can these modeled queues qualify availability.

## Decision and next gate

**Reject common adoption of selected-only registration with periodic audits.**
The full nominal regression fails the declared quality/gap rule despite speed
and memory savings. Do not select the favorable light condition or tune by door.
Keep this single option disabled and diagnostic-only (`a8d0470`), retaining all
attempts, baselines, the postprocessing serialization failure and original times.
No second cadence/RNG/recovery variant, new movement/contact or A3/A4 change is
introduced. Ground truth remains evaluator-only.

Retain observed geometric improvements, the vectorized SDF Jacobian and the fixed
graph-off/partial-batch/no-rollback/five-inlier/4 mm reference. The shared pre-A4
specification is adopted as the planned common initialization contract, not a
qualified controller. Integration still lacks a supported physical reference
**before the first A4 approach**, a closed simulation uncertainty-input contract,
reliable material/contact evidence and a complete fresh consumer path. Current
stationary evidence contains no physical pin; robot-induced diagnosis requires
the separate, currently absent pre-articulation admission/load/sweep/stop path.
The minimum next checks and simulation/hardware boundary are canonical in
[[../topics/pre-a4-initialization-and-hinge-budget|the initialization and budget topic]].
