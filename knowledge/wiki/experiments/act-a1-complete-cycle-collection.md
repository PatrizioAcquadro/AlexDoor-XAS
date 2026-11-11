# ACT A1 Complete-Cycle Collection

## Separate D4 continuation — November 8

The new D4 assignment starts from `aa328df`; the closed coordinator and paused
heartbeat remain unchanged. Evidence and job accounting are separate under
`outputs/evidence/act-a1-d4-continuation-20261108/`. The matrix remains 17/36,
including zero qualified D4 sources. Native data stay remote because local
capacity is nearly exhausted.

Both the physically complete finite-tangent nominal and the latest failed hold
reach the same reserved wrist-X target boundary, −0.885225 rad, before their
late pose error grows. The earlier saved clipping audit remains valid: clipping
could not explain its then-current push failures. The latest full hold is a
separate case. A CUDA audit of both complete records finds feasible
Jacobian-nullspace redistribution at the initial bound contacts, preserving the
six-axis linear task and all original target/rate bounds. Later recorded states
can still be infeasible; no physical recovery follows from the saved audit.

The isolated candidate passes software checks and preserves its linear task and
original target/rate limits in the native run, but is **not adopted**. Independent
readback confirms 71 redistributed commands and eight infeasible fallbacks.
The run reaches 50.641 degrees and two seconds of hold, with material error
2.262 mm rather than the earlier 15 mm drift. It then reports pre-release contact
loss at a 108 micrometre gap and an empirical load-model refusal on one shoulder-Z
acceleration sample (51.55 versus 40.26 rad/s²). Release and stabilization are
unavailable. Every actual 100 ms native-force window remains positive; 253
calibrated lower-force windows are nonpositive and four are unavailable. These
are overlapping confidence evaluations, not independent failures or proof of
zero physical force.

The ordinary stop passes. The independent six-second, fourteen-shape review
passes displacement, direction, joint travel, tool, rotation, speed and force,
but fails its empirical acceleration domain. No collision or hard joint-limit
violation is established. The original negative criteria are preserved, along
with the complete 9.85 GB source/data copy on elab3, exact hashes and released
owned processes/locks. Compact reviews are under the separate continuation
folder's `nullspace-native-review/`.

Under the updated proportional-acceptance guidance, observation confidence is
reviewed before adding controller layers.
The simpler earlier finite-tangent candidate already completes 50.522 degrees,
three seconds of hold, release and stabilization. A separate CUDA reassessment
of all 53,060 native samples now passes current/future ten-newton position support
and physical boxes using the independently qualified support extensions. Only
position support changes; its force, residual and acceleration models remain
identical. Its original negative qualification is not overwritten. Complete
observed admission and useful source data are still required; D4 remains 0/9.
The 0.545 mm historical stop is already acceptable and is not an optimization
target.

Replaying the unchanged contact monitor at its actual 60 Hz decision clock gives
12,245 positive push/hold windows, with minimum calibrated lower force 0.0213 N.
The two original nonpositive native-offset windows remain negative, both inside
a control interval. Complete negative-window controls still reject, duplicate
checks do not add evidence, and no threshold or filter changes. This diagnostic
uses recorded privileged normals; uncertainty from images is still untested.

The nominal `D4-observed-finite-tangent-N-blackwell-20261108` ran with actual
image perception and maintained observed admission on Blackwell. Its
small recorder adapter installs the original finite-tangent controller before
the first admission preview. It adds no material forecast or nullspace control.
Nine dispatch/refusal checks pass; controller sources, nominal setup, contact
monitor and entry-point qualification logic are unchanged. The original pressure
fixture remains an additional diagnostic guard, not an observation certificate.
The initial wiring test's incorrect inheritance-position assumption is retained
and corrected without a runtime change.

The job finished negative on November 9; readback on November 11 confirms
`contact_load_lost_or_uncertain` after 12 release commands (0.2/6 seconds),
following all 180 hold commands (3 seconds). Opening progresses from 49.88 to
50.84 degrees during hold and reaches 51.01 degrees at the final release sample;
this does not claim three seconds continuously above the 50.15-degree target.
Stabilization is absent. The partial recording contains 14,861 diagnostic
commands and zero eligible training windows. It cannot qualify the nominal or
unlock the eight other original conditions.

Independent readback finds positive native force at all 12,245 actual 60 Hz
push/hold decisions (minimum 0.0328 N) and positive observed lower-force evidence
in all 24,490 recorded checks (minimum 0.0199 N). One native-offset 100 ms
window is zero at 169.679 seconds; this remains a separate negative diagnostic.
All 12,797 recorded geometric admissions pass. The final observed lower-force
window is negative during release, when the saved native force is already zero.
The monitor permits unloading only after a submitted motion proves withdrawal
of every distal face. Exact proposal poses and the release latch are not saved,
so this review does not establish whether the refusal was justified or overly
conservative. The criterion is an observation-confidence and withdrawal-intent
gate; its refusal alone does not prove physical danger. No threshold changes or
phase-name exemptions are adopted.

Independent NumPy reconstruction matches all 51,188 saved native upper-force
estimates exactly, with no model, future ten-newton support, future physical-box or force
domain exits. These use privileged native normals and do not substitute for
observed perception. The original six-second continuing stop independently
passes all fourteen-shape radial/directional, joint, tool, rotation, speed,
acceleration and force bounds;
maximum tool travel is 0.233 mm. All owned processes and three kernel locks are
released, with no changes to shared reservations. The final charge is
35,696.13 GPU seconds, counted once. Original data and failures remain on
Blackwell. Explicit human consent subsequently authorized both previously
blocked transfers. All 77 raw/deployment/allocation files (8.90 GB) now have an
exact SHA256-verified replica on elab3; source hashes before and after transfer
are unchanged. Four reports and ten original RGB selections are verified locally.

Manual review covers all ten causal, same-generation selections, representing
eight distinct captures: push-last/hold-first and hold-last/release-first reuse
the same original frames across phase boundaries. The images show a nearly closed
door during approach/contact and an open door with the gripper near its panel
during push, hold and early release. Partial occlusion and sparse sampling do not
establish contact force, small separation or continuous stability. The last RGB
is acquired at 247.6 seconds, before the first native zero-force sample at
247.6125 seconds; separation and terminal refusal are not visually confirmed.
Stabilization RGB is unavailable. Negative evidence closeout is complete;
the causal withdrawal diagnosis remains unresolved. Positive-only source review
and causal training preparation were not executed. No unchanged retry is allowed.

Evidence includes `complete-finite-contact-clock-review.json`, the nominal
plan, wiring review and allocation in the separate local continuation folder.
New negative diagnostics are stored beside the remote allocation under
`D4-observed-finite-tangent-N-blackwell-20261108-negative-review/`, including
`negative-review.json`, `negative-load-support-review.json`,
`owned-resources.json` and `continuous-stop-review.json`. The local
`observed-negative-review/` also preserves transfer verification and
`manual-original-rgb-review.json`; the whole-copy receipt is
`D4-observed-finite-tangent-N-blackwell-20261108-custody.json`.
The next diagnostic must resolve the all-finger withdrawal decision from available
observed evidence before proposing any changed attempt.

## Question and current outcome

Can four existing train doors supply 36 independently qualified full-cycle and
real-history recovery conditions under the maintained controller, observed
admission and fixed scientific criteria? **The campaign resumed by explicit human
authorization on October31,2026 and remains incomplete:17/36 conditions,
comprising thirteen ordinary cycles and four complete recoveries. All nine D2
conditions are qualified.** The retained
pause baseline remains29full attempts:eight positive and21negative.

The November 6 direct human resume supersedes the November 4 pause. The
measured-release controller qualifies eight complete robot-only 0/10 N paths;
two independently measured H− support sides complete their scoped position
coverage. Actual changing-load contact remains negative: the three 0.1 N
coupled-pressure fixtures lose contact near 1.2–1.3°, and the four physically
derived push-pressure fixtures stop near 4.88–5.18° (three load refusals and one
contact loss). All seven-coordinate current/future boxes and original stops
pass for the latter four. Full payloads, unavailable phases, RGB and charges
remain preserved. A whole-command CUDA sensitivity audit finds a nonzero
quasistatic pressure contribution from tangential/rotational commands under
unequal joint stiffnesses. The single-step compensation (`ea8d5d4`) nevertheless
fails unloaded native opening in all four conditions, confusing free motion with
elastic preload. It is withdrawn; the earlier controller is restored and a
matched-elastic-plant trajectory regression preserves this failure mode. The
four 10 N counterparts remain unattempted. Independent review confirms all four
negative paths and twelve original stop segments, with frozen moving/stopped
residuals passing. The four slower pressure-matched actual-contact fixtures also
close negative: pre-release contact loss at 0.243–1.793°, followed by secondary
abort-release position refusals. Their original stops, raw data, RGB and cleanup
are independently verified. Saved original control operands expose pending FK
target distance credited against a measured gap even in explicit force-reference
mode. That distance may include retained elastic support rather than assured
closing travel; the credit is now zero only in that mode. Ordinary pose control,
all rates/extents and every qualification criterion remain unchanged. The change
passes essential unit and saved-operand checks, but the four paired actual-contact
fixtures remain negative: N 2.807°, H+ 1.230°, H− 1.442° and P 0.243°. Every
primary failure is pre-release contact loss; later withdrawal exits empirical
position support. P's final angle equals its baseline. Whole recordings and
sidecars, raw/native/model/current/future/physical checks, original stops, RGB
and owned cleanup are reviewed. This is a negative intervention, with no source
or matrix credit. Coupled arm/door dynamics must be reconciled before another
contact family; observed qualification remains pending.
See [[../topics/purdue-b1-robot-and-contact|the control contract]].

D2-RP independently qualifies the real72-command pause/313-command recovery.
D1-H- and D2-H- measure20.02mm and19.97mm lower hold heights respectively.
D2-DR confirms1.3x panel/handle mass and inertia with hinge damping6. The distinct
D2-H-/DR sources complete task-goal holding, preserving the original3s minimum,
opening requirements, release and stabilization.

D1-RV now independently qualifies its actual35nonzero injected commands and
403corrective commands. The measured14.47mm/0.03026rad handoff returns within
0.858mm/0.002824rad. Its24joint-brake intervals reduce peak joint speed from
0.22255 to0.01182rad/s; this is braking, not a zero-velocity or independent-stop
claim at0.4s. These qualified sources pass their complete native continuity,
2+4s independent stops, actual-image and GPU temporal reviews. Injected targets
remain excluded. Two reused references and fifteen new recordings fill seventeen
cells. The exact matrix cohort has 100,295 overlapping windows: 88,061 from new
recordings and 12,234 from reused references. The earlier 81,877 count covered
only the preceding fourteen new recordings. Broader selection/phase manifests
also retain historical sources and must be filtered to these seventeen IDs;
neither historical manifest totals nor overlapping windows count demonstrations.
D2-P also passes full native continuity, the unchanged3s hold, release,
stabilization,2+4s stops, actual RGB and GPU temporal preparation. Measured
base displacement is(-20,+25)mm with4deg yaw; initial shoulder/elbow deltas are
-0.05/+0.04rad and tool displacement47.63mm. Its6,045overlapping windows are
one new recording. Earlier placement failures remain negative.
Zero new optimizer updates; learned outcomes remain0/22. Development collection,
training and final test remain excluded.

The [[act-a1-complete-cycle-pause-diagnosis|pause diagnosis]] preserves the
October 27 same-door comparisons and unavailable measurements. The unchanged
D2-RP discriminator has now reproduced the same refusal at raw episode 152.0 s,
with synchronized force/feedback evidence and both independent stops passing.
That diagnostic adds no cell; the distinct pressure-v2 D2-RP candidate above is qualified. Manual finite jobs now use local, Blackwell and
elab2–5 under serialized device leases; the existing collection heartbeat is
ACTIVE and continues this same chat.

The retained pause-baseline negative full attempts comprise eight position/load-support refusals,
five perception/material/sweep refusals, two contact-load failures and one each of
prescribed geometric intersection, opening-progress exhaustion, contact-search
exhaustion, unresolved runtime stall, historical GPU contention and repaired
recorder failure. These are outcome mechanisms, not independent causal counts.
Passing components and stops never add a full cell. The first D2-H- coupled
bundle misses simultaneous gripper coverage and remains rejected. Subsequent
qualification covers the declared extrema and the actual new hold posture, with
all12cases/36stops and frozen residual/dynamic/feedback gates passing. The subsequent full source is now qualified. D2-DR's longer-budget attempt
reaches late hold but exits the empirical elbow domain; its independently
qualified successor holds at the unchanged required opening instead of seeking
maximum reach. Every preceding negative remains unchanged.

D3's lower opaque-material nominal path now passes nine robot-only cases and
27 independent stops, with zero frozen residual exceedances and all declared
moving/loaded-stop boundaries covered. The two earlier fixture negatives remain
retained: rapid integral windup and a stationary-position/reported-velocity bias.
Position-difference damping fixes the fixture without rewriting native velocity
or model/stop criteria. Only four empirical support sides are adopted at10N for
this path; no full Cartesian-box, other-condition or hardware claim follows.
The first full lower D3-N source then refuses ambiguous observed leaf ownership
before any approach command: the late selected semantic ID has only one real
finite-material witness. Independent stops pass; no source or labels are admitted.
Its original P placement still fails pedestal CAD and is excluded; passing the
robot component adds no matrix cell. A separate GPU replay of saved10Hz inputs
also yields only one independent material witness. It is a sensitivity study,
not an exact reproduction of native30Hz semantic history. Its first four views
have no mask, the21s view lacks finite two-finger support, and the useful18/25s
camera origins are too close for independent support. The subsequent native
stationary component retains those seven samples and adds three bounded head
orientations. Independent raw-joint/camera/mask review qualifies three same-ID
finite witnesses at18/28/31s and actual rigid-material ownership. Parked inspection,
healthy native feedback and2+4s stops pass. It ends at34.4s before any arm proposal;
zero source labels/windows are admitted. Calibration, identity and the three-view
criterion remain unchanged. The shared optional acquisition recipe now keeps
collection, provider and observer/IO timing aligned while preserving independent
workspace clocks; this component alone does not qualify full-path support.
The full follow-up retains three observed same-ID witnesses and admits24approach
commands. At34.8s it refuses the updated separation bound: normal-position growth
26.330mm and worst proved-sphere excess29.834mm exceed12.940mm remaining margin.
Independent raw/native-feedback and2+4s-stop review passes; zero labels/windows
are admitted. This joins D4's response-update mechanism without establishing a
shared algebra bug. The bounded diagnostic instead reaches its35.2s explicit
cutoff with a valid receipt; the earlier refusal is not reproduced. Its actual
admitted context and native stops are retained; no unchanged prefix retry is
used. A current-frame numeric study motivates stronger finite-triangle normal
certificates. Every geometric alternative keeps the entire observed core's
maximum point/model error; bounded-noise, clipped-support and collinear-refusal
checks pass. Both full follow-ups still refuse during approach: D3 at35.633s
and D4 at26.367s. Their raw/native feedback and2+4s stops independently pass;
zero labels/windows are admitted. Exact saved receipts reproduce normal-error
growth14.633mm against13.319mm on D3 and0.837mm against0.232mm on D4.
The largest growth belongs to distant body spheres with conservative remaining
whole-foreground clearance258.052mm and276.935mm, respectively. Across ALL saved
complete spheres, the same observed foreground bound and unchanged uncertainty
leave minimum residuals11.255mm and0.232mm. The global minimum and maximum growth
refer to different physical volumes. The producer now binds each sphere to its
original eroded separation margin and renewal checks every pair independently;
old receipts retain the global rule. No physical/calibration threshold or complete
error term changes. Saved-state algebra is not a full demonstration; guarded
native qualification of this shared correction is still required.

The following source62cd07d trials are also negative. D3 ends at34.4s before
any approach command with no explicit contact selection or P2P initialization.
The actual single semantic mask covers a partial wooden strip; all five measured
candidate widths are below the unchanged0.2m root criterion. This does not test
paired-margin renewal. RGB/mask/depth inspection and independent raw/native
feedback/2+4s-stop review pass. A distinct stationary component moves the actual
root50mm farther back and rotates yaw0.15rad, preserving posture, contact request,
neck trajectory, calibration and identity/finite-support criteria. Its actual
34.4s acquisition independently passes: three finite same-ID0 witnesses at18/28/31s,
actual native root/camera kinematics, parked-arm feedback, unchanged workspace,
visual review and2+4s stops. There are zero arm proposals, source labels or windows.
This qualifies acquisition only; complete path/load/contact support and the
related condition family remain unqualified.

D4 executes154approach commands and stops at27.9667s for unsupported movement/
stop support of the positive finger. Its last actual paired receipt independently
revalidates. At the refused complete sphere, eroded plane separation34.400mm is
below the observed whole-target front extent34.713mm. The nominal mapped center
already lies inside the conservative protrusion rectangle, before52.128mm of
full-pose budget and the complete radius/time expansion. Reducing pose error
alone cannot exclude that center from the current rectangle. Unknown/occupied
free-space support is a certificate state, not a native physical collision claim.
The earlier prefix refused the opposite finger in the same support family. All
raw/native feedback and2+4s stops pass; zero labels/windows remain admitted.
Suspend dependent unchanged full attempts while improving complete spatial
foreground/background support and independently qualifying any revised model.
No plane/model/physical/stop bound is reduced to close the0.313mm calculated gap.
A separate observed-only reconstruction uses the seven actual4/7/10/15/18/21/25s
inspection samples and original GPU robot filter. All voxel counts and every
original front-model bound reproduce exactly. A spatial union with the same
original global tangent reserve plus full voxel covering also contains the
refused nominal center. That representation does not solve the refusal; no new
spatial model or actor change is adopted.

A finite GPU discriminator compares48ideal paths: all four TRAIN doors and their
N/H+/H-/P setups, using maintained physical targets, targets inset by the unchanged
future physical travel, or targets inset within the main load rectangle. All4,299
controller-formula ticks per case are retained, including maintained6s approach,
ideal contact/opening through frozen goal+3degrees, hold and6s staged release.
An initial harness incorrectly used a direct linear return to closed precontact.
Its robot-only10N pilot completes one hold case, then exceeds the unchanged native
0.51rad/s limit at0.52215rad/s during release in case2; six dependent cases do not
execute. All33,448moving native samples and six stops are retained, with zero
frozen residual exceedances and independently passing stop travel/physical gates.
Allocation returncode0 does not override the negative report. That harness never
reproduced production release and qualifies no controller defect or calibration.
The corrected discriminator calls maintained `release_reference` and
`preceding_retreat_row`; all pre-release predicted q remain exactly unchanged.
First-half release goals move only normally with unchanged orientation, then
return to the latest achieved predicted pose at least5degrees earlier. The D2
approach crosscheck differs by less than0.72microradian from the prior maintained
integration. Physical-future targets remove predicted future-limit deficits;
main-load-interior targets lose original endpoint tracking in10of16conditions.
Neither result is a universal adopted fix. D3's new farther-root maintained paths
meet original endpoint ranges and predicted physical future boxes for all four
conditions; its220sampled pedestal/door CAD checks also pass the unchanged SAT
criterion. These are finite ideal geometry, with no real contact/force servo or
continuous collision proof. The new P placement independently passes native
converted-pedestal geometry, reset/visibility, actual root/joint kinematics,
visual inspection and2+4s unloaded stops. Measured relative floor delta remains
(+20,-25)mm/-4degrees and initial shoulder/elbow deltas remain-0.05/-0.04rad;
actual tool displacement is32.96mm. This adds no material identity, full motion
or contact qualification.

The first corrected native pose-controller fixture fails endpoint tracking and
loses its raw stream to a diagnostic argument-shadowing error. Its log and139.12
GPU s remain incomplete evidence. A subsequent single-case10N diagnostic saves
all15,996native samples and failure stops: opening-end error is129.68mm, already
115.21mm at the unchanged frozen47.77064degree task goal. Final commanded IK
error is9.89mm while actual-to-target displacement is125.61mm. Shoulder-X's
159.57mrad deflection times its original26.78Nm/rad stiffness matches the measured
4.2734Nm external torque; the wrist-Z command reaches its physical upper bound.
No actual hard-limit violation or production-release defect is inferred. All
frozen moving residuals and original2+4+removed2s failure stops pass. Constant10N
is the upper load/stop-model stress, distinct from the existing0.3N push target.

At constant0.3N, the same nominal actual pose controller independently passes the
complete robot-only path through frozen goal+3degrees, original endpoint ranges,
3s hold, actual staged achieved-pose release and all14moving CAD shape radial and
directional stop bounds. All18,156native samples respect the frozen residuals;
hold drift is35.93micrometers. Its actual future box still requires26.72mrad of
new10N wrist-X upper support beyond the prior scoped qualification. No such
extension is adopted. Actual contact, observation and variable force/admittance
remain absent, so this qualifies no complete source or10N model.

Related0.3N H+/H-/P paths retain all54,468native samples and nine loaded/held/removed
stops. Original endpoint, hold-drift, physical/speed/acceleration, stop-travel and
all14shape stop gates pass. H+/P have no frozen residual exceedances; their
required10N wrist-X extensions remain unqualified. H- has four shoulder-X
residual exceedances, maximum8.76mNm, at modeled50.7667degrees and the following
release. Its minimum native Jacobian singular value falls to0.00163; at the
frozen task-goal crossing it is0.06026 and the preceding opening has no residual
exceedances. The full stress test remains negative. Finite goal fields persisting
across case resets are treated as unavailable outside pose-command phases;
original raw data are preserved. The initial strict review's assertion refusal
and all GPU charges remain recorded.

The extra3degrees are an ideal planning stress, while the committed collector
already hands off at the frozen task opening. A distinct four-condition native
component using that original handoff independently passes all67,472native
samples, original endpoints, full3s holds, maintained6s staged releases toward
actual achieved prior poses and12loaded/held/removed stops. All frozen residuals,
physical/speed/acceleration limits, future physical boxes and all14moving CAD
shape radial/directional stop bounds pass. Hold drift ranges1.568–1.937mm within
the unchanged2.5mm range. H-'s minimum Jacobian singular value is0.06011; its
original full stress negative remains negative and no prefix is promoted.

These are robot-only constant0.3N paths with ideal geometric door references,
without observed scene contact, actual task force/admittance or full-source
qualification. A saved-geometry audit also verifies their complete8s stop tails
against each original latch, preserving all12segment reports and every original
bound. The full family requires a common10N wrist-X upper bound of1.228614rad,
109.684mrad beyond prior scoped support; other joint sides remain supported.

A13case/39segment robot-only calibration now independently qualifies that side:
one unloaded control and all four measured extremal postures under10N at tool
center and both distal faces/yaw±0.2rad. Existing position-difference damping
conditions all seven actual coordinates; production control is unchanged.
Native mass/impedance stability, frozen moving/stopped residuals, physical,
speed/acceleration, effort and all12loaded moving/loaded-stop boundary witnesses
pass. Radial/directional covers for all14moving shapes, joint/tool travel and
rotation also pass over each complete8s latched stop, including load removal.
Maximum tool travel from stop onset is5.84mm within the original10mm range.

The first shape review refused five removed-load continuations because it used
their fresh per-file measurement anchors. The actual stop target remains latched
once; the calibrated volume's axes belong to that original onset. All39stored
target arrays are verified identical to their case's first latch, and the full
tail is rechecked in that frame without a bound change or native rerun. The
original negative review and its charge remain. See the reusable
[[topics/purdue-b1-robot-and-contact|stop/continuation contract]].
Only the upper wrist-X empirical position support is adopted for these D3
original-task N/H+/H-/P paths and measured neighborhoods. Excess position/force
still refuses; all other calibration fields, physical/force/residual/dynamic/stop
bounds and the global recipe remain unchanged. This is no arbitrary joint-box,
other-family, hardware or demonstration qualification.

The distinct guarded D3-N full source uses this scoped support and the exact
qualified farther-root acquisition setup. Four finite same-ID witnesses at
15/18/28/31s and79approach commands retain observed ownership, but the complete
stopped negative-finger volume refuses at35.717s, about1.3s into approach. The
original eroded plane gap is19.558544mm against a19.558634mm whole-target front
extent. Observed protrusion tangent bounds are unavailable, so they cannot
certify an empty region or justify the smaller panel bound. Original measured
normal/pose/motion errors remain included. Healthy native feedback, raw integrity
and independent2+4s stops pass with no forbidden contact or physical violation;
zero labels/windows qualify. The complete negative and exact operands remain
retained; no unchanged dependent full source is launched. Required opening,3s
hold, actual contact, release/stabilization, observations and independent stops
remain mandatory.

A GPU audit reconstructs all ten actual D3 inspection RGB-D samples and exactly
reproduces the original42,212voxels,41,999connected voxels and front bounds.
The213disconnected returns include a59.101mm front feature; offline TRAIN CAD
intersection independently places the handle there at59.097mm relative to the
observed plane. These observations were present but discarded by the root's
connectivity filter. The rejected sphere's finite query instead contains1,488
connected returns behind that plane; neither count proves complete coverage.
The shared correction retains every nonrobot foreground return in the same finite
root extent as potentially attached, preserving all original padding, thresholds,
uncertainties and unavailable states. Numerical regression and same-input GPU
comparison pass. D3 now retains all213handle voxels and an87.877mm front envelope;
the original rejected sphere has19.059mm positive foreground margin after the
same full pose/motion errors establish154.547mm protrusion exclusion. This is a
single-volume diagnostic, not complete native admission. D4 retains175additional
foreground voxels and a79.791mm envelope; the same sphere still refuses with
-45.391mm foreground margin. Both original models reproduce exactly first.
The distinct D3 guarded approach/contact component then refuses at36.583s after
131approach commands, before contact. Raw/native integrity and independent2+4s
stops pass; physical violations and eligible labels/windows remain zero. Its
retained protrusion envelope now excludes the rejected ball by137.234mm, but
finite material support is-93.159mm after the unchanged pose/motion/stop errors
and5.3mm reserve. The eroded normal gap is0.322mm against the original0.5mm panel
bound. The current reference's positive material atlas covers mainly a lower
region; both other eligible roots in the exact saved publication still fail at
this query. This is unavailable observed support, not proof of physical collision.
The complete negative remains separate from full matrix attempts.

New stationary D3/D4 acquisition components test three additional intermediate-
pitch views before any arm proposal. Only actual neck targets/view times change;
mount, speed, scene, arm, load, mask/identity/metric and stopping criteria remain
fixed. Calibrated frustum planning includes the complete observed tool region in
all proposed views, with51–83pixel margins, without proving visibility or material
coverage. Actual RGB-D, finite same-ID witnesses, native physical/stop review and
material support near the fingers remain required before dependent motion.
The D3 intermediate-pitch component completes the controlled no-arm inspection
with passing raw/native, inspection limits and2+4s stops, but selected material
ownership remains ambiguous. Both eligible roots in its exact last publication
have zero finite static witnesses and zero positive dense-material clearance.
This profile is not qualified and no dependent movement follows. The original
negative, masks and394.473GPU s are retained. D4's acquisition now independently
qualifies three actual same-ID finite witnesses and the original stops.
No CAD geometry becomes an actor input or a runtime exclusion certificate.

A separate finite D4 GPU design compares six nominal base positions and four
heights, preserving N/H+/H-/P deltas and the measured fixed robot root height.
Original task-goal and+3degree stress trajectories run separately to avoid extra
held time caused by mixing goals in one batch. Maintained and physical-future-
interior controller targets give384predicted paths. CAD/pedestal, observed material,
actual force/contact, stabilization and native stop checks remain mandatory before
any candidate adoption; no physical/admission bound or task criterion changes.

D4's additional80finite GPU paths cover four contact heights and six initial
posture seeds, each with all N/H+/H-/P differences and maintained/physical-future-
interior targets. All72higher-contact paths fail original approach/contact
endpoint ranges, although later ideal opening may converge. Six tested posture
seeds at1.35m do not repair the approach. Exact finite TRAIN CAD checks find the
16tested height/variation footprints flat, inside material and unobstructed;
sloped/beveled normals do not explain these failures. Saved target-bound
clamping is retained per joint. These finite predictions prove neither native
feasibility nor global infeasibility; no setup, controller or model is adopted.
An additional paired32case reverse-path discriminator starts at retained achieved
ideal opening postures, traverses the saved geometric opening/contact goals
backward, then holds the original closed precontact target6s. All24higher-contact
endpoints still fail; all8baseline controls converge. Reversing this local branch
does not repair the approach; the higher position errors remain nearly unchanged
during the final second. The JSON summary fails on a NumPy boolean after all3,759prediction
steps are saved; the report is recovered from those arrays without a GPU rerun.
Original nonzero status, log and27.37GPU s remain charged. No native failure,
feasible production trajectory or global unreachability is inferred.
A48case static comparison then uses the maintained uncentered local IK solver
for the same16closed-precontact targets from three existing initial/achieved
postures. All12baseline solves converge; none of36higher solves meets original
endpoint ranges. This does not isolate centering as the cause or establish
unreachability; no higher-contact native attempt or controller change follows.

A separate D2 matched N/H+ planning discriminator uses the maintained6s approach:
360actual controller-formula steps, smoothstep/slerp reaching the fixed target
at70percent of the interval. All four15/20/25/30mm pairs meet endpoint tracking,
but none clears its predicted wrist future-stop box, even with predicted rather
than full-speed interval growth. Perfect target tracking remains an approximation;
contact, opening, hold, release, actual servo motion and stops are unqualified.
No posture/setup/calibration/source is adopted and no global infeasibility follows.

D4 fresh geometry at30Hz passes the prior raw27.683s sweep refusal, then stops
at27.833s when computed0.422mm growth exceeds the remaining0.194mm separation
margin. Raw integrity, feedback and2+4s stops pass; the source remains negative.
The controlled diagnostic instead admits153approach commands and refuses an
unsupported finger sweep at27.95s. It never encounters the earlier response
refusal, so the exact old receipt is not captured and no aging-credit defect is
established. Its raw feedback and2+4s stops independently pass; it adds no matrix
attempt, source or labels. The intended28.2s protocol-level cap was shadowed by
the selected episode's240s cap; the outer watchdog remained finite and the actual
refusal preceded28.2s. Subsequent diagnostic plans set the effective per-episode
budget and an explicit pre-proposal stop. Recorded RGB-D remains10Hz, so saved
input replay cannot establish exact30Hz history.

The D4 stationary review initially required tangential exclusion of the old
rejected volume. That was an incomplete reviewer criterion: the maintained
admission also permits separation from the whole observed front. A saved-operand
supplement evaluates both alternatives with every original normal/point/model,
interval, stop and material reserve retained. Whole-front separation is positive
10.493mm and complete material margin123.134mm, while tangential exclusion and
the offline spatial union still fail. The original reviewer and its negative
conclusion remain preserved with an explicit addendum; no production gate,
foreground model, calibration or bound changes. This necessary one-volume check
and qualified stationary acquisition justify one distinct guarded D4 approach/
contact component on elab4, stopped before the first opening action. Every actual
proposal still requires ordinary complete-volume admission; no source or full
motion is qualified. D3 ownership remains unsupported and dependent motion stays
suspended.

The distinct D4 observed component stops after134approach commands at36.6333s,
before contact. Raw integrity, all8,793native samples and independent2+4s stops
pass; no eligible source labels/windows. Its saved response reproduces six lost
margins among91original spheres. The worst original17.78micrometer margin cannot
pay759.41micrometers of fresh same-volume normal-error growth. This is not the
previous cross-volume aggregation defect. All30current surfaces are retained;
four independently valid constraints reproduce the selected minimum-finger-error
constraint, and none passes the same complete response gate.

Saved NumPy CAD/sweep reconstruction reproduces every original center/radius.
The six failing balls cover positive/negative finger cells; subdividing each
complete convex cell improves all child normal margins with the same interval,
physical, model and stop reserves. Child spheres do not all lie within the old
sphere, so its background certificate cannot simply be reused. The producer now
refines a positive cell when its box-to-ball plane-projection overreach exceeds
its proved margin, stopping at the existing minimum cell size. Every child takes
all original foreground/background/material/self/load/stop queries with the same
query ceiling. No acceptance tolerance or uncertainty decreases. Numerical
nominal, resolution-boundary and rigid-rotation checks pass.

The distinct refined-cover component stops at36.8833s after149approach commands,
before contact, with `unsupported_actual_robot_sweep_and_stop`. It retains
8,853healthy native samples, raw integrity, no observed force/forbidden-contact
violation and passing2+4s stops. Independent1,440sample review also verifies every
14shape radial/directional, joint, tool and rotation bound in the original6s
latch frame. At the unsupported negative-finger cell, the exact terminal ball
still loses1.61mm of whole-front separation after all original errors. Finite
material projection is positive, but the enlarged volume overlaps the conservative
relief envelope, so planar body order alone cannot prove exclusion. Saved current
RGB-D queries also remain unknown for both original radial and directional
volumes after all original normal/model/motion errors. Unknown is preserved;
no physical collision or admissible current-ray fallback is inferred. Actual
RGB at the terminal prefix was inspected. No complete contact/source, labels or
windows qualify, and dependent unchanged D4 motion stays suspended.

A separate saved-pose mathematical discriminator replaces only the loose
spherical support calculation with a coupled Jacobian enclosure and an analytical
suffix-length Taylor remainder. It keeps the same acceleration box, full stop
box/frame rotation, robot/jaw errors and every original normal/model/temporal/front
bound. All14complete foreground shape volumes pass at the original refusal.
A matching convex-background query retains all 16,384 generated vertices per
shape. It directly proves eleven shapes, including both fingers with a minimum
6.235 mm margin. The three shoulder roots also pass when each complete convex
volume is enclosed by a ball whose own foreground and historical-background
proofs independently pass. The maintained constructive background reader pays
all original calibration errors; shoulder background margins are 3.626, 2.390
and 1.193 mm. No convex margin is credited as free space for its larger sphere;
the finger outer spheres still fail their own foreground test and receive no
such fallback credit.

Independent nonlinear URDF arithmetic supplements the analytical Taylor bound:
five poses, six velocity patterns, three interval scales and acceleration-box
corners exercise all seven chain depths, every vertex of all 14 shapes and 12
projection directions. All 1,411,200 vertex checks pass across 12,600 constructed
FK states. The 1e-12 m arithmetic comparison tolerance is solely a roundoff check,
with no physical admission credit. The first convex reader's NumPy-boolean
serialization failure and all earlier unknown/negative diagnostics remain
preserved. These saved-state mathematical results preserve every original outcome,
calibration, mask and denominator.

The shared producer now implements this complete coupled-motion fallback. The
independent chain comparison also passes against the production kinematics
method. A saved D4 proposal replay uses actual GPU `PurdueFK`, the real
`PurdueIO.preview` and unchanged bounded A1 target selection with the captured
joint state, prior target and physical limits. All14root volumes pass at the
original36.8833s refusal: five use the coupled hull, three of those require
independently certified outer balls. The local separation summary is6.454mm and
the independent robot-sweep margin1.369mm. No physics step or command is delivered.
The same-estimate receipt revalidates. A separate causal reconstruction of the
older36.6167s producer and actual36.6333s response preserves the original receipt
failure, then passes with the new complete same-domain geometric margins; a
consistent increase in normal-source uncertainty still refuses. The old minimum
margin17.781micrometers becomes18.993mm for this earlier saved pose. Later observed
candidate surfaces are excluded from that reconstruction.

These software and saved-state checks establish a qualification candidate.
Complete native guarded approach/contact, full motion and related-condition
qualification remain required before any full D4 source. Existing physical,
observed-contact, uncertainty, material, identity, load and independent stop gates
remain unchanged. A convex proof never establishes free space for its larger
sphere; every former negative remains negative and no source/windows are added.

The subsequent guarded native D4 component still stops before contact after
166approach commands at37.1667s. All8,921native feedback samples are healthy,
raw integrity and physical command safety pass, and independent review verifies
the original continuous6s stop across all14shapes and every original bound.
Actual inspection/terminal RGB was reviewed. Complete transfer, owned process/
GPU/lease cleanup and all costs are retained. No component, source or window is
qualified; the earlier negatives remain unchanged.

At its terminal complete positive-finger leaf, the production coupled hull fails
whole-front separation by1.285mm before a background query can begin. The original
normal/model/temporal error is16.805mm and observed front extent35.677mm. Finite
material coverage and relative body order remain positive, while the separately
required relief exclusion remains unsupported. The original resolution is reached
without exhausting the query budget. More proof queries do not resolve this
foreground limitation.

A GPU reconstruction of all ten causal inspection views exactly reproduces every
original front-model count and bound. A spatial union retaining the same global
relief reserve plus voxel coverage also contains the query's convex mean center;
its projected exclusion still fails with all original point/motion/stop errors.
This model overlap establishes no physical collision or global infeasibility.
A common complete foreground/contact-region or acquisition/path design must
support the full motion and related conditions before another dependent native
attempt. Saved-pose enclosure improvements alone do not qualify contact.

The subsequent D3 guarded component on source `549ae0c` is also negative. It
reaches 36.583 s and 131 approach commands, with no contact or later phase.
All 8,781 native samples are healthy; delivered-command/admission/load clocks
match exactly. There are 131 admitted/delivered proposals, one terminal refusal
and 263 observed load packets, all inside the original 10 N model. The original
continuous 6 s stop passes for all 14 shapes, and actual RGB, whole-result transfer
and owned GPU/process/lease cleanup are verified. Its 619.400 GPU seconds and
unexecuted phases remain retained; no component, full source or window qualifies.

The exact negative-finger leaf reaches the original minimum resolution after
262 queries, below the unchanged 60,000-query ceiling. Its complete coupled
fallback has 32.875 mm raw normal distance, 2.239 mm motion/model radius,
20.623 mm target error and the original 83.633 mm whole-front extent: a
-73.620 mm foreground margin before any background query. This audits that
fallback, not every alternative geometric model, physical collision or global
infeasibility. Saved D3 snapshot admission and full nominal D2 compatibility do
not qualify native D3 transfer. Complete observed foreground/acquisition/path
support across the full motion and related conditions remains required; suspend
unchanged dependent attempts. The
[negative component review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D3-coupled-convex-guarded-approach-contact-blackwell-20261101/component-independent-review.json)
and [complete-domain audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-coupled-terminal-full-domain-audit-20261101.json)
retain the exact refusal. Two local review preparation errors (a removed lease key
and a copied remote frontend path) were corrected without native reruns, GPU work,
payload changes or scientific criterion changes; original readers/errors remain.

A complete saved-domain discriminator separates this D3 failure from D4. D3's
original uncertainty-enclosing sphere already excludes its original padded relief
by 95.359 mm. Its coupled fallback nevertheless used the global 83.633 mm front
everywhere. Reusing the independently justified 0.5 mm panel front leaves a
9.513 mm normal margin at this saved volume, with every original motion/model/
normal/stop error retained. The corresponding D4 sphere still overlaps the
original relief and retains its -1.285 mm whole-front result. All 14 root domains
and each terminal leaf are retained; complete convex tangent projection adds
no necessary new representation for D3.

The shared candidate therefore localizes the coupled front only after a complete
relief exclusion. It also retains that proof for both ordinary spherical and
coupled certificates and renews it from current full spatial error at dispatch,
for loaded and unloaded actions. Tangential pose changes or growth of geometry/
pose bounds can refuse even when normal bounds are unchanged. Existing physical,
material, identity, load, stopping and same-volume normal gates remain mandatory.
The 217 targeted numerical checks and the saved D3/D4 domain/uncertainty-overspend
checks pass. Actual GPU full-producer comparison against `549ae0c` now retains
its D3 refusal and admits all 14 complete volumes with candidate `68a5bcb`, leaving
5.816 mm whole-robot margin. Its 12 spatial domains revalidate at the next native
time with the same estimate, while full-point uncertainty overspend refuses.
Both D4 producers retain the unsupported-sweep refusal. This finite replay charges
20.297 GPU seconds and has verified owned process/GPU/lease cleanup. Fresh cells
and query caches plus an unavailable current rigid-target workspace are explicit;
no native tracking history or missing physical response is fabricated. A second
actual GPU comparison passes all seven original nominal phase/branch snapshots
and both receipts per state, with identical recorded cells and exact reconstruction
of every original initial-front value, count and clock. Its 65.905 GPU seconds
and owned cleanup are retained. These saved-state checks do not qualify native
whole motion or related conditions. H+ stays immutable; no new source or window
is admitted.

The distinct native `68a5bcb` approach/contact component stops after four approach
commands and no contact. Its actual response at 34.466667 s changes the complete
spatial-relief minimum from +0.556 to -17.448 mm; 24 of 36 saved exclusion domains
fail. The mean observed pose shifts 9.949 mm and 0.037797 rad while published pose
bounds grow. All 45 same-volume normal margins remain positive. This reproduces
`target_relief_exclusion_exhausted` from the actual admission and response, rather
than the older global-front-only terminal fallback. It establishes neither a
physical collision nor global infeasibility. Preserve current spatial renewal;
compare complete geometric domains and acquisition uncertainty before another
dependent native attempt, without threshold relaxation or point-specific rules.

All 8,273 native feedback samples are healthy. Four admitted proposals match four
delivered commands, with no undelivered admission; nine observed load packets
retain the original 10 N model. Original continuous 6 s/14-shape stops, actual six
RGB samples, whole 38-file transfer and owned process/GPU/lease cleanup pass.
The negative component costs 407.613 GPU seconds and adds zero source, cell or
window. Its [independent review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D3-localized-relief-guarded-approach-contact-blackwell-20261101/component-independent-review.json)
and [exact spatial-response audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D3-localized-relief-guarded-approach-contact-blackwell-20261101/exact-spatial-response-audit.json)
retain every domain, uncertainty primitive and original failure.

A subsequent complete-domain discriminator recovers every original proof cell
and all 589,824 motion/stop vertices. Full convex projection resolves 35 of the
36 domains; the elbow volume still lacks 7.729 mm when one whole-volume lever
budget is paired with its closest point. Its eight complete children instead
retain positive original-bound margins. Axis-aligned world boxes remain negative
and are preserved. Compact boxes aligned to the original physical link certify
all 36 complete domains with 52 spatial queries and no bound reduction.

The shared candidate retains these complete physical boxes alongside the original
sphere proofs and refines only when current sphere exclusion fails. An independent
CPU read verifies enclosure and every complete parent/child partition. The actual
saved response then passes all original receipt gates with 3.708 mm spatial
margin; uncertainty growth and a separately scoped tangential perturbation still
refuse. All 128 relevant numerical producer/contact/geometry checks pass. Actual
GPU full-producer and nominal-phase checks subsequently pass, while native
transfer remains negative; the original `68a5bcb` component stays negative. The
[complete compact-box discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-spatial-response-oriented-box-discriminator-20261101/review.json)
and [actual-response candidate review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-spatial-response-oriented-box-discriminator-20261101/candidate-actual-receipt-review-v2.json)
preserve the scope, original primitives and failed simpler representations.

Source `b322127` passes paired actual GPU full-producer/next-native-receipt checks
on the original D3 snapshot and all seven nominal full-phase states, while D4
retains its original unsupported-sweep refusal. These two diagnostics cost
89.439 GPU seconds and add no source. The distinct native D3 component then
refuses before contact after 90 approach commands: the actual 35.8833–35.9 s
spatial response changes its margin from +6.004 mm to -0.700 mm. All normal
margins remain positive. Its 8,617 native samples are healthy; all 90 admissions
were delivered, 181 load packets remain within the original model, and the full
6 s/14-shape stops, actual RGB, whole transfer and owned cleanup pass. The
492.303 GPU seconds and zero eligible labels/windows remain recorded. A further
complete-box support-axis projection, with all original errors and the same
5 mm resolution, still leaves unsupported domains after 544 queries. This
negative discriminator motivates no projection fallback or tolerance change.
The [native component review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D3-refined-spatial-guarded-approach-contact-blackwell-20261101/component-independent-review.json)
and [exact response discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D3-refined-spatial-guarded-approach-contact-blackwell-20261101/exact-terminal-spatial-projection-audit.json)
retain the unsupported state. Unchanged dependent D3/D4 attempts remain suspended
pending common complete observed uncertainty/acquisition/path support.

The [complete geometric discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/coupled-local-relief-complete-domain-discriminator-20261101/review.json)
and [shared-certificate check](../../../outputs/evidence/act-a1-complete-cycle-20261026/coupled-local-relief-complete-domain-discriminator-20261101/shared-certificate-saved-state-check.json)
retain every domain and original failure. An initial independent reader wrongly
required bitwise-equivalent budgets across two reference-lever transports; its
1.552/4.410 nm differences and failure remain preserved. The corrected audit
exactly recomputes the original producer's primitive arithmetic and independently
explains the published-lever difference, without modifying scientific criteria.

The separate D4 robot-only component independently passes all69,744native samples
of N/H+/H-/P through the frozen50.146446degree task opening, original3s hold and6s
achieved-pose staged release under constant0.3N diagnostic loading. It preserves
measured placement/arm deltas and the new stationary head pose. Hold drift is
1.578–1.984mm within the original2.5mm range. All12loaded2s/held4s/removed2s segments
use the original latch and pass every14shape/native/frozen-residual/physical/
dynamic/effort bound. Required10N support adds two posture sides:29.221mrad lower
elbow for H+ and78.969mrad upper gripper for H-. N/P need no position extension.

Eight additional independent native cases, including two unloaded controls and
center/two finite-face10N application at each required posture, now pass all
24continuous stop segments and actual moving/loaded-stop boundary witnesses.
Only those two empirical sides at those measured neighborhoods qualify; no global
Cartesian/joint domain, other posture, active calibration or source is adopted.
Two GPU review failures retain a cross-host calibration-path error and an unused
corrected recipe copy. The successful reviewer consumes the verified relocated
copy with identical calibration values; no native rerun or bound change occurs.

A separate D4 design retains384complete predicted task/stress paths across six
base positions, four nominal heights, four original variations and two existing
target-bound choices. Independent CUDA FK recomputes1,616,256samples and checks
original endpoints,3s hold drift, physical/future boxes and speed. Only31paths
pass; only the original1.09m/base interior-target family passes all four variations
in both studies. Its required load-domain extension remains unqualified.
The first reviewer failed on a SciPy vector-angle API shape before acceptance;
explicit scalar-axis input repairs the reviewer without changing any trajectory
or criterion, and both allocations/logs remain. A distinct96case reverse design
starts from every actual achieved held posture; all72higher cases fail, including
253.43mm best final precontact error. These finite negative designs establish no
global infeasibility. A broader243family static endpoint necessity screen varies
base position/yaw, contact fraction and height with four original perturbations
and17fixed numerical seeds. All33,048solutions are retained;749of1,944endpoints
meet original ranges, but no family passes all eight necessary endpoints. These
local solves confer no global infeasibility, whole-path, observed support, native
or source qualification.

The independent D2 design compares five common initial wrist-Z seeds for the
original nominal/H+ height pair, both physical-target and reserved-target modes,
and the exact42.028823degree task handoff. CUDA FK reviews all72,640predicted
states. All ten physical-target cases exceed the original future physical domain;
all ten reserved-target cases pass the necessary planned endpoints, hold, speed
and staged release. A distinct native test uses the original posture control.
Both cases independently pass30,976native samples at constant0.3N, full3s holds
with1.577/1.518mm drift, original6s achieved-pose release and six continuous
loaded/held/removed stop segments in the original8s latch, including all14shape
radial/directional and frozen-model checks. Nominal needs no position extension.
Against the canonical model, H+ needs measured lower-elbow43.183mrad and lower
RIGHT_GRIPPER_Y19.719mrad sides. The broader view used for the first motion review
already contained unrelated position halos; they do not establish transfer to
these postures. Each side now passes four independent cases/12segments against
the canonical model, with actual loaded moving and stopped boundary coverage,
all14shape checks and original8s latches. The scoped source recipe retains that
model and only these two measured position/load support additions.

The shared optional pose-command reserve keeps physical, A1 and admission/load/
stop limits unchanged. Source8e9cc9b passes exact GPU native-method parity on
all7,264saved control states in default and reserved modes. The first cold-JIT
verifier discrepancy remains preserved; matching initialization makes pose-error
inputs and every output bit exact without changing tolerance. A distinct full
D2-H+ TRAIN source on elab2 preserves the original posture/20mm variation,
previously qualified common pressure-v2 settings and original task goal. It
completes360approach,180contact and398push commands, then refuses at41.033333s
and6.187degrees when an actual10.726N native impulse gives a12.847N estimated
upper load, beyond the unchanged10N stop-model domain. All9,849native feedback
samples are healthy; physical80N/joint limits and1,440sample/all14shape6s stops
pass. Hold, release and stabilization remain unexecuted. No source/windows qualify.

Saved related D2-RP/DR/H-/P positives retain135one-shot gap recoveries; every one
occurs within the existing100micrometer contact tolerance. The new H+ negative
also has ten such recoveries, with a4.89micrometer preceding gap at the impulse.
This is a control hypothesis, not causal proof. A prospective shared controller
reserves geometric recovery for gaps beyond that same tolerance while retaining
ordinary positive force-error admittance. Numerical boundary and perturbation
checks are separate from full paired native contact/motion and transfer
qualification, which must precede another observed source. All original results,
force domains and physical criteria remain preserved.

Source `2cd8697` independently passes the complete native nominal/H+ pressure
pair. D2-N/H+ reach 42.938/42.979 degrees against 42.029 required; both retain
native contact through the full three-second hold, then finish six-second
release, stabilization and original six-second stops for all 14 shapes. All
48,584 task-native samples independently reproduce the frozen load estimates,
with upper peaks of 5.080/5.222 N and actual peaks of 4.891/5.020 N. Measured
hold-height variation is 19.981 mm against the requested 20 mm. Raw integrity,
physical limits and actual phase RGB review pass. Whole outputs and
616.718/635.825 GPU seconds are retained, with owned GPU/PGID/lease cleanup
verified.

The earlier H+ fixture on elab5 remains an interrupted outcome: the original
wall boundary stops it at 77.6 simulated seconds (28.365 degrees), before
hold/release/stabilization. Its 12,528 recorded task-native estimates, prefix
contact and original stops pass; whole output and 1,702.598 GPU seconds remain
preserved. Actual CPU frequencies near 800 MHz under native work are an
unexplained capacity anomaly, with no global host-policy changes. The once-only
migration to verified elab2 keeps source, fixture code, setup, load/geometry,
seeds, every physical/scientific criterion and the 1600/1800-second guards.
This correction of an inconclusive resource interruption does not erase it or
retry a conclusive physical negative unchanged.

The first guarded observed H+ source on `2cd8697` is interrupted by its6,000s
case wall limit at78.4333native s and28.692degrees, before holding/release/
stabilization. All18,825native feedback samples are healthy; every delivered
command remains physically safe and all available observed load bounds stay
inside the unchanged10N model, peaking at5.222N. The retained3,183admitted
proposals include one undelivered proposal after3,182actual guarded commands.
Independent continuous6s/14shape stops, actual RGB, whole transfer and owned
cleanup pass. This is an inconclusive resource interruption with zero eligible
labels/windows; it does not erase the earlier physical source failure.

Measured observed execution costs123–124wall seconds per native simulation
second. A distinct capacity successor preserves source `2cd8697`, the original
posture/20mm height variation, exactly two scoped canonical10N support sides,
controller gains, physical and observed gates,300sim guard and complete task
phases. Only the effective episode wall allowance becomes28,000s, with29,000s
wrapper protection, based on the original maximum whole-task phases and25%
compute headroom. The unchanged global defaults are distinct from this episode's
effective budget. The prior source is fully reviewed and released before this
single new allocation. It finishes positively in 14,255.723 GPU seconds, reaching
42.979 degrees against 42.029 required. Independent reconstruction verifies the
full 3 s loaded hold, 6 s staged release, stabilization, all 30,785 healthy native
samples, 6,172 guarded admissions/deliveries and 12,344 observed load packets.
Peak actual/estimated load is 5.020/5.222 N within the original 10 N model;
measured hold-height delta is 20.009 mm against 20 mm requested. Continuous
6 s stops pass every original radial/directional/joint/tool/rotation/speed/
acceleration/force gate over all 14 shapes. Actual phase RGB, the whole 45-file
payload plus sidecars and owned cleanup also pass.

Actual GPU temporal preparation costs another 44.443 seconds. The maintained
causal consumer, all six phases/boundaries, strictly past delivered-target
history, future action chunks, real tail masks and unchanged frozen normalizer
pass independent review. Source N15 contributes 6,184 H12/execute6 overlapping
windows as one new demonstration. This closes D2-H+ and all nine D2 conditions;
no learned-policy or other-door transfer follows. The
[independent full-source review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D2-Hplus-gap-tolerance-complete-capacity-elab2-20261101/independent-review.json)
and [qualified cohort admission](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D2-Hplus-gap-tolerance-complete-capacity-elab2-20261101/qualified-cohort-admission-review.json)
retain the preceding failure/interruption, every source ID and exact window scope.

## Observed nominal compatibility and producer cost

Source `549ae0c` completes one full guarded observed D2-N compatibility regression
with the canonical nominal load model and original physical, material, task and
stopping criteria. Independent raw reconstruction confirms 42.938 degrees against
42.029 required, continuous loaded contact and the full 3 s hold, 6 s staged
release, and ordinary 2 s stabilization. All 29,993 native feedback samples are
healthy; 5,974 delivered guarded commands match every admission clock and 11,948
observed load packets. Peak actual force is 4.891 N and the observed upper estimate
is 5.080 N within the original 10 N model. Independent continuous 6 s stops pass
all 14 shapes, including radial, directional, joint, tool, rotation, speed,
acceleration and force bounds. Eight actual phase RGB samples support qualitative
consistency; raw data establish continuity. All 59 remote result files (4.486 GB),
plus logs and accounting streams, are retained with actual GPU, process-group and
canonical lease cleanup verified. The first startup-unavailable result and a
corrected memory-stream transfer filename remain preserved.

[Independent native review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab3/D2-N-normal-refinement-complete-regression-v2-elab3-20261101/independent-review.json)
and [actual producer costs](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab3/D2-N-normal-refinement-complete-regression-v2-elab3-20261101/actual-producer-cost-review.json)
keep this regression of the already qualified nominal cell separate from the
36-condition collection. It adds zero sources/windows and does not qualify D3/D4
or the active H+ source. Actual proposal time totals 2,375.824 s; median push cost
is 0.378 s and 370 complete cover queries per command on elab3. Earlier H+ costs
use another condition/host; they cannot establish a matched collection speedup.
A separate paired RTX 4090 replay holds actual state and recorded cell cache fixed
at seven phase/branch snapshots. Both versions admit every snapshot and pass
receipt renewal at the next native tick. Complete query counts change from 2,004
to 370 at contact and 11,044 to 370 at the first push/hold/release states;
separated approach stays at 32. Original initial-front values, counts and clocks
reproduce exactly. Fresh workspace caches and one timed call per source/state
limit interpretation; these are no matched full-run throughput or stable timing
estimates. The first replay stopped on tuple/list JSON clock representation
before producer comparisons; its error and cost remain preserved. The corrected
reader keeps exact numerical equality. The
[combined review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D2-normal-refinement-complete-observed-qualification-20261101.json)
records both results and limits. No training, normalizer refit or learned success
follows.

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
| D1 | animated-door-1-88abf40 / right | Reuse C15 | Qualified new | Qualified new; pressure-v2 | Unqualified; observed robot sweep | Qualified new | Qualified new | Qualified new;0.9 | Qualified new complete recovery; task-goal hold | Qualified new complete recovery |
| D2 | door-2738468b94d74c5f / left | Reuse C14 | Unqualified; full observed source pending | Qualified new; task-goal hold | Qualified new; task-goal hold | Qualified new | Qualified new; task-goal hold | Qualified new;1.1 | Qualified new complete recovery | Qualified new complete recovery; pressure-v2 |
| D3 | psx-front-005-ee7d5c6 / right | Unqualified; updated response separation | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted; planned light 1.1 | Unexecuted | Unexecuted |
| D4 | psx-interior-wood-002-02e442c / left | Unqualified; observed sweep/response | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted | Unexecuted; planned light 0.9 | Unexecuted | Unexecuted |

The machine-readable [matrix](../../../outputs/evidence/act-a1-complete-cycle-20261026/matrix.json)
records planned and actual conditions, source/reference links, attempts, outcomes,
admitted scopes and gaps. Actual fields remain empty until measured. C14/C15 are
reused only within their saved qualified domains; they do not prove a replay of
the latest implementation. New attempt outcomes remain separate from the C03/C04 historical candidate dispositions.

Exact candidate setups are in the campaign's `setups/` directory. Nominal contact
height is 1.09 m; H+/H- use 1.11/1.07 m. P translates the fixed robot +20 mm in
world X on D1 and -20 mm on the feasible D2 placement, with +25 mm in Y
and +4 degrees yaw; D3/D4 use -25 mm in Y and
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

The October 31 resumption authorizes diagnosis, necessary corrections, qualification
and completion of the agreed conditions. Shared instability or failed independent
stopping blocks dependent execution. No unchanged retry-until-success is permitted.



## Resumed shared mechanisms and validation

Measured normal errors now propagate directly from their observed plane/model/FK
pivots to the finite fingers, without applying the same angular error through two
anchor legs. Current material-plane selection considers every root-consistent,
valid synchronized component and minimizes propagated contact uncertainty rather
than maximizing mask area. Equal best constraints remain ambiguous; identity,
pair residual and finite-geometry gates are unchanged.

The guarded pressure recipe now restores bounded positive correction when current
native force is below target while the qualification window still contains old
load. Legacy explicit recipes remain readable. No physical load, gain, rate,
contact-search, hold, opening or safety limit changed. These mechanisms are
implemented through `d538fb8` and await complete-cycle validation. Host deployment
failures and every resumed negative remain preserved with full costs. Resumed
complete candidates and nominal regressions are listed separately in
`outputs/evidence/act-a1-complete-cycle-20261026/resumed-full-attempts-20261031.json`.
They add no positive source before independent raw/stop and temporal review.

The first corrected D2-RP candidate still refuses observed load after 2.883 s
of the required 3 s hold. The prospective pressure-v2 recipe keeps the existing
0.1 N acquisition request through hold, instead of reducing it to 0.05 N before
intentional release. A saved-state GPU check also finds already-commanded normal
motion greater than the gap during D1 chatter; enabling the existing pending-motion
compensation avoids counting that correction twice. Neither change reduces a
load/admission threshold, required duration or physical stopping domain. Full
nominal and recovery validation is still pending.

A D3 matched observed-prefix replay restores initialization/tracking by choosing
automatic semantic identity after inspection. Explicit and already-selected IDs
remain pinned. D4's refused finger sphere has support from both static background
groups; foreground separation remains a separate proof, including finite relief,
pose and complete stop uncertainty.

A GPU CAD planning screen compares complete D3/D4 approach/opening/withdrawal paths
and the original height/placement/posture variations. This is planning evidence,
not observed admission, collision clearance, load/stop qualification or a new
positive source. Any selected placement still requires native whole-path support.

## D4 contact-region and error-budget discriminator

An unchanged-bound saved-state screen asks whether moving the contact region
could avoid the retained relief. Only three of 672 sampled nominal points have
positive conditional finite-material and projected-relief margins for N/H+/H−;
they lie near 1.37 m. The translated failed-domain template does not certify a
new posture, whole hand or complete motion. Earlier upper-contact studies already
retain approach failures, and the separate 243-family endpoint screen has no
complete N/H+/H−/P family. Those are finite studies, not global infeasibility.

At the original refused volume, the 65.994 mm point budget contains 19.464 mm
local-geometry and 46.504 mm dynamic-pose terms, plus 0.026 mm calibration error.
The nearest retained observed protrusion is only 45.970 mm away in projection,
inside the original 97.076 mm expanded query. Replacing the rectangle with
un-padded observed points therefore still fails this same projected exclusion.
It does not establish a physical collision or rule out a complete 3D model.
The [saved discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-projected-exclusion-and-contact-region-discriminator-20261101.json)
retains every bound and all outcomes. No changed contact setup or new dependent
native attempt follows from this screen; complete observed foreground/acquisition
and related full-motion support remain required.

## D1-P saved material-region discriminator

A CPU-only audit translates all four original conservative finger-query
rectangles over the complete saved observed-material grid, keeping the original
78.306 mm uncertainty and every excluded cell. It exactly reproduces the actual
-23.720 mm refusal. Of 40,401 lattice translations within +/-100 mm in each tangent
axis, 8,810 have positive saved material margin. All 201 pure vertical translations
remain negative; the best still lacks 8.720 mm. Raising contact alone therefore
does not repair this saved geometric query within the examined range. A separate
full-grid minimum-filter calculation exactly checks all 161,604 rectangle queries.

Regions farther sideways from the retained handle/relief merit complete-motion
design. The nearest positive lattice point moves 12 mm sideways and 23 mm upward,
but has only 0.042 mm saved margin and is not adopted. Translating the same
rectangles another +/-20 mm vertically leaves 5,170 positive proxy centers; these
are not actual varied conditions. This screen certifies no new reachable pose,
observation, changed uncertainty, robot/scene clearance, load support, continuous
path or native stop. No new native attempt, source or window follows. The
[complete outcomes and independent check](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-P-saved-material-translation-discriminator-20261101/review.json)
retain all nonpositive and unavailable states; the executable baseline is
`549ae0c` (documentation baseline `5215f34`).

An actual GPU whole-path discriminator then screens 20 common lateral/height
regions, preserving all original N/H+/H-/P floor, posture and +/-20 mm variations.
Both the frozen 41.008202-degree goal and a separate +3-degree stress are retained
under original and interior physical targets: 320 ideal paths, each with 4,110
intermediate ticks through approach, ideal contact/opening/hold, maintained staged
achieved-pose retreat/release and full 6 s stabilization. Every predicted physical
future interval, speed and pose-tracking range passes. Eleven common regions also
have positive saved material margins at all three translated height proxies.

No family fits the main empirical load-position rectangle throughout every path;
that is a support-qualification requirement, distinct from physical joint safety
or a proof that existing qualified halo/scoped/contact-free support is unavailable.
Perfect target tracking and translated old finger rectangles certify no observed
contact, pressure regulation, actual varied geometry, full native movement/load or
stops. No region/controller/calibration is adopted and no new source follows.
The [complete finite outcomes and independent review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-lateral-contact-complete-family-design-local-20261101/independent-review.json)
retain all alternatives and the ideal opening discretization. The allocation
costs 40.397 GPU seconds; owned process/GPU/lease cleanup passes. Preparation and
CPU reader errors remain preserved without extra native or GPU attempts.

A separate 160-path GPU screen restricts controller targets to the intersection
of the original physical and main calibrated position ranges, with the same
future reserve. It preserves all 20 regions, original four variations and both
frozen/extra-angle goals. All predicted future intervals fit both original
ranges; four common regions retain complete pose tracking, and only the common
+100 mm lateral/zero-height-shift family also passes every saved three-height
material proxy. Every failed tracking alternative remains. Its 29.266 GPU seconds
and owned cleanup are retained. This target choice still needs actual full
native motion, residual/load/stop support and observed material qualification;
no controller, region, calibration or source is adopted. The
[independent calibrated-target review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-lateral-contact-calibrated-target-family-design-local-20261101/independent-review.json)
retains all 160 outcomes and the perfect-tracking limitations.

A finite native component on elab3 then completes this common +100 mm lateral
region at the original frozen goal across all four robot-relative variations.
Independent GPU FK/load/geometry reconstruction verifies all 66,736 moving
native samples, original six-phase endpoints, physical and main calibrated
future-position ranges, zero torque-residual exceedances, achieved-pose retreat,
full 6 s stabilization and the original continuous 6 s/14-shape stops plus 2 s
removed-load stops. Native cost is 837.913 GPU seconds. Main position support
requires no extension. Whole transfer and owned cleanup pass.

The inherited reader also checks a whole-3 s tool bounding-box drift proxy:
H- measures 2.657 mm against 2.5 mm, and this refusal remains preserved. A
separate source-contract audit distinguishes it from the maintained final
0.5 s stability check, which passes for every condition at the unchanged 2.5 mm
threshold, with every original 3 s hold sample retained. Both reader charges
(12.089 s failed, 17.141 s complete) remain accounted. This corrects contract
attribution and qualifies no actual observed door hold. The constant 0.3 N
robot-only fixture changes no physical limits or load calibration; its common
native root does not qualify the P scene/floor/camera variation, actual material
contact, production region/target adoption or an observed source. The
[independent motion review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab3/D1-common-lateral-calibrated-target-native-family-elab3-20261101/independent-motion-review.json)
and [maintained hold-contract audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab3/D1-common-lateral-calibrated-target-native-family-elab3-20261101/independent-hold-contract-review.json)
retain both the extra proxy refusal and these qualification boundaries.

A distinct actual P scene probe on elab5 independently passes native converted-
pedestal/contact-footprint geometry, reset/visibility, actual RGB and the original
single-latch 6 s/14-shape unloaded stops. Measured floor variation is
(+20.0005,+25.0002) mm and +3.99992 degrees relative to the common N root;
shoulder/elbow deltas remain -0.05/+0.04 rad. All feedback, radial/directional/
rotation/joint/tool/speed/acceleration/force and frozen residual gates pass.
Whole transfer and cleanup pass; native/reviewer costs are 97.155/9.055 GPU
seconds. This qualifies parked placement only. The
[independent actual-scene review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab5/D1-common-lateral-P-parked-geometry-elab5-20261101/independent-review.json)
adds no material identity, full observed motion/contact, source or model adoption.

The first four stationary acquisition components remain separate initialization
outcomes. Three stop before any command because the prepared robot-only recipe
still names local runtime files absent on the remote hosts. The dependent P
component is suspended with SIGINT under its owned lease. A subsequent quality
review also dereferenced missing zero-command evaluator fields, masking the
original cause in its outer report. Original HDF5 outcomes recover that cause;
all partial files, interruptions and costs remain. The shared quality reader now
preserves zero-command initialization outcomes without fabricating phase support
or usable windows. Remote successors require all consumed geometry, load,
interface, model and workspace paths to pass host preflight; no physical or
observation criterion changes and no unchanged scientific retry follows.

After correcting recursive runtime paths and verifying the actual historical
background/workspace constructors, source `49cd6ca` independently qualifies all
four stationary N/H+/H-/P acquisitions. Each original 25.4 s inspection ends
before any arm proposal. The selected references retain 4/3/3/4 same-ID finite
witnesses, independently reproduced against actual raw camera origins, baselines
and surface observations with unchanged material/calibration limits. All 6,097
native samples per condition, parked-arm continuity, seven actual RGB views per
condition, original continuous 6 s/14-shape stops, whole transfers and owned
cleanup pass. Native costs are 196.528/326.700/322.863/826.061 GPU seconds. The
[complete acquisition-family review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-common-lateral-runtime-ready-acquisition-family-20261101.json)
qualifies stationary observed support only: zero arm proposals, source labels,
windows or new matrix cells. These inspections do not realize the requested hold
heights or qualify complete observed contact/motion.

The shared controller now optionally intersects its pose-target interval with an
explicit seven-joint domain, before applying the existing future reserve. This
exposes the same main calibrated position range used by the independently
reviewed robot-only candidate, while retaining original physical limits, default
construction, A1 joint-delta behavior, model/observation and stop criteria.
Invalid/empty or next-step-rate-unreachable target domains refuse. Numerical
boundary/default/invalid-domain checks pass. Actual CUDA parity on source
`ab5b944` covers 62,896 post-command native contexts across all four related
conditions: targets exactly match the earlier robot-only candidate, default
construction remains bit-identical, and no physical limit or rate bound changes.
This saved-state comparison does not replay native previews or physics.

The distinct guarded components on that source execute the original 360 approach
and 180 contact commands for N/H+/H-. All 540 delivered commands match admissions
and 1,080 load packets per run; peak model force bounds are 6.818/6.886/7.164 N
within the original 10 N domain. Raw feedback, measured phase endpoints, actual
RGB, original continuous 6 s/14-shape stops, whole transfers and owned cleanup
pass. Each intentional pre-push cutoff contains only 29 loaded valid samples,
spanning 0.467 s. The original 0.5 s sustained-contact gate is not yet established;
these are controlled partial executions, not qualified contact subtasks or full
sources. The maintained sustained-contact and pre-release-continuity predicates
remain false in their saved reviews. This cutoff is not a physical task failure.

P instead refuses at 27.983 s after 155 delivered approach commands, with one
refused undelivered proposal and no contact. An
[exact current-pose cover audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D1-common-lateral-P-calibrated-target-guarded-contact-blackwell-20261101/exact-current-material-cover-audit.json)
reproduces both finger clearances, 67.550/113.690 mm, and the original 94.735 mm
uncertainty budget: the limiting material margin is -27.185 mm. Robot sweep and
self margins remain positive at this refusal. Changing only the next target
cannot repair missing support at the already measured current pose. Raw rays
can see the limiting plane cell in several views; a separate saved stationary
P source audit shows why a visible center does not establish the full finite
normal-depth column and surrounding region. This is not proof of collision or
global path infeasibility, and does not justify reducing mask, geometry or error
reserves. All four component outcomes, original stops and 3,378.460 GPU seconds
remain in the
[family review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-calibrated-domain-guarded-contact-family-20261101.json).

A further [complete-path coverage discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-observed-full-path-material-coverage-audit-20261101/review.json)
queries all 3,931 saved robot-only control poses in all six phases for each of
N/H+/H-/P against that case's original stationary observed material union. It
restores each native fixture path to its original root and uses the known ideal
TRAIN hinge motion only to compare footprints in the closed reference frame.
P phase minima remain 47.596–51.671 mm, versus 246.586–250.463 mm across the other
three variants. Every query is available and exceeds the 36.004 mm sum of the
original fixed robot tracking, rotation, material-response and stopping terms.
Thus the material union is finite and nonempty; current pose/geometry uncertainty
must still fit its remaining margin. The 94.735 mm retained P budget is a design
stress comparison, not an assumed future bound. All six phases are retained,
including release/stabilization queries that do not replace the separate actual
unloaded-sweep proof. This supports investigation of general acquisition and
tracking conditioning across the full family; it proves neither future admission
nor physical infeasibility and changes no observation or acceptance criterion.

The full guarded common-region
[nominal regression](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D1-common-lateral-N-complete-observed-regression-elab2-20261101/complete-mechanism-qualification-review.json)
now independently qualifies the original complete observed cycle: 43.889 degrees
against 41.008 required, 3 s loaded hold, full 6 s release, 2 s stabilization,
3,694 guarded deliveries and 7,388 original-model load packets. The original
sustained contact begins at probe-relative 9.017 s, with no pre-release loss.
The complete original continuous 6 s/14-shape stop, actual six-phase RGB,
45-file whole transfer plus sidecars and owned cleanup pass. Its allocation
charges 5,926.536 GPU seconds. The earlier prefix remains a prefix; the selected
nominal source is unchanged.

The matched [H- regression](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab4/D1-common-lateral-Hminus-complete-observed-regression-elab4-20261101/complete-mechanism-qualification-review.json)
also independently passes the full original cycle, raw/admission/load clocks,
continuous stop, actual RGB and whole-transfer/cleanup checks: 43.731 degrees,
3,684 deliveries, 7,368 load packets and 5,428.857 GPU seconds. Its loaded-hold
height differs from completed N by -20.006 mm, retaining the original -20 mm
condition and all other setup/dynamics fields. Both native peaks and original
10 N model estimates remain within their original domains. The subsequent
[H+ full regression](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab3/D1-common-lateral-Hplus-complete-observed-regression-elab3-20261101/complete-mechanism-qualification-review.json)
also independently passes: 43.817-degree loaded holding, +19.987 mm actual height
relative to matched N, 3,693 deliveries/7,386 original load packets, original
continuous 6 s/14-shape stops, actual six-phase RGB and whole transfer/owned
cleanup. Its 11,202.895 GPU seconds are retained. All three complete height
regressions qualify this common-region mechanism; P remains unsupported. All
physical, observation, identity, error and stop gates remain.

These full cycles extend the diagnostic cutoff on already-qualified cells and
add no remaining-cell attempt, source or windows. Existing nominal regressions
and the two related height regressions have separate ledger cohorts; original
selected sources remain unchanged. The
[related regression assignments](../../../outputs/evidence/act-a1-complete-cycle-20261026/D1-common-lateral-related-full-regressions-current-20261101.json)
retain unique hosts, launch receipts and current ownership snapshots. P is not
retried. Complete related-motion qualification and improved general observed
support remain required before common-region production adoption or any full P
source.

## Shared task configuration and geometric admission cost

Eighteen of the nineteen missing conditions belong to D3/D4. Their next
intervention must jointly address contact region, robot placement/posture,
camera coverage, complete movement and changing observed uncertainty across
meaningful variations. The latest D3 response changes spatial relief margin from
+6.004 to -0.700 mm while normal margins remain positive; D4's current expanded
finger query still includes retained observed protrusions. These are failures of
observed support, not demonstrations of collision or global task infeasibility.
Further isolated saved-state fallback successes do not justify another unchanged
native attempt. All three D1 full height regressions are now closed; additional D1 view design
remains prepared and unlaunched while the shared D3/D4 problem takes priority.

Actual native D1 logs also identify persistent fine geometric partitions as a
repeated cost. The recent nominal and H+ calls retain 859 cells and check about
863 volumes; their medians are 1.068 and 2.001 s on different hosts and phases.
H- retains 286 cells at a 0.754 s median. This is a bottleneck observation, not a
controlled host benchmark. The shared traversal now coarsens only the next
partition after proving a finite nonseparated cell. It retains that fine witness,
all its octree siblings and the other complete shape roots. Every volume receives
fresh original foreground/background, physical motion, stopping and load proofs;
separation returning restores ordinary normal refinement. No clearance,
uncertainty, observation-validity or recording criterion changes.

The [isolated contracts](../../../outputs/evidence/act-a1-complete-cycle-20261026/partition-candidate-isolated-contract-review.json)
check disjoint complete box coverage, strict separation boundaries, return to
free motion and refusals for missing scene/material/load support. The first
fixture exceeded its own query budget and remains preserved separately; the
corrected nominal fixture is required to admit. A separate
[actual CUDA replay](../../../outputs/evidence/act-a1-complete-cycle-20261026/admission-partition-isolated-GPU-review-local-20261101/review.json)
compares 42 calls on seven finished native states covering all six task phases,
with original current-response receipts. All decisions and loaded branches pass.
Repeated same-state contact calls reduce 368 queries to 51, with about
0.38–0.50 s versus 0.10–0.24 s in these calls. The unchanged first calls still
reconstruct fresh workspaces. The 77.253 GPU seconds and owned cleanup are
recorded. This saved-state result is neither native transfer nor measured
whole-collection speed; a finite matched native cost comparison is still
required. No source, matrix cell or training window is added.

The [exact current-publication audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-exact-current-constraint-conditioning-local-20261101/review.json)
reconstructs both native selected plane/edge constraints within 1.8e-13 from the
actual stored RGB-D and Point2Pose packets. It evaluates every same-semantic
component with the original pairs, metric error and root-consistency checks.
The fingertip-normal minimum selects full finger point budgets of 143.63/65.15 mm
for D3/D4; other independently consistent fits give 74.96/43.55 mm and improve
the complete observed-root box bounds from 297.62/137.55 to 163.68/106.16 mm.
These alternatives retain all 22/20 measured pairs. Their different normal
metrics remain subject to original independent gates, and no old refusal is
relabelled positive.

The shared selector now ranks complete pose displacement over the entire initial
observed root, using the unchanged fit translation/rotation bounds and all
measured reference points. The current tool no longer defines the selection
objective. Finite material, full point/normal uncertainty, identity, source clocks,
scene, physical and stop criteria remain unchanged. The isolated saved-publication
selection and numerical nominal/boundary/perturbation/missing/ambiguity checks pass.
This is software qualification only. The complete-path material and placement
work remains necessary before another dependent D3/D4 attempt; a better saved
point budget does not establish complete observed execution or a qualified source.

A finite [joint task design](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-joint-task-configuration-full-family-design-local-20261101/independent-kinematic-readback-review.json)
retains 36 coupled contact-region/lateral-placement/wrist-posture families and all
288 original-goal/separate-stress N/H+/H-/P paths. Independent readback checks
all 4,659 knots and six phases per path. Eight families retain original-goal
kinematic tracking, speed and future physical limits for all four variations;
the higher-contact candidates in this screen lose tracking. Extra 3-degree
stress remains separate and does not redefine the original task. These are
perfect-target TRAIN design results, not observed movement or load/stop support.

The [full-path material and camera sensitivity](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-full-family-exact-finger-design-sensitivity-local-20261101/review.json)
then checks all 64 paths of those eight families, using the original exact
finite-finger rectangle queries and retained holes. Some D3 rectangles meet
zero-clearance cells in the frozen nominal atlas. D4's unchanged-region contact/
opening/hold minimum is 147.59 mm, with a 42.50 mm margin under its retained
current-error stress; the tested +100 mm lateral change instead loses that stress
margin. Three final-neck models keep the complete physical finger cube inside
the sensor field of view throughout these ideal paths, while arm occlusion,
actual masks and future tracking remain unmeasured. The old measured nominal
atlas and one saved current-error publication do not establish actual varied
acquisition or future admission. Thus neither a generic lateral shift nor field
of view alone resolves the shared blocker. Region/placement/posture, finite
observed support and complete-pose conditioning must be considered together.
The 41.405/8.045/8.043 GPU seconds and original coarse-ball discriminator are
retained without source or window additions.

The [paired whole-root control](../../../outputs/evidence/act-a1-complete-cycle-20261026/shared-material-and-conditioning-independent-review-20261102.json)
checks the same 64 complete paths under `e1b7653`, retaining all original finite-
finger clearance arrays exactly. With the unchanged ten-view atlas and one
saved nominal current-error publication, D4's unchanged region retains a 65.58 mm
minimum stress margin across original N/H+/H-/P contact/opening/hold paths. The
+100 mm lateral family remains negative. D3 still has zero-clearance footprints.
These frozen-publication sensitivities do not qualify varied observations,
future tracker errors, whole-robot foreground/load/stops or native movement.
Both finite CUDA allocations close cleanly and retain 16.089 GPU seconds.

A [measured component audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-material-component-mechanism-review-local-20261102/review.json)
isolates an additional D3 material bottleneck. Its original ten-view atlas has
only 21,589 positive cells and 24.59 mm maximum clearance. In the exact current
RGB-D capture, the material rule selects a 42,911-pixel side component by its
38,936 coplanar seed pixels, discarding a larger 53,857-pixel component with
10,240 independently measured coplanar seed pixels. D4 instead retains its main
163,682-pixel component. This identifies discarded measured support; it does not
make unobserved holes, glass/depth discontinuities or adjacent material known.

An [isolated component-union candidate](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-current-material-seed-union-isolated-review-local-20261102/review.json)
requires the original minimum coplanar seed independently for each component,
with the same semantic identity, depth graph, root bounds and missing cells.
An initial stale-depth seed acceptance is retained as a failed candidate; direct
same-capture coplanarity fixes that defect. The corrected copy preserves the
numerical hole/background/identity/boundary contracts, but its enlarged global
normal-depth slab loses 13,127 previously positive D3 cells and its complete-path
sensitivity remains negative. It is **not adopted**. Independent component
projection must preserve valid original cells and all actual depth gaps before
any production change. The retained D3 stationary operand package already holds
13 actual views; its final neck differs from the later 10-view component, so
inspect those recorded inputs and their clocks before collecting more data.

The production replacement projects each independently seeded component with its
own complete normal-depth column, then takes the maximum of independently
certified cell-clearance lower bounds. The
[positive native-input controls](../../../outputs/evidence/act-a1-complete-cycle-20261026/material-componentwise-positive-native-controls-production-readback-20261102/review.json)
preserve every original D3 current-cell bound while increasing positive cells
from 19,219 to 80,650. All 348,727 cells of the varied D4-P ten-view atlas and all
support clocks remain unchanged. Production exactly reproduces the isolated
component-wise candidate. The separate original D3 thirteen-view reconstruction
still has zero positive cells before and after: this is a retained negative,
not material qualification or an improvement. Its different final neck/root
cannot inherit the later positive current capture. This shared software change
qualifies no native complete movement or source.

The [D4 stationary family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-whole-root-stationary-family-20261102.json)
independently qualifies actual N on elab4 and original P placement/posture on
elab2, using immutable `e1b7653`. All ten original views and 34.4 s are retained;
six/five same-identity finite witnesses, camera/FK, parked feedback, original
continuous 6 s/14-shape stops, actual RGB and whole transfer/owned cleanup pass.
Both end before any arm proposal and add no full-source attempt, matrix cell or
windows. Stationary H+/H- would duplicate the nominal physical state; target-height
variation still requires actual guarded movement.

A [full producer replay](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-whole-root-full-producer-paired-saved-replay-local-20261102/review.json)
on D4's exact old refusal packet now passes the complete unloaded producer with
whole-root selection, including original scene, preview, load and stop checks.
It uses an ordinary per-volume normal-separation receipt, renewed at the next
native tick; exhausting its own remaining normal-error capacity still refuses.
The original fit remains negative. Two reader mistakes are preserved separately:
perturbing an unrelated geometry error and omitting the original prepaid interval
motion reserve. The corrected independent receipt review resolves neither future
observations nor native motion.

The [main-domain complete-path design](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-main-domain-task-family-design-local-20261102/independent-kinematic-readback-review.json)
restricts controller targets to unchanged main calibrated and physical joint
limits, reserving original braking, acceleration and command travel. It retains
all 36 families, 288 paths and six phases. No family meets the original pose/speed
criteria across all N/H+/H-/P variants: D3 remains negative; D4's unchanged-region
N/H-/P paths pass, while H+ reaches about 17 mm against the original 10 mm pose
tolerance. This identifies a remaining configuration/load-domain tradeoff;
neither pose tolerances nor measured calibration support are enlarged. Actual
N full movement may be qualified separately; the negative H+ prediction supplies
no permission to repeat that dependent condition unchanged.

The [reserved-target finite-material readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-main-domain-whole-root-original-atlas-full-path-local-20261102/independent-array-readback-review.json)
checks all 64 related paths, including the retained negative tracking variants.
D4 N's 4,659 projected finite-finger queries retain positive frozen-current-error
stress throughout all six phases; minimum 10.19 mm. This does not predict future
semantic masks, forces, contact dynamics or tracker uncertainty. One
[complete D4 N candidate](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-main-domain-complete-N-20261102.json)
uses immutable `e1b7653`, the unchanged original setup and ten views, and the
already supported main-calibration target reservation. It retains the original
50.146-degree goal and all full-cycle/observed/load/stop requirements, but
[the actual full attempt](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab4/D4-N-whole-root-main-domain-complete-elab4-20261102/complete-negative-review.json)
refuses at 36.05 s after 99 approach deliveries and one refused proposal. The
negative-finger shape 11 contains an unknown sweep/stop cell, with a 0.387445-rad
interface-normal cone and 0.8315 mm non-distal ordering margin. Native feedback,
raw clocks, 199 original load packets, original continuous 6 s/14-shape stops,
actual RGB and whole transfer/owned cleanup pass. This is conclusive negative
remaining-cell launch 41, charged 908.737 GPU seconds, with no eligible windows.
The whole 100-proposal timeline and exact final operands are retained. The
previous saved packet's success does not resolve evolving actual support;
further dependent D4 attempts are suspended rather than retried unchanged.

The [existing-support D3 model discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-existing-10N-support-complete-family-design-local-20261102/independent-kinematic-readback-review.json)
retains all 18 families/144 full paths and 4,516 knots per path. Using the already
adopted, scoped 10 N position sides rather than the main rectangle alone, the
unchanged-region family retains original N/H+/H-/P pose, speed and predicted
physical/future-envelope limits. Separate extra-3-degree stress still fails.
This numeric envelope does not enlarge the original measured posture/path
qualification or establish native contact/load/stops. The original main-domain
negative and a producer legacy source-label/reader arithmetic-order error remain
preserved; actual executable `a18bcd3` is verified independently.

Two [matched refined-view D3 acquisitions](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-componentwise-stationary-family-20261102.json)
now independently qualify actual N on Blackwell and original P on elab2, using
`a18bcd3`, all ten original views and no arm proposal. Three/five finite independent
witnesses, camera/FK, original continuous 6 s/14-shape stops, actual RGB and whole
transfer/owned cleanup pass. N has 74,329 positive material cells from ten views;
P has 64,389 from eight contributing views, preserving its two missing semantic
views and all ten actual captures. These different native captures are not an
exact causal cell-count comparison with the old atlas. The old thirteen-view
negative remains unchanged.

The [actual-reference full-path audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-componentwise-actual-N-P-complete-path-material-local-20261102/independent-array-readback-review.json)
then retains all eight original/separate-stress paths of that family. Original
N/H+/H- projected finite-finger queries cross zero-clearance measured cells;
P's exact finger query remains unavailable under its selected observed reference.
The complete current-error stress remains negative. Thus larger stationary
atlases, independent witnesses and predicted kinematic feasibility still do not
qualify this contact region or future movement. No new dependent D3 source is
launched. The next intervention must jointly address actual measured material,
camera/reference conditioning, contact region and feasible full placement/posture,
with original physical, uncertainty, observation and stopping criteria intact.

The [exact finite-finger failure decomposition](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-complete-material-failure-diagnosis-local-20261102/independent-array-readback-review.json)
reproduces all eight complete-path arrays. P is outside its selected reference's
finite lattice throughout every phase; its original normal-plane check passes.
N/H+/H- remain finite but cross zero-clearance cells. Missing P support therefore
cannot be described as occupied space or a failed plane test. A deliberately
permissive outer-extent calculation, discarding holes, uncertainty, kinematic and
load constraints, still finds no common translation of the unchanged loaded
paths across original N/H+/H-/P. Its 19.050 mm minimum necessary per-constraint
extent deficit is diagnostic, not an allowed tolerance increase. This only
rules out translating these frozen-reference paths; it does not establish
physical impossibility of a new measured configuration. The separately retained
extra-three-degree stress also fails. The complete-error scalar lower floor
alone does **not** exclude every other atlas point.

The [twenty-view conditioning readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-actual-ten-view-reference-conditioning-local-20261102/independent-geometric-readback-review.json)
retains every original mask, depth and capture/availability clock. P's initial
reference contributes 0.157267 rad versus N's 0.050973 rad, before the current
plane/edge term. P's selected identity is absent at 4 and 7 s. Static registered
world geometry can compare the measured camera views, but native Point2Pose
correspondences and early current-pose publications are unavailable; later root
geometry never refreshes an earlier support clock. The actual P seed is acquired
at 34.066667 s, distinct from the 34 s inspection capture.

All eight saved P roots meeting the original size minima are preserved in the
[causal-prefix reference diagnostic](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-P-measured-reference-prefix-design-local-20261102/independent-prefix-readback-review.json).
Five fail the original edge-conditioning check and one lacks the original
interior reference core. The other two, both acquired at 21 s, reconstruct zero
positive material cells using only their six actual available prefix views.
Their original and extra-stress complete paths remain negative or unavailable.
Better angle conditioning at an earlier camera pose therefore supplies no
qualified replacement seed. Native initialization, ownership, current error,
changed-path kinematics and full observed movement remain unqualified. These
saved-data studies add no source, window or matrix attempt; the one actual CUDA
FK diagnostic charges 7.034 GPU seconds with owned cleanup verified. The next
intervention must obtain a common measured reference/material region while
co-designing camera, contact region and feasible full posture, with every original
safety and observation criterion retained.

The [local normal-depth-column correction](../../../outputs/evidence/act-a1-complete-cycle-20261026/local-column-material-production-readback-20261102/review.json)
then preserves every native and same-version cell bound across seven actual
D3/D4 contexts. Complete measured depth bands within each independently seeded
component add coverage while retaining the original parent certificate, all
unknown gaps, full error primitives and original support clocks. D3 N/P coverage
becomes 143,088/66,036 positive cells; D4 N/P becomes 335,252/360,259. The old
thirteen-view D3 and both earlier P-prefix atlases remain empty. Two diagnostic
reader failures are retained: comparing a pure geometric producer with its later
native publication clock, and comparing old `e1b7653` metadata directly with the
current `a18bcd3` producer. Correct same-stage/version controls pass without
changing original native values or criteria.

The [complete-path CUDA readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-local-columns-complete-path-material-local-20261102/independent-full-path-readback-review.json)
keeps all eight paths and 48 phase outcomes, with bitwise unchanged FK, finite
query domains and original error arrays. N/H+/H- still cross zero-clearance
cells at the planned contact region, and P remains outside its selected reference
extent. No phase gains a positively supported full-error query; P's unknowns remain
unavailable. Thus
this software coverage improvement does not qualify the planned physical task,
a new source or a matrix cell; unchanged dependent attempts remain suspended.
A separate unadopted complete-column-midpoint projection probe recovers only 194
cells on one earlier P reference, with maximum clearance 1.586 mm, below even the
original fixed response-plus-stop floor after the maximum existing scene credit.
It is retained as a limited geometric result, without native or finite-contact
qualification. Actual shared reference/camera/region/posture co-design remains
required.

The [paired measured-region readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-measured-common-contact-regions-v2-local-20261102/independent-region-readback-review.json)
examines the finite prepared TRAIN panels on a declared 20 mm world-y/z lattice,
using both full finger footprints and the original N/H+20mm/H-20mm/P captures.
D3 has no common available footprint among its 3,705 positions. D4 has 850 common
available positions, 643 with positive geometric clearance and 237 retaining
positive complete frozen-current-error stress among 3,132 positions. Projection
to each measured reference is an explicit design approximation, with original
normal displacement retained; this supplies neither future observation nor
normal/scene admission. Original gaps, unavailable states and support clocks
remain unchanged. Independent manual corner/grid-slice controls and actual RGB
inspection support this finite diagnosis; it is not a global impossibility proof.

The [complete region/placement/posture design](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-contact-placement-posture-design-summary-20261102.json)
uses six separated D4 measured regions, three base-X offsets and three yaw offsets,
with a coupled lateral shift. All 54 families preserve original N/H+/H-/P and
separate extra-three-degree stress. Three factors retain 1,296 complete predicted
paths, each with 4,659 knots and all six phases: main calibrated target interiors,
physical target interiors with every empirical-domain exceedance retained, and
a common initial arm posture within the intersection of all original named
variation domains. None supplies a family satisfying all four original complete
pose/speed/physical requirements. New physical targets do not adopt any larger
force-qualified domain. The model uses ideal target tracking and the maintained
controller/release schedule; actual reset, stand-off, whole-body motion, evolving
observations, forces and independent stops remain unqualified.

The first unprojected common-posture batch correctly refuses before its first
predicted command because the original P perturbation leaves the unchanged
physical target interior. The corrected common posture keeps every original P
delta and height variation rather than clipping conditions independently. That
failed allocation and all three completed factors charge 162.574 GPU seconds,
with owned cleanup verified. Setup and reader-tail errors remain in their
records. No native attempt, source, window or matrix cell is added. The next
coupled design must consider independent lateral placement, arm branch and
camera/reference conditioning across the whole task before any physical or
collection adoption; the same failed D3/D4 sources are not retried.

The [independent lateral full-phase readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-regions-independent-lateral-full-design-local-20261102/independent-physical-kinematic-readback-review.json)
adds 108 genuinely new common-base families, retaining the old zero-offset
negatives without rerunning them. No family passes all four original complete
paths. Exact saved-array decomposition identifies a folded lower-elbow bound
in H+/P approach/contact and a lower-wrist bound on some releases; this is a
local mechanism diagnosis, not permission to extend physical limits.

The [rearward whole-task readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-regions-rearward-whole-task-design-local-20261102/independent-physical-kinematic-readback-review.json)
then retains another 108 common families. Two satisfy original N/H+/H-/P goals
across all six modeled phases, using common -100/-140 mm base-X shifts, the
original named perturbations and shared arm posture. Separate extra-three-degree
stress remains negative. Sixteen further lower measured-region families remain
negative, with a reader's inherited-parent-metadata failure preserved and its
new outcomes recomputed from unchanged arrays. All empirical 10 N position
exceedances stay unadopted. These three factors retain 1,856 complete model paths,
4,659 knots each, without new native task attempts or source/window admission.

The [calibrated camera readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-rearward-common-camera-whole-task-design-local-20261102/independent-calibrated-camera-readback-review.json)
retains both original and stress paths and 30 fixed neck poses. CUDA FK/projection
and an independent full-array projection show that the original (-0.7, 0.25) rad
head pose keeps both complete physical fingers in view across all four original
paths. The entire old multi-view root bounding box never fits one frame; that
stricter geometric probe is retained, and is not an original native admission
gate. No unseen mask, occlusion, correspondence, uncertainty bound or support
clock is created by projection. The model/camera allocations charge 207.055 GPU
seconds with owned cleanup verified.

The [new actual N/P stationary pair](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-rearward-design-stationary-family-20261102.json)
completes on elab4/elab2 using source `9df3784` and the selected -140 mm family.
Both original 34.4 s inspections end before any arm proposal. Independent reset,
parked feedback, raw camera origins, same-ID finite witnesses (eight N, six P),
all ten actual RGB-D views, original continuous 6 s/14-shape stops and whole
33-file plus three-sidecar transfer/owned cleanup pass. The same head, view
schedule, calibration and P perturbations remain. Parked N/H states coincide;
height-motion qualification is still separate. These are stationary measurements,
with zero complete-source attempts, matrix cells or eligible windows.

The [actual-reference full-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-rearward-actual-N-P-complete-path-material-local-20261102/independent-array-readback-review.json)
checks all eight original/stress paths, 4,659 knots and 48 phase reports against
the newly measured covers. N's former 1.155 m target crosses finite zero-clearance
cells; some approach and H- queries are unavailable. P's projected contact,
opening, hold and release retain positive fixed-current-error margins. Its
approach diagnostic is partly negative; projecting a distant approach onto the
panel is not itself a native material-admission rule. Frozen geometry/current
packets establish no future scene, mask or force support. Extra-angle kinematic
failures and every unadopted 10 N position exceedance remain.

A [new complete measured-region lattice](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-rearward-actual-measured-common-contact-regions-local-20261102/independent-region-readback-review.json)
uses these actual N/P covers and preserves the original height variations.
Of 3,132 finite panel candidates, 504 have four available footprints, 378 have
positive geometric margins and 103 also pass the fixed-current-error probe.
The independent corner/grid reader retains holes, unavailable values, measured
support clocks and every original error operand. Six spatially separated upper
regions are tested together with the same physical initial base/posture/head.
Their [48 complete paths](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-rearward-actual-upper-regions-full-design-local-20261102/independent-physical-kinematic-readback-review.json)
all fail the original complete pose criteria. Full-phase bound decomposition
finds late-opening lower-wrist-X and upper-wrist-Z limits; physical limits are
preserved. The saved material/model allocations charge 31.250 GPU seconds with
owned cleanup. This remains a coupled placement/contact/posture problem, not
permission to retry the failed full source or extend force domains.

The [coupled upper-region placement readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-actual-upper-regions-coupled-placement-full-design-local-20261102/independent-physical-kinematic-readback-review.json)
then tests 108 new common-base families: all six measured regions, base-Y following
the contact shift plus independent offsets, both rearward X positions and three
yaws. All 864 complete paths retain the original variation, phase, physical,
pose and speed criteria. No original four-condition family passes; this factor
charges 83.852 GPU seconds with owned cleanup. The old failed points are not
rerun, and moving the base does not transfer the previous camera masks or
current support clocks. A different common arm branch/posture still needs
complete design and fresh observed/native qualification before any source.

A [common-arm branch audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-actual-upper-regions-common-IK-branches-local-20261102/independent-common-IK-readback-review.json)
retains 288 structured IK seeds across all six measured regions and two common
placements per region, ranked by the worst complete pose violation across the
four original conditions. Forty-eight nominal precontact solutions converge;
240 failures remain. The original P offsets fit the shared physical future-target
intersection. Mathematical solver iterations do not extend a task phase.
All [384 subsequent complete paths](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-actual-upper-regions-common-branch-fullpaths-local-20261102/independent-physical-kinematic-readback-review.json)
retain those offsets and the original criteria, yet no common family passes.
Both allocations charge 55.505 GPU seconds with owned cleanup. A reader's mixed
precision rotation comparison is preserved; corrected cosine arithmetic changes
no original convergence flag or physical gate.

The [calibrated actual-view diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-actual-original-contact-visibility-diagnosis-local-20261102/independent-calibrated-visibility-readback-review.json)
checks all twenty camera transforms against raw joints and the fixed calibration.
The original contact center is in view with valid same-ID depth in seven views
of each placement. Their selected reference planes differ by 12.631 mm at that
point. A visible center establishes neither the complete finger footprint nor
coverage across the neighboring relief and finite reference perimeter. The next
reference audit preserves those original perimeters, graph gaps, seed minimum,
semantic masks and causal publication cutoffs; no reference or support clock
is retrospectively adopted. Complete tracking, observed material and loaded
native motion remain open together.

The [alternative measured-reference readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-actual-N-P-alternative-measured-reference-audit-local-20261102/independent-reference-geometry-readback-review.json)
then checks all 31 retained parallel, sufficiently large N/P surfaces. Original
seed eligibility admits 14/15 N and 10/16 P candidates; every rejection remains.
Seven N references retain positive complete finger footprints at all three
original heights, and five P references, including its selected control, retain
positive nominal footprints. All 52 available queries are independently checked;
both original control grids and every native cell match exactly. Original
perimeters, graph gaps, masks, seed minimum, depth errors and each surface's
causal capture/publication prefix remain unchanged. The CPU-only audit closes
with owned cleanup and zero GPU charge. These geometric alternatives require
prospective common reference selection and fresh tracking, full-root pose/current
uncertainty, independent witnesses and native motion/load/stops. No later
positive result reinitializes an earlier reference or qualifies a source.


The [current-seed discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/conditioned-reference-selection-actual-D3-D4-discriminator-20261102.json)
finds an area-ranking sensitivity between D4 N and P. The explicit optional
`best_conditioned_surface` recipe chooses the unique smallest existing complete
measured basis bound among the same eligible current roots, with the same exclusive
interior-pixel rule. It does not change either existing area-selection mode. The
original packet replay controls reproduce every checked pose, witness, geometry
and error field exactly. Reusing the same saved whole-leaf native map only as a
counterfactual, N and P retain eight and six independent witnesses and current
root consistency. The original finite-perimeter covers contain 581,288 N and
547,505 P positive cells; every original P clearance cell remains identical.
The [complete-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-current-conditioned-root-fullpath-diagnostic-local-20261102/independent-array-readback-review.json)
checks all eight 4,659-knot paths and six phases with original uncertainty.
Contact through stabilization has positive finite-material stress margins for
N/H+/H-/P; approach remains negative, and the original extra-angle physical failures
and unadopted force domains remain. The allocation charges 8.043 GPU seconds with
owned cleanup. D3-P has no eligible end-of-inspection current root, so this option
alone does not resolve D3. A separate coplanar-atlas consumer experiment preserves
old bounds only by independently retaining both perimeter certificates and still
has insufficient lower-height margin; it remains unadopted. Reader reconstruction,
scalar-serialization and test-fixture failures are retained. Fresh native selection,
future current errors, scene/load/stop behavior and full sources remain unqualified.


The [fresh conditioned-reference N/P pair](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-conditioned-reference-stationary-family-20261102.json)
then independently qualifies actual initialization on source `d0281c4`, original
ten-view RGB-D, reset, parked feedback, calibrated camera, identity, finite
witnesses and continuous original six-second, fourteen-shape stops. Both seeds
are acquired at 34 s and evaluated at 34.4 s. The selected pixels and unique
minimum measured basis bounds are independently reconstructed; root masks and
point coordinates match exactly. N retains four witnesses and 581,258 material
cells; P retains five and 422,185. Its measured reference differs from the saved
counterfactual, so that earlier map is not substituted. Whole stable payloads and
three sidecars per host return locally; owned process groups and GPU contexts
are absent and leases released. The pair charges 954.847 GPU seconds.
The [actual-packet full-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-conditioned-reference-actual-fullpath-material-local-20261102/independent-array-readback-review.json)
checks all eight 4,659-knot, six-phase paths under their original fixed current
errors. Contact through stabilization remains positive for N/H+/H-/P; approach
projection deficits, extra-angle physical negatives and unadopted 10 N positions
remain. This numerical allocation charges 8.044 GPU seconds with owned cleanup.
The next qualification needs actual robot movement and measured load support;
stationary acquisition does not establish future uncertainty or full observed
contact/source success.

The [first complete native family105 review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-family105-unreserved-native-complete-diagnosis-20261102.json)
retains all four original N/H+/H-/P robot-only paths at 0.3 N diagnostic pressure.
Each has 19,596 native moving samples and 1,920 stopped samples. All six phase
endpoints, original three-second hold, achieved-pose staged release, six-second
held-target stabilization, frozen residuals and continuous original eight-second
joint/tool/fourteen-shape stops pass. The future physical gate remains negative
for P: 265 approach samples exceed the lower elbow limit after the one original
stop/acceleration/measured-velocity allowance, by at most 0.003701 rad. N/H+/H-
retain positive future boxes. Whole stable payloads and three sidecars per host
return locally with owned cleanup; native jobs charge 1,709.797 GPU seconds.
Independent reviews charge 31.212 seconds, including the retained first reader
assertion at the negative physical gate. No remaining-cell attempt or source is
added.

The positive numerical design used physical-future interior targets throughout,
but the native fixture inherited raw physical target defaults and its stationary
recipe omitted the existing target-margin field. The corrected fixture wires the
same `pose_target_margin_rad` and `pose_target_domain_rad` options as the maintained
pilot. Its reserve is the original stop travel plus acceleration travel and one
maximum-speed control interval, exactly the declared design rule. Actual native
target limits must match the recipe and design before motion. Requalification
covers every original condition, all seven joints and all six phases. This fixes
a configuration mismatch; every physical, pose, rate, residual and stop criterion
is retained. Dependent 10 N calibration was withheld pending independent review
of the corrected complete native paths. Observed movement and future uncertainty
still require separate qualification.

The [corrected native family105 qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-family105-reserved-native-complete-qualification-20261102.json)
independently passes all four original complete paths with the existing shared
interior target reserve. Actual target limits match the recipe and design;
every original phase endpoint, rate, acceleration, frozen residual, held-target
stabilization and continuous same-latch stop criterion passes. Every native future
physical box is clear. The minimum joint margins are 33.280, 7.904, 32.147 and
8.064 mrad for N/H+/H-/P, respectively. Each whole stable nineteen-file payload
and three sidecars returns locally with owned cleanup. Native jobs charge
845.469 GPU seconds and independent reviews 36.218 seconds. The earlier P negative
remains; no new source or 10 N domain follows from these 0.3 N components.

The next [finite 10 N study](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-family105-reserved-actual-scoped-10N-support-manifest-20261102.json)
uses twenty required joint-side extremal postures from those actual paths, with
one original future allowance. Each posture has independent unloaded, center-load
and two distal-face/yaw cases with the original fixture, model and continuing
stop criteria. The eighty cases are assigned once across five hosts; H- has two
disjoint posture subsets. The [independent mixed-outcome review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-family105-scoped-10N-mixed-outcomes-20261102.json)
retains 74 completed cases, one failed case and five unattempted cases. Four full
groups independently qualify 64 cases; the incomplete H- group retains ten
completed cases. All 225 recorded continuous stop segments, including the failed
case, pass the original joint/tool/fourteen-shape criteria. Whole payloads and
three sidecars per host return locally with owned cleanup. Native jobs charge
5,105.654 GPU seconds. The incomplete study does not adopt a calibration domain,
an arbitrary joint box or a full source.

The failed H- lower wrist-X face-load case exhausts the fixture's reserved target
interval. Independent static equilibrium over all eighty declared cases finds
that fixed 10 Nm/rad wrist impedance needs target -0.969621 rad to balance the
0.413376 Nm external torque, below the available -0.959431 rad lower target.
The 10.190 mrad deficit cannot be corrected by longer convergence at unchanged
targets. Actual feedback, frozen residuals and the original same-latch 2+4+2 s
stops still pass. No unchanged case retry, force reduction, gain increase or
physical-bound relaxation follows.

A common corrective design reserves the signed static load-induced actuator
travel across all four complete paths and declared load cases, then applies the
original future physical reserve. It shifts the common initial elbow by
+20.322 mrad while retaining the original signed P perturbation. The
[independent whole-path numerical review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-whole-task-design-local-v2-20261102/independent-wholepath-readback-review.json)
passes all four original 4,659-knot, six-phase paths, including original pose,
rate and future physical gates. Three extra-angle stress cases remain negative.
An alternative orientation-independent lever-bound proposal also remains
negative. A duplicated-input preparation error and its executed mathematical
outputs are preserved separately; only the independently checked distinct-input
measured-load proposal supports the new design claim. No load calibration,
normal gains, force or acceptance criterion changes.

The [complete new native/stationary qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-native-and-stationary-qualification-20261102.json)
independently passes all four original complete robot-only paths and both fresh
N/P acquisitions. Native paths retain 0.3 N diagnostic pressure, all six phases,
achieved-pose release, held-target stabilization and the original eight-second
stops. Minimum original future physical margins are 43.190, 27.848, 56.260 and
28.327 mrad for N/H+/H-/P. Target parity and frozen residuals pass. Fresh N/P
references retain four/seven finite witnesses and 581,986/368,481 material cells;
selected roots, exact masks/coordinates, original ten views, actual RGB, reset,
camera and original six-second stops independently pass. The new initial arm
posture invalidates reuse of the older stationary references for this proposal.
All five whole payloads and three sidecars per host return locally with owned
cleanup. Native jobs charge 2,237.492 GPU seconds; the three independent native
review allocations charge 30.200 seconds.

The [fresh measured material-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-actual-fullpath-material-local-20261102/independent-array-readback-review.json)
checks all eight original/stress paths, 4,659 knots and six phases each. Fixed
current-error margins from contact through stabilization are positive and fully
available, including 71.015/70.277/71.475/67.841 mm minima for the four original
conditions. Negative approach projections and the separate extra-angle physical
failures remain. This diagnostic charges 7.032 GPU seconds with owned cleanup;
it supplies no evolving observation, normal, swept scene, contact or source
certificate.

Nineteen newly required joint-side extrema from those actual complete native
paths, each with one original future allowance, define 76 fresh independent
unloaded/center-load/distal-face-yaw cases and 228 original continuous stop
segments. The [necessary static target preflight](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-10N-static-target-preflight-local-20261102/review.json)
passes all 76 cases at unchanged normal impedance, force and physical pulse
reserve. It charges 7.038 GPU seconds and guarantees no native convergence or
dynamic outcome. The [new finite native study](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-actual-scoped-10N-support-manifest-20261102.json)
assigns every case and posture once across five hosts, splitting H- into disjoint
subsets. Its [independent complete qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-actual-scoped-10N-qualification-20261102.json)
passes all 76 cases, 19 postures and 228 original continuous stop segments.
Actual moving and loaded-stop boundaries, frozen residuals, physical/rate/
acceleration/effort/feedback/clocks and same-latch joint/tool/fourteen-shape
radial/directional/rotation gates all pass. All five whole payloads and three
sidecars per host return locally with owned cleanup. Native jobs charge
5,244.124 GPU seconds and independent GPU reviews 188.807 seconds. The prior
80-case negative study, missing cases and charges remain unchanged. This is
qualification of the declared coupled measured postures and task family, with
no arbitrary Cartesian-box, global/default calibration or hardware claim.

The [observed N/P prefix pair](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-observed-prefix-pair-manifest-20261102.json)
uses two local recipes restricted to their exact qualified condition and setup.
Only the verified empirical position-side widths and their 10 N force caps
change; a larger pre-existing weak-force halo is never upgraded beyond its
measured frontier. Original nominal support is unchanged, original unsupported
future intervals become supported at 10 N, and excessive force, each outside
position side, wrong condition and altered setup refuse. Global calibration,
physical/residual/dynamic/stop/observation and success criteria remain unchanged.

Each prospective native component retains the original inspection, 360 approach,
180 contact and 240 actual push commands before an explicit controlled cutoff
and original continuous six-second/fourteen-shape stops. The original required
opening remains 50.146 degrees; four seconds of push cannot qualify that goal or
a complete source. The 5,000 s case allowance follows the retained 780-command
native cost protocol; the 75 s simulation allowance and scientific criteria stay
fixed. The [independent negative-outcome review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-load-aware-observed-prefix-pair-negative-qualification-20261102.json)
closes both: N delivers 315 approach commands and P two; neither reaches contact
or push. All raw feedback, load packet clocks/models, actual RGB and original
continuous six-second/fourteen-shape stops pass. Both whole 38-file outputs and
three sidecars return locally with owned cleanup. The failures retain 1,146.9996
and 357.7641 native GPU seconds. No controlled prefix or full source is qualified.

P refuses renewed target-relief exclusion at 34.4333 s. The [exact saved-response
attribution](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D4-measured-load-aware-P-observed-780prefix-elab2-20261102/exact-target-relief-response-diagnosis.json)
reconstructs the original +2.066 mm margin and current -2.295 mm margin with the
same original complete enclosure. The nominal pose shifts 2.156 mm while current
pose error grows; geometry support is unchanged. Current surfaces already include
the complete query, and a dense same-seed diagnostic does not qualify a remedy.
A tighter native correspondence-bound counterfactual remains diagnostic: it does
not establish a physical-match certificate and is not adopted as a safety bound.
P stays suspended pending a shared perception/acquisition correction, retaining
all original uncertainty and exclusion requirements.

N refuses future position support at 39.65 s: wrist-X upper prediction 1.147537048
rad exceeds the adopted nominal 1.147524894 rad boundary by 12.154 microradians.
This is empirical support, separate from the unchanged physical limit. The
[all-posture pulse audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-qualified-shared-loaded-pulse-frontier-audit-20261102.json)
finds that adoption discarded already-qualified preregistered pulse coverage. For
every one of the 19 sides, the corrected frontier is restricted by measured loaded
motion, original loaded stop onset and the preregistered pulse target at each of
all three load sites, then intersected across sites. This uses no failure-specific
epsilon or additional empirical allowance. The [four-condition adoption review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-qualified-common-loaded-pulse-support-20261102/qualification.json)
retains original complete-path support, unchanged nominal calibration and refusals
outside each measured side or above 10 N. All physical, model, stop and observation
bounds remain unchanged. A [single distinct corrected N prefix](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-common-pulse-supported-N-observed-prefix-manifest-20261102.json)
is running on Blackwell with exact condition/setup enforcement and the same finite
780-command protocol. Its outcome is pending; P is not relaunched. These components
add no matrix attempt, source, windows or full-task claim.

The shared orientation correction uses the original measured normal and straight
edge as two orthogonal-axis certificates under the existing rigid-surface model;
[[../topics/shared-door-perception#Observed geometry and timing|the perception topic]]
gives the derivation. It preserves their original metric point, finite-triangle,
line-residual and conditioning bounds, full correspondence residuals, calibration,
time, root identity and ambiguity checks. It does not promote native registration
residuals to a physical-match certificate. Numerical nominal, perturbed and
boundary controls pass. The [paired eight-state readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/two-axis-actual-D3-D4-state-review-v3-20261102/review.json)
retains every native pair and tracking pose across D3/D4 N/P stationary and actual
refusal states. Whole-root ranking may choose a different independently measured
current component. The original test that expected fingertip-only selection also
fails with the baseline implementation; its corrected contract checks whole-root
selection after identity filtering. Initial numeric and reader assumptions are
preserved in the local evidence.

On P's actual response the same complete 19-enclosure relief proof changes from
-2.295 mm to +0.353 mm. The corrected full local-admission renewal passes and added
pose uncertainty still refuses. The [actual CUDA complete-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-two-axis-actual-fullpath-material-local-20261102/independent-array-readback-review.json)
retains all eight 4,659-tick/six-phase paths, original material availability,
complete finger queries and geometric camera coverage. Fixed-current-error
contact-through-stabilization minima remain positive; approach projections and
extra-angle physical negatives remain. The [full producer and receipt review](../../../outputs/evidence/act-a1-complete-cycle-20261026/two-axis-D4-full-producer-saved-and-current-probes-local-v2-20261102/independent-receipt-review.json)
preserves the old N empirical-bound negative. Four current-state N/P probe
receipts renew and refuse excessive uncertainty in their consumed normal or pose
primitive. Those probes explicitly repeat a prior action hypothetically; no
undelivered native command or evolved observation is reconstructed. Fresh scene
and query caches remain a limitation. GPU charges are 7.033 seconds for full paths,
7.033 for an initial incomplete-recipe setup failure and 39.531 for the corrected
producer replay, all with verified owned cleanup.

The [corrected-pulse N component](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D4-common-pulse-supported-N-observed-780prefix-blackwell-20261102/complete-negative-qualification-review.json)
finishes negative on its frozen earlier source after 164 approach commands,
8,913 healthy native feedback samples and 329 load packets; no contact or push
is delivered. At 37.1333 s, the same protected sphere has 4.208 mm of normal-error
growth against its original 3.245 mm margin, leaving -0.962 mm. The selected plane
origin and normal bound change; the two-axis full-pose correction preserves this
independent normal refusal. Actual RGB, raw/clock/load and original continuous
six-second/fourteen-shape stops pass. All 38 original files and three sidecars
return locally with owned cleanup; 456.998 native GPU seconds remain charged.
Further N attempts stay suspended while this common acquisition/normal mechanism
is diagnosed.

A [single distinct P observed prefix](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-two-axis-P-observed-prefix-manifest-20261102.json)
finishes negative on elab2 with source `da61c0d` after 98 approach commands,
197 load packets and 8,649 healthy native feedback samples, before contact or
push. The original complete target-relief exclusion has +9.266 mm at admission,
+13.123 mm for the same old remaining interval, and -0.707 mm after the new pose
update. Current position/rotation bounds grow from 18.764 mm/0.036067 rad to
35.432 mm/0.049323 rad. The [exact saved-response attribution](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D4-two-axis-P-observed-780prefix-elab2-20261102/exact-target-relief-response-diagnosis.json)
preserves every enclosure, error source and refusal. Raw/clock/load, actual RGB
and original continuous six-second/fourteen-shape stops pass. All 38 original
files and three sidecars return with actual owned process/GPU absence and lease
release; 420.414 GPU seconds remain charged. Two adapted readers fail only on an
unnecessary self-copy after saving their complete results; original scripts and
outputs are retained, the self-copy is removed and the saved results independently
read back. No simulation or scientific gate changes.

A shared measured-plane normal certificate uses two measured spanning chords of
the same planar core. If true normal m has bounded metric point error e, each
chord d satisfies |m.d| <= 2e. Decomposing m into the measured normal and its two
tangent coordinates gives a two-dimensional box whose four corners bound normal
tilt. This requires no independent errors or small-angle approximation. The
complete core's original error, original triangle conditioning/hemisphere,
straight-edge, all native pairs, physical/model/calibration/time and downstream
normal/pose gates remain. A first triangle-only construction still leaves the
retained N response negative by 0.053 mm; that negative design is preserved.
Independent normal/tangent extents provide a stronger general certificate.
[Ten original D3/D4 publication controls](../../../outputs/evidence/act-a1-complete-cycle-20261026/normal-span-actual-D3-D4-state-controls-20261104/review.json)
retain native poses/pairs and improve whole-root point-error certificates. Both P
responses and the corrected N exact saved response renew, while excessive
consumed uncertainty refuses. Correlated metric-limit numerical perturbations
also pass. The implementation is frozen at `15decf6`.

The [independent actual CUDA complete-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-span-actual-fullpath-material-local-20261104/independent-array-readback-review.json)
retains all eight 4,659-tick/six-phase paths, original material availability, full
finger queries, geometric camera coverage and negative approach/physical stress
cases. Original four contact-through-stabilization fixed-current-error minima are
81.884/81.785/81.891/77.845 mm. These remain diagnostic margins, with no future
normal/scene/contact or observed-motion guarantee. The [full producer receipt readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/normal-span-D4-full-producer-saved-and-current-probes-local-20261104/independent-receipt-review.json)
retains both old empirical-bound N refusals. Six paired hypothetical current N/P
probes renew and refuse excessive consumed normal or pose uncertainty. The
corrected N candidate uses a complete unloaded separation proof; its baseline
uses a loaded geometry proof. No prior action or observed contact is relabeled,
and fresh scene/query caches remain a limitation. The two allocations charge
7.030 and 46.555 GPU seconds, with verified owned cleanup. The full-path wrapper's
inherited eight-view description is corrected as metadata: the worker actually
requires all ten original views, and no observation changes.

A [distinct prospective N/P observed component pair](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-span-observed-prefix-pair-manifest-20261104.json)
is dispatched once on Blackwell/elab2 with source `15decf6`. Source archive,
recursive config references, runtime constructors, actual UUID/ownership/empty
contexts/capacity pass before launch; both native GPU process groups and leases
are verified. Each retains fresh ten-view inspection and the original finite
360 approach/180 contact/240 actual push endpoint, fixed full-task goal, physical,
load, measured uncertainty and independent six-second stop criteria. Both finish
negative before contact or push. The [independent negative closeout](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-span-pair-negative-qualification-20261104.json)
retains N's 323 delivered approach commands, one rejected undelivered proposal,
647 load packets and 9,549 healthy feedback samples; P retains 170 deliveries,
341 load packets and 8,937 healthy samples. Original raw/clocks, saved verified
load packets, actual RGB and all original continuous six-second/fourteen-shape
radial/directional/joint/tool/rotation/speed/acceleration/force stop gates pass.
Each whole 38-file payload and three sidecars return with actual owned process
and GPU absence and canonical lease release. Charges remain 1,171.268 and
478.673 native GPU seconds; zero unsafe commands are recorded.

The [exact N attribution](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D4-normal-span-N-observed-780prefix-blackwell-20261104/exact-negative-mechanism-diagnosis.json)
reproduces the future-load refusal at 39.7833 s. Wrist upper prediction
1.148007724765 rad exceeds the original native-resolution loaded boundary
1.147981762886 rad by 25.961879 microradians; current measured position remains
inside support. The full two-second contact-free stop margin is -338.924 mm, so
the weaker-domain fallback remains unavailable. No empirical side is extended.
The [exact P attribution](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D4-normal-span-P-observed-780prefix-elab2-20261104/exact-negative-mechanism-diagnosis.json)
reproduces the same-sphere response refusal at 37.2333 s: normal growth 2.036 mm
against 0.633 mm original margin leaves -1.403 mm. The plane pivot shifts and its
angular error grows; the aggregate anchor-position error alone decreases.
Every spatially associated sphere and measured primitive is retained. An initial
diagnostic reconstruction mismatch of at most 3.638e-12 rad is resolved by using
the original native float32 velocity arithmetic, without changing its comparison
or any physical tolerance; the failed reader is preserved.

Two [diagnostic alternatives remain unadopted](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-span-pair-diagnostic-limitations-20261104.json).
Integrating speed and acceleration together would reduce N's joint forecast,
but the 0.51 rad/s criterion is sampled, not a hard continuous actuator cap:
reference clipping is 0.45 rad/s, while maintained actuator limits are much
higher. Current acceptance cannot supply the missing future invariant.
Independent normal certificates from all six currently eligible P fits, aligned
to the unchanged full pose with their original errors, still refuse its exact
same-sphere response. No favorable plane, smaller uncertainty or position
exception is adopted.

The [completed causal recorded-frame CUDA replay](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-certificate-recorded-frame-replay-local-20261104/independent-causal-readback-review.json)
processes all 374/399 available P/N observations in order, including each separately
identified exact final publication and all ten inspection times. Each case retains
341 unavailable events; P retains three older certificates and refuses three at
the original horizon, while N retains none. All support clocks and ages pass
independent readback. Its new P/N references differ from native by 370.884/188.284
mm, so the smaller replay errors are not evidence of a native correction. Missing
30 Hz inputs remain unavailable. The closed allocation charges 250.582 GPU seconds
with verified owned cleanup; all original native negatives remain.

The [exact native certificate diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/native-temporal-and-joint-certificate-static-design-20261104/review.json)
instead ages the previously published certificate through the real response time,
retaining every primitive, pivot and support clock. In both P contexts it remains
current-root consistent, has smaller complete-root error and passes the saved
response. Original normal-span P combined error is 58.610 versus 66.326 mm.
All six fresh complete pose/normal alternatives retain that current negative.
This is a static counterfactual, not delivered motion or future source support.

The shared implementation now retains at most one previous measured local
certificate after a valid fresh current constraint, current native identity and
current root-depth check. It pairs the actual constraint with its own camera,
support, map/reference and selection, ranks complete pose plus normal error on the
entire original measured root, and pays the original linear/angular aging until
expiration. Loss, recovery uncertainty, missing/ambiguous current geometry,
reference/selection/generation changes and expired support cannot use history.
Native pose and articulation diagnostics remain current. No model, physical,
load, uncertainty, normal, material, scene or stop criterion changes.

[Numerical/session checks and twelve exact saved-publication controls](../../../outputs/evidence/act-a1-complete-cycle-20261026/bounded-certificate-actual-D3-D4-state-controls-20261104/review.json)
retain exact single-publication parity with `15decf6`, native pairs/masks/poses and
all bounds/clocks. Independent whole-root error calculations agree. Two P controls
reconstructed from their own prior published primitives and actual raw-frame
camera retain the real 33.333 ms age and pass the saved response; excess consumed
uncertainty still refuses. Original prior model-pair counts are unavailable and
no native-stream equivalence is asserted. Actual CUDA producer, complete-motion
and related-condition qualification remain required before further source
collection. No source, matrix attempt or windows are added.

The [actual CUDA complete-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-bounded-certificate-actual-fullpath-material-local-20261104/independent-array-readback-review.json)
retains all eight original 4,659-knot/six-phase paths, geometric FOV, material
availability, contact-to-stabilization fixed-current-error margins and negative
approach/physical stress cases. It grants no future evolving-observation support.
The [full-producer receipt review](../../../outputs/evidence/act-a1-complete-cycle-20261026/bounded-certificate-D4-full-producer-saved-and-history-probes-local-20261104/independent-receipt-review.json)
retains the two old N empirical-bound refusals and verifies eight hypothetical
current receipts, including the separately identified prior-published P
certificate control. Original consumed-error overspend refuses. The reader saved
its complete successful result before a post-write self-copy error; independent
readback retains that error and verifies the saved report. Charges are
7.035 + 55.824 GPU seconds with owned cleanup. Neither control actuates a command.

The [distinct bounded-certificate P component](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-bounded-certificate-P-negative-qualification-20261104.json)
finishes negative from source `f63b65b` on elab2 after nine approach deliveries,
ten admission records including one undelivered refusal, nineteen load packets
and 8,293 healthy feedback samples. No contact or push is reached. Original
raw/load/clocks, actual RGB, six-second/fourteen-shape stops, whole 38-file plus
three-sidecar transfer and owned cleanup pass; 422.278 GPU seconds are retained.
The final certificate is current and only 16.667 ms old, with no earlier
certificate retained at this packet. The complete positive-finger query has
+36.744 mm finite planar coverage, but -9.830 mm whole-front separation and
-58.295 mm relief exclusion. Unknown space stays unknown. Baseline/current
reconstraint on this same publication agree exactly; a two-ULP diagnostic
angular recomputation discrepancy changes no physical or calibration bound.
Full motion, related-condition and complete-source qualification remain pending;
counts stay 17/36 with no added source or windows.

An [independent saved-native position-domain audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-whole-motion-position-domain-audit-20261104/independent-readback-review.json)
checks all 78,384 recorded samples from the four complete N/H+/H-/P 0.3 N
robot-only paths, including preload/settle and all task phases. All measured
positions, actual targets and hypothetical future 10 N position-box gates remain
inside the existing certificates. Thirty-two phase-boundary controls agree with
the original scalar gate. This does not turn the 0.3 N paths into actual 10 N
motion, observed-geometry admission or source qualification.

The exact observed N refusal instead retains 18.100 mrad of current-position
margin and 17.798 mrad of target margin at the wrist, while its complete original
future interval exceeds empirical support by 25.962 microradians. The original
physical limits and independent stop results remain separate. Configured target
limits provide physical/load-compliance headroom and are wider than the measured
position box, but actual target membership already passes in this negative.
Clipping targets merely to that measured box would therefore neither change the
four saved paths nor prevent this refusal. Any prospective control correction
must jointly preserve full-path tracking, force-compliance target offsets and
complete future/stop support across the original conditions. No new speed
invariant, empirical extension, target recipe or unchanged observed N retry is
adopted. The independently closed P negative above remains separate from this
saved-array audit.

The [fresh-negative whole-path readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-bounded-certificate-fresh-N-P-wholepath-stress-local-20261104/independent-array-readback-review.json)
retains all eight 4,659-tick/six-phase ideal paths with the actual latest N/P
observed covers. Full-finger approach deficits remain about 15.480 mm in N/H
and 24.888 mm in P, while fixed-current-error contact-to-stabilization margins
remain positive. Original extra-angle physical negatives also remain. Frozen
observations and legacy diagnostic limits do not establish future moving scene,
normal, contact, load, stopping or source qualification.

The [original P front reconstruction](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-bounded-certificate-P-front-coverage-local-20261104/audit.json)
exactly reproduces all ten views and 34,669 observed voxels. All 3,058 protrusion
voxels and disconnected foreground remain. Even a union of individually padded
returns fails the original complete relief exclusion. A different, unadopted
height-layer model retains the same global tangent padding and additive height
reserve, and localizes only the higher returns for each measured height level.
Its use must be restricted to a completely supported finite observed root.
The [whole-root surface audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-height-layers-whole-root-collider-audit-20261104/review.json)
and [independent solid-polytope optimization](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-D4-height-layers-whole-root-collider-audit-20261104/independent-solid-polytope-readback.json)
pass all 35 candidate layers across the actual D4-P, D4-N and D3-N scenes against
their complete fixed TRAIN collision surfaces. An unseen high-attachment control
fails as required. At the saved P query, the 16 mm layer gives a 43.221 mm front
bound and +101.040 mm spatial exclusion, with every original error reserve
retained. The original plane tolerance is 4 mm; an earlier unadopted interactive
1 mm estimate is superseded, not used to qualify the model. These are fixed-domain
geometric checks. Optional producer and remaining-receipt finite-domain
revalidation now retain every higher return and original global reserve. The
new certificate rechecks complete normal separation as well as spatial root and
higher-return exclusion, including loaded receipts; convex and outer-ball proofs
remain distinct. Numerical boundary/perturbed/missing-normal controls and affected
admission/session regressions pass. Actual saved full-producer, related complete
motion and native qualification remain separate. The implemented layers now
exactly match all 35 independently validated full-collider layers and finite
domains. The [original-packet CUDA producer review](../../../outputs/evidence/act-a1-complete-cycle-20261026/finite-height-whole-root-and-original-producer-qualification-20261104.json)
keeps the disabled model/producer unchanged in all three scenes. On P, the enabled
producer admits the exact earlier-refused packet with two complete-volume
certificates. Five remaining-interval receipts pass, while coherent finite-root
and normal-primitive perturbations reach the height-proof refusal. N retains its
independent future-position refusal. D3's default packet is already admitted and
uses no height layer; this is no new D3 native success. The 70.801 GPU seconds
include both retained D3 reader-preparation failures, with owned cleanup. Original
packets, selected poses, clocks, all foreground and reserves remain unchanged;
no source or matrix cell is added. Four local allocations charge 38.288 GPU
seconds, including both retained reader-preparation failures, with owned cleanup.

The [finite-height/feedback P prefix negative review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D4-finite-height-feedback-P-observed-780prefix-elab2-20261104/complete-negative-qualification-review.json)
closes the distinct source `1d6995d` outcome after all 360 approach commands and
eight contact commands. The next proposal is refused before delivery at
40.533333 s. All 9,729 native feedback samples, 369 admission records, 737 load
packets, original clocks and actual RGB are retained. Independent six-second,
fourteen-shape continuous stopping, whole 39-file/three-sidecar transfer and
actual owned GPU/process cleanup pass; 801.246 native GPU seconds remain charged.
Push, hold, release and stabilization are unavailable. No source, window or matrix
attempt is added by this component qualifier.

The [whole-motion saved-array comparison](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab2/D4-finite-height-feedback-P-observed-780prefix-elab2-20261104/whole-motion-load-history-gap-review.json)
reproduces the original float32 future-position refusal exactly. Shoulder-Z's
upper forecast exceeds its measured support by 3.496 microradians while current
position retains 15.933 mrad of support. No measured control position is outside
that support. The original contact-free alternative is unavailable. This
establishes neither a physical-limit violation nor permission to extend the
failed boundary. After mapping both actual root transforms, contact goals differ
by at most 76.888 micrometers from the robot-only reference. The observed prefix
has zero evaluated contact, while the complete reference holds 10 N throughout
all six phases. Those loaded trajectories and separate unloaded posture probes
do not establish whole-motion extrema at zero or changing load. The initial
unmapped world-goal comparison is retained separately and superseded.

The [complete unloaded counterpart study](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-unloaded-complete-manifest-20261104.json)
assigns one distinct full 0 N N/H+/H-/P robot-only trajectory to each of four
compatible owned elab hosts. It preserves the original 4,659 commands, fixed
0.5 s feedback, signed variations, all six phases, three-second hold, achieved
staged release, six-second stabilization, physical/future limits, frozen residuals
and original one-latch 2+4+2 s/fourteen-shape stops. The independently closed N/H+/P
components below pass; H− remains pending after a healthy running inspection.
Constant 10 N cases are retained without repetition. A copied N preflight's old
10 N-only scope assertion failed before launch; its corrected unloaded scope and
all original checks passed before the sole dispatch. Prepared independent GPU
readers may run only after actual completion, whole transfer and owned cleanup;
negative or interrupted outcomes require their actual witnesses. Even a passing
0 N component cannot qualify 10 N support, intermediate-load motion, observation,
contact or a complete source. The independent observed N prefix remains running
on Blackwell and is left undisturbed. The matrix remains 17/36.

The [first three unloaded component qualifications](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-unloaded-first-three-qualified-components-20261104.json)
independently pass all original six-phase endpoint, hold, achieved release and
stabilization criteria, frozen moving/stopped residuals, physical/future limits
and original one-latch 8 s/fourteen-shape stops. Their 58,788 moving and 5,760
stopped native samples, nine stop segments and whole 19-file/three-sidecar
payloads per condition are retained. Actual owned cleanup passes for all three
native jobs and CUDA reviews. The completed native allocations charge 645.693
GPU seconds and the independent reviews 27.148 seconds. Only the declared force
and case name differ from each corresponding constant 10 N case; controller,
setup, original physical/rate/target/stop limits and all phases remain unchanged.
H− is explicitly unavailable to this partial qualification.

The [saved-array full-history comparison](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-zero-ten-first-three-position-support-audit-20261104.json)
checks all 117,576 native samples from these six complete 0 N/10 N histories.
All measured positions remain supported. N/H+ retain the required future-position
domain at both loads. P instead has 37 unloaded native samples whose original
future 10 N position/stop boxes exceed current support: upper shoulder-Z by at
most 26.482 microradians and upper wrist-X by 41.278 microradians. Original scalar
refusals agree at the full-history extrema. Thus the loaded trajectory does not
bound the unloaded trajectory, even though both satisfy physical and stopping
criteria. No calibration, numerical allowance or criterion is changed. Finish H−
and qualify the necessary measured support at the original required load and
load sites before adoption; intermediate/changing-load and observed complete
motion remain separate requirements. This partial study adds no matrix cell,
source or training window.

The [complete unloaded qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-unloaded-qualified-components-20261104.json)
now closes H− as well: all four original 0 N components independently satisfy
the original six-phase, physical/future, frozen moving/stopped model and one-latch
8 s/fourteen-shape stop criteria. All 78,384 moving and 7,680 stopped samples and
twelve stop segments remain; complete transfer and owned cleanup pass. Native
charges are 1,238.775 GPU seconds, reviews 36.197 seconds. The
[full eight-history position audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-zero-ten-complete-four-position-support-audit-20261104.json)
retains H−'s additional 62.841-microradian future wrist-X deficit, alongside P's
two deficits. No measured position is outside scoped support. No position bound
is changed, and intermediate/changing-load trajectories remain unqualified.

The [observed N negative](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D4-finite-height-feedback-N-observed-780prefix-blackwell-20261104/complete-negative-qualification-review.json)
retains 360 approach, 180 contact and 91 incomplete release commands, all 631
admitted proposals, 1,263 load packets and 10,781 healthy native feedback samples.
There is no push, hold or stabilization. No command is unsafe or eligible for
teacher supervision. The original six-second/fourteen-shape stop, actual RGB,
whole 38-file/three-sidecar transfer and actual owned cleanup pass; 2,670.174
native GPU seconds remain charged. All contact commands have zero native force
and no loaded contact. The teacher begins release after that physical failure;
the later observed safety refusal is `contact_acquisition_budget`. These causes
remain separate, and the incomplete release stays incomplete.

The [whole-motion ramp diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D4-finite-height-feedback-N-observed-780prefix-blackwell-20261104/whole-motion-contact-ramp-lag-diagnosis.json)
finds 1.661–1.686 mm normal endpoint lag in every original 0 N/10 N N/H+/H−/P
ramp, against unchanged 1 mm compression. Observed N independently has 1.706 mm
pose error and 0.671–0.716 mm native separation at the end of contact. Passing
the original robot-only 10 mm endpoint tolerance therefore does not establish
actual contact. Rather than changing seek time, compression or acceptance
criteria, the optional shared controller now uses the previous dispatched goal
to supply the part of the reference increment not already paid by feedback.
Numerical ramp/balance/saturation/pure-preview/default and guarded-stop controls
pass. Actual CUDA saved-input controls, all original complete 0 N/10 N motions,
independent stopping and observed contact remain required; no old positive or
negative is reclassified, and no calibration, source or matrix cell is admitted.

The [actual CUDA original-history controls](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-reference-original-history-controls-local-20261104/review.json)
for source `5317e9c` pass all 34,392 original pose-command boundaries across the
eight complete 0 N/10 N histories. Both disabled absolute/default and prior
feedback modes remain bitwise equal to `1d6995d`. New proposals preserve pure
preview and original target/rate bounds. Actual environment preview/dispatch/
reset/stop methods, exercised with official Lab pose math and a fake plant,
verify cold start, commit-only-on-dispatch, copied goals, rejected-preview
immutability and history clearing for joint commands, reset and independent
idempotent stop. These are software/history controls, not changed-controller
native trajectories. The allocation charges 8.040 GPU seconds with owned cleanup;
66 focused numerical/dispatch/stop checks and the documentation contract pass.
An initial linter call used a runtime without Ruff; the installed local Ruff
passes and no dependency is installed.

The [complete zero/ten native qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-8-qualified-components-20261104.json)
now closes all eight original N/H+/H−/P cases at 0 N and constant center-site
10 N for source `5317e9c`. One fixed causal-reference controller retains all
4,659 original commands, six phases, physical/future and reserved-target limits,
frozen moving/stopped residuals, rates and effort, three-second hold, achieved
staged release, six-second stabilization and original one-latch 8 s/fourteen-shape
stops. Normalized pose dispatch equals repeated pure previews, and joint commands,
reset and stop clear reference history. All 156,768 trajectory samples, 15,360
stopped native samples and 24 stop segments pass independent review. Whole
transfers and actual owned cleanup pass. Native charges are 2,697.612 GPU seconds;
readers charge 51.303 seconds. Normal ramp error ranges from −11.207 to 20.911
micrometers, retaining positive margin against unchanged 1 mm compression.
H−'s largest phase endpoint error is 7.681 mm, within the original 10 mm range.
This necessary tracking result does not establish physical door contact.

The [calibration-baseline reconciliation](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-calibration-baseline-reconciliation-20261104.json)
corrects the prior four-case support interpretation: those readers compared the
actual frozen component recipe, which predates the separately qualified
condition-scoped 17-side/68-case support. Their nine N/P requirements remain
fixture-relative; they do not enumerate new needs beyond the latest support.
The load-model, recording and stop implementations remain unchanged between
measurement source `7bd2ea5` and `5317e9c`, and the calibration records differ only
in independently measured position tolerance and force caps. The original
reader error and all original records remain.

The [complete latest-support audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-complete-eight-latest-qualified-all-coordinate-position-requirements-20261104.json)
compares every coordinate of both complete load histories against that existing
support. All actual measured positions are supported. The single original
stop/acceleration/measured-velocity future allowance leaves nine uncovered
condition/joint sides: two each for N/H+/H− and three for P. The
[scalar-gate audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-complete-eight-latest-qualified-position-support-audit-20261104.json)
reproduces every original decision across 156,768 samples, retaining 2,647 future
refusals and all current-position positives. No bound is extended or failure
reclassified; deriving complete support still requires all seven coordinates.

The [new required-load study](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-required-load-support-manifest-20261104.json)
derives the exact uncovered extrema from all eight complete histories plus that
single allowance. Every posture retains the other six measured coordinates.
Each has the original unloaded, center-10 N, face-0/+0.2-rad and face-1/−0.2-rad
cases, with unchanged all-seven-coordinate finite boundary servo, pulse,
convergence, physical/model and continuing 8 s/fourteen-shape stop criteria.
All 36 new case payloads differ from the previous 68; no already-qualified case
is repeated because of the stale comparison. The
[necessary static CUDA target check](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-support-static-preflight-local-20261104/review.json)
passes original physical/pulse reserves for every case, charging 7.028 GPU seconds
with owned cleanup. This does not establish dynamic convergence or native support.
The [completed required-load qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-required-load-support-qualification-20261104.json)
independently passes all 36 cases and 108 original stop segments. All-seven-joint
finite servo/pulse convergence, loaded moving and stop boundary coverage, frozen
residuals, physical/rate/effort limits, clocks and fourteen-shape original latch
bounds pass. It retains 167,420 trajectory feedback samples and 69,120 stopped
samples. Whole original payloads and three sidecars per job transfer completely;
actual owned process/context absence and canonical lease release pass. Native
charges are 2,358.284 GPU seconds; independent CUDA readers charge 107.847 seconds.

The [merged condition-scoped support](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-qualified-common-loaded-support-20261104/qualification.json)
uses the conservative common frontier across center and both face/yaw sites,
requiring actual moving and original stop-onset coverage capped by the original
preregistered pulse. It updates those nine sides within the existing 17-side
condition records and preserves all unaffected prior 68-case coverage, fields
and the original main domain. Every coordinate of all eight complete 0 N/10 N
histories and the single original future interval passes; all 156,768 original
scalar decisions are checked. Outside each new frontier and 11 N still refuse.
Physical/force/stop/model and observation limits remain unchanged. This supports
only the declared conditions, exact setup and measured complete-motion scope;
it does not qualify arbitrary Cartesian boxes or changing/off-center-load motion.

The [distinct causal-reference N/P observed transfer](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-observed-prefix-pair-manifest-20261104.json)
is dispatched once on Blackwell and elab2 after those prerequisites. It retains
the original TRAIN assets, signed setups, ten inspection views, 360 approach,
180 contact and 240 push commands, finite budgets and original 2+4 s/fourteen-shape
stops. Actual runtime verifies 0.5 s feedback and causal-reference mode. The
metadata-reader and RGB-D entry errors remain preserved; the existing Isaac entry
resolves the libraries before the only native launches. No raw recording or
native sample payload goes outward. A controlled prefix cannot admit a source
or training window.

The [P negative qualification and shared diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-P-negative-qualification-20261104.json)
closes the actual refusal after 211 approach commands: 212 admission records
include one undelivered rejection, 423 load packets retain the terminal check,
and all 9,101 native feedback samples are healthy. Contact and all later phases
are unattempted. Original raw/clocks/load masks, actual RGB, continuous
six-second/fourteen-shape stops, all 38 original files plus three sidecars and
actual owned cleanup independently pass. Native cost is 770.931 GPU seconds;
the negative adds no matrix attempt, source or window.

The exact saved shape-11 convex sweep/stop has a −1.080 mm whole-front margin
at the original uncertainties. Its current CAD plane gap of 47.204 mm does not
certify the uncertain future volume or establish a physical collision. The
retained certificate is 83.333 ms old; the available 16.667 ms-old alternative
has larger whole-root and witness-normal uncertainty, so forcing freshness is
not a demonstrated correction. The original whole-root selection is reproduced.

Original CUDA robot-occlusion masks and all ten native views reproduce the
seed front models exactly across four D3/D4 observations. The new P seed has
15,333 voxels and no protrusion layers; its already measured coplanar atlas has
56,000 voxels and 9,466 protruding returns. Merely broadening that domain does
not solve the complete proof: its best unchanged height-layer certificate still
has negative spatial and normal margins. No broader domain is adopted. Current
depth reobservation of atlas normal baselines leaves both P controls unchanged
and improves the old N witness-normal bound by only 0.034 mm. Historical D3
normal-selection reconstruction differs from its saved pose by 3.565 mm; that
baseline stays unavailable rather than being treated as a successful control.
The D3 front-model reconstruction is a separate, exact comparison.

All diagnostic preparation failures and partial results remain retained. The
first front-comparison allocation completes three D4 cases before an embedded
remote D3 config path fails; only the unprocessed D3 case runs in the successor.
Both owned GPU leases are cleaned; their total 28.287 GPU seconds remains
charged. Saved-reader import/interface/serialization and historical readback
failures are preserved separately. No collector is retried, and no source,
recipe, calibration or physical/observation threshold changes follow.

The [N contact/push diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/blackwell/D4-causal-reference-N-observed-780prefix-blackwell-20261104/whole-contact-push-pressure-diagnosis.json)
closes the other actual negative after 360 approach, 180 contact and 93 push
deliveries. All 633 admissions/deliveries, 1,267 load packets and 10,789 healthy
native samples remain; hold, release and stabilization are unattempted. Contact
is independently established at native time 43.0167 s, then the load lower
window crosses zero at 44.95 s. Actual native force is zero with a 3.047-micrometer
positive gap, despite the teacher's lagging 0.0356 N qualification window.
Material and complete sweep margins remain positive. This is physical loss,
not an unsupported-geometry refusal. Actual RGB, raw/load clocks, original
six-second/fourteen-shape stop, whole 38-file/three-sidecar transfer and owned
cleanup pass; all 2,836.819 native GPU seconds remain charged. The
[pair closure](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-reference-observed-pair-negative-qualification-20261104.json)
adds no matrix attempt, source or window.

The optional causal pressure lead uses consecutive completed native intervals.
Its [numerical audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-pressure-lead-numerical-controls-20261104/review.json)
reproduces all 93 original push feedback signals and verifies
an ideal first-order lag factorization plus 30 paired scalar comparisons. The
actual controller already anticipates the whole reference, including pressure
corrections. That omitted path invalidates the first-order model as a transfer
argument; those numerical controls establish no native task success. The
original 0.1 N target, gain, rates, extents and independent criteria remain.

The first four pressure fixtures stop after one inspection command because the
additional task-load recorder consumed a manipulation-only force field during
inspection. The candidate pressure loop is unexercised. All four raw recordings,
five native samples each, original six-second/fourteen-shape stops, actual RGB,
whole 17-file/three-sidecar transfers and owned cleanup pass. Original telemetry
recovers their actual process-group GPU identities; it is not presented as a new
live observation. All 225.410 GPU seconds remain charged. The common recorder
now preserves raw inspection while restricting its additional task-load audit to
manipulation. Synthetic contract checks retain four native task samples and the
original 10 N rejection of an 11 N upper bound.

The [corrected complete pressure family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-pressure-taskphase-complete-family-manifest-20261104.json)
assigns N/H+/H-/P once each on Blackwell/elab4/elab3/elab2, preserving original
signed setups and matched seed. All four now close
[negative after independent review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-causal-pressure-taskphase-negative-qualification-20261104.json).
Contact is lost after 95–100 push commands, at 1.23–1.32 degrees against the
50.146-degree goal. Original 0.1 s normal-load lower windows become negative;
hold and stabilization are unattempted. Later failure withdrawal exits empirical
wrist-X support in N/H+/H- and shoulder-Z support in P. This is distinct from a
physical joint-limit failure. All current/future physical boxes, raw clocks,
14,188 task-native samples, recorded causal controls, actual RGB, original
six-second/fourteen-shape stops, whole 25-file/three-sidecar transfers and owned
cleanup are independently reviewed. Original load-domain and future-domain
failures remain negative. All 1,146.468 native GPU seconds and 6,516,295,452
original transferred bytes are retained. No calibration boundary or criterion
is changed to admit these results.

The [shared mechanism/readback audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-existing-native-continuous-gap-recipe-audit-20261104.json)
checks all 393 saved push controls plus 141 gap/pending-motion/rate/extent cases.
In the ideal whole-reference-feedforward model, a second pressure derivative
can introduce an unstable alternating mode; the earlier lag-only factorization
therefore supplies no stability guarantee for this runtime. This calculation is
not a fitted native plant or a counterfactual success claim.

The [distinct native-gap family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-native-gap-pressure-complete-family-manifest-20261104.json)
uses completed native mean force, continuous measured-gap correction minus
already-commanded normal advance, and zero derivative lead. All four now close
[negative](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-native-gap-pressure-complete-negative-qualification-20261104.json):
contact is lost after 90–97 push commands at 1.15–1.26 degrees, before hold or
stabilization. Later failure withdrawal exits empirical wrist-X support in
N/H+/H- and shoulder-Z support in P. All 14,100 task-native samples, 371 causal
controls, raw clocks/RGB, physical boxes, original six-second/fourteen-shape
stops, whole 25-file/three-sidecar transfers and owned cleanup are reviewed.
All 1,145.422 native GPU seconds and 6,503,172,642 original transferred bytes
remain. No physical violation is inferred from an empirical-domain refusal;
all lower-load and future-support failures remain negative. The family is
unadopted; no unchanged retry is assigned.

The [whole-motion pressure/hinge audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-whole-motion-hinge-pressure-balance-audit-20261104.json)
uses the four original signed contact geometries and six completed native
histories. Across the original 50.146-degree motion, the normal torque lever is
0.19714 m. The unchanged actual hinge damping is 4 N m s/rad, so a normal-only
steady balance at the requested 1 degree/s requires 0.35413 N. The old 0.1 N
request corresponds to 0.28238 degree/s in that model. Joint and implicit-actuator
damping name the same drive and are counted once. Saved native inertia and
parked authored COM predict unloaded decay of 0.7911/s; actual saved free-coast
medians are about 0.8021/s. This difference is retained and supplies no fitted
force multiplier. Authored poses are not presented as simulated moving states.
Startup acceleration, tangential friction, distributed contact and nonlinear
servo interaction remain outside this consistency calculation.

The [prospective pressure design](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-motion-consistent-pressure-control-design-20261104.json)
uses the existing explicit `push_force_n` field. The shared rule is
`max(original acquisition request, damping * reference speed / normal lever)`;
it yields 0.35413 N for all four original cases. Acquisition and hold stay at
0.1 N. The 27 parameter-variation balance checks and 160 bounded force/gap/
pending-motion cases pass numerically. This changes a control request, while
load/contact acceptance thresholds, the original 10 N stopping model, 80 N
physical limit, gains, rate/extent, dynamics, variations and full phases remain.
The calculation establishes no counterfactual physical completion.

The [motion-consistent pressure family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-motion-balanced-pressure-complete-family-manifest-20261104.json)
now [closes independently negative](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-motion-balanced-pressure-complete-negative-qualification-20261104.json)
in all four original conditions. N delivers 536 push commands and reaches
8.611 degrees, retaining positive original lower-force windows. It stops when
the frozen moving-model upper estimate reaches 12.133 N against the original
10 N domain. Actual native contact peaks at 9.788 N, below the separate 80 N
physical limit. H+/H-/P lose contact after 95/92/88 push commands at
1.276/1.234/1.163 degrees; their original lower windows become negative.
Later withdrawal exits empirical wrist-X support for H+/H- and shoulder-Z for P.
Physical current/future bounds and all four original six-second/fourteen-shape
stops independently pass. No hold or stabilization qualifies. All 14,852
task-native samples, 811 causal controls, original raw/RGB, whole payloads plus
sidecars and actual owned cleanup are reviewed. The 1,163.771 native GPU seconds
and 6,586,748,344 transferred original bytes remain accounted. No source or
controller is adopted; further pressure-parameter variants are suspended.

The [saved controller-state diagnostic](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-motion-pressure-selected-controller-array-diagnostic-20261104.json)
separates measured normal pose error from the previous elastic target and the
original arm compliance across all four available trajectories. Both pose and
pressure loops integrate the normal goal. This supports testing a shared
finite-normal controller, without establishing that this coupling alone caused
every native failure. The first diagnostic reader's incorrect phase-code
assumption is retained and corrected using the production phase table; no native
condition was repeated because of that reader error.

The optional `pose_normal_compliance` implementation preserves measured-error
integration and causal reference tracking on the other five axes while using
absolute position feedback along the current requested tool normal. It removes
previous normal target memory and refuses a numerically unreachable pure normal.
The bounded target is the only retained command state. Inner impedance, all
physical/rate/reserved-target limits, stopping, admission and load criteria are
unchanged. Baseline source is `b55785e`; this prospective implementation starts
from `c13028b`. The [actual CUDA controls](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-compliance-rank-guard-original-history-controls-local-20261104/review.json)
preserve disabled default/feedback/reference results exactly across 34,392
commands from all eight original complete 0 N/10 N histories. Enabled command
bounds, repeated previews and actual environment dispatch/reset/stop history
semantics pass. These are single-command counterfactual and software checks,
not changed-controller rollouts. Full original four-condition 0 N/10 N native
qualification, including the original 1 mm contact ramp and independent stops,
precedes any door-contact or observed-source successor. The separate observed
P geometry proof remains unresolved. The matrix stays 17/36.

The [first finite-normal required-load native case](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-compliance-N10-negative-qualification-20261104.json)
closes independently negative on Blackwell, with the actual option enabled in
source `4619165`. After 540 pose deliveries, the original necessary 1 mm ramp
margin fails: normal error is 5.331 mm and margin is −4.331 mm. Approach/ramp
10 mm/5 degree endpoint ranges pass; opening, hold, release and stabilization
remain unavailable. All 3,120 native samples, whole nineteen-file plus
three-sidecar transfer and actual owned cleanup are verified. Original frozen
moving/stopped model residuals, current/future physical boxes and all three
one-latch 2+4+2 s/fourteen-shape stops pass. Empirical current and original future
position support fail in 1,815 and 2,095 samples respectively; no failed-point
extension follows. The other declared full-path qualifications were left running at that check;
the subsequent six-case reconciliation below preserves their actual outcomes.

The [shared complete-history compliance audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-finite-normal-complete-original-compliance-diagnostic-20261104.json)
uses all eight original complete native histories and their recorded stiffness.
At the new negative witness, `n.T Jp diag(1/Kp) Jp.T n * 10 N` predicts 5.272 mm
versus the measured 5.331 mm. Actual linearized target deflection is also
5.331 mm. Finite normal feedback removes pose integration into contact but loses
constant-load rejection. This static diagnostic neither proves dynamic contact
stability nor selects a new gain. A shared causal compliance/load correction and
complete related-condition qualification are required. No native door attempt,
observed source, calibration or criterion change follows from the candidate.
The family was unadopted at that intermediate snapshot; complete closure follows below.

The [reconciled six-case readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-compliance-six-reviewed-two-pending-20261104.json)
independently closes the N/H+/P complete zero-load paths and all original stops
positive in their component scope. H+/P's paired 10 N ramps also fail, at normal
errors 5.219 and 5.750 mm, with negative original 1 mm margins and empirical
position-support exits. Their original frozen moving/stopped residuals, physical
current/future boxes and one-latch eight-second/fourteen-shape stops pass. The
complete mixed recordings remain unchanged; zero-load results do not promote
the failed loaded cases. Six declared cases are reviewed; only H-'s original
zero/ten pair remains pending at its last owned healthy check. No native source,
calibration or controller adoption follows.

The [complete eight-case finite-normal closure](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-finite-normal-eight-closed-qualification-20261104.json)
independently verifies H-'s final zero-load path and 10 N negative, closing all
eight declared outcomes. H-'s ramp normal error is 5.433 mm. All four zero-load
full-path components and stops pass; all four required 10 N ramp/empirical
position gates fail. The 90,864 native and 15,360 original stop samples, whole
87,367,177 original bytes and 1,702.885 native GPU seconds remain accounted.
No allocation remains active in this family. No controller or source is adopted.

A [complete-history fixed-wall boundary audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-exact-load-cancellation-contact-feedback-rejection-20261104.json)
rejects blind exact load feedforward. For normal compliance C and positive wall
penetration h, exact causal load cancellation gives `F_next = h/C + F_previous`;
there is no finite contact equilibrium in this scalar boundary model. This is
not a native passivity or dynamics result and does not select a replacement gain.

The shared implementation now supports explicit `normal_reference_only` teacher
pose commands during existing force-admittance control. It keeps delivered
elastic support and pays only the causal normal goal increment, removing the
second normal pose-error integration. Other pose axes, original force requests,
gains, limits, phases and stops remain. Acquisition before force support and
intentional release use ordinary pose feedback. Actual teacher IO propagates the
same mode through preview, admitted proposal and dispatch; no evaluator,
force-truth or new learned observation enters the controller. Baseline is
`e96758e`, with no pre-existing tracked changes. The [actual CUDA history audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-incremental-normal-original-history-controls-local-20261104/review.json)
preserves default, causal-reference and disabled finite-normal results bitwise
across 34,392 original commands. Enabled targets, queued mode, preview/dispatch
and reset/joint/stop histories pass their software scope. The [bounded scalar
admittance audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-incremental-normal-bounded-admittance-scalar-controls-20261104.json)
retains all original compliance ranges, gain, pressure requests and clipping;
it does not prove native stability. Full original 0 N/10 N paths with explicit
mode entry/exit and original 1 mm ramp/hold/release/stabilization/stops are still
required before any door-contact successor. No matrix cell or source is added.

The [original eight-case transition family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-incremental-normal-zero-ten-native-manifest-20261104.json)
from `a11f4f1` is fully closed in the
[independent outcome reconciliation](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-incremental-normal-eight-closed-qualification-20261104.json).
All four zero-load complete six-phase paths and original stops pass their
robot-only component scope. All required 10 N ramps pass their original 1 mm
margin, but opening errors are 21.360 mm (N), 20.548 mm (H+), 21.688 mm (H−)
and 21.137 mm (P), failing the original 10 mm range. Hold/release/stabilization
are unavailable for those loaded cases. Original physical current/future boxes,
frozen moving/stopped residuals and one-latch eight-second/fourteen-shape stops
pass; empirical current/future positions do not. Whole transfer, actual owned
process absence and lease release are verified. No component promotes a failed
loaded assignment, contact, calibration or scientific source.

The [shared whole-motion covector diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-incremental-normal-all4-elastic-covector-diagnosis-20261104.json)
uses measured native Jacobians, delivered joint targets and original motion
stiffness across all four completed available motions. Transporting only the
change of the normal joint-load covector predicts the opening drift with
linearized residuals of 0.038, 0.009, −0.820 and 0.274 mm for N/H+/H−/P.
These are saved-trajectory diagnostics, not results of a changed closed loop.
Full compliance-change compensation counts both Jacobian factors and roughly
doubles the missing term. Neither that compensation nor exact force cancellation
is adopted. The original reader's unavailable cold-entry goal error is retained
and corrected by zero transport when dispatched geometry history is absent.

Optional `pose_normal_transport` implements that common geometric increment
with no fitted coefficient, force truth, changed gain, domain or threshold.
History is committed only on actual dispatch and cleared on mode exit, joint,
reset and stop. Fixed geometry produces zero transport for any elastic offset.
Baseline `45ebf24` has no pre-existing tracked changes. Essential numerical,
coupled-axis, clipping, teacher IO and setup controls pass. The
[actual CUDA saved-history review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-history-closure-local-20261104/review.json)
verifies all 34,392 original commands, bitwise default/finite/prior-mode parity,
new target/rate bounds, pure previews and actual environment history semantics.
Its initial audit-generation syntax failure is retained with accounting and
owned cleanup; no native motion is repeated. Original four-condition 0 N/10 N
full native paths, mode transitions, 1 mm ramps and all stops remain required.
No native door or observed-source successor is admitted by software results.

The [geometric-transport eight-case family](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-zero-ten-native-manifest-20261104.json)
from `f3a1905` now has complete
[outcome reconciliation](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-original-eight-closed-outcomes-20261104.json).
Seven original complete robot-only path/stop cases pass. H+ at zero load fails
one original native shoulder-acceleration sample at release entry: 7.929262
against 7.876929 rad/s². The
[all-eight native limit diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-all8-native-limit-diagnosis-20261104.json)
retains the witness and unchanged bounds. The outcome-routed H+ reader verifies
both complete pose paths, frozen moving/stopped residuals, physical current/
future boxes and all original continuing eight-second/fourteen-shape stops,
while preserving the zero-load acceleration negative and separate loaded
component positive. Whole original 101,277,473 bytes, 156,768 native and 15,360
stop samples, 1,901.827 native GPU seconds and owned cleanup are retained.
No controller or source is adopted.

The [all-seven-coordinate empirical readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-all-eight-latest-qualified-all-coordinate-position-requirements-20261104.json)
retains three uncovered condition/joint sides from complete available motion.
N's loaded release has 12 original future refusals with maximum deficit 0.077 mrad;
H− also exits current/future measured support. H+/P are covered. The original
scalar gate and full coordinate comparison remain separate, and the native
acceleration negative remains explicit. No support frontier is extended.

The [all-eight mode-exit diagnosis](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-normal-transport-all8-release-reference-handoff-diagnosis-20261104.json)
finds that release correctly rebases to achieved pose, but causal feedforward
still subtracts the previous ideal force goal. Every first release command asks
for only 5.639 micrometers of translation and zero rotation; stale reference
steps instead pay 0.041–9.605 mm of prior tracking/compliance error. H+ at zero
load receives 2.337 mm of artificial reference step at its acceleration witness.
This is a shared mode-history defect, not a reason to change acceleration bounds.

At baseline `25b5b09`, with no pre-existing tracked changes, the optional transport
mode now uses measured tool pose as reference origin on its first ordinary-pose
command. Its delivered target, original feedback/centering/gains, commanded path,
limits and stops remain. Later dispatches resume ordinary goal history. The
[actual CUDA handoff review](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-release-handoff-controls-local-20261104/review.json)
reproduces all eight old native target commands exactly, verifies corrected
measured-origin increments, invariance to stale-goal position/orientation,
pure previews and exactly one recorded dispatch handoff. Default/finite/old-mode
math remains bitwise identical across 34,392 original command boundaries;
165 focused CPU regressions and documentation checks pass their software scope.
These one-command controls do not qualify changed dynamics. All eight original
0 N/10 N paths, phases, empirical decisions and stops remain required before any
dependent stress, door-contact, observation or source successor.

The [distinct complete moving N probe at the original required 10 N](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-complete-moving-10N-N-manifest-20261104.json)
finishes negative on elab4. It uses the qualified frozen physical fixture
`d0281c4`, the same common initial posture, target reservation and gains, and
requests all 4,659 original six-phase commands with achieved staged release,
held stabilization and original continuing stops. This differs from the
completed 0.3 N paths and short joint-side pulses; the N empirical boundary is
not extended by its earlier 26-microradian deficit.

The [independent negative review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab4/D4-complete-moving-10N-N-elab4-20261104/complete-negative-qualification-review.json)
retains 15,996 healthy native samples through the complete approach, ramp and
opening. Approach/ramp endpoints pass. At the frozen original opening endpoint,
actual position error is 87.570 mm and orientation error 0.130285 rad, exceeding
the unchanged 10 mm/5 degree limits. Hold, release and stabilization remain
unavailable. All moving/stop frozen residuals, actual current/future physical
limits, feedback, speed and acceleration checks pass, with no effort saturation.
The original continuous 2+4+2 s/fourteen-shape stops independently pass against one
original latch; their 1,920 native samples are retained. Whole 19-file plus
three-sidecar transfer and actual owned cleanup pass. The native allocation
charges 209.582 GPU seconds and its independent CUDA review 9.050 seconds.

The [whole-motion mechanism readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab4/D4-complete-moving-10N-N-elab4-20261104/whole-motion-mechanism-diagnosis.json)
separates a 5.574 mm target-FK endpoint error from 87.982 mm actual-to-target
motion. Shoulder-X target/position difference reaches 127.943 mrad at that
endpoint; the gripper command occupies its upper target bound for 5,416 native
samples. These observations identify loaded tracking and limited command freedom
without proving a particular controller repair. The next shared correction must
jointly address measured load/compliance, target reserves and all original
N/H+/H-/P complete paths, preserving physical/model/future/pose/stop criteria.
The composed center-site load is a robot-only intervention, with no observed
door contact or full-source success. All dependent unchanged native attempts
remain suspended. The first preflight used the workstation runtime path; that
zero-launch setup failure is retained, and the actual elab4 runtime passed before
the sole launch. No calibration domain, source, window or matrix cell is added.

A shared optional [[../topics/purdue-b1-robot-and-contact#Common benchmark controller and stop|outer pose-feedback mode]]
addresses the observed elastic bias without changing the inner actuator gains.
It integrates measured pose correction on the previous delivered target using
an explicit time constant, preserves the original nullspace/rate/reserved-target
bounds and has no hidden state beyond the delivered target. The default remains
the original absolute IK. Numerical bias-rejection, invalid-state, pure-preview
and saturation controls pass. An [actual CUDA saved-path audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-actual-saved-path-controls-local-20261104/review.json)
verifies exact original-default tensor parity and bounded, repeatable candidate
proposals on all four complete 0.3 N paths and the 10 N negative prefix. The
0.5 s candidate time constant is declared for qualification; no favorable
native parameter selection or trajectory equivalence is inferred from saved
single-packet proposals. Original clipping and missing later phases remain
explicit. These saved controls alone do not adopt the controller for collection.
The [four-condition native manifest](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-complete-required-load-manifest-20261104.json)
assigns each original condition exactly once across elab4/elab3/elab5/Blackwell,
using frozen source `7bd2ea5`. Actual runtime wiring, required 10 N force and
GPU UUID/context/PGID/canonical ownership are verified on all four. The
[independent four-condition closure](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-native-qualified-components-20261104.json)
now passes all original six phases, frozen residuals, physical/future limits and
continuous 2+4+2 s/fourteen-shape stops with that single fixed controller. The
78,384 native motion samples and 7,680 continuing-stop samples are retained.
Opening position errors are 0.124 mm in N, 0.112 mm in H+, 5.482 mm in H- and
0.094 mm in P, all within the original 10 mm criterion. H-'s larger error does
not justify a different tolerance or controller. Whole transfers and actual
owned cleanup pass. Native costs total 1,299.104 GPU seconds, and the four
independent CUDA reviews add 36.196 seconds. The earlier 87.570 mm N failure
remains negative with its original source and cost.

These are complete center-site required-load robot-only components, with ideal
geometric goals and a composed load. They do not qualify observed door contact,
the full source or a larger empirical position domain. The changed trajectories
require [17 new measured joint-side postures](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-whole-motion-10N-support-manifest-20261104.json),
derived from every actual native sample with one unchanged stop/acceleration/
measured-velocity allowance. The 68 distinct cases retain unloaded controls,
center 10 N, face0/+0.2 rad yaw and face1/-0.2 rad yaw, original seven-coordinate
boundary servo/pulse limits and 204 continuous stop segments. A necessary
[static CUDA torque-balance check](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-whole-motion-10N-static-preflight-local-20261104/review.json)
passes all 68 original physical target reserves; native convergence, residual,
loaded coverage and stop qualification remain pending. It charges 7.036 GPU
seconds with owned cleanup. The [complete 68-case independent qualification](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-pose-feedback-whole-motion-10N-qualification-20261104.json)
now passes all 17 postures across the four conditions: 319,920 moving native
samples, 130,560 continuing-stop samples and all 204 original stop segments.
Loaded moving and original stop-onset coverage pass at all three sites, with the
preregistered pulse caps retained. Whole outputs and actual owned cleanup pass.
Native cost is 4,505.595 GPU seconds and independent reviews add 173.489 seconds.
Only the conservative common three-site measured frontiers enter the new
condition-scoped calibrations; outside-side and 11 N checks still refuse. The
global default calibration, physical and residual limits remain unchanged.
Observed native transfer, full-source qualification and matrix/window credit
remain separate. Elab5's initial zero-launch Torch preflight failure remains
retained; its existing native-job Python passed before the sole launch, without
installing dependencies.

The [D3 terminal-view discriminator](../../../outputs/evidence/act-a1-complete-cycle-20261026/D3-common-terminal-view-static-design-local-20261102/all-P-roots-original-union-independent-review.json)
checks the only original recorded view with available common N/P selector choices,
at 21 s. Both eligible P roots retain zero material cells under the original union
operator, even when independent static-world depth proofs are queried after all
ten views become available at 34 s. Exact causal-prefix control arrays match.
The 21 s references and every view keep their own clocks; this is no earlier
native initialization or later-data transplant. The separate maximum-of-individual-
certificates probe is not the maintained union and supplies no new task support.
D3-P actually initialized at 34.067 s in the earlier run, after the unsupported
34 s sample. A terminal-view change alone therefore has no qualified full-task
solution from these measured alternatives, and no new D3 native retry follows.

A [finite matched native cost comparison](../../../outputs/evidence/act-a1-complete-cycle-20261026/admission-cost-matched-native-pair-20261101.json)
uses one original D2 nominal setup on elab5, the same 780-command prefix including
240 prescribed push commands, and unchanged recording/physical/observation/stop
requirements. Its first baseline reaches 360 approach and 40 contact commands,
then hits the 2,000 s compute allowance. The
[independent interruption review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab5/D2-N-admission-cost-baseline-guarded-prefix-elab5-20261101/complete-interruption-review.json)
retains 400 deliveries, one admitted-but-undelivered proposal, 802 load packets,
zero labels/windows, original continuous 6 s/14-shape stops, actual RGB and whole
transfer/cleanup. It charges 2,099.729 GPU seconds. Its 401 geometry calls cost
899.404 s; the recent median is 2.834 s for 284 checked volumes. These counts
confirm persistent repeated work, without establishing an actual speedup.

The proportional observed whole-prefix forecast is 4,084.722 s. A single
capacity successor declares 5,000 s case/5,200 s wrapper allowance for both
members of the same fixed comparison. It also interrupts during push after
737/780 deliveries, retaining one admitted-but-undelivered proposal and 1,476
load packets. The [second interruption review](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab5/D2-N-admission-cost-baseline-capacity-guarded-prefix-elab5-20261101/complete-interruption-review.json)
independently passes raw/native/model/clocks, original continuous 6 s/14-shape
stops, actual RGB and whole transfer/cleanup, preserving 5,105.248 GPU seconds.
Its 738 geometry calls cost 3,036.970 s; the last 180 median is 4.997 s and 589
checked volumes. The measured 5,397.820 s whole-prefix projection supersedes the
older low forecast without erasing it. The 8,000/8,500 s capacity successor now
[qualifies the complete declared prefix](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab5/D2-N-admission-cost-baseline-capacity-v2-guarded-prefix-elab5-20261102/complete-prefix-qualification-review.json):
360 approach, 180 contact and 240 push deliveries, exactly 780 proposals and 1,560
original 10 N load packets, with no undelivered proposal. Native feedback,
original continuous 6 s/14-shape stops, actual RGB, all 47 original files plus
three sidecars and owned cleanup independently pass. Whole allocation wall is
3,968.382 s and charged GPU time is 3,973.382 s; 780 admission calls total
2,184.937 s. These are measured values of this finite run, with both earlier
interruptions and their costs retained, rather than a speedup claim.

The single prepared optimized `8139034` member is freshly staged and dispatched
on the same elab5 after baseline closure. All five active numeric input files
match after label/documentary-scope normalization; the original setup, seed,
780-command/240-push cutoff, observation pipeline, 10 N support and stopping criteria are
unchanged, with matching 8,000/8,500 s capacity. Source-archive verification and
actual-runtime constructor preflight pass. The current material-column change
is not substituted into either frozen comparison source. The optimized member
now [independently qualifies its full declared prefix](../../../outputs/evidence/act-a1-complete-cycle-20261026/remote-results/elab5/D2-N-admission-cost-optimized-guarded-prefix-elab5-20261101/complete-prefix-qualification-review.json):
780 deliveries, 1,560 original load packets, zero undelivered proposals, original
continuous 6 s/14-shape stops, actual RGB and whole 47-file/three-sidecar transfer
and owned cleanup pass. The controlled endpoint is the same 38.4 s/3.558794-degree
partial opening; it is not the complete task.

The [completed cost-pair readback](../../../outputs/evidence/act-a1-complete-cycle-20261026/admission-cost-matched-native-pair-complete-review-20261102.json)
measures whole allocation wall of 3,968.382 versus 2,473.502 s, a 37.67% reduction
or 1.604 ratio in this one same-host/configuration pair. The 780 geometry calls
total 2,184.937 versus 1,051.936 s; recent-180 median call wall is 3.192 versus
1.232 s and median checked volumes fall from 335 to 53. Both original physical
and scientific contracts still pass. Actual charge is 2,478.502 GPU seconds for
the optimized member, with all earlier interruption charges preserved.

A bounded streaming comparison retains all 67 raw HDF5 datasets and both
original stop files. Sixty-five datasets, including depth, native feedback,
control, evaluator and commands, and every stop operand are identical. The RGB
and its derived three preprocessed color channels differ; mean absolute RGB
channel difference is 0.780/255, with 95th percentile 3/255. Current perception
normal-cone and load-normal estimates also differ, while original force upper
bounds and all 10 N verification fields match. The rendering cause is not
independently established. Thus this is a measured single-prefix cost result,
not a pixel-identical perception-input causal experiment, repeated-run variance
estimate, universal multi-host speedup or full hold/release qualification. Neither
member nor either interruption adds a matrix attempt, selected source or window.

## Resources, archival and closeout

The latest historical authorization removes cumulative GPU-hour, allocation-count
and retained-data administrative ceilings. It does not remove physical safety,
scientific admission, actual disk-capacity checks or finite per-job watchdogs.
The historical pause is preserved; October 31 authorization restores execution
within the same scientific scope. Earlier 12 GPUh,
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
not transfer ownership. Current authority includes the local
RTX 4090, Blackwell and elab2–5, with explicit transfer approval for source, four
TRAIN doors, robot assets, configurations/calibration and perception runtimes.
Remote deployment is not scientific equivalence. Fresh host/runtime checks,
matched observed-input diagnostics and full-cycle attempts retain separate counts. Default watchdogs are 2400 wall/
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


## November 6 measured-release native qualification

The resumed `0edb99f` family independently passes all eight original N/H+/H−/P
0/10 N complete six-phase paths and one-latch eight-second, fourteen-shape stops.
Whole outputs and three sidecars are retained; owned process/GPU absence and
lease release are verified. The unchanged shoulder acceleration limit is
7.876929 rad/s²; H+ now peaks at 1.874785 rad/s² at zero load. The historical
7.929262 failure remains negative. These are robot-only fixed-load components.

The [complete support audit](../../../outputs/evidence/act-a1-complete-cycle-20261026/D4-measured-release-all-eight-closeout-and-support-20261106.json)
checks all seven coordinates against the latest independently qualified support.
N, H+ and P have zero current or single-future-interval exits at both loads.
H− retains 746/1,118 current and 1,381/1,417 future exits at 0/10 N, despite passing
physical/model/stop checks. Its dependent contact attempts remain suspended;
no empirical boundary is extended. Changing-load contact is the next independent
control question for the three supported conditions. The matrix remains 17/36.
Native execution cost was 1,858.813907 GPU seconds; independent CUDA review cost
29.234440 seconds, separately recorded from geometric profiling and its failed
reader preparations.


The static-depth query profile identifies repeated recorded-image validity and
minimum queries as the main D4-P replay cost. An exact integral validity index
passes the existing occupied/unknown/background and continuous admission tests,
plus 304 nominal, boundary and perturbed rectangle comparisons against raw pixels.
The complete saved D4-P producer retains every diagnostic field, query count and
negative decision exactly: warm profiled calls average 0.387 s versus 0.430 s
before the change (about 10% lower) on the local workstation. The first tiled
candidate was slower and remains a retained negative. These are saved-state
profiling results; a complete observed collection timing comparison is pending.

The actual N terminal snapshot also retains its preceding admitted proposal at
44.9333 s, one control interval before the preserved 44.95 s physical contact
loss. Its exact action, world endpoints, estimate and causal publication can be
combined with the finished raw joint state and original ten RGB-D views. GPU
endpoint readback agrees within the existing numerical comparison. Three paired
full-producer replays retain every diagnostic except wall time exactly. Raw-array
rectangle counting averages 1.577 s over the two warm profiled calls; indexed
counting averages 1.414 s, a 10.35% reduction. Cold calls are 4.037 and 3.685 s.
The reconstructed traversal/cache history differs from the original live call,
which checked 70 volumes; replay queries are 64 cold and 53 warm. These timings
therefore establish replay cost and proof parity, not native collection savings.
The remaining profile is dominated by subdivided background checks. Both failed
GPU preparations and the completed comparison, totaling 45.4055 GPU seconds,
are retained with owned cleanup. The canonical result is
`D4-N-reconstructed-producer-paired-cost-review-20261107.json`; no source or matrix
cell is added, and the original physical failure remains negative.


The three subsequent changing-load actual-door fixtures remain negative. N/H+/P
lose native contact after 100/91/97 push commands, at about 1.29/1.17/1.25°;
hold and stabilization are unavailable. The later abort-release guard exceeds
empirical position support and becomes the outer exception. That terminal error
must not obscure the preceding contact loss. All original continuous stop reviews
pass; whole original payloads, masks, RGB and charges are retained.

Collection reports now preserve `task_failure_before_exception` and
`exception_phase` when a recorder guard raises after the probe failure latch.
The same exception still terminates collection and remains the error/traceback;
qualification, control and safety behavior are unchanged. The complete causal
command readback matches delivered normal increments within 0.03 µm in its
linearized diagnostic. The next control question is motion/pressure consistency,
using the existing hinge-damping/speed/lever derivation, rather than extending
the later release boundary to relabel these failures.


The two whole-family-derived H− sides now pass eight independently reviewed
finite 0/10 N center/face-yaw cases and all 24 original same-latch stop segments.
Only those two position-support sides are added at the verified 10 N cap; all
other calibration fields and the previous 17 sides remain unchanged. All
156,768 original trajectory states and their single original future intervals
pass the resulting support audit; outside-position and 11 N controls refuse.
This is scoped model support, not an observed demonstration or a Cartesian box.

The next four-condition actual-contact fixture uses the existing physical rule
`max(contact_guard, hinge_damping * planned_angular_speed / normal_torque_lever)`:
4 Nm·s/rad, 1°/s and 0.197141 m imply a 0.354129 N push request. Contact and hold
requests remain 0.1 N. The mechanical controller is unchanged from qualified
`0edb99f`; `8bc4571` adds diagnostic failure context and exact static-depth query
reuse. No gain, force/model threshold, motion variation or stop criterion changes.
The original three negatives and earlier pressure-target negatives remain valid;
new complete changing-load outcomes are required before any observed source.


### Material motion on the force axis

After the four negative gap-credit interventions, CUDA native inertia readback
covers 232 saved postures from all eight original robot-only paths and both
four-condition contact families. The unchanged local bilateral-contact model
has no unstable poles in its 120 active-force postures. Increased damping is
not justified by this result; a derived critical-damping diagnostic destabilizes
part of the explicit numerical model and remains unadopted. These calculations
exclude unilateral impacts, nonlinear geometry and friction.

An optional `force_reference_material_flow` teacher mode now replaces only the
reference-clock normal placement with the current measured material placement.
It requires explicit force-reference control. The reference angle, other five
pose axes, injected disturbance displacement, force compression and every
physical/continuity/load/stop criterion remain unchanged. Causal native samples
still determine material placement; unloaded coasting cannot qualify a source.
Defaults preserve previous behavior. Essential geometric controls and local
models with completed-interval displacement pass. All four complete native
attempts then close negative: last-push angles 0.589°, 4.936°, 2.670° and 0.686°.
H+ has 0.617 s of hold and a completed abort release/stabilization; the others
retain secondary empirical withdrawal refusals. Raw/native/model/all-seven
current/future/physical/clock, original stops, actual RGB, whole outputs and
owned cleanup are independently reviewed. No source or matrix credit is added.
Normal-only transport leaves the gripper orientation on the reference clock.
The resulting 0.065–3.216° frame mismatch moves distal extrema by 30 µm–1.503 mm
in the saved geometric diagnostic; this is not counterfactual native evidence.
It motivates checking coherent material-frame transport before another contact
attempt. No actuator gains or acceptance limits change.


### Coherent material-frame contact reference

The normal-only negative family exposes a split reference: translation follows
the moving material while orientation remains on its time-based reference.
An optional `force_reference_material_frame` now applies one rigid transform to
the complete fresh contact goal and its local pressure, tangential compensation
and injected disturbance. It requires force-reference control and is distinct
from the retained normal-only mode needed by historical saved-data readers.
It does not transform an already transported goal again. Fixed frames preserve
goal values exactly; normal compression and full relative pose are conserved.
Braking, measured-origin return, actuator impedance, empirical/model bounds,
physical limits and all original task/contact/stop criteria remain unchanged.

Essential nominal, perturbed and fixed-frame controls pass. The same local
bilateral-contact model, now with coherent hinge-driven pose transport and causal
completed-interval displacement, is stable at all 120 active-force postures.
These numerical results do not qualify nonlinear native contact or observations.
All four changed native paths from `0963741` close negative: last-push angles
N 1.606°, H+ 0.471°, H− 1.983° and P 0.511°; none attempts hold. N first reaches
the original position-halo force upper bound, 10.550 N against qualified 10 N,
with physical and current/future position bounds clear. The other three lose
loaded contact before secondary abort-release empirical position exits. Their
native force peaks and model force bounds remain distinct; unavailable estimates
are preserved. No source or matrix credit is added.

Independent review covers 14,940 native task samples, original 6 s/14-shape stops,
all-seven current and single original future domains, physical limits, causal
clocks, RGB, whole original payloads and owned resource cleanup. All 836 push
compression updates reconstructed from the original acquisition ramp and saved
completed force/gap samples agree with the normal offsets in original float32
goal packets within 5 nm. Command-frame consistency alone does not establish continuous
contact. Dependent contact variants are suspended while the full unilateral
contact, sampled pressure and measured-gap recovery mechanism is audited; the
stable fixed-J bilateral model excludes these effects. All thresholds and prior
negative outcomes remain unchanged. Observed qualification remains open.


### Completed native gap for control

The whole-family sampled-contact audit identifies a temporal mismatch: pressure
recovery reused the maximum separation of the preceding native interval as the
current gap, even when its final sample shows that closure already happened.
Across 39 completed jobs in ten retained families, 966 of 8,014 saved pressure
updates change when only that operand is replaced. The largest unnecessary
closure request is 17 µm. These are single-update diagnostics on recorded inputs,
not counterfactual trajectories or newly qualified demonstrations.

The next displacement command now uses only the preceding interval's last native
separation. A missing endpoint remains unavailable; earlier maxima cannot fill
it. The original interval maximum and full missing-sample mask still determine
contact qualification. Acquisition, push/hold support and continued-contact
release recovery share this timing rule. Pressure feedback remains the original
mean/window selected by the recipe; no force maxima substitute for means.
Unloading priority, admittance gain, clipping, recovery gating, empirical load
support, physical limits and all full-task/stop requirements remain unchanged.
The trace records the actual control gap and its source separately. Moving,
closing, unavailable and invalid endpoint controls pass. All four complete changed
native paths from `012e826` then close negative: last-push angles N 1.932°,
H+ 0.464°, H− 1.614° and P 0.486°. Each loses loaded contact before hold and retains
a secondary abort-release empirical position exit. All 931 recorded endpoint
operands and original qualification maxima/missing masks independently agree;
815 reconstructed push normal offsets match original float32 goals within 5 nm.
Review covers 15,856 native task samples, original 6 s/14-shape stops, every
current/single-future empirical and physical domain, RGB, whole outputs and owned
cleanup. No source/matrix credit is added. The timing fix alone is insufficient;
no thresholds or calibration domains change.

### Whole rigid support point and posture

A saved-data diagnostic uses the unchanged UMI collision convex meshes, canonical
tool translation and actual completed poses on twelve native contact paths.
In the endpoint-gap family, orientation changes move the leading rigid support
by roughly 20–50 µm relative to the tool origin, with per-command changes up to
11 µm. Scalar 0.1 N elastic deflection is of comparable magnitude; this comparison
is not measured force or a fitted causal explanation. A local plane predicts
recorded endpoint gaps only approximately and does not replace finite observed
material, relief or swept-space certificates.

A separate CUDA FK audit covers all 156,768 native samples of the eight original
completed 0/10 N robot-only paths, preserving unavailable pose goals and all
prior outcomes. The normal contact covector includes the moment arm of the
current rigid support point, unlike the tool-origin position covector. These
calculations expose a model/control boundary across full posture and load
histories; they do not qualify a changed controller. Further native variants
remain suspended while actual shape motion, unilateral pressure, mode transitions
and original full-task constraints are reconciled.


### Rigid support reference relative to mode entry

The optional whole-frame mode now computes the normal support of the unchanged
full contact convex meshes under the actual and requested orientations. Their
difference is a geometric origin offset; only its change relative to mode entry
is applied. It does not integrate old offsets or infer force from target error.
This preserves the acquired load at entry while accounting for rigid fingertip
motion that the tool-origin point model omits. Contact orientation and tangential
reference stay unchanged. History is committed after a delivered mode command;
ordinary poses, braking and measured-origin return end that history.

The pressure, gap, load-window and continuity implementations are unchanged.
Compression gains, rates/extents, native force means, actuator impedance, frozen
six-dimensional residual-load calibration and every original physical/full-task/
stop criterion remain unchanged. The support geometry does not replace observed
finite material, relief or clearance evidence. Defaults and normal-only modes
keep their previous command behavior.

Nominal rotations, active-edge switches, rigid co-motion, fixed geometry and
invalid-input controls pass. CUDA FK on all 156,768 original native samples
checks 28,632 prospective mode commands across the complete eight 0/10 N paths;
maximum geometric offset is below 1 mm and maximum step is 27 µm on these saved
histories. An additional 1,883 commands in twelve completed contact histories
satisfy the same geometric identities; four disabled-mode histories stay
unchanged. These are geometry/control-reference audits, not counterfactual
physical trajectories. The complete changed native family from `519c347` also
closes negative: last-push angles N 0.458°, H+ 1.449°, H− 3.430° and P 0.527°.
All four lose loaded contact before hold, followed by empirical position refusals
during abort release. Hold and stabilization remain unattempted. Independent
review verifies all 16,852 native task samples, 1,184 full-mesh reference operands
and 1,068 pressure updates. Recorded normal goals agree within 5 nm after
accounting separately for the geometric reference. This is command parity, not
contact qualification. Current and single-future empirical exits are retained;
physical boxes, original 6 s/14-shape stops, raw feedback, actual RGB and whole
output/owned cleanup reviews pass. The four negative outcomes add no source,
window or matrix credit. The reference correction alone is insufficient.
Further dependent variants require a shared diagnosis of unilateral impact,
pressure sampling, posture and complete transitions, with all original criteria
and the frozen six-dimensional load model preserved.


### Limits of reduced unilateral dynamics

A CUDA diagnostic extends the fixed-posture model with a frictionless unilateral
normal impulse, original native mass matrices and actuator gains, full rigid
support moments, measured-frame motion, causal pressure updates and tangential
correction. At 120 previously measured postures, three local initial conditions
are compared over 20 seconds with matched bilateral runs: equilibrium, 2 µm
separation and 0.2 mm/s opening-velocity perturbation. The unilateral cases reach
8.112 µm gaps but retain at least 0.0667 N in the original 0.1 s force window;
none reproduces the native loss below 0.02 N. Native failures remain decisive.
This reduced model omits evolving inertia/Jacobians, distributed/frictional
contacts, jaw compliance and joint-domain clipping. Its favorable result limits
its explanatory scope and cannot justify a gain change or contact qualification.

The original recordings retain forces, gaps and actuator diagnostics but lack
complete native contact-point/normal/actor-pose histories across the terminal
exception. A diagnostic sidecar now writes each original contact packet,
measured joints/poses and native inertia at the matching 240 Hz feedback clock.
It returns the original packet unchanged and saves original fixture data before
closing the new stream. All four original N/H+/H−/P runs validate 16,852 sidecar
packets against their pressure streams: clocks, phases, q/dq/ddq, Jacobians,
targets, expected/feedback torques and normal-force sums agree exactly. Inertia
remains positive, and both finger poses/full joint states are retained. All 17
native arrays in each condition exactly match the previous same-host baseline.
Original stops, raw data, RGB and whole-transfer/owned cleanup pass; all tasks
remain negative with primary contact loss and secondary empirical release
refusals. This validates observability, with no source or matrix credit.

| Condition | Native push packets | Positive force | Both fingers loaded | Finger switches |
| --- | ---: | ---: | ---: | ---: |
| N | 320 | 236 | 2 | 8 |
| H+ | 1,144 | 942 | 2 | 7 |
| H− | 2,436 | 2,005 | 3 | 15 |
| P | 372 | 321 | 15 | 23 |

Switch counts omit intervening zero-force packets; packet counts are not
qualified loaded windows. The pressure center spans roughly ±19 mm in tool Z.
Mass-matrix changes from push entry are 0.47/1.64/3.95/0.50%. Actual right-jaw
motion reaches 25.4/29.0/74.5/27.3 µm; the pooled gripper metric includes left
jaws and cannot establish right-jaw deformation. Contact moments include measured
normal forces only, not the complete frictional wrench.

The actual Purdue configuration uses `IdealPDActuatorCfg`. Reconstructing its
explicit PD calculation from preceding q/dq/gravity and delivered targets exactly
matches every saved computed effort across all 16,852 packets, including clipping
to applied efforts. The diagnostic gravity getter records current-state model
gravity, however; using it as the preceding interval's feedforward introduces up
to 0.00964 N m error. `TemporalExpertRecorder` now describes that clock correctly;
no recorded values, controller behavior or criteria change. Normal-only command
and body-output dynamics retain unexplained contact residuals; friction, other
constraints and numerical terms remain unseparated. This supports further
measurement, not a gain change or a new contact qualification.

Evidence: `D4-contact-manifold-four-negative-closeout-20261107.json`,
`D4-contact-manifold-family-outcome-review-20261107.json` and
`D4-contact-manifold-native-torque-closure-audit-20261107.json` in the campaign
evidence directory. Native controller source remains `519c347`; the diagnostic
clock description and completed-family review follow baseline `57863e4`.

A separate nominal native diagnostic adds a filtered Tensor API view for the two
exact finger–panel pairs, recording normal and tangential patches at the original
240 Hz clock. Normal and friction getters share count/start buffers, so every
returned array is copied before the next getter. All 3,484 completed packets
verify; filtered and original normal force/moment sums agree exactly, and all 17
saved native arrays exactly match the previous same-host nominal recording.
The additional measurement therefore preserves this failed trajectory. Raw data,
original load/current/future/physical checks, 6-second fourteen-shape stops, RGB,
whole transfer and owned cleanup are independently reviewed. The task remains
negative at 0.458°, with no new demonstration or matrix cell.

During 320 push packets, friction is nonzero in 264 and normal force is positive
in 236. The 28 zero-normal/nonzero-friction packets reach only 0.00807 N friction;
retain these solver readbacks instead of reconstructing them from an assumed
instantaneous Coulomb cone. Adding the measured tangential wrench reduces the
largest per-joint command-dynamics RMS residual from 0.04968 to 0.000326 N m;
body-output residuals similarly shrink. This identifies a major omission in the
normal-only dynamics audit, not a validated controller intervention. The frozen
six-dimensional load model, gains, actor inputs and scientific criteria remain
unchanged. The subsequent H+/H−/P measurements independently verify the same readback,
including exact normal force/moment sums and all 17 baseline native arrays in
each condition. Complete-motion diagnosis remains necessary before a shared
correction. Evidence is in the nominal run's
`independent-native-friction-review.json` and
`complete-native-friction-observability-and-negative-outcome-closeout.json`.
The reader's initial NumPy-integer JSON serialization error is retained; its
correction changes only output encoding and adds no GPU allocation.

The complete four-posture friction review covers 16,852 native packets and
7,021,207,743 original bytes. All original tasks remain negative at the same
0.458/1.449/3.430/0.527° last-push angles; no hold, stabilization or source is
qualified. Every original task/load/current/future/physical check, stop, RGB pair,
whole transfer and owned cleanup is reviewed, retaining the primary and secondary
failure masks. Adding the measured tangential wrench changes the largest
per-joint push command-dynamics RMS residual as follows:

| Condition | Normal-only residual (N m) | Normal plus friction (N m) |
| --- | ---: | ---: |
| N | 0.0496753 | 0.000325755 |
| H+ | 0.0519848 | 0.000279792 |
| H− | 0.0620566 | 0.000334474 |
| P | 0.0468029 | 0.000304929 |

These measurements identify why a normal-only local model failed to reproduce
the native dynamics. They do not identify a stable closed-loop correction by
themselves. Any elastic compensation must preserve actual free-reference motion;
the previously withdrawn frozen-q compensation remains rejected. The full-family
review is `D4-native-friction-family-complete-review-20261107.json`; native
controller source remains `519c347`, with no gains or criteria changed.

A subsequent CUDA counterexample screens a proposed increment of measured
joint-target tracking error against the original controller and the withdrawn
frozen-q correction. Sixteen measured four-posture compliances and the recorded
friction inputs define 96 matched first-order cases. The proposed correction
reaches 4.147 mm normal error during free motion, versus 0.122 mm for the original
and 33.285 mm for the withdrawn correction; its loaded cases also lose force.
It is rejected before any production edit. The original toy plant now also loses
force when driven by the recorded friction, unlike the earlier normal-only
stable model. Prescribed friction, fixed geometry, first-order plant dynamics and
absence of native joint inertia/contact redistribution limit this result; it
neither validates a replacement controller nor predicts a new native trajectory.
The existing feedback time constant is used as a matched plant approximation,
not an identified native response. All model failures and the owned CUDA charge
are retained in `D4-causal-elastic-tracking-screen-negative-closeout-20261107.json`.

The next shared intervention addresses the target integrator rather than
canceling an inferred external load. During explicit normal-reference commands,
optional finite tangential impedance subtracts `J (target - q)` from the pose
error before the existing feedback gain, and removes the normal component before
the damped pose solve and tangential reference feedforward. With a fixed goal
and an already achieved target, blocked elastic displacement no longer grows the
target. The explicit normal pressure reference and existing geometric transport
remain separate. Default/prior modes retain exact saved-history results.

The isolated candidate passes actual-CUDA controls at sixteen measured
four-posture Jacobians: blocked-target no-windup, no damped-solve normal leakage,
free tangential reference, nullspace memory, bounds and preview purity. The
implemented option then passes the operational numerical regressions and 34,392
original CUDA command-history checks, including actual environment preview,
dispatch, joint override, reset, stop and all eight measured release handoffs.
No gains, force thresholds, scientific criteria or actor inputs change. These
are software controls; full native N/H+/H−/P × 0/10 N paths and original stops
must still qualify before a new contact attempt. Evidence is in
`finite-tangent-essential-CUDA-candidate-review-20261107.json` and
`D4-finite-tangent-controls-local-20261107/review.json`.

All eight native outcomes at source `700191d` are independently closed. The four
complete 0 N robot-only paths and original same-latch eight-second, fourteen-shape
stops pass. Every required 10 N case stops at the original ramp endpoint position
gate: N 11.4444 mm, H+ 11.0698 mm, H− 11.7677 mm and P 12.5259 mm against 10 mm.
The separate necessary normal compression margins stay positive; orientation,
physical current/future limits, rates, reserved targets, effort, frozen moving/
stopped residuals and original stops pass. Opening, hold, release, stabilization
and loaded mode exits remain unavailable. All 90,864 native samples, 15,360 stop
samples, 76,445,206 original bytes and actual owned cleanup are retained in
`D4-finite-tangent-eight-outcome-closeout-20261107.json`.

All seven current and single-future position coordinates pass using the latest
condition-specific support already independently qualified before this run,
including the two November 6 H− sides. The first derived N/P comparison mistakenly
used the historical calibration bundled in the job; its false refusals and masks
are preserved and superseded. Only previously qualified position tolerance and
force-cap arrays differ; the frozen dynamic model is identical. No support
extension or threshold change is introduced. The family remains unqualified at
10 N and adds no door-contact or scientific source evidence.

A subsequent all-posture causal audit finds that the tangential ramp error
matches the measured `J (target - q)` displacement within 0.082 mm. Its entry-
relative counterpart is only 0.048–0.361 mm in the saved operands; this is an
algebraic counterfactual, not a corrected native rollout. Optional finite
impedance now preserves the already delivered joint-target preload at mode
entry and subtracts only subsequent elastic displacement. It does not infer or
cancel an external force, fit a gain, freeze actual robot posture or change the
normal pressure channel. The offset is seeded only on dispatch, retained during
that mode and cleared on ordinary pose, joint, stop and reset transitions.

Essential CUDA controls cover all eight actual entry offsets, preserved preload,
blocked elastic equilibrium, free tangential reference motion, independent normal
increments, bounds and invalid inputs. Production CUDA checks cover all 34,392
original commands, eight actual-method entry histories and all eight measured
release handoffs. Default/prior behavior remains exact when the new reference
offset is absent. Original full native 0/10 N paths and stops are still required;
no contact or source is qualified by these software controls. Evidence is in
`finite-tangent-entry-preload-essential-CUDA-screen-20261107.json` and
`D4-entry-preload-controls-local-20261107/review.json`.


The entry-preload native family at `5a5caad` is independently closed across
141,656 movement samples, 15,360 original stop samples and 97,055,489 original
bytes. All four 0 N paths pass the full original endpoints, hold, achieved
release, stabilization, history, physical/model/current/future support and
same-latch eight-second/fourteen-shape stops. At 10 N, N/H+/P fail the opening
endpoint at 83.497, 80.858 and 82.883 mm, also beyond the original orientation
limit. H− fails normal reachability before completing opening, with earlier
acceleration and empirical-support violations. All four loaded paths leave
previously qualified position support; later phases stay unavailable. H−'s three
stop segments fail speed, acceleration and frozen-model criteria. Its shoulder
velocity alternates near ±4.47 rad/s, independently confirmed by position
differences. Original failed stop predicates remain explicit in the readback;
none is discarded because another check fails first. The canonical family result
is `D4-entry-preload-eight-outcome-closeout-20261107.json`.

Forward-only native inertia readback covers 356 candidate and qualified-baseline
postures without advancing dynamics. The seven-joint local explicit-PD stop
linearization has radius 1.0201 at the failed H− stop region, versus at most
0.9746 for the baseline samples. At the first failed-stop posture, its negative
pole is −1.01864 and its velocity direction has cosine 0.9999993 with the observed
alternating mode. The measured early amplitude ratio is 1.01980. This supports
posture-dependent local stop instability, not a gain change. Full twenty-joint
unconstrained models also predict instability in stable baseline cases, exposing
missing constraints; they are rejected as qualification gates.

Whole-path readback separates 78.65–105.15 mm loaded tangential errors from
18.78–26.31 mm normal error. The cumulative omitted full-offset covector term is
only 1.01–1.43 mm, so replacing that scalar term alone is unsupported. A separate
saved-state elastic-reference comparison transports the initial stiffness-
weighted reference with the measured tool frame. On the independently qualified
10 N histories, tangential residual RMS falls from 41.6–44.8 mm for a fixed joint
reference to 2.30–2.74 mm. This static algebra is an unimplemented geometry
hypothesis at that checkpoint, not a corrected trajectory, force-feedback
measurement or contact qualification.

The subsequent isolated candidate passes 48 matched elastic-plant cases with
free translation/rotation, steady/changing load, fast/slow response and blocked
motion. At fixed geometry its commands exactly match the prior seeded finite
controller; blocked targets do not drift. All sixteen saved native geometries
preserve entry elastic pose, rigid-coordinate invariance and the matched
transported equilibrium without reseeding from later delivered offsets.
The optional runtime now retains that six-axis tool-frame elastic reference,
transporting it with measured tool rotation and stiffness-weighted Jacobians.
Actual CUDA checks cover all 34,392 original single-command history boundaries,
eight real entry offsets and eight measured release origins, plus actual-method
preview/rejection/dispatch/mode-exit/joint/stop/reset semantics and original rate/
target bounds. Defaults and prior modes remain bitwise unchanged. Evidence is
`D4-tool-frame-elastic-reference-essential-CUDA-screen-20261107.json` and
`D4-tool-frame-elastic-reference-controls-local-20261107/review.json`.
These are software and matched-plant controls, not new native trajectories.
The qualification harness now applies original current/single-future empirical
support and physical predicates before dispatch/wrench updates and after each
completed four-sample native interval. The existing single future allowance is
paid once, using the latest already qualified scoped support. All eight positive
baselines and four earlier 0 N components pass the guard; all four earlier 10 N
negatives are detected first at future-support exits. Empirical refusal remains
distinct from physical safety. Float32-rounded boundary-test preparations and an
old metadata-field preflight refusal are preserved; only test representation and
the actual closure-field reference were corrected.

At `38a8efe`, all eight unchanged original 0/10 N cases were dispatched once on
Blackwell and elab2–4. Fresh archive/runtime/UUID/ownership/capacity checks pass.
One actual startup snapshot confirms four owned running allocations and the
new reference/guard flags, with no reported error. Later progress is unpolled;
no path, stop or source is newly qualified. The checkpoint is
`D4-tool-reference-software-and-owned-startup-checkpoint-20261107.json`.
All original full-path and continuing-stop criteria remain required before
contact/source adoption. Refusals retain actual completed intervals and leave
unexecuted phases unavailable; counters advance only for delivered commands.

All resources are closed. A completed inertia reader left only its own auxiliary
Omniverse Hub process in the allocation group; the GPU was already unused. Only
that verified PID received TERM, with group identity and actual charges retained.
The first reference-frame algebra run exposed nonorthogonality from float32 FK
rotation blocks; standard rigid-rotation normalization fixes its representation,
with that failed preparation retained. No native criterion changes. Evidence is
in `D4-entry-preload-native-and-mechanism-closeout-20261107.json`.


The first `38a8efe` outcome is an empirical refusal at N0N, not a physical
failure. At 35.9833 s during opening, three native packets leave the original
single-future lower shoulder-X support by at most 10.71 microradians. All current
position, physical, rate/reserve/effort and frozen-model predicates pass, as do
all original same-latch 2+4+2 s/fourteen-shape stops. Whole transfer and owned
cleanup are verified. The 10 N case is unattempted; opening remains incomplete
and hold/release/stabilization unavailable. The canonical result is
`D4-tool-reference-N-first-refusal-closeout-20261107.json`.

On 1,918 exactly matched original native references, qualified `0edb99f`, fixed
joint `5a5caad` and tool-frame `38a8efe` have common-prefix opening RMS position
errors of 0.160, 0.881 and 1.118 mm. The last available errors are 0.180, 1.253 and
1.688 mm. Available tracking remains inside the original pose criteria, but this
does not establish the missing complete path. The new empirical refusal is
preserved without changing support or thresholds. At the single wake snapshot,
P reports its 0 N case complete and continues at 10 N; H+/H− are healthy at 0 N.
Those declarations remain unreviewed and later progress is unpolled. Complete
family outcomes are required before a common mechanism or calibration decision.


The complete tool-frame family is closed in
`D4-tool-reference-whole-family-closeout-20261107.json`: seven actual cases and
one unattempted N10N case, with 114,620 movement and 13,440 stop samples. H−/P0N
and 10N, plus H+0N, qualify as five complete robot-only components. H+10N refuses
future empirical support at 33.35 s; N0N retains its earlier refusal. All seven
actual cases pass physical limits, frozen models, current position support and
original continuing stops. No demonstration or global calibration is added.

The shared secondary posture objective previously used physical target ranges.
At both actual refusal witnesses its nullspace step points slightly toward the
binding empirical edge. An isolated alternative, normalized by already qualified
condition support, points inward by about 0.33–0.34 mrad per control step at the
isolated screen's default gain 0.1, while leaving all six primary axes unchanged.
That scale is not the native configured gain of 2.0. The all-family CUDA audit covers 23
saved cases and 91,621 boundaries. It checks soft-cost descent and pose
projection on qualified/current histories; earlier rank-deficient failed states
remain diagnostic, not qualified controls.

The optional soft preference is now implemented with default behavior preserved.
It retains the existing centering gain and physical/target/rate bounds. Actual
CUDA original-history, entry/preview/dispatch/reset/stop and all eight measured
release controls pass. Matched free/changed-load/blocked-task elastic plants
preserve primary motion while allowing redundant-posture movement. Evidence is
`D4-tool-reference-whole-family-soft-posture-objective-CUDA-screen-20261107.json`,
`D4-soft-posture-controls-local-20261107/review.json`,
`D4-soft-posture-eight-actual-handoff-controls-CUDA-20261107.json` and
`D4-soft-posture-matched-elastic-plant-CUDA-screen-20261107.json`.
This is software qualification; new guarded native paths/stops remain required.
No failed-point support extension, criterion change or unchanged retry is made.


The `ab87675` midpoint-preference configuration is independently closed negative
in `D4-soft-posture-negative-whole-family-closeout-20261107.json`. Four actual
0 N approaches refuse only original single-future empirical support at
7.23–7.67 s. The binding upper joint is wrist X for N/H+/H− and shoulder Z for P.
Current empirical support, all physical/target/rate/effort and frozen-model
predicates pass, as do all original 2+4+2 s/fourteen-shape stops. The complete
7,152 movement and 7,680 stop samples are preserved. All four required 10 N cases
are unattempted. Normal-reference/preload mode is never entered; its fields stay
unavailable, rather than being treated as failed or completed mode evidence.

The configuration is rejected. The prior weighted cost can descend while a
different coupled face is exhausted; task-null projection alone cannot certify
operational support through the motion. The next diagnosis examines all seven
future faces jointly over 27 saved cases and 92,449 boundaries, including all
six actual refusal witnesses. At each witness, a rate-bounded scalar nullspace
interval can restore the existing future box with negligible pose projection
error. This is static algebra, not an actuator prediction or implemented
controller; it does not establish future velocity, load evolution or complete
motion. Evidence is
`D4-posture-whole-family-coupled-feasibility-CUDA-diagnosis-20261107.json`.
No threshold, support, gain or phase relaxation is made, and no further native
variant is launched. One early-outcome reader initially assumed an approach
endpoint existed; the missing-field failure and GPU charge are retained, and the
corrected reader explicitly leaves the unreached endpoints unavailable. All
owned resources are released; the matrix remains 17/36.


The subsequent finished-native response audit covers all 27 actual cases and
98,569 control intervals, including held-joint stabilization only where reached.
It preserves the exact identity between measured motion, delivered target change
and elastic-offset change. Assuming that offset stays constant gives eight
falsely favorable original future-support predictions; observed offset changes
reach 5.333 mrad within one control interval. Conversely, 863 intervals have a
delivered target outside measured empirical support while current measured
positions remain inside, including qualified H−10N components. Clamping targets
to measured support would therefore discard valid loaded behavior.

All four same-host unloaded approach pairs have exactly identical native goal
coordinates/clocks, setups and six relocated input payloads. Their measured
joint differences reach 0.120–0.173 rad, supporting the changed redundant-posture
interpretation rather than a changed task reference. The earlier isolated
midpoint plant/static preference screens used gain 0.1. All native setups use
2.0; the corrected 48 matched-plant cases pass at that value. These fixed-geometry
controls still do not qualify native posture evolution or reverse its rejection.

A proposed four-substep position-integration model also fails the original
float32 arithmetic comparison on every history, with a 3.815 microradian peak
position/velocity defect. Its conditional acceleration/velocity intervals remain
model diagnostics and cannot establish native reachability. This is not a new
numerical tolerance, a universal resolution bound or a change to existing
qualified stop evidence. Both that model refusal and a prior recipe-path reader
preparation error are retained. All five new allocations are closed and charged
once. No controller, native successor, support/criterion change or demonstration
is added; the matrix remains 17/36. The next coupled posture proposal needs an
independently supported causal actuator response over complete motion and mode
transitions. Evidence is
`D4-posture-causal-response-and-model-validity-closeout-20261107.json`,
`D4-posture-whole-family-causal-response-CUDA-diagnosis-20261107.json` and
`D4-soft-posture-actual-gain-matched-elastic-plant-CUDA-screen-20261107.json`.


A read-only native actuator journal establishes the causal command clock without
changing control. It copies full-system inertia, armature, gravity/Coriolis,
actual simulation effort, COM Jacobian and composed wrench after command writes,
then native states after the physics update. On original N0/10N finite controls,
all 2,016 existing feedback packets are identical with and without the observer;
all original continuing stops pass. Its 5,856 initial paired packets also verify
the pre-physics explicit-PD effort clock. Interrupted close handling preserves
an unmatched prestate with its poststate unavailable.

The four full-family diagnostics are now independently closed positive. All
eight original N/H+/H−/P0N/10N paths pass complete phases, original 1 mm ramp,
hold, achieved release/stabilization, physical/target/rate/effort, frozen models,
all seven current/single-future empirical predicates and original 2+4+2 s,
fourteen-shape stops. The 156,768 movement/settle/preload and 15,360 stop packets
join the journal exactly. All original files and sidecars are transferred and
owned resources released. Executable `ab87675` runs the already qualified
`0edb99f` configuration, with finite tangential impedance and soft posture off.
This is a baseline response diagnostic, not a new controller or demonstration.

The unconstrained twenty-coordinate inverse remains inaccurate. The robot URDF
contains two WSG mimic relations; motor armature must also enter inertia. A
nominal eighteen-coordinate reduction lowers arm-velocity prediction errors to
microradians/s. An initial demand for bitwise identical mimic positions fails;
that stronger-than-needed input test and its charge are retained. Positions are
not replaced, and no physical tolerance is widened. Projecting next momentum
instead accepts actual measured state, including tiny native constraint defects.
The model uses source-defined mimic relations, armature and passive damping,
with no fitted parameters. Over all 172,128 packets, peak arm-velocity error is
7.023 microradians/s. Algebraic command-impulse and float32 input perturbations
pass without imposing a new bitwise state gate.

This supports a causal one-physics-step velocity operator. It does not establish
next-position integration, a complete changed-controller trajectory or camera,
material, contact or observed-source admission. Actual wrench truth belongs only
to this independent diagnostic and cannot become an observed-controller input.
The next posture proposal must retain delivered target elasticity, primary
motion, all original future-support faces and mode transitions. Every earlier
negative, preparation failure, unavailable field and resource charge is retained;
the matrix remains 17/36. Evidence is
`D4-actuator-response-whole-family-closeout-20261107.json`,
`D4-actuator-response-all-native-packet-joins-20261107.json` and
`D4-actuator-response-whole-family-constrained-momentum-CUDA-20261107.json`.


The isolated four-substep extension uses the supported constrained momentum
operator, original full gains, retained elastic targets and primary motion. A
causal load residual uses only the last completed interval. Across 39,184 original
intervals, its position error reaches 28.664 microradians; even known current
fixture forcing does not make fixed geometry and position integration exact.
Algebraic perturbation controls pass without fitting a correction or changing
criteria. This model remains an approximate proposal, not an admission gate.

Four independent native jobs test 64 saved full states across the original
conditions, loads and phases. Seven command axes and a command-response task-null
direction are perturbed with both signs at 0.001 and 0.003 rad. Of 2,176 declared
trials, 2,078 dispatch; 32 remain unattempted at reserved target boundaries and
66 follow two zero-perturbation empirical future refusals at loaded N/H− hold
states. Their physical/current/rate/acceleration predicates pass. The refusals
are preserved, with no dependent retry. The 960 available central pairs cover
15,992 native packets, including four original final 2+4+2 s stops. All dispatched
predicates are independently reproduced, original stop payloads join the
journal exactly, complete originals transfer, and owned resources close.

Native position-response error peaks at 11.562 microradians and velocity-response
error at 135.233 microradians/s. Zero-command repeat positions are identical.
An exact error decomposition using measured native velocities attributes up to
12.104 microradians to position updating and less than 0.979 microradians to the
integrated velocity-response model. No precision floor, fitted correction or
acceptance buffer is inferred. These local interventions do not qualify a new
controller or continuous trajectory. Next work is a coupled posture preference
using measured seven-joint margins and approximate actuator response, with
original guards, gains, history and full-motion qualification preserved. The
matrix remains 17/36. Evidence is
`D4-four-substep-response-whole-family-closeout-20261107.json`,
`D4-four-substep-causal-response-whole-family-CUDA-20261107.json`,
`D4-four-substep-native-interventions-whole-family-CUDA-20261107.json` and
`D4-four-substep-native-position-error-decomposition-CUDA-20261107.json`.


The next optional posture implementation uses a coupled all-face preference,
while retaining the already bounded primary target and original gain 2.0.
A scalar correction remains in the kinematic Jacobian nullspace; the constrained
four-substep actuator response supplies its approximate effect on measured
joint margins. Physical target/rate limits remain separate from empirical
support. The score uses the original measured-speed allowance once and cannot
create an admission refusal, even when negative. Loaded targets are not clipped
to measured support. The API defaults unset and is separate from midpoint-only
centering.

An initial objective also penalized proposed endpoint speed. Its 34,392
single-command controls passed, but the 48-case closed-loop posture screen
failed convergence: a one-tick inward correction could be no better than rest.
An exact scalar witness retains that failure. The measured-margin correction
passes the same full-history and 48-case free/changing-load/blocked elastic
plant controls without changing acceptance criteria. The largest primary-target
difference is 2.68e-15 in the matched plants, and the largest final redundant
coordinate falls from 0.85 to 0.45056 rad. These are model controls, not native
trajectory or contact qualification.

Production math matches the isolated version on all 34,392 commands. The
unchanged default pose math and disabled actual-method previews retain parity.
Eight original saved cases pass pure preview, exact dispatch, entry reference,
no reseeding, rejected-preview invariance, measured-origin exit, joint override,
idempotent stop and reset controls. One initial method assertion compared a raw
quaternion with the normalized queued request; its failure and cost remain,
and canonical-request preview/dispatch equality passes all eight cases. The
224 focused tests and lint pass. All six allocations, including both failed
controls, are closed and charged. The new source still requires all eight
original guarded 0/10 N native full paths and stops before contact or source
adoption; the matrix remains 17/36. Canonical source-control evidence is
`D4-measured-coupled-margin-source-controls-closeout-20261107.json`.


All eight declared `6a27ffb` native outcomes are closed. H−0N passes the original
six phases, 1 mm ramp, hold, achieved release/stabilization, frozen models, all
seven current/future predicates and original fourteen-shape 2+4+2 s stops.
N/H+/P0N stop during approach at 7.50/7.05/7.167 s; H−10N stops during hold at
68.333 s. All four failures are original empirical single-future refusals;
current measured support and every physical, speed, acceleration, effort,
reserved-target, frozen-model and stop predicate pass. The other three 10 N
cases remain unattempted. The global preference configuration is rejected;
one robot-only component adds zero demonstrations.

The complete custody review retains 41,208 movement and 9,600 stop packets,
83 original files plus twelve sidecars, and 58,362,101 original bytes. All owned
resources close. The readout's initial obsolete midpoint metadata assertion,
its allocation, and a closeout return-code assumption are retained. Native
processes return zero after preserving a guarded negative report; scientific
outcomes come from the report and independent predicates. No outcome is inferred
from process exit alone.

A whole-reference audit compares all five actual cases with prior paths using
identical setups, programs and calibration inputs. All four failure witnesses
have exactly identical goal coordinates, phases and control clocks. The prior
path retains 2.718/8.076/6.461 mrad at the terminal Y lower face for N/H+/P,
and 9.645 mrad at the shoulder Y lower face for H−10N. The new loaded posture
differs by up to 0.13024 rad in one joint. This is whole-trajectory drift, not
support for a new precision floor or a failed-point calibration extension.
The completed H−0N release is based on its achieved pose, so its differing
release goals are explicitly excluded from an exact paired-goal interpretation.
Nullspace decomposition is diagnostic and does not supply an executable
one-tick repair. Next work is whole-motion posture/reference continuation with
primary motion, changing load and transitions retained, using the qualified
paths as controls. The matrix remains 17/36. Evidence is
`D4-coupled-margin-whole-family-closeout-20261107.json` and
`D4-coupled-margin-whole-reference-posture-CUDA-diagnosis-20261107.json`.


A causal virtual-reference screen receives only each current recorded goal,
its last delivered virtual state and the first measured configuration. All
eight complete 4,659-tick programs pass the unchanged supported future box,
physical reference and rate checks. Independent post-command FK also passes
the original six pose endpoints, 1 mm ramp and final half-second hold drift.
These 37,272 virtual ticks are not loaded native motion. The initial screen's
error samples precede its reference update; delivered endpoint claims use the
independent post-command readback instead.

A whole-motion perturbation plant then uses each qualified native physics
packet's measured inertia, mimic constraints and original explicit PD gains.
Zero perturbations reproduce the nominal recording by construction. For the
previous finite-reference controller, the model nevertheless reproduces all
seven independently available native support outcomes: five positives and the
two N0/H+10 negatives. Pre-release model/native joint differences peak at
0.138 mrad; predicted first exits precede native refusals by 45.8 and 29.2 ms.
The original N10 remains unattempted in that native family. These controls
support rejecting candidates in isolation, not a native position certificate;
nonlinear inertia, Coriolis and fixture derivatives remain unqualified.

A causal virtual-path null restorer retains only five of eight constant-load
support outcomes. At its three failure witnesses, task-space deviation dominates
the remaining null-coordinate difference. A second isolated design instead
seeds a complete joint reference from delivered targets minus their transported
six-axis entry elasticity. Current commanded goals advance that reference;
only bounded delivered motor targets commit its state. The prior sixteen whole
model control histories remain bitwise identical. Seven of eight constant-load
paths retain original support, and all eight original pose, ramp and hold
checks pass in the model. N10 still leaves wrist-X lower future support at
control tick 3,831 during hold, reaching a 0.685 mrad full-path deficit.
No production integration or native attempt follows this incomplete result.
The virtual reference uses the physical joint span for centering while motor
target bounds remain unchanged; the next matched control must preserve the
original reserved-target centering metric before attributing a design effect.

Both designs retain all four loaded variable-force stress negatives. The plant
has no actual contact-pressure loop, so these stress responses do not establish
the full teacher's variable-load behavior. The maintained teacher uses completed
native contact intervals; observed runtime admission remains separate. Hard position, speed and
acceleration predicates pass all 32 model paths in the second design; empirical
exits are retained separately. Next work must resolve the full loaded hold,
actual causal outer force loop and preview/dispatch/antiwindup transitions with
all original criteria before production or complete native qualification.
All five allocations close and add 160.186542 GPU seconds, with no new native
or observed launch. The matrix remains 17/36. Canonical evidence is
`D4-whole-reference-continuation-isolated-closeout-20261107.json`,
`D4-causal-virtual-reference-whole-motion-independent-CUDA-review-20261107.json`
and `D4-transported-joint-reference-whole-motion-independent-CUDA-review-20261107.json`.


A matched complete control restores the original reserved-target centering
metric in the transported joint reference. All sixteen nominal-replay and
prior-controller histories remain bitwise identical. Original model support
returns to five of eight: N0/N10/H+10 remain negative during opening, while all
eight pose endpoints pass. The earlier seven of eight also changed the posture
metric. Neither result supports production adoption or another centering retry.

A common stop/support audit reads 32 actual 2+4+2 s profiles across five native
families, preserving 61,440 packets. At all ten empirical-refusal states, the
actual stopped range plus the original velocity/acceleration delay stays inside
the original position domain, with a smallest 4.688 mrad margin. The largest
recorded joint excursion among the 31 hard-motion positives uses 30.15% of its
original joint stop bound. The failed loaded H− entry-preload stop is preserved
at 4.470 rad/s. These are retrospective observations, not a smaller prospective
bound or a waiver. Configuration-only replay also reproduces all ten group and
terminal refusals in the actual position-support contract; full observed
geometry/admission is not inferred from that isolated check.

Blackwell independently reads native inertia and Jacobians at 301 complete
20-joint recorded postures, including anchors and first/maximum-speed/final
samples from all three stop segments. Every saved joint representation matches
exactly. The readback advances no dynamics or teacher task. Whole custody closes
16 original files plus three sidecars, totaling 16,009,808 bytes. Workstation
analysis runs independently under its own GPU lease.

The stopped-PD reconstruction uses the actual per-recipe gains
`[2560,1280,640,1280,80,160,100]` and unchanged damping ratios. A first reader
incorrectly treated feedforward buffers on passive mimic followers as dispatched
efforts. With the actual actuator mask, all twenty simulation efforts reconstruct
exactly across 15,360 original stop packets; constrained one-step velocity error
peaks at 3.437 µrad/s. Followers still contribute mass, armature and passive
damping through the physical 18-coordinate mimic tangent. Frozen stop matrices
at all 301 poses have radius below one for all 31 measured hard-motion positives;
the unstable H−10 anchor has radius 1.018603, reaching 1.019592 along its profile.
The maximum relative eigenpair backward error is 6.40e-16. This separates the
recorded control, but unqualified nonlinear inertia, gravity and contact-force
derivatives still prevent a prospective position certificate.

All six allocations close and add 112.730304 GPU seconds. The initial continuation-
anchor reader failure, passive-follower effort reconstruction failure, and two
zero-GPU configuration-input shape preparations remain preserved. No controller,
calibration, stop envelope or guard changes. Next work is a causal state/command-
dependent stop-support proposal, using complete nominal, boundary, perturbed and
variable-load stops, the original 10 N scope and unstable control. Retrospective
travel and frozen poles alone are insufficient; every original physical gate
and prior outcome remains. Canonical closeout is
`D4-original-centering-and-stop-mechanism-closeout-20261107.json`, with
`D4-whole-native-stop-versus-support-forecast-corrected-CUDA-20261107.json`,
`D4-stop-profile-causal-PD-masked-momentum-CUDA-20261107.json` and
`D4-whole-native-stop-inertia-physical-tangent-CUDA-review-20261107.json`.
The matrix remains 17/36.

An isolated prospective stopped-PD model uses full current state, native inertia,
the physical mimic tangent and only the last completed momentum residual.
Across eight original complete stop profiles (15,360 native packets), the first
six seconds have at most 56.76 µrad position error. Load removal raises position
error to 2.776 mrad and directional excursion underprediction to 1.019 mrad;
constant residual is therefore not a stopping envelope. The frozen model's
arbitrary time-varying force ball (norm at most 10 N at the known application
point) contains these eight profiles. Fifty-six independently maximizing input
sequences reach its final faces with 1.46e-16 rad arithmetic defect. This is a
finite-horizon linear-model property, not a nonlinear physical bound. The other
24 profiles retain unavailable full causal inputs; the original unstable control
remains rejected.

Native qualification selects the smallest original future-support margin in
each of six phases of all eight complete baseline histories: 48 exact full-state
inputs, all locally stable, with the closest original margin about 34 nrad.
A predeclared smooth 2 Hz unilateral 0–10 N load runs for six seconds, followed
by two seconds without load, under original stopped gains and 2+4+2 s artifacts.
The first four workers failed before their first trial on a ProxyArray/Tensor
adapter mismatch. A Blackwell preflight then caught confusion between the
gripper-link origin and distal UMI point (245 mm apart). All five failures,
68+17 original files, 12+3 sidecars, charges and unattempted trials are preserved.
Zero exit codes never override the incomplete/error reports. Three dependent
workers stayed unlaunched after the frame mismatch.

The common correction uses the maintained URDF operational offset and rotation
for application point, normal and COM-to-point Jacobian. All 48 source states
join exact native clocks and joint packets; the resulting point Jacobian agrees
with the original operational feedback within 1.03e-7 m/rad. Corrected model
inputs replace the rejected link-origin preparation without deleting it.
Blackwell executes its first two eight-second probes before the other three
independent hosts are dispatched. All 48 then complete 144 original stop segments
and 92,160 native packets. Whole custody closes 216 files plus twelve sidecars,
122,844,418 bytes, with actual owned cleanup on every host. All original tool,
joint and 14-shape radial/directional/rotation bounds, physical position/speed,
frozen-load residuals and current empirical position checks pass. The vectorized
full-joint geometry agrees with the maintained CPU consumer within 7.78e-16.
Explicit PD efforts reconstruct exactly; constrained one-step velocity error
peaks at 50.71 µrad/s under this variable forcing. The frozen model following
the predeclared input has at most 72.31 µrad position error; all 48 native paths
stay inside its 10 N model envelope. Unseen force directions, application changes,
state perturbations and complete new-controller paths remain unqualified.

The unchanged recipe acceleration check passes 46/48 states. N/P approach stops
from the loaded-labeled histories reach 8.261/10.937 rad/s² at the first native
step, versus the original 7.876929 recipe value. Direct physical velocity changes
confirm both peaks; they are not reset-derivative artifacts. All 48 stay inside
the preexisting main load-calibration acceleration domain (16.529308 rad/s² for
that joint). Runtime uses recipe acceleration for the next motion sweep and a
separate acceleration domain in load estimation; neither is an independent
hardware acceleration specification. Keep both legacy negatives and audit these
consumer contracts before any change. Thirteen global future forecasts also
remain negative during already-stopped continuations. No criterion, tolerance,
calibration or envelope is changed. Saved states provide no complete-path or
demonstration credit. All ten final native/review allocations close, adding
896.903017 GPU seconds; the exact continuation ledger is 186 closed allocations
and 7.069997 GPU hours, including all prior failures.
Canonical records are `D4-prospective-stop-forecast-local-closeout-20261107.json`,
`D4-variable-load-stop-pretrial-ProxyArray-closeout-20261107.json`,
`D4-variable-load-distal-geometry-preflight-closeout-20261107.json`,
`D4-variable-load-distal-operational-J-native-clock-review-20261107.json` and
`D4-variable-load-distal-stop-six-phase-native-manifest-20261107.json`.
The complete result is `D4-variable-load-distal-stop-six-phase-closeout-20261107.json`;
acceleration evidence is in `D4-variable-load-distal-stop-acceleration-gate-origin-CUDA-20261107.json`
and `D4-variable-load-distal-stop-acceleration-domain-contracts-CUDA-20261107.json`.
The collection remains 17/36, with no controller, criterion or support change.

The subsequent actual production-contract replay calls the unchanged NumPy load
and next-motion methods on all 92,160 packets. It reproduces 2,664 position-halo
force-upper refusals and 2,040 stop-force-upper refusals; the two acceleration
witnesses have no production acceleration-domain refusal. This is a method
replay with recorded robot feedback, not observed geometry/admission.

A general source correction computes the farthest-corner norm of the existing
three-dimensional force uncertainty box. The old radial bound remains a
rounding fallback. No calibrated joint error, force cap, motion/geometry/stop
criterion or estimated wrench changes. Exact paired production calls increase
load-plus-stop-force acceptance from 87,456 to 88,176 packets: 720 previously
unnecessary refusals are removed, and every old acceptance is preserved. Load
estimation alone passes 90,000 packets, versus 89,496 previously. All common
wrenches, residual errors and residual bounds are bitwise unchanged; normal
lower bounds never decrease. An existing independently calibrated weak-normal
formula activates additionally at 453 packets because its force domain can now
be established more tightly.

Independent CUDA checks cover all eight box corners, 65,536 nominal/boundary/
perturbed vector cases and every native force norm in these 92,160 packets.
Minimum recorded force-coverage margin is 0.02494 N. These checks introduce no
new empirical tolerance; the unit cap tests retain an outside-domain refusal.
All 165 focused tests and lint pass. Canonical result is
`D4-force-box-bound-correction-closeout-20261107.json`, with
`D4-actual-stop-load-and-next-motion-consumer-contracts-20261107.json`,
`D4-force-box-paired-production-consumer-contracts-20261107.json` and
`D4-force-box-CUDA-native-and-perturbed-qualification-20261107.json`.
All prior 46/48 legacy stop outcomes and thirteen stopped-state global future
negatives remain unchanged. The matrix is 17/36; no native teacher retry is
justified by replay acceptance alone.

### November 7: prospective force direction and distal-cover stress

The next frozen-model screen uses the same 48 full saved phase states, original
stop gains and support predicates. It selects the worst original supported
face/time over the operational center and 30 nominal distal-cover vertices,
then constructs a maximizing world-force history with norm at most 10 N.
Exact model witnesses agree within 4.45e-16 rad; float32 force commands are
rounded inward only where necessary. All 48 unrestricted-ball histories
predict original speed or stop-travel failures, with speed up to 1.828 rad/s;
five also cross an empirical position face. They remain unattempted. All
include tensile intervals, so they are mathematical stress witnesses rather
than demonstrated door-contact histories.

A separate exact support-function screen keeps compression unilateral. A
normal-only subset has no original model gate failures and peaks at 0.153 rad/s.
A tangent-to-compression ratio at most 0.5 still produces 48 model gate negatives,
two empirical-face negatives and peak speed 0.832 rad/s. This ratio is a model
reference subset, not a new admission rule or measured future-force contract.
No force cap, support, gain or physical criterion is changed.

Only one protected ball-witness interval was executed natively, followed by
1,919 unforced intervals. Actual full state, distal-cover application and root/
mass readback agree. All original tool, physical position/speed, joint-travel,
14-shape and residual checks pass. The legacy motion-acceleration predicate
fails, and two samples exceed existing main load-model acceleration support at
wrist-Z/gripper-Y; those are empirical-domain negatives, not hardware limits.
Exact PD dispatch matches; one-step full velocity error is 83.2 µrad/s and
planned position error 18.03 µrad. A 4.565 µrad late frozen-envelope escape at
7.992 s occurs where its width has decayed to 4.45e-15 rad. No failed-point
precision floor or additional uncertainty margin is introduced.

The canonical inputs are `D4-force-footprint-stop-six-phase-inputs-20261107.json`
and `D4-contact-cone-stop-subset-model-screen-20261107.json`. Native evidence is
`D4-force-footprint-protected-prefix-native-manifest-20261107.json`, its WHOLE
transfer/independent review, and
`D4-force-footprint-prefix-acceleration-and-model-defect-20261107.json`.
A matched normal-versus-added-tangent prefix uses the same full state, point and
compressive force; it is a separate causal diagnostic, never qualification of
the suspended full programs. The matrix remains 17/36; no new runtime bound,
controller, contact-feedback input or demonstration is adopted.

The native matched pair is now closed: full initial q/v, mass and latched targets
are bitwise identical, application point is identical, and compressive commands
agree within 0.95 µN float32 rounding. Added tangent peaks at 4.472 N. Gripper
speed rises from 0.04876 to 0.40068 rad/s. Its 6.319 mrad travel exceeds the
unchanged empirical 5.847 mrad joint-stop bound by 8.07%; wrist-Z/gripper-Y
acceleration also leaves main load support at two samples. The normal control
passes every original predicate. Both retain original hard position/speed, tool,
14-shape and residual checks. These prefixes are fixtures with load removed
after four intervals, not actual door-contact episodes or executed full model
negatives. The finite model predicts speed over 0.51 rad/s for every full
friction-subset program, which remains unattempted. Whole transfers preserve
45 original files plus six sidecars across the three prefixes; owned cleanup
passes. Six closed allocations cost 185.323 GPU seconds including all numerical
screens and independent reviews. All 193 continuation allocations are closed.

Canonical closure is `D4-force-footprint-and-contact-tangent-closeout-20261107.json`;
causality is in `D4-contact-tangent-matched-prefix-causal-comparison-20261107.json`
and the corresponding manifest and independent GPU review. Next work is an
isolated causal disturbance-aware stopping response using completed joint-output
feedback, original 10 N/physical/14-shape scopes and full coupled dynamics;
prescribed fixture force must never become observed feedback. Original complete
controller paths and contact qualification remain necessary. No source bound,
criterion, controller or demonstration is adopted from these diagnostics.

### November 7: causal calibrated stopped-response prototype

The isolated correction keeps the measured stop target and original gains, then
adds only the certain part of the previous completed joint-output residual.
For each joint it applies `-sign(r) * max(abs(r) - b, 0)`, with the unchanged
calibration's component error budget. No prescribed fixture force is an input.
The controller checks current main position/speed/acceleration support and exact
completed-sample time/generation; unsupported input gives zero correction.
Eleven actual-method contracts cover uncertainty, missing/stale/contradictory
feedback, pure preview, reset and generation changes.

The coupled-model screen retains 960 trajectories: 48 phase states, five force
programs and four controller/input conditions. Ideal or conditionally trusted
residual correction passes 48/48 friction-subset model cases, but the strict
calibration mode passes only 22/48 under abrupt forcing. Those negatives and
arbitrary-ball failures remain; conditional inputs are never promoted to verified
feedback. Smooth rotating friction passes all 48 strict model cases without
calibration exits. Independent Blackwell auditing joins 44,952 actual original
native samples: all 44,948 calibrated samples satisfy the existing torque error
budget. Four out-of-domain samples remain excluded even though their observed
errors also fit. An initial metadata-adapter failure and short-job historical
GPU-PID unavailability are preserved in the operational closeout, not fabricated.

A native matched comparison uses identical full initial state, point and smooth
rotating force, with norm at most 10 N for six seconds and two seconds removed.
The unchanged control fails original tool, joint-stop and 14-shape checks;
its maximum segment tool travel is 10.422 mm and a segment terminal speed is
0.03538 rad/s. The correction passes all original checks, reducing those values
to 0.545 mm and 0.00329 rad/s. Actual dispatch uses only the preceding measured
joint-output residual; feedforward and all twenty actuator efforts reconstruct
exactly, with no calibration rejection or target/gain change.

The first positive is reused through exact full-state/point/force/helper parity.
Forty-seven further stops on Blackwell and elab2/3/4 complete all 48 saved states
across the six original motion phases. All 92,160 controlled native packets pass
original physical position/speed, tool, joint-travel, 14-shape, load residual and
current-position predicates. All observer inputs stay inside existing main load
calibration. The two N/P loaded-approach legacy moving-acceleration negatives
remain: 46/48 pass that stricter historical combination. Maximum segment tool
travel across the family is 1.032 mm; maximum terminal speed is 0.004968 rad/s.
Actual controller and feedforward reconstruction errors are zero. Frozen
replay with the delivered inputs differs by at most 28.64 µrad; this is a
retrospective model check, not a prospective stopping certificate.

Canonical result is `D4-causal-stop-response-six-phase-closeout-20261107.json`;
the manifest is `D4-causal-stop-smooth-six-phase-native-manifest-20261107.json`.
The prototype is `causal_residual_stop_controller_20261107.py` in the ignored
campaign evidence directory, used by the native workers. WHOLE transfer and
owned cleanup preserve 239 original files plus fifteen sidecars for the 48
controlled states. Fifteen closed phase allocations, including the failed
adapter and all independent reviews, cost 1,163.909 GPU seconds. All 208
continuation allocations are closed, with no cumulative administrative cap.
Production software remains `69fce43`, native environment source `6a27ffb` and
optional posture settings unset. The collection remains 17/36; these stopped
components add no demonstrations or optimizer updates. Maintained optional
integration, default/history/reset validation and all eight original complete
0/10 N controller paths/stops precede source adoption, any smaller geometric
stop bound, or another observed teacher attempt.


### November 7: maintained stopped-response integration, disabled by default

Baseline `bbd42d9` is tracked clean before implementation. The maintained
controller and environment now support the qualified causal response without
changing any current collection recipe, target, gain, calibration or stop bound.
The load estimator's existing torque-origin validation and component error
budget are shared directly. Only completed post-latch samples are consumed;
control-tick handoff, pure preview, repeated stop, reset and moving/default
feedforward behavior pass numerical checks on the actual environment methods.
Recording retains its event callback. Essential input tests reject wrong torque
origin/units/sign/health, contradictory clocks, unverified calibration and
unsupported position/speed/acceleration. These are software contracts, not new
native motion or teacher evidence. Canonical lifecycle result is
`maintained-stop-response-actual-lifecycle-contracts-20261107.json` in the ignored
campaign evidence directory. All eight original complete 0/10 N paths/stops
remain the next native qualification; original unsafe and out-of-calibration
controls stay negative. Collection remains 17/36 and optimizer updates remain zero.


### November 7: maintained stopped response passes eight original full paths

Source `968704e` runs all four original robot-only conditions at 0 and 10 N,
with unchanged motion controller, setup, full six-phase path, 1 mm ramp,
three-second hold, achieved release and stabilization. The stopped-only response
is explicitly enabled in these diagnostics. All eight cases and their original
2+4+2 s continuations pass independent physical, target/effort, current and
single-future empirical, frozen-load, joint-travel and 14-shape checks. Across
172,128 native packets, the 15,360 stopped intervals join exact preceding real
joint-output feedback to the unchanged component error budget and actual
full-system feedforward. Each case starts with one explicitly missing input;
no subsequent input leaves calibration. There are 5,760 nonzero corrections.
One anchor persists across each complete eight-second continuation and episode
reset clears the observer before the next moving path. Maximum segment tool
travel is 0.259 mm; maximum segment terminal joint speed is 0.000306 rad/s.

All original output transfers preserve 100 files and twelve sidecars, totaling
243,986,030 bytes, with actual owned process/GPU absence and canonical lease
release. Four ordered independent readers per condition check causal dispatch,
whole outcomes, actuator clock and constrained-dynamics diagnostics. Inherited
review prose describing only the older moving controller is corrected with the
original derived reports retained; raw artifacts and numerical outcomes are
unchanged. Mimic position/velocity identity negatives remain diagnostic, never
new refusal gates. Eight closed phase allocations cost 2,070.535 GPU seconds;
all 216 continuation allocations are closed. Canonical closure is
`D4-maintained-stop-response-whole-family-closeout-20261107.json` in the ignored
campaign evidence directory, alongside the closed manifest and WHOLE receipts.

This qualifies the maintained response for the tested complete robot-only
fixture paths. Defaults and current collection recipes remain disabled. There
is no new observed contact, teacher demonstration, optimizer update or smaller
geometric stop bound; collection remains 17/36. Next work is a prospective
closed-loop stopping-support proposal using current state, the original 10 N
scope and the same calibrated component uncertainty. Unknown future delivered
corrections must not enter its bound. Retain the unstable H− entry-preload
control, arbitrary-force/out-of-calibration negatives and frozen-model limits.


### November 7: prospective support and implicit-drive negatives; shared stop scoring

The prospective frozen closed-loop envelope uses only initial coupled state,
31 distal contact-point alternatives, the original 10 N ball at every physics
interval and the existing component error budget. A change of state coordinates
retains the shared force history across the causal one-sample compensation delay.
The certain-residual remainder is bounded by twice the same component interval,
a consequence of its measurement error and confidence deadzone. An independent
superset face witness matches within 4.45e-16 rad and all qualified prefixes of
the earlier synthetic trajectories remain contained. Position support closes at
48/48 anchors, but no anchor closes all original travel, speed and calibration
acceleration conditions. This is a negative support proposal, not a new runtime
bound or evidence of an actual physical violation at every anchor. An initial
scalar roundoff check used inconsistent units; its failed allocation and script
are preserved before the operand-based arithmetic check is corrected.

A separate solver-implicit model keeps nominal gains, target, effort limits and
strict completed-residual feedback. It reproduces the original explicit control
exactly, then evaluates 480 trajectories and 301 actual inertia snapshots across
32 old profiles. The implicit frozen physical tangent is stable at all snapshots,
including the earlier unstable H− profile, whose explicit negative is retained.
The initial partial screen counts omit terminal settling. Rescoring the saved
trajectories with the original terminal criterion leaves zero fully screened
abrupt-friction cases; the original report is retained beside the correction.

One native implicit-drive diagnostic is preserved. An initial API-layer mismatch
fails before trial dynamics, leaving one unattempted state. The corrected
Isaac Lab PhysX adapter verifies gains, latched targets, drive-cap readback and
physical total-effort capacity after reserving feedforward. Its 1,920 completed
intervals pass physical position/speed, original joint travel, all fourteen
shapes and the unchanged calibrated torque-error budget. Peak measured residual
error is 0.004819 N m, with zero budget exceedances, zero input-domain rejections
and exact feedforward/explicit-buffer reconstruction. Internal solver drive
torque remains unavailable. The original two-second terminal speed is
0.06115 rad/s against 0.01, so the candidate stays negative and dependent trials
are suspended. No physical trial is repeated to repair that outcome.

The maintained `stop_continuation_outcome` centralizes the original complete
sample contract without changing thresholds or acceptance. It also reports
`stop_not_settled` explicitly. Replay of 27 complete native segments preserves
all 26 positives and one negative exactly; absent model channels remain
unavailable. The shared scorer prevents the isolated terminal-preflight omission
from being silently promoted to qualification. Focused unit coverage includes
nominal, exact-boundary, perturbed, nonfinite and missing cases.

Canonical results in the ignored campaign evidence directory are
`D4-prospective-causal-stop-frozen-support-20261107.json`,
`D4-solver-implicit-causal-stop-isolated-screen-20261107.json`,
`D4-prospective-support-and-native-implicit-negative-closeout-20261107.json`
and `shared-stop-scoring-native-and-model-parity-20261107.json`.
Both native diagnostic transfers preserve 41 original files plus six sidecars;
all owned processes, GPU contexts and leases close. All 222 continuation
allocations, including numerical and pretrial failures, are accounted for.
Collection remains 17/36, with no new demonstrations, optimizer updates,
implicit controller recipe or geometric stop bound. Next work audits actual
contact/free-space evidence and geometric stopping consumers before another
controller variant or observed teacher attempt.


### November 7: observed stopping-domain audit and historical-view filtering

Read-only review of 35 nonempty stored D3/D4 admission logs retains all 5,390
proposals: 5,372 local geometric admissions, fourteen unsupported actual sweep/stop
volumes and four weak-position future-support refusals. Nineteen empty admission
logs remain empty, not passes. These heterogeneous historical runs do not form
an independent success denominator. No recorded complete contact-free stop
margin is positive. The producer already checks weak position support against
complete observed stopping clearance and nonzero target motion through its full
horizon; a weak current force cannot replace the future load domain. Later N
geometry-admitted proposals still end in native contact loss. No smaller stop
bound or unchanged contact retry follows from this audit.

A conservative historical-camera filter reduces repeated background-view work.
It excludes only cameras that cannot contain the queried ball's center, with
propagated arithmetic bounds and the original padded-image convention. All
surviving or ambiguous views use the same complete ball/depth/occupancy reader;
calibration conditions, current material exclusion and source clocks remain
unchanged. Nominal, image-edge, immediate representable-neighbor, nonstandard
projection and perturbed camera/unknown-depth controls preserve the original
reader's outcomes.

One owned workstation GPU allocation reconstructs the exact complete saved N
admitted and P refused producers, with three calls per mode and original inputs.
All six paired decisions, proof margins and diagnostics except timing are exact.
For N, 133,584 old per-view calls reduce to 31,414; 58,911 distinct culled actual
view/center pairs are independently rejected by the original zero-radius reader.
Warm profiled N wall time falls from 1.520 s to 1.016 s (33.14%). P uses no
uncached historical spherical queries in this replay, retains its unsupported
finger-volume refusal and shows no measured benefit. These are replay timings,
not native collection throughput. No physics steps, commands, demonstrations or
optimizer updates are added. The owned GPU process and lease close; its 39.439
GPU seconds are booked once, bringing continuation accounting to 223 closed
allocations.

Canonical ignored evidence: `D3-D4-actual-observed-geometry-stop-consumer-audit-20261107.json`
and `D4-actual-observed-center-frustum-paired-review-20261107.json`. Original
negative histories, source candidates, paired profiles and actual GPU identity
remain preserved. Collection is 17/36. Next work addresses material/visibility,
whole-path empirical support and sustained contact together before new observed
collection.


### November 7: current moving-force observability and direct normal support

All 16,852 original four-posture friction packets are joined exactly to recorded
robot-feedback acquisition and availability. The unchanged calibration and actual
production load contracts retain 16,842 supported packets and ten position-domain
refusals. Only completed joint-output torque, expected robot dynamics, Jacobian
and robot tool orientation enter the estimator. Independently measured normal
and friction wrenches enter scoring afterward. Direct projection of the same
seven-joint error box into tool axes identifies a nonzero tangential external
force at 477 accepted push dispatches. No calibrated current component has the
wrong sign. At 32 of these dispatches, at least one certain component changes
sign in the following 240 Hz interval. This refutes a guaranteed unchanged future
sign; it neither predicts a counterfactual controller failure nor identifies
material-frame friction from robot tool axes. A moving cancellation law is not
adopted.

The current normal-force lower bound had summed separate worst-case world-axis
errors before projection onto the requested normal. Those maxima can require
conflicting signs of the same joint-error variables. The maintained estimator
now uses the exact scalar support `abs(n @ pinv(J.T)[:3]) @ b`, retaining the
existing valid bound for floating-point ties. The same projection transports
the weak scalar calibration through the difference between requested and tool
normals. Its independently calibrated scalar coefficients and force cap remain
mandatory. Calibration, force-norm upper limits, weak-position/acceleration
domains and normal-angle/sample-age reserves remain unchanged.

Paired old/current production calls retain every original domain refusal and
all wrench, force-upper, residual-error/budget, position-halo and scope outputs
exactly. Normal lower bounds never decrease and remain below independently
measured total compression on all supported packets. The final bound is tighter
in 7/20/48/6 N/H+/H−/P packets, with maximum gain 0.008081 N. The separately
qualified weak scalar branch dominates most packets; the mathematical improvement
does not resolve the physical contact losses.

Independent CUDA enumeration covers 58,748 native or generic normal/mapping
cases, with all 128 vertices of each identical error box. Generic cases reject
the zero-uncertainty negative control; corner-support defects remain within
operand-scaled arithmetic bounds. The final full-branch pair additionally checks
50,556 production calls at the original tool normal and two predeclared ±3°
diagonal perturbations, preserving the same ten domain refusals per normal and
all upper-force/wrench/error/scope outputs. All supported lower bounds still
cover measured total compression. These perturbations are numerical controls,
not a newly qualified normal domain. Enumeration of both normal and difference
projections covers 101,112 cases, including zero and nonunit differences, with
all 128 error-box vertices. Three owned local CUDA allocations cost
51.3286 GPU seconds in total. The first two actual monitor identities are
recovered after completion and preserved alongside the coordinator's already-finished
snapshots; no live identity is invented. Processes and leases close, and all
226 continuation allocations are accounted for. No physics steps, commands,
demonstrations or optimizer updates are added.

Canonical ignored evidence: `D4-moving-observability-and-direct-normal-support-closeout-20261107.json`,
`D4-moving-tangent-residual-observability-local-20261107/report.json`,
`D4-direct-normal-support-paired-local-20261107/report.json` and
`D4-direct-normal-transport-paired-local-20261107/report.json`. Original measured
failures, all masks, paired bounds, pre-change source and charges are preserved.
Collection remains 17/36. Further contact work must model the changing
normal/tangential dynamics and causal sample timing together with complete
posture/material/visibility support; this bound correction alone does not
justify another unchanged native teacher attempt.


### November 7: complete moving-contact inertia and frame-covariant readback

Before the readback, selection freezes all 32 calibrated push dispatches with a
certain tangent component reversing in the next 240 Hz interval, and 32 nearest
distinct calibrated nonreversal controls in the same push. Five native endpoints
per command deduplicate to 293 exact full20 joint states: 61/100/113/19 for
N/H+/H−/P. Four owned hosts read full mass, armature, gravity/Coriolis and tool/
finger COM Jacobians after ordinary bootstrap, without a new contact motion
trial. These are current robot-model readbacks, not original historical full20
dynamics measurements. Original clocks and full q/dq write/read are exact.

The first four readers fail before reading any saved state because of an
unjustified bitwise root-frame equality requirement. A corrected nominal-only
reader then fails before its 61 states on an equally overstrict untransformed
gravity-invariance control. All five failures, 354 unattempted state reads and
original payloads remain preserved. The common correction records both root
frames and transforms gravity as well as Jacobians/wrenches. A mass-weighted COM
gravity map retains raw native gravity and derives its canonical-source
representation separately. Independent frame controls reject the unchanged-
gravity negative control. No physical, empirical or numerical acceptance domain
is widened to obtain a pass.

Independent CUDA scoring covers all 64 commands and 256 actual native intervals.
Full measured velocity differences, actual applied arm efforts and distributed
normal/friction patches close the seven arm rows. Measured mimic velocity
nonidentity remains present, with a 0.00166 m/s peak; the constrained eighteen-
coordinate response uses the physical mimic basis without altering measured
states. Unrecorded historical other-joint motor efforts are never zero-filled.
The largest per-joint RMS residual ranges from 0.136 to 4.188 mN·m using only the
arm block, versus 6.626 to 9.460 µN·m with the measured other-joint inertia term.
That contribution peaks at 0.01602 N·m and occurs in reversals and controls alike.
It explains the old balance residual but does not establish the cause of contact
loss. Subsequent effort, force and acceleration remain scoring-only truth.

The frozen current hand-response operator includes full inertia, armature,
physical mimic coupling and passive follower damping. Its relative difference
from the consistent arm7 operator is only 2.2–4.8 parts per million. Normal/tangent
normalized coupling spans 0.209–0.225; this dimensionless descriptor is not a
failure fraction or prospective contact certificate. Positive reduced inertia
and an independent zero cross-mass-block control support the calculation.
Changing contact/friction memory and sampled control remain unresolved together
with full posture/material/visibility support; another unchanged teacher trial
is not justified.

All four successful readers, five failed readers and the independent local review
close, costing 136.2783 GPU seconds. WHOLE transfer preserves 144 original files
and 27 sidecars, totaling 106,191,822 original bytes. The short local CUDA review
finishes between monitor samples: its historical GPU PID is unavailable, while
the actual CUDA device result, wrapper UUID/owned process-group launch and cleanup
remain recorded. An initial closeout assertion on this missing receipt is
preserved, then resolved by recording the unavailable state without a rerun or
invented identity. Fresh six-host inspection finds no compute contexts or active
episode leases. Continuation accounting is 236 closed allocations and 8.12579 GPU
hours. Collection remains 17/36; all original physical/contact/load/stop negatives,
zero optimizer updates and unadopted controller/stop-bound status remain.

Canonical ignored evidence:
`D4-current-full20-moving-contact-complete-closeout-20261107.json`,
`D4-moving-contact-full20-covariant-native-manifest-20261107.json` and
`D4-current-full20-moving-contact-review-local-20261107/report.json`.
The five pretrial failures and frame controls have separate preserved records.
Maintained model source is `01f814a`; no maintained controller changes in this
readback milestone.


### November 7: conditional contact-patch timing control

The same frozen 64 commands/256 intervals support a separate CUDA diagnostic on
all 1,024 recorded friction patches. Original finger link transforms transport
the same material point between endpoints, without matching or inventing patch
identity across intervals. Current full20 finger COM Jacobians and measured
unprojected velocities supply a separate velocity model. The prepared TRAIN D4
hinge supplies conditional door-point kinematics; historical native door-root
poses and body twists are unavailable. Model versus recorded finger link position
difference peaks at 0.477 µm, without making model-derived velocities direct
historical body-twist measurements.

All 32 selected raw tangent-force reversals survive both fixed and endpoint tool
frames. At these selected dispatches, the largest force-coordinate effect of
tool rotation is 0.0000280 N; normal-force projection into tool tangents peaks
at 0.00265 N, versus 1.312 N for the friction projection. This distinguishes measured friction effects from
a coordinate-change explanation but does not identify a causal compensation.
All 32 zero-normal/nonzero-friction actor intervals remain present. Signed work
scores are retained without a fitted cutoff. Material-point finite-displacement
relative velocity differs from the endpoint model mean by up to 0.00741 m/s.
Consequently interval-average force and endpoint velocity do not provide exact
continuous contact work or an identified memory law. Neither the signs of these
scores nor presumed persistence of current force qualifies a controller.

The single owned CUDA review costs 9.0463 GPU seconds. It records its actual
live PID/UUID/context internally before completion, then closes its process,
context and lease. Accounting is 237 closed continuation allocations and 8.12830
GPU hours. No native simulation, controller, tolerance, stopping bound, source,
optimizer update or demonstration is added; collection remains 17/36. Canonical
ignored evidence is `D4-moving-contact-patch-timing-complete-closeout-20261107.json`,
with the frozen plan and complete signed patch results beside it. Source is
`1b34721`. Next work needs an informative prospective native material/friction
intervention under the original full-motion and support criteria, after checking
existing controls and the actual material override interface.


### November 7: causal friction intervention and direct native pose/velocity pairing

The nominal material control keeps native controller `519c347`, the exact N
setup/load calibration, original recipe, camera, gains and task/stop criteria.
It changes static/dynamic friction only on the two right finger rigid bodies
and Panel, with native material readback; restitution and other bodies remain
unchanged. All 53,956 native task packets verify exactly zero friction forces,
including 200,704 returned friction slots. The task reaches 50.550165° against
50.146446° required, with 180 hold, 360 release and 120 stabilization control
intervals. Hold remains loaded/contact-valid; its minimum angle is 49.885461°
under the unchanged hold-variation criterion. Release and the final unloaded
window pass. Actual RGB pixels show the expected opening sequence; contact
surfaces remain partly occluded, so images do not certify contact or observed
material admission.

Independent reconstruction retains positive original 0.1 s load-lower windows
(minimum 0.056713 N), a 3.602910 N estimated force-upper peak, supported current
positions, physical/future physical limits and all six-second/fourteen-shape
stops. Complete pressure-component qualification is nevertheless negative:
3,296 required-10 N future boxes cross the lower empirical `RIGHT_SHOULDER_X`
face, first at 131.904167 s. Maximum excess is 0.000726758 rad, with worst future
lower endpoint −1.147179304 versus supported −1.146452546 rad. No bound, force
scope, calibration or tolerance is changed. The saved report explicitly excludes
teacher/data use while preserving its original-criteria physical success flags.
This is a causal material contrast with N's original 0.458° failure, not a new
realistic-material solution, source or demonstration. Related friction-zero
conditions remain suspended.

An independent measurement campaign then adds direct native finger/Panel COM
velocities, actor-local COM poses and Panel actor transforms to all four original-
friction histories on their original hosts. It writes no materials, controls or
physical states. Every legacy contact-packet value and all seventeen saved arrays
remain byte-identical over 3,484/4,320/5,596/3,452 N/H+/H−/P packets. All four
original tasks remain negative, including their primary/secondary failures and
release load-domain errors. Original physical stops and actual RGB pairs are
reviewed; no unavailable old state is zero-filled or retroactively inserted.

The exact repeated trajectories associate new direct twists with the frozen
293-state readback. Full20 COM `J*dq` differs from native linear velocity by at
most 4.322e-9 m/s and angular velocity by 1.019e-8 rad/s. Native contact-point pose
increments still differ from endpoint velocity means by as much as 0.0173883 m/s
across the complete histories. On the original 256 selected intervals, the
largest COM discrepancy is 0.00526159 m/s; projection of measured full-coordinate
increment minus mean joint velocity contributes up to 0.00525095 m/s. Nonlinear
pose quadrature and native/readback pose-increment terms peak at 0.00011824 and
0.00015156 m/s. The explicit decomposition closes within 1.31e-18 m/s arithmetic
residual and preserves both reversal and matched groups. This locates an omitted
position-update term; it does not prove a sole contact-loss cause, prospective
force persistence, exact continuous friction energy or a new stopping bound.

The material control and direct-state campaigns preserve 148 original files and
15 sidecars, totaling 16,004,910,612 original bytes. All eight native/review GPU
allocations close, bringing continuation accounting to 245 closed allocations
and 8.84595 GPU hours. Original masks, interruptions, current/future distinctions
and all negative qualifications remain. Collection is 17/36, optimizer updates
zero, and no controller, numerical default or dataset source is adopted.

Canonical ignored evidence:
`D4-zero-friction-causal-complete-review-closeout-20261107.json`,
`D4-original-friction-body-and-position-complete-closeout-20261107.json`,
`D4-original-friction-body-twist-family-independent-review-20261107.json` and
`D4-original-friction-native-position-increment-review-local-20261107/report.json`.
Coordinator baseline is `e7c52f5`; native control remains `519c347`.

The next prospective control isolates TGS external-force integration. Actual
baseline startup logs warn that `enable_external_forces_every_iteration=False`
may cause noisy velocities. Installed backend documentation recommends the option
for this symptom, and the external Alex readiness helper enables it. Additional
stabilization already defaults false, so disabling it again would be unchanged.
The proposed control changes only the force-integration flag with original
friction, retaining every existing criterion; it is not yet a validated fix.


The single-flag native control is now closed negative before contact. All four
samples of its first approach command reproduce `inconsistent_single_tool_wrench`
against the original calibration, with consistency error about 201 times its
existing bound. All native normal and friction patches are empty. At 34.4 s,
stationary q and gravity feedforward are exactly equal to the original-friction
baseline, but projected joint force minus applied effort peaks at 4.275786 N·m,
versus 9.536743 µN·m originally. The applied-effort comparison is diagnostic only;
no replacement feedback, bias fit, new tolerance or calibration is introduced.

The complete original recording and sidecars are preserved (26 files plus three
sidecars, 1,085,096,490 bytes). Integrity, independent six-second fourteen-shape
stops, actual precontact RGB and owned-resource cleanup pass. Contact, push,
hold, release and stabilization remain unattempted, so contact timing and the
full-cycle response under this flag are unavailable. No numerical default is
adopted and no teacher is added. Native plus independent GPU review charge
180.339617 GPU seconds; all 247 continuation allocations are closed. Canonical
result: `D4-TGS-force-feedback-control-closeout-20261107.json`. The common
position/contact discrepancy remains unresolved; the next preregistered nominal
control isolates PGS with original materials, feedback and acceptance criteria.


The single PGS control is closed negative: 145 push commands reach about 1.02°,
then contact is lost before hold. Abort release reaches 251 of 360 commands
before wrist-X empirical refusal; stabilization is unattempted. Independent
review retains all four current-domain and 210 required-future-10 N domain exits,
original physical limits and all six-second fourteen-shape stops. Peak native
normal force is 3.016 N; recomputed supported force upper bound peaks at 3.111 N.
Actual RGB, whole transfer (29 original files plus three sidecars,
1,681,899,085 bytes) and owned cleanup are complete. PGS is not adopted.

The largest contact-point displacement/endpoint-velocity component discrepancy
is 1.866 mm/s versus 4.358 mm/s in original N, on different trajectories. Five
complete recorded histories contain 20,596 native samples; all applied PD efforts
reconstruct exactly using preceding gravity. Current elastic-reference algebra
separates normal virtual wrench, other-axis and joint-torque-null contributions
to the scalar normal-support coordinate. Other-axis contributions during push
reach -0.403 to +0.208 N across the original family, versus a 0.1 N target. These
are virtual reference coordinates, not external-force measurements or a causal
contact explanation. A geometry-only alternative preserves virtual normal effort
at 1,353 exact causal dispatch states, but no nonlinear or native qualification
follows. The original reader incorrectly permuted native `xyzw` quaternions as
`wxyz`; its commanded-normal interpretation is invalidated and retained. The
corrected reader checks the normal against recorded material-reference axes and
finds a largest increment difference of 0.0691 µm, superseding 0.106 µm. The
transport alternative is not dispatched as a contact repair. The next isolated
candidate audits the motion projector's normal virtual-effort leakage while retaining the other five pose components.

Canonical result: `D4-PGS-and-elastic-coupling-closeout-20261107.json`. Four GPU
allocations, including three independent readers/screens, charge 248.795149 GPU
seconds; all 251 continuation allocations are closed. Collection remains 17/36.
No gains, material coefficients, empirical support, tolerances or task criteria
change. All failed and unattempted phases and reader-preparation errors remain.


An ignored force-orthogonal projector now passes nine essential CUDA controls
at sixteen recorded whole-family Jacobians, using synthetic goals and preloads.
The candidate preserves the other five requested pose components, explicit
normal-reference increment and existing preload; opt-out outputs remain bitwise
identical. Repeated previews, common rotations, invalid mode and input rate/limit
algorithms pass. The maximum virtual normal-effort increment in the controlled
nonnormal test falls from 0.2515 N to arithmetic roundoff. This is a synthetic
algebra test, not measured external force, actual command replay, nonlinear free
motion or native load qualification. No maintained source or collection recipe
changes. Canonical result: `D4-force-orthogonal-projector-controls-closeout-20261107.json`.
Its 7.030644 GPU seconds are booked once; all 252 continuation allocations are
closed. Actual causal replay and matched free/perturbed dynamics are the next
required steps before any native successor.


The corrected quaternion reader and actual causal command replay are closed in
`D4-force-projector-causal-replay-closeout-20261107.json`. The transport identity
still covers 1,353 states, with a 3.33e-16 N virtual-reference algebra defect.
Original wrong-axis results and charges remain available through
`normal-wrench-screen-quaternion-invalidation-20261107.json`.

The separate command replay reconstructs 5,144 actual requests across original
N/H+/H-/P and nominal PGS histories. Exactly one first approach state per history
is unavailable; no nearest or future state fills it. Exact current q, Jacobian,
previous delivered target, command clock and dispatched normal history reproduce
the original requested and delivered targets within 2.384e-7 rad, supported by
float32 arithmetic. Candidate opt-out remains bitwise identical and all five
release handoffs retain the achieved-pose semantics. Maximum nonnormal virtual
normal-effort increments fall from 0.0202–0.0593 N across the histories to
6.24e-5–6.54e-5 N, including float32 requested-target rounding. Maximum requested
joint changes are 2.12e-5–6.15e-5 rad. These are single-command counterfactuals on
fixed recorded states, not measured contact force or changed-path qualification.
The two GPU allocations charge 17.082159 seconds and are closed; matched coupled
motion is still required before any native candidate assignment.


The subsequent conditional full-motion model rejects the isolated projector
before native assignment. Twenty-eight 4,659-control-tick paths include eight
nominal zero-perturbation replays, eight original-controller controls, eight
candidate 0/10 N paths and four candidate changing-load paths. Nominal replay
is exact by construction. All eight original controls pass the modeled original
predicates. The candidate misses at least one pose endpoint in all eight
constant-load paths, leaves required future support in seven and fails the
original 1 mm ramp in three. All four changing-load paths also fail; matching
original variable-load controls were not run. All 28 modeled physical position,
speed, acceleration and right-arm effort-cap predicates pass.

The projector changes normal pose displacement while preserving the other five
requested pose components. In the model, normal error accounts for 99.03% to
almost 100% of opening-position error, including unloaded cases. Removing a
virtual target-effort increment does not preserve actual free motion or contact
force. Trajectories after a first modeled refusal are diagnostic continuations,
not guarded native dispatches; late large errors are conditional on the frozen
inertia/forcing and position-integration assumptions. Full fourteen-shape stops,
other-joint effort caps and changed contact physics are unqualified. No native
candidate or dependent contact trial is launched.

A separate read-only scalar audit reconstructs the original outer loop across
the five completed teacher histories. All 1,358 pressure/gap calls use exactly
the previous completed interval's recorded feedback. The four replayed helpers
are AST-identical to frozen source `519c347`. Original ramp and feedback history
recover compression omitted from the transition trace; 2,113 contact/push goals
reconstruct against recorded panel poses and shape offsets with a 50.63 nm
maximum discrepancy. One declared precontact approach sets only the material
plane origin; this is not a fitted calibration or physical tolerance. Pending
advance remains a recorded past input, not independently recomputed geometry.
No omitted or duplicated outer correction is identified, and no future force
or counterfactual contact response is qualified.

Evidence is `D4-force-projector-whole-model-and-teacher-chain-closeout-20261107.json`,
`D4-force-projector-whole-coupled-model-local-20261107/report.json` and
`D4-teacher-normal-command-chain-review-20261107.json`. The model charges
46.356157 GPU seconds; all 255 continuation allocations are closed. Collection
remains 17/36, with no maintained controller, recipe, domain or criterion change.
Next work concerns contact-surface normal kinematics and the actual four-substep
motion response, including finger/jaw rotation and changing load, while retaining
full free-reference motion, achieved release and the causal pressure/gap loop.


A subsequent CUDA collision-support audit closes all 20,596 actual native samples
across the original four histories and PGS. It compares closed-finger geometry,
measured jaw-position FK and direct recorded actor collision geometry at exact
clocks. Zero-jaw, independent vertex, axis-boundary/perturbed and rigid-coordinate
controls pass. During push, jaw changes relative to normal-mode entry peak at
45.99 nm and direct-body differences from the closed model at 106.13 nm. These
results do not justify a jaw patch. Rigid angular support changes span maxima
of 1.33–5.06 µm per native step across the histories. A first reader failed its
independent vertex-control batch dimension before reading native histories;
its script, log and 7.026087 GPU seconds remain. The corrected audit charges
10.057673 seconds. Evidence is
`D4-contact-surface-jaw-geometry-v2-local-20261107/report.json`.

A forty-path conditional model then compares the existing 60 Hz teacher shape
reference with the same reference plus native-step rigid shape transport. All
sixteen previous nominal-replay/original-controller histories remain bitwise
identical. Both shape clocks preserve all eight constant-load 0/10 N paths,
including original base-goal endpoints, 1 mm ramp, support, physical position,
speed, acceleration and right-arm effort caps. Each retains four changing-load
negatives; the model has no actual outer contact-pressure loop, and one H− stress
case per clock also fails a modeled physical predicate. Late post-refusal paths
remain diagnostic continuations. Neither variable contact behavior nor native
stops are qualified by this model.

The intermediate update retains the pure-normal pose direction and original
native rate, joint-target domain and total control-interval budget. Saturation
commits only the pose motion actually delivered by the quantized target. The
last world goal includes that payment before the next absolute outer command,
avoiding a second payment of the same shape change. Other requested pose-axis
defect peaks at 1.52e-18 in the double-precision model. Limited updates remain
recorded. Eight isolated CUDA controls at 448 recorded Jacobian/target postures
cover fixed geometry, preview purity, float32 target arithmetic, original limits,
saturation, coordinate rotation and the outer-command boundary. The actual
float32 other-axis defect peaks at 1.542e-7 in pose units, checked against an
operand-derived arithmetic allowance rather than a physical tolerance.

The nominal native 0/10 N component completes once on owned Blackwell with
frozen source `ab87675` plus the isolated geometry hook. Both cases pass the
independent original six phases, base-goal endpoints and 1 mm ramp, three-second
hold, measured release/stabilization, current/single-future 10 N support,
physical/rate/acceleration/effort limits, frozen moving/stopped models and
one-latch eight-second fourteen-shape stops. Every new native increment also
runs the original movement guard. The causal journal retains 43,032 packets,
including 3,840 stopped packets, and independently joins all 21,474 intermediate
targets. Geometric increment reconstruction differs by at most 7.71 nm, normal
payment by 1.22e-15 m, and archived float32 world goals reproduce exactly. Original
preview equality is checked against the first actual native target; final
targets are not copied onto earlier substeps.

The first motion reader compared a shape-adjusted controller goal against a
reported original base-goal endpoint. That reference-association failure and
its 7.030110 GPU seconds remain. Reconstructing original base goals from the
unchanged plan repairs the comparison for all normal phases, preserving the
original thresholds and controller-goal diagnostics. No native run is repeated.
Whole custody closes 26 original files and three sidecars, 76,332,001 bytes, with
actual GPU/PGID/lease ownership and release established. The successful motion
and causal-clock reviews charge 11.062401 and 7.027951 GPU seconds; target custody
charges 13.080754 seconds and the native job 441.873830 seconds.

A separate static code and data audit blocks A1 source adoption. In 3,554 of
3,579 unloaded intervals and 3,576 of 3,579 loaded intervals, targets vary across
the four native steps. Maximum changes from the first target are 64.37 and
76.53 microradians. The current teacher records one first command at 60 Hz;
the A1 joint executor clears normal pose mode and does not run this prototype.
No single first, final or mean target reproduces the exact nonconstant profile.
This audit runs no policy evaluation, training or additional physics. All related
conditions and contact successors remain suspended while the common A1 source
boundary is unresolved. Next work preserves the existing held 60 Hz motor-target
and recording contract through a causal command correction. This qualified
robot-only component adds no source, observation, variable-contact or teacher
credit.

Evidence is `D4-native-rate-shape-preflight-closeout-20261107.json`,
`D4-native-rate-shape-matched-model-local-20261107/report.json`,
`D4-native-rate-shape-software-controls-local-20261107/report.json` and
`D4-native-rate-shape-zero-ten-native-manifest-20261107.json`. The model and
software controls charge 62.485277 and 7.031963 GPU seconds. Complete nominal
custody and the A1 limit are recorded in
`D4-native-rate-shape-nominal-complete-closeout-20261107.json` and
`D4-native-rate-shape-A1-action-contract-audit-20261107.json`. All 264 continuation
allocations are closed, with no active or prepared job. The verified matrix
remains 17/36: thirteen ordinary and four recovery cycles.


### Causal held-target shape proposal preflight (November 7, 2026)

The native-rate reference cannot supply the existing A1 action contract. An
isolated successor therefore keeps one float32 motor target for each 60 Hz
interval. Current full twenty-coordinate mass, armature, impedance and measured
state, together with only the last completed affine response residual, predict a
bounded four-step shape-transport reference. A scalar correction matches its
predicted endpoint normal coordinate of tool position plus rigid collision
support. No future plant geometry or fixture-force schedule enters the proposal.
The model remains a controller preference, with no new force, support or stop
certificate. All gains and original criteria remain unchanged.

The endpoint derivative can have either sign. The initial positive-only check
was an unjustified refusal, corrected to use the finite nonzero signed response.
That failure and its 8.036316 GPU seconds remain. A later fixed-geometry test
passed the wrong positional argument; its original reader and 8.037706 seconds
remain, with no native rerun. Eight software controls then pass at 64 actual
states, including strict inward float32 rounding at original rate/domain faces,
pure previews, fixed geometry, other-axis numerical closure and the coupled
four-step response. Seven adapter controls at the same states establish pure
preview/dispatch identity, one target across four dispatches, explicit incomplete
past-packet rejection, ordinary A1 bypass and unchanged float32 simulator default.

All 52 conditional whole paths reach 4,659 control ticks. Forty prior histories
remain bitwise identical. All eight new constant 0/10 N paths pass the original
base endpoints, 1 mm ramp, current/future support, position, speed, acceleration
and arm-effort predicates. Four changing-load paths remain negative; H-minus
also fails original speed and acceleration. The teacher outer pressure loop is
absent, and post-refusal continuations are diagnostic model paths only. These
results do not qualify native motion, changing contact, stops or demonstrations.

The nominal native 0/10 N successor is prepared with the unchanged full program
and original eight-second fourteen-shape stops. It additionally requires all
four actual journal targets to equal the preview and an exact ordinary A1
joint-delta roundtrip. Related and contact jobs remain suspended until informative
complete qualification. Evidence is
`D4-held-causal-shape-preflight-closeout-20261107.json`,
`D4-held-causal-shape-matched-model-local-20261107/report.json` and
`D4-held-causal-shape-zero-ten-native-manifest-20261107.json`. All 269 continuation
allocations are closed at preflight closure; five new allocations charge
132.981625 GPU seconds including both failures. Collection remains 17/36, with
no new source, optimizer update or adopted production controller.


### Exact A1 float32 target repair (November 7, 2026)

The first held-target native nominal component stops in unloaded opening at
42.6 seconds. All four physical targets are identical, but the recorded joint
delta reconstructs one adjacent float32 value for RIGHT_WRIST_X: the difference
is 2.3283064365386963e-10 rad, exactly one target ULP. This is a demonstrated
action representation mismatch, with no observed physical gate failure on the
available prefix. All eight original native 0/10 N paths reconstruct exactly.
The new outcome remains negative; its allocation return code zero does not
override the incomplete report and assertion. It charges 198.291507 GPU seconds.

Whole custody retains 23 original files and three sidecars, 25,916,269 bytes.
The journal contains 10,228 delivered native movement packets and the saved
measured-feedback stream 10,224. The final four feedback packets were not saved
after the assertion; they remain unavailable and are never filled with actuator
commands. Current and original single-future 10 N position, physical, speed and
acceleration gates pass on all journal states. Frozen measured residuals pass
where recorded and in all stopped segments. Independent original latch-frame
eight-second, fourteen-shape CAD, tool, joint, speed, acceleration and force
checks pass. Later unloaded phases and the entire loaded case are unattempted.

The proposal now canonicalizes its target through the existing bounded A1
joint-delta operation before preview, admission and dispatch. The encoded delta
must replay exactly. At an original rate/domain face, actual float32 delta
neighbors move inward; no gain, domain, physical criterion or numerical
acceptance range is relaxed. Only actual delivered normal payment enters goal
memory. Five controls at 39,589 native states pass 197,945 target cases, including
both faces, zero crossing and midpoint. Eight software and eight Warp-adapter
controls pass at 64 states. All eight constant-load conditional whole paths
retain original predicates, with forty prior histories bitwise identical. Four
changing-load negatives, including H-minus speed/acceleration, remain.

Evidence is `D4-held-causal-shape-A1-roundtrip-failure-audit-20261107.json`,
`D4-held-canonical-A1-grid-controls-20261107.json` and
`D4-held-canonical-A1-preflight-closeout-20261107.json`. The corrected nominal
native successor is prepared in
`D4-held-canonical-A1-zero-ten-native-manifest-20261107.json`; all 274 allocations
are closed at this checkpoint. Its complete original movement, stop, exact
profile and predictor-input custody remain required before related/contact
progression. Collection remains 17/36 with no adopted source or controller.


### Loaded response singularity and minimum motor correction (November 7, 2026)

The canonical A1 held-target nominal native run completes and independently
qualifies its original 0 N motion and stops. Its 10 N case fails during opening
at native time 59.820833 s, control interval start 59.816667 s. RIGHT_SHOULDER_Y
acceleration is 8.1623945236 rad/s², above the original 7.8769288063 rad/s² limit.
Physical acceleration is the only failed predicate; available current and single
future 10 N support, physical position, speed and frozen moving/stopped residuals
pass. Independent original eight-second, fourteen-shape continuous stops pass.
Later loaded opening, hold, release and stabilization remain unavailable.

One of four native steps in the last held-target interval was delivered before
the guard refused further advance. Its final measured feedback and declared
goal position were not recorded. Whole transfer preserves 26 original files and
three sidecars, 62,221,373 bytes; actual owned process/GPU absence and canonical
lease release pass. The process return code zero does not override the negative
whole report. All 6,569 causal proposal inputs and actual motor targets replay
exactly; 6,568 available declared goal positions join exactly. All complete
intervals retain four identical targets and exact ordinary A1 reconstruction.
Readers preserve the partial interval and scope full qualification to 0 N.
Earlier reader preparation failures and misleading inherited positive summaries
are retained alongside their corrections; no native rerun repairs metadata.

The common failure is the inverse of a nearly zero dynamic endpoint response
along one kinematic pure-normal direction. At the refused state a 20.9 nm desired
endpoint increment produces an unbounded 62.5 mrad motor request before the
original target/rate limits. The successor uses the minimum motor norm solution
within the nullspace of the other five instantaneous reference axes. It retains
the same current coupled 20-coordinate response and last completed residual,
original gains, pseudoinverse cutoff, target/rate domain, A1 mapping and all
physical and empirical criteria. No damping or threshold is fitted to the failure.

On all 6,569 actual proposal states, the old targets reconstruct exactly and the
new unbounded solution satisfies the endpoint and other-axis constraints with
no larger motor norm. At the refused state the largest proposed motor correction
is 17.93 µrad. This is a frozen-current counterfactual, not executed physics.
Eight software and eight Warp-adapter controls pass at 64 recorded states.
The 52-path conditional model retains forty prior histories bitwise and passes
all original predicates on eight new constant 0/10 N paths. All four changing-load
negatives remain, including H-minus speed/acceleration. The outer pressure loop
and changed nonlinear contact response remain unqualified.

Evidence is `D4-held-canonical-A1-mixed-native-closeout-20261107.json`,
`D4-held-minimum-motor-current-response-local-20261107/report.json` and
`D4-held-minimum-motor-preflight-closeout-20261107.json`. At this checkpoint all
284 continuation allocations are closed, charging 9.493105594 GPU hours. The
nominal robot-only successor is prepared in
`D4-held-minimum-motor-zero-ten-native-manifest-20261107.json`. Related conditions
and contact successors remain suspended pending complete independent nominal
motion, stop and action/input qualification. No controller, source, calibration
or demonstration is adopted. Collection remains 17/36.


### Minimum motor nominal native qualification (November 7, 2026)

The minimum motor norm held-60-Hz nominal component independently passes both
original 0 N and 10 N paths: all six phases, original base 1 mm normal ramp,
three-second hold, achieved-pose release, stabilization, current and single
future 10 N support, physical limits, speed, acceleration, effort and frozen
moving/stopped models. Both continuing stops pass the original one-latch
eight-second, fourteen-shape CAD, tool, joint, rotation and force criteria.
The earlier scalar-direction inversion failure and all original bounds remain.

All 9,318 control intervals contain exactly four identical native motor targets
and reproduce the ordinary A1 joint-delta target exactly. All 7,158 correction
receipts reconstruct from current coupled state and only the last completed
residual; actual targets, declared current goals and delivered payments join
exactly. The journal has 43,032 native packets, including stops; all 39,192
non-stopped feedback packets are verified. The pre-physics PD effort clock is
independently checked. Effort commands remain distinct from measured feedback.

Whole transfer preserves 26 original files and three sidecars, 69,459,609 bytes,
with actual owned process/GPU absence and canonical lease release. The native
allocation charges 711.572580 GPU seconds; three independent GPU readers charge
26.122010 seconds. All 288 completed continuation allocations are closed,
totaling 9.698020758 GPU hours. No changing-load, pressure-loop, actual contact,
observation, source, policy or teacher qualification follows from these components.
Collection remains 17/36.

Evidence is `D4-held-minimum-motor-nominal-complete-closeout-20261107.json` and
the nominal run's independent motion, held-profile, proposal and actuator-clock
reviews. Independent H+/H−/P 0/10 N whole-path components are prepared on the
original elab4/elab3/elab2 hosts with exactly the same bundle and all original
criteria, in `D4-held-minimum-motor-related-zero-ten-native-manifest-20261107.json`.
Each outcome still requires whole transfer, owned cleanup and independent
complete qualification. Contact successors remain suspended.


### Related P qualification and missing contact response data (November 7, 2026)

Related P independently passes both minimum motor norm 0/10 N paths with the
same nominal-qualified bundle and every original complete movement, current and
single future 10 N support, physical, rate, acceleration, effort, frozen model
and eight-second fourteen-shape stop criterion. All 9,318 A1 target profiles and
7,158 current/last-completed proposal and goal-payment joins are exact. The
43,032-packet native effort clock passes, with 39,192 non-stopped measured
feedback packets. Whole custody retains 26 original files and three sidecars,
69,036,841 bytes; actual owned process/GPU absence and lease release pass.

The P native allocation charges 725.051138 GPU seconds and three readers charge
26.130495 seconds. All 292 completed continuation allocations are closed,
totaling 9.906682322 GPU hours. H+/H− remain running and unqualified at this
checkpoint. Their results are independent; P does not qualify either posture.
The closeout is `D4-held-minimum-motor-P-complete-closeout-20261107.json`.

An independent Blackwell measurement now adds only full native mass, armature,
targets and actual drive gains to the original same-host contact trace, retaining
source 519c347, setup, calibration, materials, controller and all criteria. The
old trace's arm inertia block cannot substitute for the coupled 20-coordinate
response. This measures missing inputs rather than retrying the negative for
success. Complete custody, exact original outcome, all seventeen historical
arrays and all prior contact/body packets are required before associating the
new data with that trajectory. Missing predecessors remain unavailable.
The manifest is `D4-original-friction-full20-native-readback-manifest-20261107.json`.
No changing-pressure, actual-contact, observed source or teacher qualification
is added. Collection remains 17/36.


### Full20 contact inputs recovered with exact trajectory association (November 7, 2026)

The read-only Blackwell measurement preserves all seventeen historical arrays
in both values and bytes, every prior contact/body packet, and the original
outcome on all 3,484 native states. Primary pre-release contact loss and secondary
load-envelope refusal remain negative; the original process return code one is
retained. All added full mass, armature, targets, drive gains and current q/dq
fields are finite and available. Their original arm-block subsets are exact.
There are 3,483 adjacent native intervals; the first predecessor is unavailable.
No missing state is zero-filled and no forward response is yet qualified.

Whole custody retains 28 original files and three sidecars, 1,663,908,509 bytes,
with actual owned process/GPU absence and canonical lease release. The allocation
charges 234.784750 GPU seconds. All 293 completed continuation allocations are
closed, totaling 9.971900308 GPU hours; H+/H− are the two remaining active native
qualifications at this checkpoint. The closeout is
`D4-original-friction-full20-readback-closeout-20261107.json`.

Source inspection confirms that expected load already uses full mass times full
acceleration plus gravity and Coriolis. This is not an omitted arm/non-arm inertia
term to patch. Next analysis uses the recovered inputs for the causal controller
response and pressure/geometry interaction, preserving both old failures and
all original physical, empirical and stop criteria. No teacher source is added;
collection remains 17/36.


### Eight held-target paths and actual-contact causal inputs (November 7, 2026)

H+/H− independently pass their original complete 0/10 N movements and continuing
stops, completing the eight-path N/H+/H−/P minimum motor norm component family.
Every original base 1 mm ramp, endpoint, three-second hold, achieved release,
stabilization, current/single-future 10 N support, physical, rate, acceleration,
effort, frozen moving/stopped model and one-latch eight-second fourteen-shape
stop criterion passes. All 37,272 A1 profiles, 28,632 causal proposal/goal/payment
joins and 172,128 native effort-clock packets pass. Their 156,768 non-stopped
feedback packets are verified. Both H whole transfers and owned cleanup pass.
Earlier negatives and all four changing-load model failures remain preserved.
The family closeout is `D4-held-minimum-motor-full-family-closeout-20261107.json`.

The real-contact Temporal recording provides 871 declared physical world-pose
commands. Exact request/state/dispatch/application and applied-target custody
joins 870 to current full-system state and completed predecessors. The first
has no preceding native mass sample and stays unavailable. The 109 active
normal-reference commands, 29 contact and 80 push, all have complete causal
inputs. No images are read and no later force or pose enters the proposal.
Actual subsequent states and contact loss are retained solely for scoring.

The fixed-current, last-completed-residual proposal satisfies every endpoint
and other-five-axis linear constraint. All 109 proposed four-step forecasts
remain inside original physical, speed and acceleration limits; none is target
limited. The largest quantized motor correction is 6.44 µrad. The predicted
geometry change reaches 0.219 µm, while original endpoint surface prediction
error reaches 12.2 µm and model/native surface closure differs by 0.261 µm.
These observations cannot establish a response or safety bound, a new closed
path, pressure stability or contact success. The original primary contact loss
and secondary load-model refusal remain negative.

The initial analysis used a remote-only asset path and failed before proposals;
its charge and original script are retained. The corrected analysis requires an
identical local URDF hash. Across these two allocations all 303 completed
continuation allocations are closed, totaling 10.723980664 GPU hours. Evidence
is `D4-actual-contact-current-input-custody-20261107/report.json`,
`D4-actual-contact-minimum-motor-response-v2-local-20261107/report.json` and
`D4-actual-contact-causal-response-study-closeout-20261107.json`.

One nominal actual-contact intervention is prepared in
`D4-held-minimum-motor-actual-contact-native-manifest-20261107.json`. It applies
the same eight-path-qualified geometry operator while retaining original source
519c347, setup, pressure/gap/feedback law, normal-mode activation, gains,
materials, calibration, complete task and original stops. Joint-only inspection
is unchanged; installation precedes the first physical pose command. Read-only
native target and response journals preserve causal custody. This is a
privileged component with no observed-source or teacher credit. Related contact
successors remain suspended. Collection remains 17/36.


### Actual contact outcome and material sampling clock (November 8, 2026)

The single nominal minimum motor norm intervention from local milestone
`a18e8f5` remains negative. Original push has 80 commands and ends at 0.458°;
the new path has 345 commands and reaches 1.957°. Both lose contact before
hold and encounter a secondary empirical position refusal during abort release.
The new path records every one of its 4,540 native packets and 1,135 complete,
exact A1 intervals. Original six-second fourteen-shape continuing stops and
actual pre-physics effort clocks pass. Current position support has two exits;
the unchanged single-future support audit has 147. Physical future boxes remain
clear. Positive contact lower-force windows do not pass. No hold, stabilization,
complete task, pressure component or source is qualified.

All original outputs transfer whole, 32 files plus three sidecars, totaling
1,833,669,240 bytes; owned GPU/process cleanup and lease release pass. Native
contact normal sums, filtered/raw forces, full-state clocks and friction data
are independently checked. Recorded head RGB preserves the door and gripper
view at push and abort release, but cannot establish micron-scale contact or
observed-source admission. All six new allocations close, including failed
proposal-reconstruction controls, bringing the continuation ledger to 309
allocations/10.811842272 GPU hours. The closeout is
`D4-actual-contact-minimum-motor-closeout-20261107.json`.

Independent scalar reconstruction retains six target disagreements among
374 proposals, up to 1.1921e−7 rad; the remaining 368 are exact. A reader's
extra quaternion normalization is corrected and its original result retained,
but this does not resolve the disagreement. Cross-device arithmetic and missing
actual root quaternion remain separate unresolved inputs; no numerical tolerance
is adopted. Original physical task negatives do not depend on this control.
The next read-only journal adds actual pre-physics root pose.

The executed-target surface forecast differs from the achieved surface by up to
10.9 µm during push, versus a predicted geometric correction below 0.21 µm.
The material-clock audit independently advects the already declared virtual
reference point around the known hinge. It uses completed current angle/velocity
or the last completed acceleration; actual next angle is scoring only. The
reference moves up to 28.5 µm per interval. Current velocity forecasts its normal
travel with 0.107 µm RMS/0.285 µm peak error during push; adding completed
acceleration has 0.131 µm RMS/0.538 µm peak error. These are reference-motion
scores, not fitted contact predictions or a physical bound.

One new nominal component is prepared in
`D4-causal-material-clock-native-manifest-20261107.json`. Only active push/hold
material-frame requests are advanced by current measured hinge velocity over
exactly one existing control interval. Essential rigid point/orientation,
handedness, inverse, zero-speed identity and invalid-clock controls pass. All
original setup, geometry proposal, pressure/gap law, gains, material, load,
physical, task, release and continuing-stop criteria remain unchanged. Related
native contact conditions stay suspended. Collection remains 17/36; no observed
source or teacher credit is added.


### Material-clock negative and scoped tangential-impedance experiment (November 8, 2026)

The causal material-clock N intervention from `e4a1c15` is independently closed
negative. It has 97 push commands/0.612° before primary contact loss, followed by
the original release position-support refusal. All 888 complete A1 profiles,
126 geometric proposals/paid goals and 97 current-motion/predicted-request joins
are exact; actual pre-physics effort and all original six-second fourteen-shape
stops pass. Native root inputs remove reconstruction ambiguity in this recording;
the earlier six mismatches stay preserved. Correctly applying a current-velocity
reference forecast does not establish stable contact.

All 3,552 native packets, contact force sums, current/full-state clocks and
original pressure/current/single-future audits are retained. Whole transfer is
33 files plus three sidecars/1,678,972,629 bytes, with owned process/GPU absence
and lease release verified. Four new allocation charges plus the integration
audit close once, bringing the continuation ledger to 314 completed
allocations/10.887384493 GPU hours. Evidence is
`D4-causal-material-clock-closeout-20261108.json`.

The separate saved-data integration audit covers 180,220 packets over all eight
qualified robot-only paths and both actual-contact negatives. Semiimplicit Euler
has roughly 0.96–1.40 µrad RMS arm error during original opening/release;
explicit and trapezoidal alternatives are worse. Actual-contact push has larger
semiimplicit residuals, up to 115.8 µrad, and last-completed residual changes up
to 128.9 µrad. These formulas use measured post-velocity solely for diagnostic
scoring. No new causal model, fitted coefficient, safety margin or precision
allowance is adopted.

One nominal actual-contact experiment is prepared in
`D4-finite-tangent-actual-contact-native-manifest-20261108.json`. It enables
only the existing `700191d` finite tangential impedance, keeps the prior
minimum-motor geometric proposal, and disables material anticipation. This
addresses the common coupling between accumulated other-axis elastic pose error
and unilateral normal contact. The original normal pressure/gap law, native
feedback, gains, materials, setup, complete task and all physical, empirical,
single-future and continuing-stop criteria remain unchanged. The load model is
identical to `519c347` by source comparison; that revision adds only the optional
pose mode and its recording metadata.

The prior `700191d` family passed all four complete 0 N paths but failed every
10 N ramp tracking endpoint at 11.07–12.53 mm. Those original tracking criteria
and failures remain unchanged and the full 10 N controller domain remains
unqualified. This single actual-contact experiment is a mechanism diagnostic,
not a replacement qualification for that domain. Expected 0.1 N pressure targets
never replace measured load or the unchanged 10 N future/stop audits. All related
conditions remain suspended and no teacher/source credit is permitted. Collection
remains 17/36.


### Completed finite-tangent contact task with strict qualification failures (November 8, 2026 UTC)

The single nominal experiment from `c968deb`, isolated source `700191d` plus the
unchanged minimum motor norm held-target bundle, completes 50.522076° opening
against 50.146446°, three-second hold, six-second achieved release and two-second
stabilization. Original materials, setup, gains, native feedback, pressure/gap
law, normal activation, load model and acceptance criteria remain unchanged;
material anticipation is off. The original report passes the complete task.

Independent readers verify all 13,265 exact A1 intervals, 53,060 native packets,
12,274 bitwise-exact current-root causal proposals and paid goals, explicit
pre-physics effort clocks, native state/contact sums, and the original continuous
six-second fourteen-shape stop. There are no unavailable or partial intervals.
All current empirical positions and original future physical boxes pass. Frozen
load estimates remain within 10 N, with a 4.690655 N upper peak and a 4.049628 N
actual normal-force peak. Eight selected original RGB frames across all six
movement phases were visually reviewed; their visible panel/gripper motion does
not establish micron contact, force continuity or observation identity.

Strict independent pressure qualification is negative. The original future box
has 3,717 shoulder-X lower-support exits during push, at most 1.264 mrad, and 287
wrist-X lower-support exits during hold/release, at most 6.724 mrad. These are
empirical support gaps, not physical limit violations. Of 48,980 push/hold
100 ms force-lower windows, two are nonpositive, with a −0.016776 N minimum.
One corresponding actual native normal-force window is zero. The original
control-level continuity flag stays positive and is preserved separately;
it cannot replace this independent native-window check. Applying maintained
`01f814a` uncertainty arithmetic to the same data improves the lower bound but
retains both nonpositive windows. No threshold, support domain or error budget
is changed, and no force gap is relabeled as uncertainty alone.

The full 40-file output and three sidecars (9,205,859,826 bytes), original
outcomes, failures and owned cleanup are retained. Three allocations close once,
bringing continuation accounting to 317 closed allocations and 11.376352 GPU h.
The closeout is `D4-finite-tangent-actual-contact-closeout-20261108.json`; causal
support/normal comparison is `independent-saved-support-and-normal-diagnostic.json`
inside its finished run. The 1 GB native sidecar is reviewed incrementally with
the original comparisons and denominators, without retaining all raw rows.

All four historical finite-tangent 10 N endpoint failures remain failed and that
controller domain remains unqualified. The completed physical task does not
qualify pressure, observed geometry/identity/material/finite stopping, a source
or a teacher demonstration. Related contact successors remain suspended.
Collection stays 17/36. Next work separates independent original-10-N empirical
support measurement from causal diagnosis of the actual 100 ms gap, preserving
full movement, realistic load variation, original force and stop criteria.


### Separate support measurement and coupled timing test (November 8, 2026 UTC)

A streaming audit of the completed finite-tangent trajectory retains 4,618
push/hold native zero-force runs. The longest has 24 samples, exactly 100 ms,
with a 37.812 µm maximum observed separation in the surrounding 17 control
commands. Its causal operands show normal unloading after a force peak,
followed by delayed gap recovery. These observations do not identify a unique
plant cause or establish a new tolerance. The full native force and the
original control-level loaded flags remain separate.

Across all 12,065 push commands, current-velocity prediction of the already
known virtual material reference has 0.264 µm RMS / 0.881 µm peak normal error,
versus 14.337 / 19.584 µm for the held current frame. On the seven reference
commands overlapping the zero-force window, peak error changes from 12.720 to
0.717 µm. Adding last-completed acceleration has worse full-path peak error;
no parameters are fitted. Actual future door motion is scoring only. This
reference forecast does not establish force or a physical response bound.
The audit is `D4-finite-contact-material-clock-diagnostic-local-20261108/report.json`;
its one closed allocation brings continuation accounting to 318 / 11.380265 GPU h.

One explicit fourth cell of the two-factor nominal contact experiment is
prepared and dispatched once: `700191d` finite tangential impedance plus the
already verified current-velocity material-reference clock. Neither factor
alone established strict contact qualification: the earlier minimum-motor and
timing-only trials lost contact; finite tangent alone completed the physical
task but failed strict force/support checks. The combined trial changes no
setup, gain, material, pressure/gap law, empirical/physical limit, full-task or
original six-second stop criterion. Its manifest is
`D4-finite-tangent-material-clock-native-manifest-20261108.json`. All historical
10 N tracking failures remain failed; this diagnostic grants no controller,
observed-source or teacher qualification. Related conditions stay suspended.

Independently, the original joint/load fixture tests the actual coupled
shoulder-X/push and wrist-X/release future extrema, preserving all other measured
coordinates. Each posture has unloaded, center-10-N, and two distal-face/yaw-10-N
cases, original 0.5 mrad seven-joint pulses, frozen residuals and continuous
2+4+2 s fourteen-shape stops. These jobs are separate from contact control and
cannot automatically establish a Cartesian envelope or adopt calibration.
The manifest is `D4-finite-contact-N-required10N-support-manifest-20261108.json`.
Initial preflight schema/static-copy errors and one unsupported driver-purpose
rejection are preserved with zero GPU charge. The rejected driver had no native
allocation, live launcher or lease; the corrected label uses the supported
mechanism purpose. Actual GPU PIDs, UUIDs and separate canonical leases are
verified for started jobs in each deployment.


The finite shoulder-X support subset is independently qualified: all four cases,
three required 10 N loaded boundary and stopped-coverage checks, original frozen
moving/stopped residuals, seven-joint physical/rate/acceleration/effort criteria
and same-latch eight-second fourteen-shape stops pass. Whole output is 28 files
plus three sidecars / 14,761,553 bytes; owned cleanup passes. The closeout is
`D4-finite-contact-N-j1-support-closeout-20261108.json`. Two allocations close
once, bringing continuation accounting to 320 / 11.458237 GPU h. Wrist-X is still
running. No calibration or Cartesian support is adopted; the original completed
contact task's strict force/support failures and matrix 17/36 are preserved.


### Qualified finite nominal future-position support (November 8, 2026 UTC)

Wrist-X independently passes all four original unloaded/10 N cases and loaded
boundary/stop coverage, with no frozen residual exceedance, clipping, physical,
rate, acceleration, feedback or original continuous eight-second fourteen-shape
stop failure. Both coupled postures therefore pass eight cases and 24 stop
segments. Whole originals are 56 files plus six sidecars / 30,091,303 bytes;
all support-job resources are released. Only the new wrist allocation and its
GPU reader are booked here: continuation accounting reaches 322 closed
allocations / 11.632135 GPU h. Shoulder costs are not charged again.

`D4-finite-contact-N-qualified-required10N-support-20261108/qualification.json`
prepares condition/path-scoped support from that independent evidence. Only the
lower shoulder-X and wrist-X position extents change; all other calibration
fields, force caps and previous support are identical. All 53,060 completed
finite-tangent N native samples pass current and the original single-future
seven-coordinate checks at the required 10 N. Both exact qualified boundaries
pass; 11 N and one original 0.5 mrad pulse beyond each boundary still refuse.
The original calibration's 4,004 future refusals remain recorded separately.
No current sample needs the newly added band, and no residual, force, physical
or stop criterion is altered. This is finite measured-posture/path support,
not an arbitrary Cartesian envelope or automatic transfer to another trajectory.

The closeout is `D4-finite-contact-N-required10N-support-closeout-20261108.json`.
Maintained recipes and the running combined contact test retain their original
inputs. The completed task's strict force failure, including the actual 100 ms
zero-force window, remains negative. No pressure, global controller,
observed-source or teacher qualification follows; collection stays 17/36.


### Combined contact timing outcome and remaining empirical support (November 8, 2026 UTC)

The combined finite-tangent/current-material-clock N trial from `2c78f31` is
independently closed negative during hold. It reaches 51.850548° after 11,194
push commands, then records only 165 hold commands (2.75 of the required 3 s).
Four final samples exceed original elbow upper empirical support by at most
0.706166 mrad; no preceding task failure is reported. Release and stabilization
are unavailable. All physical current states and original future boxes pass.

All 11,899 A1 intervals, 47,596 native packets, 11,388 exact current-root causal
proposals and paid goals, 11,359 current-input/predicted requests, native effort
clocks, contact-stream sums and original continuous six-second fourteen-shape
stops pass independent checks. Every available push/hold 100 ms force-lower
window is positive, with a 0.006122 N frozen minimum; four final windows remain
unavailable under the position guard. Actual native force-window minimum is
0.021709 N, with no complete 100 ms zero window; the longest zero run is 23
native samples (95.833 ms). This is prefix evidence, not complete qualification
or permission to change the original window criterion.

Original future support has 3,556 excluded intervals. Numeric comparison with
the separately prepared finite-only N support removes shoulder exits but leaves
286 future intervals and the same four current elbow failures. That artifact
is not automatically transferred to the new trajectory. Frozen/modeled upper
force peaks at 7.782284 N versus 7.028226 N actual. Available RGB frames across
approach/contact/push/partial hold remain visible; later phases are missing.

Whole output is 33 files plus three sidecars / 8,201,361,396 bytes; all owned
resources are released. The native job and three GPU readers close once:
326 continuation allocations / 12.081889 GPU h. Evidence is
`D4-finite-tangent-material-contact-closeout-20261108.json` and its run's
`independent-combined-support-force-diagnostic.json`. Historical finite-only
force failure, all 10 N tracking negatives, unavailable states and all original
criteria are retained. No source, pressure, global controller or teacher
qualification is granted; collection remains 17/36.

Earlier independently measured H− elbow-upper and wrist-lower frontiers are
broader and use identical frozen dynamic-model fields. Their condition/posture
scope prevents automatic N adoption. The next support study tests those
pre-existing frontiers at the actual combined-N coupled postures, with original
three-site 10 N perturbations and continuous stops, rather than expanding a
threshold to the failed contact point. Dependent contact retries remain
suspended until informative independent support verification.


The prospective transfer study is staged as
`D4-shared10N-frontier-N-transfer-manifest-20261108.json`: elbow upper
−0.312805806 rad and wrist lower −0.914791286 rad are taken from the earlier
qualified H− calibration, not fitted to the new N refusal. The new combined N
prefix requires −0.416832882 rad elbow upper and −0.913824428 rad wrist lower
under the unchanged original future allowance. At each selected complete N
posture, all other measured coordinates are retained. The fixture checks
unloaded and three 10 N sites/yaws, seven-joint 0.5 mrad motion, frozen residuals,
physical/rate/acceleration/effort/feedback limits and the original continuous
eight-second fourteen-shape stops. The complete transfer study closes mixed. Elbow upper passes all four cases;
wrist lower completes unloaded and center cases, fails the face0/yaw +0.2 case
while conditioning and leaves face1/yaw −0.2 unattempted. The wrist's broad
frontier is not qualified. Every available native sample passes frozen residual,
physical, rate, acceleration, effort, feedback and clock checks; all 21 original
continuous stop segments, including the failed case's same-latch eight seconds
and fourteen-shape covers, independently pass. Whole transfer is 53 original
files plus six sidecars / 28,816,788 bytes, with owned resources released. The
native reports determine scientific outcomes even though both outer allocation
records return zero. Evidence is `D4-shared-frontier-mixed-outcome-closeout-20261108.json`.

Reconstructing all 1,834 conditioning intervals across the three wrist cases
matches delivered targets within 0.120 µrad. The first undelivered wrist target
would spend the original 0.5 mrad physical reserve by 1.198 µrad. The failed
transferred frontier lies 0.967 mrad beyond the maximum original wrist future
requirement of either the completed finite-only path or combined prefix. The
failure remains a physical target refusal; neither its desired tracking band nor
its pulse/reserve is changed. Static force-balance screens retain all measured
other-joint coordinates, the original four load/site cases, fixed stiffness and
one frozen stationary residual allowance. They reject the broad frontier but
leave approximately 0.065 mrad minimum wrist margin at the original path's actual
future requirement across both entire saved paths. Zero load, signed load and
native-wrench controls pass; the largest saved static-wrench discrepancy is
0.0000374 N m. This approximation does not qualify dynamic pulses, conditioning,
contact or stopping.

The separate `D4-combined-N-scoped-support-manifest-20261108.json` therefore
prepares two informative studies: transfer the earlier N shoulder-lower extent
at the changed combined-path posture, and qualify its complete/prefix-derived
wrist-lower requirement. All other measured posture coordinates are retained;
all eight exact planned cases pass the static target-reserve preflight. Both use
the unchanged original fixture, seven-joint 0.5 mrad pulse, three 10 N sites/yaws,
frozen model and continuous eight-second fourteen-shape stops. Fresh host/source/
capacity/UUID/lease preflights pass. There is no automatic transfer from the
finite-only shoulder posture, no reuse of the failed broader wrist extent and no
new force, residual, physical or acceptance threshold. Original missing hold,
release and stabilization, historical 10 N tracking negatives and strict force
failures remain. Before the two new native allocations, 333 continuation
allocations close / 12.280223 GPU h. No maintained recipe, default calibration,
source, teacher or matrix claim changes; collection remains 17/36.


### Independently qualified combined N position support (November 8, 2026 UTC)

The two changed-posture studies from `719f480` independently pass all eight
original cases and 24 continuous same-latch stop segments. This includes wrist
face0 at the original complete/prefix-derived requirement; the broader H− wrist
transfer remains negative. Every original moving/stopped frozen residual,
physical, speed, acceleration, effort, feedback and clock predicate passes, as do
loaded boundary coverage and all fourteen-shape stop covers. Whole transfer is
56 original files plus six sidecars / 29,896,047 bytes; all owned resources are
released. The closeout is `D4-combined-N-scoped-support-closeout-20261108.json`.
The two native jobs and two serialized GPU readers close once, bringing the
continuation ledger to 337 allocations / 12.528750 GPU h.

The separate `D4-combined-N-qualified-scoped-support-20261108/` artifact combines
these eight cases with the independently qualified four-case elbow transfer.
Twelve measured cases and 36 original stop segments support only the two saved N
trajectories and exact coupled postures. The prior finite-only N support is
retained; only wrist-lower and elbow-upper extents grow, and only the newly
qualified elbow-upper side's force cap changes from 1 to 10 N. All other force
caps, residual coefficients, physical limits and calibration fields stay identical.
Boundary 10 N controls pass; 11 N and one original pulse outside each measured
frontier still refuse. No arbitrary joint-product box or condition transfer is
claimed.

All 53,060 finite-only and 47,596 combined-prefix samples pass the original
single-future seven-coordinate 10 N predicates. These checks do not replace
current-state load admission: a zero-tolerance side can be outside the estimator's
main domain without appearing in the motion-only side predicate. The frozen
`700191d` load estimator is therefore independently recomputed on all 47,596
combined-prefix current samples with the new qualified support. All estimates
are available and stay within 10 N (peak 7.782284 N); all 45,436 original strict
100 ms push/hold lower windows are positive, minimum 0.006122356 N. The four
original estimator refusals and unavailable estimates remain in the original
report. Original 4,004/3,556 future refusals, finite-only actual force gap and
strict force failure, broad wrist-transfer failure and all four finite 10 N
tracking negatives are preserved. Completing the remaining hold, release and
stabilization is still unvalidated.

One new nominal complete actual-contact component is dispatched once as
`D4-qualified-support-contact-N-native-manifest-20261108.json`. Fresh source,
runtime, UUID, owner and capacity preflight passes; the actual CUDA process and
canonical lease are verified during initial inspection. It retains the
exact previous combined `700191d` controller, held minimum-motor bundle,
current-velocity material reference, initial setup, original materials, force and
gap laws, physical limits and all original task/three-second hold/achieved release/
stabilization/current/single-future/strict force/six-second fourteen-shape stop
criteria. The intended change was independently measured N position support. However,
the finished consumer audit below shows that this file was not actually selected.
All other conditions remain suspended until an informative complete independent review.
This privileged component grants no observed geometry, source, global 10 N
controller or teacher qualification; the matrix remains 17/36.


### Actual load-calibration consumer binding (November 8, 2026 UTC)

The first qualified-support nominal successor is negative in hold at the same
51.85055° / 2.75 s point as its baseline. The intended support file existed in its
new deployment, but `recipe.load_calibration` retained the older rigid-support
deployment's absolute path. The actual remote file predates the allocation and
matches the original baseline, not the intended qualified artifact. Preflight
checked that referenced inputs existed and separately validated the new JSON;
it did not compare the real consumer's constructed model with that artifact.
The runtime correctly refused the old domain. This rollout therefore does not
exercise or invalidate the independently qualified support.

All 94 compared pressure, full-torso actuator and both stop-array fields exactly
match the previous same-host negative, including missing-value masks. Independent
reviews verify 11,899 exact A1 profiles, 47,596 native packets, 11,388 causal
proposals, 11,359 material requests, effort clocks and original six-second
fourteen-shape stops. The four current estimator refusals and 3,556 original
future refusals remain. Actual RGB selections across available phases were
reviewed; their pixels are not identical to the earlier rendering. Release and
stabilization remain unavailable. Whole transfer is 33 original files plus three
sidecars / 8,201,427,541 bytes, with owned cleanup. The native job and three GPU
readers close once: 341 continuation allocations / 12.977673 GPU h. Evidence is
`D4-misbound-qualified-support-contact-closeout-20261108.json` and the deployment's
`actual-consumed-calibration-path-diagnosis.json`.

The correction explicitly points the recipe at its deployment's qualified
`load.json`. The pressure fixture uses a shared loader guard that compares the
resolved path and complete parameters before constructing its model. Preflight
executes that exact consumer function: the nominal input passes, while the
historical path and changed parameters refuse. Before the first physical pose,
the live fixture separately checks and records the constructed model in
`actual-load-model-binding.json`. This closes the missed input connection; it
changes no force estimate, residual bound, physical limit, controller, material,
gain, setup or task/stop criterion. A distinct single nominal component is
dispatched once under `D4-bound-qualified-support-contact-N-native-manifest-20261108.json`.
Fresh preflight and actual CUDA ownership pass; before-motion readback confirms
the resolved intended path and exact constructed model. The completed original
output is now independently reviewed below. Related conditions stay suspended;
this component adds no source, controller-domain or teacher credit.

### Bound support resolves admission; hold still loses contact (November 8, 2026 UTC)

The corrected `700191d` nominal component actually consumes the independently
qualified N calibration. All 49,548 native current estimates are available and
inside 10 N, peak 7.782284 N; every original single-future seven-coordinate box
passes measured support and physical limits. The native force peak is 7.028226 N.
The previous calibration exception is eliminated, without changing control,
material, gains, pressure, uncertainty, physical or task/stop criteria.

The task remains negative: independently reconstructed controlled opening is
51.379074° versus 50.146446° required, but hold records 173 intervals / 2.883333 s
of its required 3 s before pre-release contact loss. One complete original native
100 ms force window is zero. Twelve strict force-lower windows are nonpositive,
minimum −0.243347 N, with zero unavailable estimates. The original continuity
failure reproduces from contact manifolds with no loaded/valid annotation mismatch.
The six-second achieved release completes, followed by two seconds of stable
unloaded motion. Final separation is at least 22.243 mm; release and stabilization
do not repair the earlier failed hold. The native report has no exception.

Independent readers verify all 12,387 exact A1 profiles, 49,548 full actuator
packets, 11,396 causal proposals and 11,367 material requests. Original six-second
fourteen-shape stops and all physical states pass. Nine original RGB selections
across all available approach/contact/push/hold/release/stabilization phases are
reviewed with causal capture availability; visibility does not establish force.
Whole transfer preserves 40 original files plus three sidecars / 8,684,783,180
bytes. The native job and three GPU readers close once with owned cleanup:
345 continuation allocations / 13.438652 GPU h. All prior failures, interruptions,
unavailable states and charges remain; the matrix stays 17/36.

Evidence is `D4-bound-qualified-support-contact-closeout-20261108.json` and the
run's `independent-pressure-review.json`,
`independent-negative-complete-native-outcome.json` and
`independent-contact-loss-operands.json`. Next work diagnoses whole-path
unilateral support, material/hold reference, finite elastic response and phase
transition before another informative native intervention. No unchanged retry,
threshold relaxation, broad 10 N qualification or observed-source adoption follows.


### Whole-path unilateral response forecast diagnosis (November 8, 2026 UTC)

Actual CUDA scoring covers all 23,670 normal-mode intervals of the combined and
finite-only complete recordings. Each prediction uses the current full-torso
state/inertia, already dispatched held target and last completed affine residual.
Future physical state and force enter scoring only. No controller parameter,
integration law, gain or qualification criterion is fitted or changed.

The largest rigid-support surface errors occur immediately after recontact:
78.313 µm combined and 44.006 µm finite-only. The preceding native force is
3.489 N / 3.166 N, while all four next forces fall to about 0.170–0.426 N.
The frozen affine term carries that impact through the next four predicted
steps. These are contact-transition prediction errors, not new physical bounds.
The final failed combined hold interval itself has only 1.065 µm surface error,
despite its zero force and roughly 40–52 µm positive native gap. Model error at
earlier impacts therefore does not by itself establish the cause of final loss.

A matched saved-data alternative averages only the four native residuals in the
already completed control interval, with identical current state, target and
future scoring arrays. It reduces combined push/hold surface RMS from
1.973/5.185 to 1.040/1.892 µm, and finite-only push/hold RMS from 1.750/3.697 to
0.800/1.224 µm. However, both initial-contact transients regress: peak error rises
from 5.309 to 20.606 µm and RMS from 1.318 to 2.294 µm. The alternative is not
adopted and does not justify another native contact retry. Whole pose/normal
pressure, material reference and phase-transition behavior remain unresolved;
the combined hold's measured material error grows from 0.085 to 6.632 mm.

Both original 100 ms zero-force windows and all task/support failures remain.
Four saved-data GPU allocations close with owned cleanup, reaching 349
continuation allocations / 13.457394 GPU h. Evidence is
`D4-whole-contact-causal-forecast-closeout-20261108.json` and
`D4-whole-contact-mean-residual-forecast-closeout-20261108.json`, with complete
scored arrays, transition masks and copied reader sources. These are model
diagnostics; source, contact qualification and the 17/36 matrix do not advance.


### Outer tangential integral above finite impedance (November 8, 2026 UTC)

The next audit reconstructs all 23,668 original offset updates followed by a
supported-phase command on both complete N histories, using frozen `700191d`
code and original geometry/pose/error/enable operands. The two unused final
post-hold updates are outside this delivered-update denominator. The combined
path integrates 6.173 mm of tangential compensation during its incomplete hold
and reaches the original 7.5 mm extent; the finite-only path integrates 1.632 mm
during hold and peaks at 2.807 mm over the whole path. All original outcomes remain.

The outer compensator was still integrating measured material-position error
while the inner controller explicitly removed tangential elastic-error
integration. The correction retains the already applied material offset during
the actual normal-reference plus finite-impedance mode. It preserves entry
preload, ordinary compensation, existing pressure control, gains, materials,
support, physical limits and all original task/contact/stop criteria. No new
setup flag or filter parameter is introduced. The actual caller's four mode
combinations route this behavior only when both existing flags are active.

Default and explicitly nonfinite behavior is bitwise identical to all 23,668
reconstructed original updates. Entry-boundary, blocked-error reversal and mode
exit regressions pass. Actual CUDA tests compare both outer loops on 20 free and
20 blocked reduced plants, with rotated frames and perturbed Jacobian metrics.
Retaining the offset prevents additional blocked-reference accumulation. The
largest free tracking error is 0.999 mm, versus 0.374 mm with the old integral;
both stay inside the original 2.5 mm material-drift guard. Supplied reduced-model
position limits and the original rate limit pass. These are control design
tests, not native robot dynamics, force, friction, load-support or stop
qualification. The changed complete contact path is still untested.

Evidence is `D4-outer-tangent-integral-whole-path-audit-20261108.json`,
`D4-outer-tangent-integral-correction-closeout-20261108.json` and
`D4-outer-tangent-mode-CUDA-controls-20261108/report.json`. Its GPU allocation
closes with owned cleanup: 350 continuation allocations / 13.459906 GPU h.
The matrix remains 17/36, with no source or broad 10 N controller adoption.


### Remote whole-evidence review and isolated native successor (November 8, 2026 UTC)

The workstation cannot currently hold another complete native payload. A full
copy of the prior negative, 40 original files plus three sidecars /
8,684,783,180 bytes, is streamed directly from Blackwell to elab3 without local
large-file staging or deleting either original. Before/after source inventories
and every destination SHA256 match. Actual destination capacity is checked;
no cumulative storage or GPU-hour ceiling is introduced.

Independent readers run against the whole replica with frozen `700191d` math.
The first attempt fails before reading data because its child environment loses
the installed Isaac Sim `h5py` path. Preserving the official runtime's inherited
Python path repairs this infrastructure fault. Both attempts and their charges
remain. The complete rerun executes all original A1, pressure, current/single-
future, physical, effort, proposal, material-request and six-second fourteen-
shape stop readers. All 49,548 time/phase, availability, upper-force and strict
100 ms lower-force decision masks match exactly; the twelve failed lower-force
windows and native negative remain. A metadata assertion originally demanded
bit identity of derived cross-host scalars. Its two-ULP minimum difference and
complete maximum array differences are recorded, with no changed input,
criterion, outcome or additional GPU allocation. Review resources are released.
This establishes review readiness, not another demonstration.

One nominal successor is dispatched once on owned Blackwell. Its source is the
original `700191d` controller with only the `016f5ef` outer-helper and actual-caller
change overlaid. Restoring those two edits reconstructs the original module AST;
the minimum-motor bundle, material clock, scoped N load support, physical setup,
materials, gains and every original force/task/stop criterion stay unchanged.
Before motion, live readback verifies the exact source and constructed load
model. Its first actual finite-mode update retains the applied offset exactly.
The latest bounded health check verifies the owned GPU PID and process group;
the complete physical outcome is still open. Related H+/H-/P tests remain
suspended. The prepared remote review includes the actual outer consumer and
original full-task force/phase/RGB checks, after finished whole custody and owned
cleanup. Partial motion cannot qualify this privileged component or an observed
source.

Evidence is `D4-remote-WHOLE-review-readiness-closeout-20261108.json`,
`D4-remote-review-runtime-environment-failure-20261108.json`,
`D4-retained-outer-source-overlay-isolation-review-20261108.json` and
`D4-retained-outer-tangent-N-native-manifest-20261108.json`.
There are 352 closed continuation allocations / 13.488624 GPU h, plus the open
native job. The matrix remains 17/36 and optimizer updates remain zero.


### Retained outer offset native negative and normal-gap rate diagnosis (November 8, 2026 UTC)

The single `700191d-outer-016f5ef` component is independently closed negative.
Actual source and constructed scoped N load support match before motion. All
7,483 outer updates join exact native/teacher clocks: 7,332 finite-mode offsets
remain unchanged and 151 ordinary updates reproduce the original helper. This
confirms the implemented routing, not physical contact qualification.

Push ends at 32.457127° versus the unchanged 50.146446° requirement. The last
native impact is 11.942854 N and the independently recomputed upper estimate is
13.470716 N, beyond the original 10 N verified support. This is not a small
numerical excursion. Actual physical limits and current joint-position support
pass, while 2,228 original single-future position-support checks refuse. One
original 24-packet / 100 ms native force window is zero immediately before the
impact; the strict lower-force minimum is −0.027046 N with no unavailable
estimates. Hold, release and stabilization never execute and remain unavailable.
No broader load support or threshold change follows from this failure.

Independent reviews verify 7,843 exact A1 profiles, 31,372 native actuator packets,
7,332 causal proposals, 7,303 material requests and the original six-second
fourteen-shape stops. Six causally available original RGB selections are reviewed;
the gripper and door remain visible, but pixels do not establish force or
observed material ownership. Whole direct remote custody verifies 36 original
files plus three sidecars / 5,919,844,403 bytes with source stability and exact
SHA256. Native and reader resources close once, reaching 354 continuation
allocations / 13.811090 GPU h. All original failures and charges remain, with
zero new demonstrations and the matrix still 17/36.

The next scalar audit uses all 12,274 finite-only, 11,396 bound-combined and 7,332
retained-outer normal-feedback inputs. It verifies 30,915 preceding native-force
and endpoint-gap joins. Frozen rate-stage arithmetic exactly reconstructs every
one of those 31,002 inputs. The geometric gap term supplies 73.1%, 78.9% and
78.6% of positive push rate-stage travel, respectively, even though all positive
gap corrections are within the original 100 µm contact tolerance. It supplies
97.7% in the bound-combined failed hold. This identifies a substantial shared
normal-reference input, not the cause of the native outcome or a qualified fix.

The original teacher trace is recorded before the caller adds post-command
compression. An initial reader fails on phase-free metadata; a subsequent empty
direct-compression comparison is explicitly rejected as evidence. Both attempts
remain. The final audit therefore isolates the exact rate stage at a neutral
compression strictly inside the original extent; it does not claim actual
extent clipping, delivered compression or a replayed physical alternative.
The existing separated-gap recovery gate would not trigger on these inputs,
but that same-input comparison is not adopted. Full unilateral release/recontact,
variable-load/posture response and phase transitions still require support.
The original predictor is prepared for scoring on the complete newly available
push prefix, with missing hold explicit and future physical values scoring only.

Evidence is `D4-retained-outer-tangent-contact-closeout-20261108.json` and
`D4-three-history-normal-gap-rates-20261108/report.json`, with complete inputs,
original rejections, source readers and remote original custody preserved.

The prepared original-predictor review now scores all 7,332 normal-mode intervals
of the new completed prefix on actual elab3 CUDA. It uses the known held target,
current full-torso inertia/state and last completed residual; future state and
force enter scoring only. The worst support-surface error is 84.779 µm, with
532 push intervals having four zero-force native packets. The final interval's
surface error reaches only 1.119 µm while its four native forces are
0/0/0/11.942854 N. A small position-forecast discrepancy therefore does not
establish the cause of the pressure impact. Missing hold stays explicit. No
model, gap gate, controller, gain, support domain or threshold is adopted.
This reader closes with actual GPU/PGID absence and canonical lease release:
355 continuation allocations / 13.815895 GPU h. Its complete scores remain on
elab3, with the local report and source under
`D4-retained-outer-whole-contact-forecast-closeout-20261108.json` and the saved
remote review kit. Collection remains 17/36.


### Recovered compression and separated-gap force-reference routing (November 8, 2026 UTC)

The original scalar reference is now recoverable without inventing missing trace
fields. The frozen contact ramp and its first loaded-contact latch initialize
compression; all subsequent recorded force/gap inputs reproduce its bounded
updates. Independent original USD reads verify units, contact point, frame and
hinge. The reconstructed complete request includes the actual tangent offset,
material-frame transport and rigid-shape support correction. All 12,425 available
finite-only float32 world requests match exactly. All 11,367 bound-combined and
7,303 retained-outer pre-clock world requests match within the existing float64
arithmetic budget, maximum difference 5.56e−17 m; their complete request-to-native
clock associations were already independently verified. The 360 early contact
requests in those two remote recordings remain outside this direct comparison.

No regulated normal reference reaches its original extent. The geometric gap
term therefore contributes the same 73–79% of positive push reference travel
identified before extent clipping; this is no longer a neutral-reference-rate
assumption. Requested poses still do not establish actual physical tracking or
contact. All original force gaps, impacts, empirical-support failures and missing
phases remain negative.

The actual force-reference push/hold caller now uses the existing stateful
separated-gap recovery. Inside the original contact tolerance, force-error
admittance establishes pressure; geometry recovery acts once on a measured
separation and resets after actual native recontact. The ramp/acquisition and
ordinary pose routes retain their original behavior. Force targets, pressure
coefficient, unloading, rate/extent limits, setup, materials and all original
physical/support/contact/full-task/stop criteria remain unchanged. Actual-caller
checks cover missing/negative/nominal/boundary/perturbed gaps, repeated separation,
recontact, unloading and acquisition/support transitions. The first test had
incorrectly assumed push and default hold targets were equal; that test error
is preserved and corrected without changing either target.

Evidence is `D4-original-normal-compression-and-world-request-recovery-20261108/report.json`
and `D4-force-reference-separated-gap-software-closeout-20261108.json`.
This is a common control-routing correction with native qualification still
pending. It adds no source, broad controller domain or demonstration; related
contact variants remain suspended and collection stays 17/36.

The isolated native successor retains the preceding `700191d-outer-016f5ef`
archive except the actual force-reference gap-routing condition from `a182f0b`;
removing it reproduces the preceding module exactly. Its complete support closure
matches the committed implementation. Actual source and constructed scoped N
support are verified before movement. Finished independent review confirms all
1,616 consumed gap-route updates: none requests geometric recovery, including
374 positive inputs below the original 100 µm tolerance. All 1,645 actual finite
outer-offset updates retain their value; 151 ordinary updates match the original
helper. These implementation checks pass while physical qualification fails.

Push ends at 5.331619° controlled opening versus the unchanged 50.146446° goal.
The original causal loaded-contact gate fails at probe time 35.916667 s:
0.008206662 N filtered force versus the unchanged 0.02 N minimum. Its measured
6.332997 µm gap, footprint, material position/orientation and physical state pass.
All 6,464 native 100 ms windows remain strictly positive, with that same minimum,
but one original empirical lower window reaches −0.007533418 N. Forty push
control intervals have four zero-force packets; the longest run is five control
intervals, 83.333 ms. There are no unavailable lower-window inputs. These distinct
force and geometry criteria remain intact; positive past force cannot establish
current stable pressure or repair loaded-contact loss.

All 10,544 task-native packets and 2,636 control intervals have complete feedback.
Current and original single-future support/physical checks pass for all seven
joints, with no model failure or support-domain exit. Peak actual force is
4.049628 N and recomputed upper force is 4.230325 N. The original six-second,
fourteen-shape stops pass. Hold is absent. The 360 release and 120 stabilization
commands are preserved after the failure; they do not complete the missing hold
or restore full-task continuity. All 47 original payload/sidecar files,
2,831,045,008 bytes, remain on Blackwell and match the elab3 replica by SHA256.
All nine available causal same-generation RGB selections are inspected; the two
hold selections remain unavailable. The images show approach, modest panel motion
and withdrawal, without establishing micrometer gaps, force or observed-source
perception qualification.

Original ramp/acquisition and feedback reconstruct the new scalar reference.
All 1,616 available push world requests match their original pre-clock inputs
within the existing float64 arithmetic budget, maximum difference 2.78e−17 m;
the separate complete reader already joins those requests to actual Temporal
commands. The 180 contact requests remain outside this direct comparison. No
regulated reference reaches its original extent and the gated gap contribution
is zero. Independently replaying every original loaded-gate update exactly
reproduces the pressure refusal. Requested/reference parity remains separate
from achieved contact. The earlier three histories and their negatives remain.

The unchanged fixed-current full-system four-step predictor also scores all
1,645 newly available normal-mode intervals on CUDA: 29 contact and 1,616 push.
Its worst surface discrepancy is 15.864 µm; the final unloaded interval is only
1.053 µm despite four zero-force packets. Hold and a release forecast are
unavailable in this receipt set. These scores are not a contact-force predictor,
physical response bound or explanation of unilateral pressure stability.

The first independent review stops on missing immutable planning context. Its
reader also carried the old `gap_catchup is None` expectation. The repaired reader
checks the actual routing with the unchanged original recovery helper; all force,
physical, support and task acceptance criteria remain. Original planning constants
are verified against the original baseline plan. Completed profile/stop readers
are reused. Both the failed allocation and a CPU preflight missing-plan error
are retained. A separate arithmetic reader's variable-shadowing error and partial
output are also retained before its corrected namespace completes; no experiment
or control change results from those reader repairs.

All four allocations close, including the failed reader: 497.197955 GPU seconds.
The continuation ledger now has 359 closed allocations / 13.954006 GPU h, with
all owned PGIDs/GPU contexts absent and canonical leases released. Evidence is
`D4-separated-gap-force-reference-contact-closeout-20261108.json`,
`D4-separated-gap-normal-reference-reconstruction-v2-20261108/report.json` and
`D4-separated-gap-whole-contact-forecast-elab3-20261108-health.json`.
The matrix stays 17/36. Next work addresses shared causal unilateral pressure
stability across the four whole histories and phase transitions. Related native
variants remain suspended; no unchanged retry, gain/filter fit solely to pass,
empirical-support widening, criterion relaxation or teacher credit follows.


### Force-only recovery inside the qualification distance tolerance (November 8, 2026 UTC)

The four retained histories expose a control-mode distinction. Ordinary position
control already closes normal pose error; explicit force-reference mode removes
that action. Reusing the contact-qualification distance tolerance as a deadband
in the latter mode leaves small measured zero-force gaps to pressure-reference
accumulation. The new negative has nine such recorded inputs whose reference-only
travel time at zero force exceeds the original force window. That calculation
ignores queued actuator and panel motion and is not a physical latency bound.
The earlier D2 impulse negative and qualified D2 ordinary-position behavior
remain decisive boundaries against a general reversal of the recovery rule.

The actual helper now permits one geometric recovery for a positive available
endpoint gap after actual zero native force only in explicit force-reference
mode. It retains state until positive native recontact rearms it. No recovery
starts from positive force, a missing gap or a zero/negative gap. Ordinary
position mode keeps the original contact-tolerance boundary. Contact acquisition,
normal targets/gain, unloading, rate/extent, material model, empirical support
and all original force/physical/full-task/stop acceptance criteria stay fixed;
the 100 µm qualification distance and 0.02 N loaded minimum are unchanged.

The independent saved-input audit covers 32,531 original push/hold updates.
It identifies 203/273/325/24 additional one-shot opportunities across finite-only,
bound-combined, retained-outer and separated-gap histories, respectively.
Production helper decisions agree with the independent rule for every input;
ordinary-position decisions agree exactly with the frozen helper. Replacing
only the helper restores baseline `8293497` module bytes. Ninety numerical
probe checks pass, including actual support-closure execution for push/hold,
small positive/boundary/perturbed gaps, repeated separation, missing observations,
recontact, unloading and acquisition/mode transitions. Contact qualification
still refuses unloaded samples that satisfy the distance tolerance.

Evidence is `D4-force-reference-recovery-deadband-audit-20261108/report.json`
and `D4-force-only-small-gap-production-history-review-20261108.json`.
These checks establish the software intervention and preserved criteria, not
an alternative physical trajectory, stable nonlinear contact or a teacher source.
A distinct isolated full native qualification is required before related
conditions proceed. Collection remains 17/36 and prior negatives remain intact.


The one nominal qualification is dispatched under
`D4-force-only-small-gap-N-native-manifest-20261108.json`. Its source revision
`700191d-force-only-gap-20f89ed` changes only the recovery helper from the preceding
`700191d-outer-gap-a182f0b` archive; restoring the prior helper reproduces every
module byte. The loaded helper matches the committed implementation. All setup,
load support, low-level controller bundle and material-clock bytes are identical.
Actual source and constructed support are bound before movement, and the real
GPU process, UUID, PGID and canonical lease are verified. The completed native
outcome is negative: controlled opening is 24.590940° against 50.146446° required,
with pre-release contact loss at 124.650 s. Hold is unavailable. Abort release
and stabilization complete but do not repair the original loss.

Finished-only independent elab3 review verifies all 6,940 actual gap decisions
against a separate literal rule, including 208 real recovery pulses, and exact
retention on 6,969 finite outer-offset updates plus 151 ordinary updates. All
31,840 task native packets have valid feedback and frozen load-model estimates
within the original 10 N range; actual force peaks at 6.603082 N and the
recomputed upper bound at 7.423907 N. Four original 100 ms windows (native indices
29,916–29,919) have zero mean force. The strict empirical lower bound reaches
−0.018070 N, with no unavailable samples. Current positions pass; 2,397 original
single-future boxes exceed only measured shoulder-X lower support. Its required
minimum is −1.148206770 rad versus measured −1.147716284 rad, a 0.000490486 rad
support deficit, not a physical-limit violation or permission to widen support.
All physical states, fourteen-shape six-second stops, actuator clock, causal
proposal and material-request readers pass. All original acceptance criteria and
negative outcomes remain unchanged.

Whole custody preserves 45 original run files and three sidecars totaling
6,158,972,694 bytes on Blackwell and elab3, with identical source-before/after
and destination SHA256 inventories. All nine available causal same-generation
RGB selections are inspected; hold images remain unavailable. Pixels do not
establish micrometer gap, load support or observed-source qualification. Native
and review PGIDs, known GPU contexts and canonical leases are released. The two
actual allocations add 1,156.683692 GPU seconds; all 361 continuation allocations
are closed at 14.275307 GPU hours. No source, teacher, label or matrix credit is
added; collection remains 17/36. Evidence is
`D4-force-only-small-gap-contact-closeout-20261108.json`.

A common saved-input audit covers 39,471 original push/hold controls across the
five actual-contact histories. The changed path has 329 zero-feedback inputs in
208 episodes, with 208 actual recovery pulses and 40 episodes of further positive
gap growth. Its final episode begins with a 35.390 µm pulse and reaches a measured
62.905 µm input gap, 27.515 µm beyond the first value. Original pending FK advance
is retained separately; it is not guaranteed delivered gap-closing travel.
These records identify a sustained-separation/reference question, not a causal
failure certificate or a simulated alternative physical outcome. Evidence is
`D4-five-history-zero-force-episode-audit-20261108/report.json`. Further work must
address unilateral pressure/recontact stability, variable-load posture support,
material geometry and full phase transitions together. Related native variants
remain suspended; no unchanged retry or gain/filter/threshold fit follows.

The separate shoulder-X lower-support fixture now closes positive after
independent review of all four original cases: unloaded, center 10 N, face0 with
+0.2 rad yaw and face1 with −0.2 rad yaw. All seven original 0.5 mrad joint pulses,
frozen residual/feedback/effort/physical/speed/acceleration criteria and four
continuous eight-second stops pass. Their twelve segments retain one latch per
case and all fourteen shape covers. All three loaded motion and loaded-stop
cases cover the exact −1.148206770 rad boundary at the
original coupled posture; the extreme required future box occurs during abort
release. No original physical or acceptance limit changes.

Whole custody preserves all 28 run files and three sidecars, 14,756,201 bytes,
with source-before/after and local SHA256 inventories identical. The first reader
wrapper refuses an unsupported purpose name before allocation, GPU context or
lease acquisition. That failed plan, log and zero-GPU result remain preserved.
A distinct corrected attempt uses the existing numerical-review category and
the identical scientific reader; no wrapper acceptance list or scoring rule is
changed. The reader's actual CUDA identity, original outputs and both owned
native/review cleanup records are retained.

A separate artifact changes only the prior calibration's measured shoulder-X
lower position support, preserving all other position sides and every force,
residual, speed/acceleration, physical and task field. Its scope is the completed
force-only N trajectory and the independently qualified coupled posture, with no
arbitrary joint-product box, other condition or unseen phase guarantee. Original
packet rescoring verifies all 63,680 current and exactly one original future
position checks, with zero new refusals. The original 2,397 support refusals
remain in their original review; the contact failure, zero force windows and
missing hold remain negative. Original actor inputs and maintained recipes are
untouched. Evidence is
`D4-force-only-small-gap-N-qualified-scoped-support-20261108/qualification.json`
and `D4-force-only-small-gap-required-support-closeout-20261108.json`.

The two real allocations add 309.350929 GPU seconds. All 363 continuation
allocations are closed at 14.361238 GPU hours, with no owned job open. Collection
remains 17/36. Shared unilateral pressure/reference/recontact stability and full
observed motion remain open; no contact retry, gain/filter fit or criterion
relaxation follows from the support qualification.


The five-history virtual-reference factorization closes with 39,616 original
normal-mode updates (12,274 / 11,396 / 7,332 / 1,645 / 6,969). Source helper AST
parity, native-before/command/trace-after clocks, previous delivered targets and
actual moving stiffness are verified. With full compliance C = J K^-1 Jᵀ,
virtual wrench W = C^-1 J(target − q), and unit pose normal n, the existing scalar
is nᵀCW / nᵀCn. It contains nᵀW plus compliance coupling from the other five
axes. W is virtual elastic state, not measured contact force. Direct scalar output
is unavailable; reconstructed source math is labeled accordingly.

All five analytical diagonal/coupled/rotated/zero-reference controls pass, with
fixed-geometry transport exactly zero. Across all originals, displacement
reconstruction defect is at most 5.56e-17 m or rad, scalar decomposition defect
2.23e-15 N, and source-float32 versus independent-float64 transport difference
0.287 nm. Larger older hold components are retained, including 29.701 µm in the
bound-combined history; they do not by themselves establish a causal defect.
In the latest path the other-five contribution is at most 14.572 nm per update.
The final five zero-feedback controls have 13.662 nm absolute transport-component
sum and 27.515 µm additional measured gap growth. Component sums are not bounds
on a changed nonlinear physical outcome. No transport, controller, gain,
threshold, model, original result or source membership changes.

An initial extraction selected a reports-only local directory and failed before
any GPU allocation; that failure is preserved. Corrected extraction reads the
original complete remote custody. The single CUDA allocation, its full outputs,
known PGID/GPU context absence and canonical lease release are verified. Its
7.051108 GPU seconds are charged once: all 364 continuation allocations close
at 14.363197 GPU hours, collection still 17/36. Evidence is
`D4-elastic-reference-factorization-closeout-20261108.json`; the shared sustained
separation/reference/recontact question remains open without an unchanged retry.


The causal endpoint-gap prototype uses the original full-system four-step
current-inertia model and last completed affine residual. A current native
interval-mean force of zero and a positive last endpoint gap enable it. The
measured material frame keeps its original one-control-interval velocity forecast.
The desired remaining gap is g − min(g, original normal rate × dt); only its
positive deficit against predicted existing surface travel requests another motor
target correction. This is a stateless geometry proposal, not another accumulated
gap reference or a force/contact/stopping bound. Positive-force unloading, original
pressure scalar/gains/rate/extent, ordinary/acquisition/release phases and all seven
motor rate/domain limits remain unchanged.

The whole completed N record supplies 6,969 normal-mode intervals. Of 329 eligible
zero-force positive-gap inputs, 129 proposals change and none reaches motor
rate/domain saturation; the other 6,840 targets remain identical. Ten nominal,
rate-boundary, already-covered, perturbed, missing, positive-force and nonpositive
gap controls pass. The actual shared helper and material planes reproduce the
independent proposal's canonical float32 targets exactly on all intervals. The
changed recorder callback and native propose method then execute on nine selected
original cases, including acquisition, loaded/covered separation and the five
terminal zero-feedback controls. Scalar source targets, callback provenance,
payment, repeated preview purity, phase/no-op branches and stale-feedback refusal
pass. The constructor, dispatch physics and changed native contact remain untested.
The unchanged original predictor's worst surface error is 25.800 µm; all original
zero-force windows and missing hold remain negative. Candidate nominal endpoint
residuals do not establish nonlinear contact or force performance.

The first reader fails on float32-to-float64 indexed assignment and retains its
original source, error and charge. A distinct corrected reader succeeds. A later
actual-caller reader saves passing scientific controls and proposal arrays, then
fails on a reporting variable shadowing the phase vector. That nonzero return
remains intact. Exact CPU comparison of all nineteen saved array fields against
the successful shared-helper run, plus a control on the repaired reporting loop,
recovers the already completed evidence without a GPU rerun; the missing original
forecast report is referenced to its prior successful run, not relabeled.
All four allocations and owned cleanup records remain preserved, adding 69.101278
GPU seconds. All 368 continuation allocations close at 14.382391 GPU hours.
Evidence is `D4-endpoint-gap-response-prototype-closeout-20261108.json`.

The candidate native bundle is prepared locally with launch readiness false. It
retains the frozen outer pressure module, original shape operator, full-system
model, material clock and setup. Its explicit load artifact adds only the previously
independently measured shoulder-X lower side. Complete-output proposal/context/
paid-goal readers and actual-consumer preflight remain required before one full
nominal native qualifier. No maintained recipe, source membership, teacher,
optimizer update or matrix cell changes; collection stays 17/36.


Complete native-output reader readiness now passes independent original-trace
reconstruction for all 6,940 push/hold contexts, including 329 zero-force positive
endpoint gaps. A separate CUDA run validates the new original-shape-plus-gap
reader assembly on all 6,969 normal-mode intervals: original shape targets are
bitwise exact, inactive targets and all seven motor rate/domain limits remain,
and rebuilt contexts equal separately extracted original pressure operands. It
retains 129 changed numerical proposals, without inferring a physical trajectory.
The finished-only driver preserves the original custody, profile, effort, material
request, outer/scalar-gap, pressure, support and continuous-stop readers. The
official-runtime preflight passes the actual source/model/path guards and the
explicit independently measured additional shoulder lower side, with force caps
and outside-support refusals unchanged.

The first native qualifier fails before contact on the first common inspection
transition: `KeyError('native_mean_normal_force_n')`. The new cache consumer had
assumed every transition carried contact feedback, while inspection intentionally
supplies only its validity result. The original recording retains two states, one
inspection command and 0.016667 s; all task phases and endpoint controller use
are unavailable. Original two-plus-four-second continuing stop reports pass,
without constituting a new independent fourteen-shape or contact certificate.
All 25 original run files and three sidecars (4,941,250 bytes) are SHA-verified
before/after transfer and on elab3. Nonzero return, traceback, unavailable phases,
known process/GPU cleanup, released lease and 45.323851 GPU seconds remain intact.
Evidence is `D4-endpoint-gap-inspection-startup-failure-20261108.json`.

A separate v2 bundle clears the unavailable endpoint cache and returns after the
unchanged successful inspection transition. Every physical phase retains required
published force/endpoint input, and an original transition failure cannot advance
the cache. Nine essential CPU controls execute the real transition method across
inspection, six physical phases, missing noninspection force and original-super
failure. The recorder's start, installation, context and close methods and all
numerical target math remain unchanged; no saved-model GPU screen is repeated.
Preparation harness errors are retained separately without GPU allocations. The
new official-runtime preflight passes. One corrected full nominal qualifier is
running on owned Blackwell; actual load-model/path and outer-source bindings
are verified at startup, with complete contact/force/support/task/stops/RGB pending.
The elab3 finished-only review plan and deployment metadata are staged, gated on
complete WHOLE custody and native owned cleanup. Related variants remain suspended.

The new reader's 27.514034 GPU seconds and failed bootstrap are charged once;
370 continuation allocations close at 14.402624 GPU hours plus one open native
job. All earlier outcomes and criteria remain; no teacher, source, matrix or
optimizer credit is added. Collection is still 17/36.


The corrected complete nominal endpoint-gap native qualifier closes negative at
32.432183° of 50.146446° required, in push at probe time 152.966667 s. A single
native sample exceeds the existing verified 10 N upper load domain: 10.448960 N,
with independent frozen-estimator recomputation matching to rounding. Estimator
exception count is zero; that is distinct from the failed force-domain predicate.
The measured normal component peaks at 8.371987 N and cannot stand for total
normal-plus-friction load. All 34,556 original 100 ms native force means remain
positive, minimum 0.038542 N; original calibrated lower windows remain positive,
minimum 0.022403 N. The original current position support has zero exits, while
52 single-future empirical position boxes leave measured support. Original hard
physical boxes and all six-second fourteen-shape continuing stop predicates pass.
Hold, release and stabilization are unavailable after the push exception. None
of these component positives establishes a complete demonstration.

All finished custody/profile/effort/material-request/outer/scalar-gap/proposal
readers execute. Actual contexts reconstruct from preceding published feedback
and current material clock for all 8,639 push controls. All 8,668 held normal
targets and paid goals are bitwise exact, with zero disagreements; 609 contexts
are eligible and 215 change targets. All 39 original run files and three sidecars
(6,721,834,622 bytes) match source-before/after and the elab3 replica. Six causal
same-generation RGB views are manually inspected across inspection, approach,
contact and available push; they are readable and retain door/tool visibility,
without pixel-scale force/contact or observed-source claims. Missing later views
are preserved. Native and original-reader allocations add 1,365.064685 GPU seconds;
their known owned process groups/GPU contexts and canonical leases are released.

A separate CUDA score uses the existing current-full20/last-completed-residual
predictor with the actual known dispatched target, scoring all four subsequent
native states and forces only after prediction. This is the first such score on
the changed complete path. Worst surface prediction error is 82.475 µm, not a
physical bound. At the final interval, predicted anchored gaps are 1.470, −0.907,
−3.350 and −5.856 µm; actual normal forces are 0.347, 8.372, 0 and 0 N. The new
endpoint response adds no correction in this current interval because the model
already predicts closure. The largest eight actual force intervals likewise add
no current extra correction, and six follow a changed one; effects of prior
targets and reference memory remain possible, without a causal conclusion.
Across all 8,639 available gap contexts, 945 model trajectories have a negative
interior gap without a negative endpoint, and 1,619 intervals record force while
all four model gaps are nonnegative. These are diagnostic geometry classes, not
new contact, safety or acceptance rules. Endpoint closure alone is not a contact
force model; neither disabling a current correction at the peak nor clipping only
an endpoint is justified as a complete mechanism repair.

The diagnostic adds 19.292041 GPU seconds and closes its owned resources. All
373 continuation allocations close at 14.787168 GPU hours, with no active job,
new teacher/source membership, matrix cell or optimizer update. Evidence is
`D4-endpoint-gap-response-N-native-closeout-20261108.json`. Further work must address
shared unilateral recontact over the held interval, elastic reference/payment,
coupled material/pose motion, normal-plus-friction load and feedback delay before
another physical successor; retain the 52 empirical support exits separately.
No thresholds, gains, filters, recipes or calibration are changed to pass this
negative. Related variants remain suspended; collection stays 17/36.


The subsequent current-state spring/reference audit reconstructs all 8,668 source
targets and paid goals exactly. Base, original shape and endpoint contributions
use the same actual measured state and drive stiffness. The endpoint addition
changes virtual spring storage by up to 3.015830 mJ; this is not measured contact
energy, work or force. At the final force peak, prior paid normal memory is
149.207034 µm, raw reference increment 22.519072 µm and effective increment
−126.687962 µm, while the current extra correction is already zero. Algebraic
spring/task/null and reference identities pass. Kinematic payment prevents
accumulation and does not measure actual travel; removing it is not a justified
repair. Evidence is `D4-endpoint-reference-energy-closeout-20261108.json`.

The isolated whole-interval successor treats recontact bidirectionally across all
four native predictions. It preserves the actual commanded other-five reference
axes, uses original seven motor domain/rate bounds and four nominal nonpenetration
inequalities, first chooses the closest attainable original-rate endpoint and then
minimizes stiffness-weighted command increment energy in the two-dimensional
nullspace. Empty or degenerate nominal feasibility falls back to the original
command; it introduces no physical or empirical admission refusal. These are
causal model guidance constraints, not a contact-force law or safety certificate.
The initial material-axis-frame result remains intact; the successor corrects the
frame to the actual command quaternion and independently checks nullspace and
quantization contributions. Nine analytical nominal/boundary/coupled/infeasible
controls pass. All 609 eligible proposals are feasible, changed and scalar/batch
bitwise identical, with 388 backoffs; loaded, missing and nonpositive-gap branches
and original A1 replay/motor bounds pass. Nonlinear nominal gap minimum is
−2.125330 nm, and maximum linearization defect 0.182655 µm, without interpreting
either as a physical margin. The maximum virtual target spring storage decreases
from 4.938951 to 4.376180 mJ on these saved inputs, but mean increment energy rises;
no blanket energy reduction or changed-trajectory claim follows.

The actual staged recorder context and adapter proposal/preview/dispatch methods
pass twelve original-input fixtures, including acquisition, advance, backoff,
loaded and final controls. Four material clocks, targets and paid-goal memory are
exact; loaded/missing/nonpositive/stale and all excluded-phase controls pass.
Dispatch uses an explicit underlying shim; constructor, native physics, contact
forces and complete task behavior remain unqualified. No maintained runtime,
gain, pressure law, calibration, physical/empirical criterion or observed source
is changed. Four allocations add 153.578649 GPU seconds and close their known
process/GPU contexts and canonical leases. All 377 continuation allocations close
at 14.829828 GPU hours. Complete-output reader assembly and official-runtime
preflight are required before a new native qualifier. Evidence is
`D4-whole-interval-recontact-model-closeout-20261108.json`; collection stays 17/36.


The new whole-interval complete-output reader assembly independently matches all
8,668 raw-input shape/candidate targets and added reference payments to the
separately verified proposal, with 8,639 preceding-feedback/current-clock push
contexts and 609 eligible inputs. Original motor bounds and inactive behavior
pass. The original archived physical outcome is not rewritten. Its 42.007259 GPU
seconds close once with known process/GPU absence and canonical release, bringing
378 completed continuation allocations to 14.841497 GPU hours. Exact source hashes
bind the actually exercised callback/adapter/helper to the new native bundle and
its future finished-review copy. Official-runtime preflight passes actual
constructed-model binding, original measured-support/outside controls, pressure,
source, phase and existing physical/stopping criteria. The inspection transition
body is byte-identical to the previously exercised nine-case lifecycle; changed
callback/proposal methods retain their separate twelve-fixture scope.

One new full nominal N native qualifier is dispatched on Blackwell using the
shared bidirectional four-step proposal and the same original setup, calibrated
load, pressure and full-cycle/stopping criteria. The actual initial GPU/process
group and canonical ownership are verified, along with constructed qualified
load and original outer-source bindings before physical motion. Whole-interval
controller installation and complete native outcome remain pending at this first
snapshot; neither the software proof nor startup qualifies contact or a teacher.
The original complete force/current-plus-singlefuture/physical/task/hold/release/
stabilization/continuous-six-second-fourteen-shape and RGB readers are prepared,
with the new target/context/paid-goal reconstruction replacing only the prior
endpoint proposal reader. They require finished original evidence, whole payload
and three-sidecar custody, and actual owned cleanup before dispatch. All related
contact variants remain suspended. Evidence is
`D4-whole-interval-native-readiness-closeout-20261108.json`; collection remains
17/36, with one open native allocation and no optimizer update.


The full nominal whole-interval native successor ends negative at 45.067589° of
50.146446° required. Contact continuity fails at probe time 210.316667 s during
push; hold is absent. The original abort withdrawal and two-second stabilization
complete. They do not establish full opening, a held task or repaired continuity.
The original report records no exception or physical violation. Its 52,400 native
samples and all original 45 run files plus three sidecars are preserved; source
before/after and the elab3 replica match for all 9,190,437,020 bytes. Blackwell
originals remain retained, with no local bulk copy. Actual owned native processes,
GPU context and canonical lease are released; 1,715.110170 GPU seconds are booked
once.

The first finished reader stops at an obsolete immediate-predecessor assertion:
it expects the finite-tangent baseline, whereas this successor correctly records
the endpoint-gap v2 predecessor. The base controller itself matches. The failure,
32.580063 GPU seconds and completed original profile/stop outputs remain retained.
A distinct corrected pressure reader validates the exact current lineage and
unchanged planning ancestor; its valid case passes and eight wrong-lineage cases
reject. All pressure, calibrated support, physical and task/stop predicates remain
unchanged. Completed custody and continuous-stop reports are reused byte-for-byte.
The corrected remaining review is running on owned elab3. Full target/context,
paid-goal/request and manual causal RGB closeout are pending; no complete
independent qualification is inferred from the partial review.

The pressure readback already establishes additional negatives. All 48,320
original native 100 ms push means are positive, minimum 0.022398 N, while their
original calibrated lower windows reach −0.048805 N. The independent frozen load
upper peak is 7.754009 N and remains within the original 10 N domain; its maximum
recomputation defect is 7.994e−15 N. Estimator and fixture model failures are zero.
The actual normal component peaks at 7.234842 N during withdrawal. Current
measured position support has zero exits, but 1,015 original single-future boxes
leave empirical support, requiring shoulder joint 1 down to −1.149035743 rad.
That empirical deficit is distinct from physical safety: hard physical boxes,
control states and original six-second fourteen-shape stops pass independently.
One measured release handoff is retained, without treating abort completion as
a successful demonstration. No support range, gain, filter or tolerance is changed.

The 380 closed continuation allocations total 15.326967 GPU hours, plus one open
corrected reviewer. An unchanged full-four-step predictor score is prepared for
the complete new path after the running review closes and fresh ownership is
verified. It will compare original geometry and force outcomes without claiming
a force bound or changed physical trajectory. Related physical trials remain
suspended pending a shared contact/reference/pose mechanism correction and
qualification. Collection remains 17/36, with no source membership, teacher or
optimizer update added. Operational evidence is
`D4-whole-interval-finished-reader-lineage-controls-20261108.json`, the completed
native WHOLE/cleanup receipts and `continuation-cost-and-state-20261106.json`.


The original whole-interval N independent closeout now completes negative. All
12,109 delivered targets and paid goals reconstruct bitwise; all 12,080 push
contexts and material-clock requests match. Nine causal original RGB selections
are manually inspected, covering eight distinct capture indices: push-last and
release-first share one available frame. All are readable and retain door/tool
visibility; this is no pixel-scale contact/force or observed-source certificate.
Hold images and the physical hold remain unavailable. Actual native force means
are positive across all 48,320 original 100 ms windows, but 29 original calibrated
lower windows are nonpositive, with minimum −0.048805 N. The 1,015 original
single-future empirical support exits all occur during push, starting at absolute
time 151.654167 s; current measured support and hard physical boxes remain clear.
All original six-second fourteen-shape stops pass. None repairs incomplete
opening, lost continuity or missing hold.

An unchanged causal four-step predictor scores the complete 12,109-interval path
using only current state/inertia, the dispatched target and the last completed
residual. Future recorded states and forces are scoring targets only. Worst
surface defect is 93.921229 µm. Across 12,080 push contexts, 1,868 nominal
trajectories cross internally without a negative endpoint; 2,974 intervals have
positive recorded force despite all four nonnegative nominal gaps. These are
diagnostic geometry classes, not contact or safety criteria. At the final
interval the predicted gaps are 84.695, 112.933, 139.058 and 161.489 µm; recorded
rigid-surface scores are 84.270, 112.366, 138.503 and 160.868 µm, and all four
actual native forces are zero. The predictor therefore already represents the
terminal separation; its largest final defect is only 0.621059 µm. This does not
prove a force law or replacement trajectory.

A separate authority audit reconstructs every eligible original scalar target
exactly and retains all seven original motor domain/rate limits plus all four
nominal gap inequalities. Removing only the exact-other-five commanded-pose
equality makes the requested endpoint nominally attainable in all 157 eligible
inputs, including the four previously infeasible polygons. At the final interval
that equality permits only 26.944925 µm closing response of 187.516031 µm requested;
the seven-variable model can attain the endpoint. Its quadratic minimizer,
however, reports 21 unresolved status flags despite small primal defects. The
linear command-reference changes reach 11.473522 mm off axis and 0.027425476 rad;
they are not achieved physical pose deviations or qualified task tolerances.
Five nominal, boundary, limited, perturbed and infeasible solver controls pass in
both official runtimes. A local preparation syntax failure is retained separately,
without GPU allocation or native attempt.

The current native adapter pays only the normal component into its reference
history, relying on exact preservation of the other five command components.
An unrestricted seven-motor correction would invalidate that assumption. The
model result is therefore not adopted: a reliable joint pose/recovery allocation,
consistent reference payment and complete mode/history validation are needed
before a native successor. Preserve original pressure, empirical/physical limits,
full-cycle and stopping criteria; do not widen support or fit gains/thresholds to
these negatives. The native run, failed first reader, corrected reader and two
diagnostics retain all costs and release their known owned resources. All 383
continuation allocations close at 15.367263 GPU hours, with no active or prepared
GPU job, new teacher, source membership or optimizer update. Evidence is
`D4-whole-interval-recontact-N-native-closeout-20261108.json`; collection stays
17/36. D3/D4 complete observed paths and real collection geometry cost remain open.


The isolated pose-aware successor now has a complete software/model screen, not
a native qualification. It keeps the original seven motor bounds and four linear
nominal surface inequalities. A first quadratic stage minimizes the other five
command axes' elastic energy using the native coupled compliance; a second
minimizes motor spring increment energy while retaining that task displacement
and endpoint. No arbitrary translation/rotation weighting is added. Existing
OSQP supplies independently checked primal and stationarity residuals; an
unqualified numerical result falls back to the original proposal. The CPU-only
LP/QP library algebra follows the unchanged CUDA state/inertia/response model;
no physics or GPU model is moved to CPU.

All 157 eligible whole-interval inputs and 609 prior endpoint inputs solve, with
zero candidate or old-objective OSQP failures. The old 21 SLSQP status flags
remain in their original diagnostic; they are no longer used as evidence of a
minimum. Quantized targets retain the original domain/rate bounds and exact A1
replay. The whole-interval model's largest other-five increments are 2.185439 mm
and 0.006970938 rad; the prior endpoint path preserves those axes to numerical
precision. Linear gap constraints are nominal guidance: nonlinear prediction
still reaches −24.724895 µm, and no force or physical error bound is inferred.
These frozen-input scores do not execute an alternative trajectory.

The prototype pays only the original normal shape increment, plus all six axes
of the additional recovery command. Translation and a left world-frame XYZW
rotation commit after dispatch, while preview remains pure. Seven nominal,
boundary, perturbed, duplicate-constraint and infeasible solver controls pass.
Zero-extra arithmetic, independent Hamilton-product and pure-repeat checks pass.
The first constructor/callback family retains a failed cross-reader bitwise
quaternion check and leaves its second source unexecuted. A single-fixture
diagnosis shows bitwise-identical target, Jacobian, translation and extra payment,
with an 8.88e−16 quaternion difference. The installed reduced quaternion product
subtracts order-one intermediates when producing tiny vector components. A
separate operation-count bound and independent Hamilton product pass 384 controls,
each with a rejecting out-of-bound perturbation. This changes only numerical
comparison, not a pose tolerance or the controller. The corrected actual
constructor/recorder/propose/preview/pre/apply review now passes all 34 fixtures,
17 from each archived source. Real constructor wiring uses the actual URDF and
gains. Pure preview, exact target, dispatched six-axis payment, mode exit, all
four held apply callbacks, material clocks and loaded/missing/nonpositive/stale
feedback controls pass. An explicit original-dispatch shim and original state
fixtures do not qualify changed physical history.

The same production finished reader independently assembles all 20,777 held
inputs from both complete archived prefixes. All 766 eligible targets match the
independent scalar model; every inactive target, zero extra payment and paid
reference remain bitwise identical to the original. Full reference translation
and an independent quaternion numerical contract pass. The reader retains
original causal native state/inertia, last-completed residual and material clock;
no future physical pose or force enters a proposal. Ten wrong pressure-reader
lineages reject without changing any scientific pressure predicate. The real
recorder's only difference from the tested source is descriptive mode metadata;
its exact inverse substitution restores the complete tested source. Original
inspection/feedback method bodies remain identical.

Blackwell's existing solver runtime and fresh official preflight pass without
installing dependencies. Exactly one full nominal pose-aware native qualifier
then executes with the original physical, force, empirical, full-cycle and stop
criteria. It ends negative during push at absolute 262.270833 s: shoulder Z
acceleration −45.513767 rad/s² is outside ±40.259247, and gripper Y −39.205082 is
outside ±38.197281. The model upper load is unavailable there; the finite upper
peak elsewhere is 6.137695 N. The refusal is an empirical-support failure, not
independent evidence of unsafe physical force. The exception report has no final
opening outcome; its last diagnostic trace reaches 47.492645° versus 50.146446°
required. Hold, release and stabilization remain unavailable.

The complete original readers validate raw integrity, all native feedback,
13,162 exact actual targets/reference positions, 13,133 causal push contexts and
330 scalar recovery proposals, including independent quaternion payment checks.
Actual 100 ms force means are positive across all 52,532 required native windows,
minimum 0.019679 N. Preserve eleven finite nonpositive calibrated lower windows,
minimum −0.039807 N, plus four unavailable windows. Current position support has
zero exits; the original single-future audit retains 2,568 empirical exits.
Hard physical boxes pass. The direct stop report remains positive, while the
independent original six-second, fourteen-shape audit fails the empirical
shoulder Z acceleration face at 45.265678 rad/s²; all its other original bounds
pass. Six causal original RGB selections are manually viewed; no missing phase,
microscopic force, material or observed-source claim is inferred from pixels.

Whole custody passes all 42 original payload/sidecar files totaling 9,300,585,333
bytes, with source-before/source-after/elab3 replica hashes identical. Blackwell
originals and the elab3 replica remain retained, with no bulk local copy or
removal. The native allocation charges 1,815.732883 GPU seconds, the complete
independent review 135.023984, and the whole-motion forecast 25.540352. All three
owned process groups, GPU contexts and canonical leases close; all failures and
masks remain. Continuation accounting is 391 closed allocations / 15.988224 GPU
hours before the next isolated model screen.

The unchanged full20 causal CUDA predictor scores all 13,162 normal intervals
using current state/inertia, the last completed residual and the already known
held target. Future physical state and force are scoring targets only. Preserve
333 all-four-zero-force push intervals and 140.815407 µm worst surface error.
At the sole acceleration guard interval, the predictor independently anticipates
−45.517582 rad/s² shoulder Z and −39.205383 gripper Y before dispatch. Thus the
position/rate and four nominal gap constraints omit an already predictable
original-domain acceleration failure; neither an empirical extension nor a
newly fitted controller gain is justified.

An isolated successor keeps the same lexicographic pose/motor objective and
four nominal gap inequalities, adding the affine acceleration response of every
arm joint at all four native steps within the unchanged calibrated domain.
The complete three-history CUDA screen evaluates 330 new pose-aware inputs,
157 whole-interval inputs and 609 earlier endpoint inputs: 1,096 eligible inputs
in total. No quantized target leaves the original predicted acceleration domain.
Four nominal, boundary, perturbed and infeasible algebra controls and all four
loaded/missing/nonpositive branches pass for every history. Nineteen inputs
have a feasible original LP and a passing pose stage, but the second energy QP
fails its accepted status or numerical contract. Their original-proposal
fallbacks remain, including six on the new native history. They are numerical
failures, not physical infeasibility or qualified changed motion.

After fixing six task axes, the seven-motor energy problem has one remaining
coordinate. The next isolated solver intersects its scalar inequality interval
and minimizes that quadratic directly, independently checking the original
primal, equality and stationarity accuracy. Five further scalar energy controls
pass. Its full 1,096-input screen retains five crossed intervals from first-stage
primal imprecision: two on the new native history and three on the whole-interval
history, with zero on the older endpoint path. Predicted quantized acceleration
exits remain zero. Preserve both complete screens and every original fallback.
The allocations charge 115.822105 and 113.657429 GPU seconds and release their
known owned GPU contexts, process groups and canonical leases, closing accounting
at 393 continuation allocations / 16.051969 GPU hours.

The corrected numerical-anchor solver retains a candidate only under the SAME
original independent primal, equality and stationarity contract. No tolerance,
empirical domain, physical limit, force threshold or gain is enlarged. All 1,096
eligible inputs now solve, including all nineteen original energy-stage fallback
inputs and all five crossed scalar intervals, with no quantized predicted
acceleration-domain exit. Eleven numerical-library controls pass on Blackwell,
including rejection of a material anchor defect. The maximum energy-stage primal
residual is 2.150587e−10 within its stated numerical contract. Preserve all prior
numerical failures and source versions. Nominal nonlinear gap remains as low as
−24.724895 µm on the old whole-interval path, not an adopted physical tolerance.

The actual constructor/recorder/context/propose/preview/pre/apply family passes
19 new-history, 17 whole-interval and 29 older-endpoint fixtures: 65 total,
including all nineteen earlier numerical fallback inputs. Exact URDF/gains,
current/last-completed state, pure preview, exact target and dispatched full6
reference payment, original inactive/loaded/missing/nonpositive/stale controls,
phase exits and four identical held apply callbacks pass. The original-dispatch
shim does not execute native physics. The same production reconstruction function
then passes every one of 13,162 + 12,109 + 8,668 = 33,939 original held inputs.
Its 1,096 eligible candidates match independent scalar proposals; all inactive
targets and paid memory remain identical to the originals. Quaternion payment
retains independent Hamilton-product and operation-count checks.

While preparing native wiring, source inspection catches an unexecuted recorder
installer reading `env.load_model`, which the native environment does not expose.
The recorder actually constructs `self.fixture_model` under its existing exact
path/parameter guard. Only that domain-source expression is corrected; the
qualified real adapter constructor, controller helpers and recorder lifecycle
methods remain unchanged. CPU execution of the real expression with the actual
guard-bound original model passes and rejects the old environment expression
and an altered calibration. An initial library-control import-path error is
preserved as zero GPU work and zero native attempts; the exact frozen source
path fixes the control's startup. No native error is relabeled or removed.

Eleven wrong pressure-reader lineages reject, while every scientific pressure
predicate remains identical to the completed original reader. The native
recorder's remaining differences are descriptive mode/calibration readback;
inverse substitution restores the corrected source. Setup, qualified load,
shape/clock, original pressure recorder, materials and frozen low-level source
bytes remain identical. Local, Blackwell and future elab3 reviewer hashes match.
Fresh official-runtime preflight passes original support and outside cases,
constructed calibration path/parameters and real recorder call, outer/source
binding, force-only caller, inspection lifecycle and existing solver-library
checks. No dependency is installed.

The full nominal acceleration-aware D4 qualifier closes negative at 15.580474°
versus 50.146446° required, with contact loss at probe-relative 84.383333 s.
Hold is absent; abort release, stabilization and the original continuous
six-second/fourteen-shape stops pass. All 18,096 actual 100 ms force means are
positive, minimum 0.016101 N; calibrated lower means are positive, minimum
0.002375 N, with no unavailable window. Current and single-future empirical
position support, physical boxes and original required-10 N support all pass.
The exact original load-filter reconstruction nevertheless falls below the
unchanged 0.02 N loaded-force floor. Positive force and qualified sustained
load are distinct criteria; neither threshold changes.

All 4,553 delivered targets and paid references reconstruct exactly, including
80 eligible recovery proposals; all 4,524 material requests and nine causal RGB
selections are reviewed. Hold remains unavailable. The unchanged full-path
predictor retains 81 all-four-zero-force push intervals and 29.546677 µm worst
surface error. Predicted and actual original-domain acceleration exits are zero;
this nominal model is not a physical response or force bound.

A complete proposal audit finds that 56 of 80 recoveries remove existing planned
approach even though the original nominal endpoint already reaches the requested
plane within the original acceleration domain. At the failure, the solver removes
4.114936 µm of predicted approach to impose zero endpoint gap. This is a command
mechanism, not a verified physical counterfactual. An isolated correction retains
the original target and payment when that additional recovery is unnecessary;
the original solver and all physical, empirical, force, phase and stop criteria
remain unchanged. Complete four-history CUDA qualification now passes all 1,176
eligible model inputs with zero fallback or quantized predicted acceleration-domain
exit. All 825 already sufficient original approaches retain their target and zero
extra payment exactly. The actual constructor, recorder context, pure preview,
dispatch, mode exit and four held apply callbacks pass 83 + 19 + 16 + 30 = 148
fixtures, including every newest eligible input and all earlier numerical fallback
fixtures. The explicit original-dispatch shim executes no native physics.
Complete production-reader assembly passes all 38,492 held inputs, with all 1,176
independent scalar candidates, exact inactive targets/reference memory and
independent quaternion-operation checks. The same reconstruction function is used
in the prepared finished reader.

Twelve wrong pressure-reader lineages reject, while every scientific pressure
predicate remains identical. The exact corrected recorder fixture-model binding,
original lifecycle and all setup/load/material/control criteria remain unchanged.
Fresh official Blackwell preflight and local/native/elab3 source parity pass.
The one changed full nominal qualifier closes negative. Its original report
records 50.478609° versus 50.146446° required and a passing opening requirement,
but hold ends at 2.65 s with no qualified held angle and pre-release contact
loss. Abort release and stabilization are recorded; the original direct stop
reports pass. Full independent force, empirical current/single-future support,
physical and six-second/fourteen-shape stopping reviews are complete. Physical,
position support, required-10 N support and original stops pass; the complete
task and 133 calibrated hold lower windows remain negative. Related physical variants remain suspended; no
observed teacher source is admitted.

For the preceding acceleration-aware negative, whole custody verifies 48 files /
4,685,718,741 bytes on Blackwell and elab3.
Native, independent reader and forecast allocations charge 803.092604,
63.264163 and 14.208161 GPU seconds; all three owned resource groups close.
The completed model, caller and reader allocations additionally charge
124.088554, 133.769495 and 122.289329 GPU seconds, preserving every earlier cost.
Continuation totals are 405 closed allocations / 17.095649 GPU hours, with no
active owned job. Collection remains 17/36, with no new source or window.
Evidence is `D4-acceleration-aware-recontact-N-native-closeout-20261108.json` and
`D4-preserve-planned-approach-native-readiness-closeout-20261108.json`.

The load-filter audit exactly reconstructs all 5,544 recorded trace values and
loaded flags. The final four native forces are all zero, their largest measured
gap is 0.536436 µm within the original 100 µm gap tolerance, and the footprint,
material position/orientation and physical checks pass. The preceding normal
feedback is already zero with positive gap. Across the complete push there are
81 zero-force intervals, zero missing gaps and zero gap-tolerance exits. The
failure therefore cannot be replaced by a gap-tolerance or physical-safety
claim. The original filtered load is 0.016100931 N against 0.02 N required.

The proposal audit retains all eighty inputs: fifty-six have negative additional
closure, up to 7.539756 µm, while the original nominal endpoint already meets the
recovery goal within the unchanged acceleration domain. The isolated correction
preserves that existing pressure approach exactly, rather than demanding another
zero-gap equality. This is neither an inferred force law nor evidence of a changed
physical trajectory. The additional-recovery solver, numerical contracts,
calibration, native gains, material properties and all original qualification
criteria stay unchanged. Eleven algebra controls and four inactive-input controls
are retained and pass on all four histories. Full native and observed-source
qualification remain pending. A CPU audit's initial missing
remote source path is preserved with zero GPU/native allocations; the exact
original Git AST fixes only its source loading.


The new CPU reconstruction exactly reproduces all 15,146 original load-filter
values and loaded flags, including 13,967 push and 159 hold rows. Push has 734
all-four-zero-force intervals but no missing gap or gap-tolerance exit; its
minimum filtered load is 0.031147 N. Hold has eleven all-four-zero intervals.
At the only post-establishment loss, probe-relative 244.416667 s / absolute
278.833333 s, all four forces are zero and their separations rise from 4.410748
to 106.379397 µm. The original 100 ms filtered force is still 0.110512 N,
above the 0.02 N loaded minimum. The original maximum-gap predicate fails;
no force floor, gap tolerance or filter is changed.

The whole transition already loses task posture validity at hold 2.20 s,
while loaded contact and the physical checks still pass. Material error then
crosses 10 mm; at contact loss it reaches 15.206833 mm with orientation error
0.066520 rad. Every hold footprint and physical row remains inside its original
checks. Door speed rises from 0.197629°/s at hold entry to 0.667217°/s at loss.
The final raw row is 51.163864°; it is a diagnostic sample, distinct from the
original report's opening statistic and absent held endpoint.
The original last-0.5-second conditioning trend is -0.002552163 per second;
its unchanged nine-second hold/release reserve projects sigma to 0.019956498,
below twice the original 0.01 damping. Actual hold sigma never crosses 0.02.
This is a controller planning reserve, not a physical safety violation or a
proof of global unreachability. The next correction must address the combined
posture, paid reference, pressure and moving door across the transition.

Whole custody now matches the complete payload and three sidecars: 49 files /
10,395,037,705 bytes across Blackwell and elab3. The initial HDF5 replica differs in one
byte. Two circumscribed write/read checks do not survive subsequent complete
re-verification; their failures, divergent block and complete divergent replica
are preserved. Transferring the exact source HDF5 into a new inode and reopening
it passes the full SHA256 check, followed by complete stable-source/replica
verification. No cause is inferred and the Blackwell original is untouched.
The native allocation charges 1,984.000518 GPU seconds and its known owned
processes and lease close once. The independent original-reader review, eleven causal original RGB selections
and unchanged whole-path CUDA forecast are complete. No native successor or
teacher cell is added. Evidence is
`D4-preserve-planned-approach-N-native-closeout-20261108.json`.


The final independent review reproduces all 14,155 delivered normal-mode targets
and paid references, including 742 eligible recovery inputs and 273 actual
changes. All 56,504 original native 100 ms force windows are positive, minimum
0.024095 N. The calibrated lower windows are positive throughout push
(minimum 0.010793 N), but 133 of 636 hold windows are nonpositive, minimum
-0.438486 N, starting at absolute 278.283333 s. None is unavailable. The frozen
model upper peak is 9.179812 N, distinct from the actual normal peak 8.005346 N.
Current and single-future empirical position support, original required-10 N
support, physical boxes and all original six-second/fourteen-shape stops pass.
These results preserve the complete hold/posture/continuity negative.

The unchanged causal predictor scores all 14,155 normal intervals on actual
owned CUDA, with future states and forces used only for scoring. Push/hold retain
734/11 all-four-zero-force intervals; peak surface forecast errors are
49.037324/38.579715 µm. Predicted and actual acceleration-domain exits are zero.
At the final hold interval the preceding published force is 0.351135 N, so the
zero-force recovery is inactive, while the nominal forecast opens to 143.741521 µm.
Actual rigid-surface scoring reaches 143.884743 µm; the native contact separation
is a distinct measurement. This association does not prove an alternative
control would succeed, and no new gate, force law or physical retry follows.
The two final independent allocations charge 167.376227 and 27.569386 GPU seconds;
all owned resources close. The original HDF5 transfer anomaly remains preserved.

The user then closes this coordinator campaign for independent friction review
and D4/D3 assignment in “Astra2”. The heartbeat is verified paused; all six host
GPU contexts and checked GPU/project locks are released. No new experiment,
variant or scientific analysis starts after that request. The earlier matched
smooth-friction candidate comparison has already completed, followed by the
48-state stopped-component and eight whole robot-only path qualifications;
these must not be described as an unexecuted candidate or a teacher success.
The final handoff is `FINAL-HANDOFF-20261108.md` in the campaign evidence folder.
