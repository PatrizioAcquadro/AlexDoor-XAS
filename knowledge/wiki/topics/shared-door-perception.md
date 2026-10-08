# Shared Door Perception

Phase 6.0 is **unqualified**. 6.0B retains static calibrated RGB-D/multiview
geometry with frozen GroundingDINO, native SAM3 and DINOv3. The former geometric
and custom material trackers, full-state replay orchestration and SAM3 video
comparison are retired. Their results remain in
[[experiments/b1-perception-findings|Perception Findings]]. The optional CAD-free
Point2Pose prototype is implemented under
[[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0]];
its diagnostics do not qualify a release or enable contact. An opt-in
[[../experiments/p2p-sam3-unified-frontend|unified official SAM3/SAM3.1 video path]]
uses one persistent full-leaf text identity for initialization and every subsequent
mask. It skips separate SAM2 segmentation and DINOv3 extraction while retaining
observed RGB-D geometry, TAPIR, SuperPoint, registration and TSDF. The next
experimental frontend is SAM3 with periodic detection reconditioning OFF. A single
bounded SAM3.1 OFF comparison changes only period16→0 and completes both original
openings; it has a smaller SAM memory footprint but weaker nominal physical-target
accuracy. The selected recipe balances correct-pose continuity and complete latency
while retaining SAM3's measured GPU/CPU history growth. Operational defaults,
older baselines and all prior results remain unchanged.

## Recording, inspection and storage

`recording/b1.py` and `recording/b1_runtime.py` retain synchronized metric RGB-D,
valid-depth masks, arm/neck proprioception, intrinsics and optical camera poses.
`b1.rgbd.v1` stores N+1 observations including reset/terminal, N applied commands,
separate annotations, calibration and a factual outcome. Observation selection
excludes asset identity, split, expert angle, phase and simulator labels.

`scripts/collect_perception.py` is maintained for separately authorized collection.
It validates the frozen train/development corpus, rejects test identities and uses
one fresh Isaac CUDA process per episode. Worker failure stops admission;
`--resume` skips only complete validated episodes. Qualification records are unchanged.
No collection or training was started by this cleanup.

All 50 `datasets/b1/perception/engineering-v2` episodes remain intact, including
calibration and metadata, for 6.0B and future common train/development qualification.
They are engineering recordings, not matched policy demonstrations.
`recording.b1.episode_paths` validates membership, conditions, duplicates and split
isolation without loading visual features. Superseded `engineering-v1`,
`inspection-pilot-01` and `inspection-tallest-01` recordings were removed after
preserving their headers/results. Selected models remain in `models/perception/`;
historical records and the cleanup inventory are under
`outputs/b1/perception/evidence/`. Removed ignored payloads are not Git-recoverable.

`configs/perception_inspection.json` retains the common 25 s scan, seven sample
times and simulated 10-degree upward camera mounting adjustment. The arm holds
its parked pose while the neck scans; contact, door motion, tool drift and tracking
checks can stop inspection. Physical neck limits are unchanged. This mounting
proposal is not hardware calibration. Depth is ideal rendered optical-axis depth,
not a real ZED stereo-error model. Runtime and recorded mount transforms stay aligned.

Inspection retains the 10 mm arm drift, 0.01 rad door motion and 0.1 rad neck
tracking limits. Float32 roundoff contributes 32 epsilon in meters/radians;
neck tracking additionally allows one 60 Hz tick at the common 0.4 rad/s limit
(6.67 mrad). Reports retain base limits, measured values and allowances. This
observation guard does not relax dynamic freshness or qualification error budgets.

## Static scan — 6.0B

`GeometryProvider` requires `scan_fusion="object-v1"` from
`configs/perception_geometry.json`. `scan_state` exposes static object evidence;
`update(sensor)` retains the provider interface but returns an invalid
`DoorEstimate` (`static_scan_only` for normal input) and no policy encoding.
The scan supplies no qualified runtime state or admission to move. A missing
required RGB-D/calibration input has an explicit rejection reason.

`ModelWorker` sends only RGB bytes to an isolated process. GroundingDINO uses
`door.` with 0.30/0.25 thresholds; native SAM3 uses positive normalized box prompts
and confidence 0.5. DINOv3 excludes CLS/register tokens and preserves the
letterbox mapping. All weights stay frozen on CUDA. The worker's local dependency
overlay isolates NumPy/SAM3 from the external Isaac installation; image inference
needs no `decord` or video worker.

`CueEngine` keeps capture and measured availability separate, one request in
flight and cancelled-request generations. Semantic requests target 5 Hz through
25 s; pending results may complete afterward without reading a later observation.
Nonmonotonic sensor time resets the episode; gaps clear pending inference.
Reset clears static memory and public estimates. Input selection uses only sensor
keys, never annotations or asset labels.

`Surface` preserves disconnected measured components, dense extrema, supported
edges and packed per-observation membership with calibrated poses, capture and
availability times and observed/clipped/unobserved border status. Extraction
re-samples residual depth after each plane fit. Enclosed semantic omissions recover
only valid measured inliers on a seeded plane; missing/off-plane depth and the outer
silhouette are not filled. Partial-view borders cannot contract earlier measured
material. Contradictory perimeter claims are removed without erasing observations.

`ScanMemory` consolidates registered overlapping support while retaining competing
associations and original references. Relief attachment requires an observed internal
seam; proximity, parallelism, color and shared masks do not establish ownership.
Seam association checks every component pair in the same captured frame/mask.
Surrounding support is fixed only conditionally on every retained object alternative
and observed separating borders. No area winner, handedness, nominal dimensions
or universal bottom-frame rule resolves ambiguity.

Static panel-edge alternatives remain distinct from physical hinge hypotheses.
Observed separated, well-conditioned cylinder arcs may support a predominantly
vertical axis with explicit metric/angular bounds and a calibrated floor intersection.
Leaf-plane support is excluded from the hardware fit. Missing fits do not establish
that hardware is physically absent. Ideal-depth tolerance and fit sensitivity are
assumptions, not qualified real stereo/calibration uncertainty.

`footprint_support` checks both complete finite distal contact covers against
observed rasters, holes and clipping. Repeated collinear extrema fail as
`degenerate_finger_face`. The maintained projected mesh-band covers preserve the
canonical geometry/calibration; they are neither measured pad area nor soft-finger
mechanics. See [[topics/purdue-b1-robot-and-contact|the robot contact contract]].
`ObservedSpace.query` uses caller-supplied covering balls for the relevant continuous
sweep/stop envelope. Calibrated rays distinguish free cover, occupied support and
unknown space; returned clearance includes the certified ray volume's sides.
It assumes resolved pixel support and ideal depth, not arbitrary unseen thin objects.

These queries supply geometry only. Consumers still need the full robot/tool/leaf
sweep, uncertainty, reachability, response, load and stopping bounds. Human review
of the two pilot regions establishes only their displayed local leaf role.
Other observations on a fused plane cannot refresh an occluded selected patch.
Endpoint IK and a small free-space ball do not certify a continuous path or contact.

## Supported diagnostics

From the repository root, select a fresh output directory:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py smoke \
  --output outputs/b1/perception/NEW_SMOKE
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py diagnose-scan \
  --output outputs/b1/perception/NEW_SCAN
```

`smoke` runs frozen inference on one train image. `diagnose-scan` consumes only
0–25 s on `door-2738468b94d74c5f` and `animated-door-1-88abf40`, nominal/light.
Metadata selects recordings; annotations do not enter inference. Reports retain
candidate reprojections, geometry, finite-cover queries, timing and ambiguity.
The static commands perform no dynamic tracking or full-state evaluation.
Point2Pose diagnostics and setup are described below.
Execution success does not imply offline, dynamic or release qualification.

## Panel tracking prototype — 6.0C

`point2pose_runtime.tracking_provider` adds `PanelTracking` to the existing
`GeometryProvider`. Default 6.0B behavior and policy encoding remain unchanged.
Initialization uses automatic 6.0B components from the common inspection's first
completed view (4 s), with an immutable same-acquisition RGB/mask/depth/K/joint
packet. Reciprocal existing registration checks consolidate duplicate observations;
distinct supports remain independent hypotheses. Before native initialization, candidates lacking distributed measured interiors
are rejected individually and recorded. A diagnostic selects the first eligible
automatic candidate explicitly without labels or later truth-based reselection.

The official CAD-free pipeline supplies segmentation, distributed point tracking,
registration, map, graph and TSDF. The pinned source receives two reproducible
corrections at installation/worker startup: f2m recomputes loss from current
inlier support, residual and the existing jump guard; SDF registration returns
the selected refined pose and its statistics. An unsupported/invalid result marks
the object lost while preserving its finite fallback pose for diagnosis. Frame-to-map
registration continues for lost objects, allowing supported recovery under the
same ID. The unchanged final SDF support gate retains the pre-refinement cluster
fallback and recomputes residuals/inliers for the returned pose. No threshold,
model, ground-truth input or operational/contact admission rule changes. The
tracked patch validates pinned source, preserves pre-fix files and declares
`native_fixes` in runtime metadata. A regenerated SAM2 mask must retain five
spatially distributed measured interior references of the source component.
This checks essential support without treating legitimate panel relief as frame
contamination or requiring identical boundary pixels. It does not certify every
extra mask pixel's physical ownership; ambiguous component relationships remain
ambiguous. Positive SAM2 prompts are unchanged from upstream: actual calls use five
raster-sampled positive points, distinct from these interior preservation checks.
Mask expansion is measured, but absolute leaf ownership is not certified.

The renewal criterion now counts final registration inliers, retaining the
fewer-than-ten, mask-area and 15-degree view gates and native lost-state sampling
suppression. The prior unbounded variant and both memory failures remain preserved.
`point2pose_renewal.py` limits each candidate to 120 active references, sized from
the largest complete-baseline population and measured 1.875 MiB/point TAPIR causal
state. It defers whole native sampling batches when capacity is unavailable.
Models, candidate selection, native point selection, promotions, numeric gates,
graph and TSDF remain unchanged.

The isolated `offline_episode(..., allow_partial_reference_batch=True)` experiment
admits the prefix of the fully filtered native batch that fits the remaining
120-reference capacity. Native selection/order and pending-point checks are retained;
zero capacity still defers sampling. The complete pair gains 480 correct nominal
poses but loses 119 light poses, with 169/315 more inaccurate acceptances.
Whole-batch admission remains the default; the partial option is diagnostic.
The independent `offline_episode(..., refit_seed_rollback=True)` diagnostic
preserves a RANSAC hypothesis's own valid seed when its weighted SVD refit loses
five-point support. It rechecks that seed on the same remaining pool at 4 mm,
then returns/removes exactly those recomputed inliers. Valid refits, sampling,
ranking, selection, SDF and later support/jump/renewal checks remain unchanged.
Rollback is disabled by default. It may preserve wrong hypotheses as well;
see [[../experiments/point2pose-refit-hypotheses|isolated evaluation and limits]].
See [[../experiments/point2pose-isolated-improvements|isolated comparisons]]. This switch does not disable the
graph; graph-off experiments must select that independent control explicitly.

A new reference can sponsor one retirement after native promotion and three
metric inlier observations on the actually published post-graph pose/map. This
uses the existing promotion streak. Retirement requires more than the existing
pending TTL of 15 visible, valid, masked, low-uncertainty outlier observations;
loss, missing native visibility or invalid depth freeze the counter and prevent
retirement. At most three points per candidate retire per frame. A planar convex-hull/cell guard
protects geometric distribution. No age-only or invisible-point eviction occurs.
Compaction copies retained TAPIR query/features/causal rows into independent CUDA
storages; stable IDs, CPU historical landmarks, keyframes and graph references
survive and slots never become new material IDs. Allocator reservation may remain
after live storage is freed. Historical graph/keyframe/TSDF growth is not bounded
by the active-reference cap.

Post-graph support is additionally recomputed for retirement and audited separately
from inherited frontend statistics; native acceptance and graph publication are
unchanged. Neither internal confirmation nor metric self-consistency verifies
material identity. Native visibility can disagree with independent projected-depth
visibility; the guard does not establish protection of every physical occlusion.

Diagnostic traces now save the actual `SAM2.add_new_prompt` calls and original
input masks at initialization, separately from the interior preservation checks.
They also observe native promotion attempts, retained observations, geometric
spread, whether its check is enabled/passes and the actual promotion decision. Observation
does not change native decisions or require ground truth.

Pending geometric confirmation remains disabled in the paper configuration. The
bounded graph-on comparison completes both saved CUDA openings, but light
precision regresses, nominal retains a severe accepted graph outlier and final
loss, and physical occlusion
protection remains unresolved. The policy is an unqualified prototype; see
[[../experiments/b1-perception-findings|renewal evidence]].

The complete graph-off ablation (`0e6185f`) retains all other controls and original
initializations. It removes the two original event errors and improves primary
light correct availability, but worsens nominal availability, terminal loss and
conditional target tail. Native promotion can still change pending coordinates;
already-confirmed primary landmarks are not globally revised. Whole-batch renewal
can stall at 92 active references because the next 30 would exceed 120, while
late support falls below five and lost-state renewal stays suppressed. No common
replacement is adopted; graph-on remains the default. See
[[../experiments/point2pose-bounded-global-graph-ablation|sequence results and reference/TSDF history]].

Three isolated graph-off follow-ups complete six original openings. Partial
admission gains correct nominal poses but adds inaccurate acceptances in both
conditions. Six mm and enabling the existing 8 mm pending spread check worsen
both complete conditions, so 4 mm and disabled geometry confirmation stay the
existing defaults. Rounding saved pixels also worsens median/p95 error; original
SAM2 prompts are preserved. Correct hypotheses can be missed by sampling or lost
during clustering despite available 4 mm support. These are prototype diagnosis
results, not material/contact qualification. See
[[../experiments/point2pose-isolated-improvements|complete isolated results and next problem]].

The subsequent own-seed refit rollback completes both original openings but
gains 211 correct nominal 10 mm/5-degree poses while losing 386 light and adding
wrong acceptance in both. It preserves supported seeds mechanically, including
wrong ones; support alone cannot select a reliable pose. Its common adoption
and combination with partial admission are rejected. It remains an explicitly
disabled diagnostic; five inliers, 4 mm and normal jump/SDF gates stay fixed.
Saved guard refusals and unsupported tails motivate separate recovery and
correspondence questions, with no additional correction implemented. See
[[../experiments/point2pose-refit-hypotheses|accuracy, continuity, memory and retained wrong hypotheses]].

With `estimate_init_pose=false`, map M remains the first optical camera frame.
The observed zone frame defines O separately; its fixed transform to the material
zone is Z. Compose `W_Ct(FK) * Ct_M(Point2Pose pose) * M_O * O_Z` exactly once.
Do not substitute `init_pose` for the current `pose`. Distributed references
estimate candidate pose even when the selected contact face is uniform or covered.
Measured transformed root-plane support and informative residual motion establish
only the root candidate's rigid relationship; an included handle/relief is not
attached just by the mask. Tangential motion on a stationary plane is insufficient.
Articulation uses the frozen observed initial pose and uncertainty-aware vertical
residual consensus. A current accepted pose supports relative angle independently
of displacement-selected hinge samples; neither invents a closed task protocol nor
a loaded-contact reference.

`point2pose_impact.transport_impact` is evaluator-only. It measures the transported
observed zone point, its oriented +X normal and full rotation separately. For a
saved world target `G`, conditional propagation is `Z_est * inverse(Z_expected) * G`,
with `Z_expected` obtained by transporting the observed seed with evaluator truth.
This removes seed alignment bias and assumes the target's intended local coordinates;
it does not evaluate actual A4 predictions, contact admission or missing A3 geometry.
Finite lost poses retain their rejection state; missing poses/targets stay missing.
The geometry adapter now freezes an explicit observed reference and separates
current-image angle support from displacement-selected hinge evidence. Vertical-model
residuals require uncertainty-aware sample consensus; origin envelopes propagate
initial uncertainty and rotation-lever feedback. Missing floor/calibration bounds
remain absent. Native SDF Jacobian assembly is vectorized in NumPy; paired CUDA
refinements retain samples, iterations, RNG and decisions, with no end-to-end
150 ms qualification. See [[../experiments/p2p-observed-geometry-and-sdf-jacobian|the separate intervention results]].
The full static A3 reference additionally needs observed closed yaw and supported
hinge geometry. An observed relative-axis fit cannot supply those missing fields.

The zone anchor is immutable. `PurdueFK` expresses the tool relative to it;
tangential slide, normal separation and orientation are separate from tracking
error. Finite distal covers query the measured region, preserving holes and
clipping. The allowable region is eroded by the covers and relative uncertainty,
including `2*r*sin(delta_angle/2)` with the actual rotation origins and finger
lever arms. The 1 cm/5 degree perception gates are not physical slide limits.
Current measured depth across both complete footprints is also required for
contact geometry; a localized zone can remain available while these checks fail.
Existing relevant-space/obstacle queries remain maintained, but a complete dynamic
hand/arm sweep and load/stop admission is still 6.0D work.

Primitive local geometry, dynamic pose, calibration/FK and temporal growth are
counted once. Initial map depth belongs to local geometry; current depth and
metric registration residual belong to dynamic pose. TAPIR's dimensionless
uncertainty is never interpreted as meters. Missing calibration/motion bounds
remain `None`; rendered depth residuals are diagnostic envelopes, not hardware
metrology. Existing footprint tolerance is deducted once from the additional
composite erosion. No prediction or native frozen pose advances support time.

Each worker is isolated from Isaac/6.0B; only whitelisted observations and packed
array bytes cross IPC. One in-flight request and one replaceable latest capture
bound the queue. Models load before acquisition; startup is reported separately.
Completion includes snapshot copying, candidate preparation, IPC and queue waiting; release
never backdates a source or rewrites an older estimate. Support expires at 150 ms.
Once immutable automatic seeds are ready, the stateless 6.0B CUDA worker is released.
Reset increments generation, clears candidates/selections/queues and terminates
then recreates the prepared Point2Pose process before new acquisitions, clearing
SAM2/TAPIR/map/keyframe/optimizer/graph/TSDF state. It also reopens the seed worker.
Ordinary frame gaps only invalidate semantic requests; they do not reload models.
Tracking failure latches unavailable until an explicit episode reset;
native reacquisition retains the original zone and identity in the same episode.

Use `scripts/perception.py point2pose-smoke`, `point2pose-live-smoke`,
`point2pose-replay` and `point2pose-live` with fresh output directories. See
`models/perception/README.md` for isolated installation and command details.
Four full chronological pilot recordings are sampled at 20 Hz (stride three);
full and observable denominators are retained separately. Visible reference
support is scored independently from tracker acceptance using evaluator-only
truth projection and measured depth. Every loss remains in the denominator.
Live observer cases run in fresh serial Isaac processes with a parked arm;
visibility faults are synthetic input faults, not physical occluder validation.
Setup, numerical regressions and execution success are distinct from useful
availability, accuracy and the unchanged official qualification gates.

### Serial offline evaluation

`scripts/perception.py point2pose-offline` uses a separate synchronous coordinator.
All four pilot recordings are processed at their original 60 Hz, including the last
observation. Each request completes before the next frame; neither computation
time nor a replaceable queue selects inputs. Initialization searches automatic
candidates from 4 s through the common inspection, at the existing semantic period.
The first eligible candidate stays the primary diagnostic without truth selection.
Two additional fresh processes per recording assess initialization and the first
second after the seed. No failed attempt is silently restarted.

Every native pose is composed into the calibrated world and compared with the
leaf motion at its exact source row, relative to the immutable seed. Truth stays
in the evaluator. Seed error is zero by alignment and does not establish absolute
initial pose accuracy or candidate ownership. The mask-preservation guard reports
each candidate; offline diagnosis can continue after a rejection, but that
candidate remains rejected by integration. Operational initialization remains strict.

Incremental `frames.jsonl` retains finite lost poses, invalid poses, missing
initialization and terminal unprocessed rows. Native tracking and measured-support
integration have separate coverage, conditional errors, loss/recovery timelines
and full/after-seed/observable denominators. Observability uses truth-projected
seed geometry and measured depth independently of tracker acceptance; it remains
unknown before a seed. Error distributions exclude the zero seed row, while
coverage retains it. Startup and request/IPC/native compute latency are separate;
the 150 ms deadline never affects offline results. No local contact state is
published and all official qualification/contact flags remain false.

SIGINT closes the workers, retains the completed prefix, counts the interrupted
request and remaining scheduled rows as unprocessed, and stops the campaign
before another attempt. Failed processes still retain their partial results and
continue to the other planned attempts; an explicit user interruption does not.
Unstarted full recordings remain in the campaign's expected-frame denominator.
Unstarted initialization windows have unknown lengths until an automatic seed
exists. Worker constructors also close a process interrupted during startup.
The first real campaign was stopped by the user; the retained 4–10 s drift audit
is documented in [[../experiments/b1-perception-findings|B1 perception findings]].

For a bounded diagnostic, `offline_episode(..., capture_window_s=(4.0, 10.0),
registration_diagnostics=True)` processes only that inclusive acquisition window
and preserves original HDF5 row/frame IDs, including interrupted suffix counts.
For the focused full-sequence graph ablation,
`offline_episode(..., use_key_frame_graph=False)` selects the existing native
graph-off path. The default remains enabled. Keyframe creation/promotion, bounded
renewal/replacement and TSDF integration/rebuild rules are unchanged; fusion uses
unoptimized keyframe poses. Registration/resource diagnostics accept the absent
graph and retain zero graph updates/counters. This measures the total sequence
effect, including subsequent correspondence, RNG and fusion history, rather than
isolating one displayed pose.

An explicit bounded window initializes automatic candidates inside that window,
even after the static inspection schedule ends; the frozen recipe is unchanged.
The optional native `registration.jsonl` observes registration inputs, hypotheses,
inliers/residuals, SDF poses before/after refinement, frontend decisions and graph
updates before publication. Observers copy intermediate values and return the
original native results unchanged; they do not alter thresholds, models or
tracking decisions. This trace requires diagnostic mode and receives no truth.
`resources.jsonl` inventories live tracker query/features/causal CUDA storages,
retained frame masks, every candidate's active/historical references, keyframes,
LM graph variables/factors and TSDF buffers. Unique storage is counted separately
from allocator reservation and temporary peaks. Published source points retain
the current post-graph landmarks for all selected IDs, including same-frame
retirements. Renewal events and active query IDs make compaction auditable.
It also saves all extracted object-track pixels, visibility/uncertainty values,
map coordinates/IDs and per-frame SAM2 masks, including excluded references.
Its export time is separate metadata; total request time includes diagnostic work
and is not an operational performance measurement.

## Maintained estimator-independent interfaces


`DoorEstimate` retains legacy timestamp/validity/confidence, world hinge frame,
panel rotation, signed angle, dimensions and local/world contact pose. Positions
are meters, rotations matrices and articulation radians. These fields remain
consumed by 6.1; retirement does not reinterpret their complete-state semantics.

Operational support carries leaf identity, hinge alternatives, closed reference,
angle/panel support and explicit contact selection. Axis-free `DoorEstimate.local`
retains independent local geometry, identity and dynamic support.
`validate_local_contact` verifies local support only and admits no action/load.
Selections identify an action source; target changes declare a predecessor.
Visual-reference visibility and selected-contact freshness remain distinct.

`FieldSupport` records capture, availability, actual support time, episode generation
and conservative position/rotation bounds. Missing bounds remain missing. Static
support has no age-only expiry; dynamic support expires within 150 ms, with only
1 ns clock roundoff. Prediction or a copied timestamp cannot refresh observations.
Reset generations prevent cross-episode reuse. `select_contact` ranks only
producer-verified reachable proposals; it does not invent reachability or silently
switch the action source's selected material point.

`validate_geometry` dispatches explicit `legacy-full-state`/`operational-v1` profiles.
`admit_action` checks every credible hypothesis for the same patch/world trajectory,
summing geometric/robot error, rotational travel, latency and stopping margins
against both footprints and relevant clearance. Unloaded admission additionally
requires observed positive separation. Missing required bounds refuse admission.
Current operational adapters still require supported hinge hypotheses; future
Cartesian provisional admission must be implemented explicitly.

`RobotFeedback`/`admit_load` require fresh documented torque/device semantics,
bounded robot-only residual/load evidence and verified contact/stop behavior.
`PurdueIO.robot_feedback()` and current recordings explicitly report absent torque,
even when commands contain effort values. The observed monitor blocks loaded
execution; supplied torque alone cannot bypass the missing load/stop model.
Legacy truth-based expert/stop monitors remain separate consumers.

`B1Observer` combines a future provider's finite static/recent encoding with current
nine joint positions and velocities, preserving causal timing/reset checks.
`features_available` is separate from profile validity; provisional features do not
authorize policy commands. ACT/Diffusion × A1–A4, matched data, normalization and
stop-only runner/adapters remain maintained. Final raw/live encoding equivalence
and physical rollout validation await a qualified provider.

`b1.perception.release.v2` retains immutable named artifact identities,
`visual_dims`, `warmup_s`, `max_gap_s`, inspection and true offline/dynamic/frozen
flags. `verify_artifacts` checks exact names/bytes before future loading. Release
v2 remains `legacy-full-state`; no operational migration or qualified release is
introduced. Static `PrototypeRecipe` cannot satisfy those release flags.

Pure scientific metrics in `perception/evaluation.py` retain full-state errors,
finite/rejected/accepted quantiles and the original gate denominators. Every
contact/push/hold sample counts; missing/rejected states do not disappear. Per-door
coverage and joint accepted-state precision must reach 95%; empty accepted sets
fail. Required positions/dimensions retain 1 cm limits and rotations/angle 5 degrees.
Operational chosen-contact scoring remains future 6.0E work. Detailed provisional,
feedback and staged qualification requirements are canonical in
[[implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0]].

The [[pre-a4-initialization-and-hinge-budget|shared pre-A4 protocol]] defines
stationary pin versus bounded diagnostic initialization before policy dispatch,
separates known simulation floor/FK from correspondence envelopes and verifies
the unchanged A4 command sensitivity. No initialization motion/contact path is
implemented.

The [[../experiments/p2p-selected-registration|selected-registration comparison]]
retains the graph-off/partial-batch/no-rollback/five-inlier/4 mm reference and
reproduces its preserved complete poses with the vectorized Jacobian. Periodic
alternative audits reduce memory/latency but regress the nominal tail without a
native loss signal; the new option remains disabled diagnostic code. Nine native
frames represent 150 ms only in dense 60 Hz replay; live queue drops and material
recovery remain separate unqualified behaviors.
