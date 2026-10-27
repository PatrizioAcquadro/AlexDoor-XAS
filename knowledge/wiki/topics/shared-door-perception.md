# Shared Door Perception

Phase 6.0 remains **unqualified**. Inspection, static scans, Point2Pose smoke,
live/replay and serial evaluation use one persistent SAM3 text/video frontend
with `door surface`. `configs/point2pose_selected.json` is the common runtime
recipe: reconditioning OFF, bounded causal history, graph OFF, partial reference
batches ON, rollback OFF, all-query TAPIR and renewal-only depth filtering.
TAPIR, SuperPoint, registration, TSDF, calibrated full-scene RGB-D and camera/FK
remain. Five inliers/4 mm and all action/policy/contact thresholds are unchanged.

Setup and commands are in [Perception Model Setup](perception-model-setup.md).
[[../experiments/b1-perception-findings|Historical findings]] and the linked
experiments own results; [[../implementation_phases/phase-6-0-operational-perception-and-contact|Phase 6.0]]
owns remaining acceptance work. Older custom trackers and the rejected Point2Pose
variants are historical source, while their saved evidence remains available.

## Module layout

All imports begin with `alexdoor_xas.perception`:

| Previous module | Maintained import |
|---|---|
| `point2pose_runtime`, `point2pose_worker`, `point2pose_patches` | `point2pose.runtime`, `point2pose.worker`, `point2pose.patches` |
| `panel_tracking` | `point2pose.tracking` |
| `sam3_frontend`, `sam3_runtime`, `sam3_worker`, `sam3_memory` | `sam3.frontend`, `sam3.runtime`, `sam3.worker`, `sam3.memory` |
| `point2pose_offline`, `point2pose_replay`, `point2pose_live` | `diagnostics.offline`, `diagnostics.replay`, `diagnostics.live` |
| `point2pose_diagnostics`, `scan_diagnostics` | `diagnostics.smoke`, `diagnostics.scan` |

Other Point2Pose helpers follow the same `point2pose.<suffix>` convention;
shared scoring lives in `diagnostics.metrics` and `diagnostics.impact`.
Contracts, geometry, provider and control remain at the common level. `ipc` owns
private worker framing and array serialization; model-specific initialization and
runtime environments stay separate. No legacy import wrappers are maintained.
Saved-data readers were migrated; retired launchers use their recorded revision.

CLI names are unchanged. Live commands do not enumerate recordings. The live
matrix returns failure for a failed child, absent report or recorded execution
failure; diagnostic quality remains a separate report outcome. Serial evaluation
retains every original timestamp and all failure/availability accounting.

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

All 50 `datasets/recordings/b1/perception/engineering-v2` episodes remain intact, including
calibration and metadata, for 6.0B and future common train/development qualification.
They are engineering recordings, not matched policy demonstrations.
`recording.b1.episode_paths` validates membership, conditions, duplicates and split
isolation without loading visual features. The earlier October 2 cleanup removed
superseded `engineering-v1`,
`inspection-pilot-01` and `inspection-tallest-01` recordings after
preserving their headers/results. Selected models remain in `models/perception/`;
historical records and the cleanup inventory are under
`outputs/evidence/b1/perception/`. Removed ignored payloads are not Git-recoverable.

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

`session.PerceptionSession` coordinates SAM3 and optional P2P in their separate
CUDA runtimes. `SessionProvider` supplies the maintained `update(sensor)`,
`scan_state`, `static`, `last_cue`, `reset()` and local tracking interface. It
returns invalid `DoorEstimate` objects and no policy encoding; this integration
admits no motion. The numerical `GeometryProvider`/`CueEngine` remain useful
geometry/event harnesses with injected cues, without an operative image-model path.

The coordinator snapshots exactly the observed sensor keys: RGB, full metric
depth/validity, intrinsics, camera pose from the same joints/FK, frame and acquisition
time. SAM3 runs once for each processed capture. Its IDs/masks accompany that same
packet through geometry and P2P, including after initialization. Neither an older
mask nor a cropped/resegmented image is substituted. Geometry extraction retains
its existing measured component/plane/edge membership checks; semantic membership
alone does not establish leaf ownership. GroundingDINO, the box-prompt static
worker and the separately executed SAM2 path are retired; DINOv3 remains retired.
`sam3.SAM3_REVISION` owns the retained source revision.

No first-frame detection is required. Empty output reports `no_candidate` and
continues the same text session. Multiple IDs report `selection_required`;
`provider.select_object(sam3_id, geometric_object=None)` selects explicitly. A
single ID is latched automatically; later IDs cannot replace it. Competing geometric
roots under that ID report `ambiguous_geometry`. A selected static surface ID can
resolve the root only through observed registration to current support. Missing
or insufficient measured support postpones initialization without re-prompting.
The runtime exposes the SAM3 ID, geometric surface, fixed zone, native object ID
and original support times as one binding, rather than a mask-list index.

Maintained tracking ingress accepts `--selection` JSON through the same selection
API. It records `sam3_id`, an observed `criterion`, and either `geometric_object`
(a session-local static surface ID) or `reference_pixel: [u, v]`, with optional
`after_s`. A pixel chooses only the uniquely eligible measured surface containing
it on the actual seed capture, after the requested time; zero or competing supports
keep the same SAM3 session waiting. Pixel coordinates depend on the displayed view;
the applied frame/time and all alternatives are recorded, and replay cadence can
change that capture. This is explicit reference selection, not leaf/contact
qualification. Static IDs are local to one scan and must not be blindly reused
across different acquisition schedules. Unique eligibility may initialize without
a request; no first-root, area or semantic-score winner resolves ambiguity.

`initialization` reports `inspection_complete`, `reference_state`, and `p2p_started`
independently. Alternatives are available during inspection without native startup.
Neither 25 s, a selected SAM ID, nor a missing reference starts P2P. Identity cannot
change even between a request and its first processed capture. Complete reset
clears the request as well as models, bindings and queues; explicitly configured
command requests are reapplied to the new session and recorded again.


SAM3 starts with the first processed acquisition. Geometry fusion runs at the
existing semantic period during the prescribed stationary inspection (4–25 s).
The inspection protocol must remain stationary. At 25 s the static surfaces,
scene cloud and their acquisition/support times freeze; later moving leaf depth
never enters that map. P2P initializes from the first eligible **current** capture
at or after 25 s, retaining its own immutable measured reference and zone. These
are distinct from older inspection support. `current_space` and `scan_state.space`
use a bounded latest full-scene RGB-D view for surrounding obstacles/free-space
queries, separately from the frozen inspection map. Missing scene support remains
unknown, and late depth does not retroactively refresh inspection geometry.
Published scan snapshots preserve each acquisition's actual completion time across
replacement; the seed reference receives its complete publication time. Earlier
inspection support and the immutable seed never acquire later tracking timestamps.

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

## Panel tracking prototype — 6.0C

`point2pose.runtime.tracking_provider` creates the shared session provider.
The original SAM3 worker stays alive when P2P initializes and throughout native
tracking; no periodic text prompt or second segmenter is introduced. An absent
selected ID supplies an empty mask and explicit semantic loss, even if a finite
native fallback exists. Same-ID return is reported separately from material
recovery. After semantic loss, local material identity remains unavailable as
`material_recovery_unverified`; native geometric recovery alone cannot certify
that the originally observed material returned.

The pinned CAD-free pipeline retains the loss/SDF publication fixes, clustered
registration, full-frame 480/four-iteration TAPIR, SuperPoint, vectorized SDF
Jacobian and 120 active references. Partial batch admission preserves native point
order. The existing renewal-only 10 mm/radius-2 depth gate leaves initialization
sampling unchanged. Retirement still requires promoted metric inlier support,
visible outlier observations and geometric distribution; historical landmarks
and TSDF/keyframes survive active TAPIR compaction. No age-only eviction or new
quality/performance variant is part of this integration.

External masks, all-query TAPIR and renewal filtering are runtime options;
`diagnostic_only` is false. Initialization still requires five distributed measured
interiors and strict same-capture mask preservation. Registration traces observe
behavior without relaxing that guard. Native SAM2 imports previously activated
BF16/TF32 even without SAM2 inference; the worker now enables the same numerical
context explicitly before construction and through requests. Startup patches the
pinned segmenter package to remove its eager SAM2 import, preserving original
source backups. SAM3 internal tracker modules and GTSAM's ISAM2 are unrelated to
the retired separate SAM2 package and remain maintained.

Bounded SAM3 history preserves native useful conditioning, attention memories,
pointers and identity metadata without unbounded RAM offload. Original frame
indices remain; accessing a released historical input fails explicitly.
[[../experiments/p2p-sam3-selected-development|Selected development]] owns the
historical 31 s results. Starting during inspection changes SAM3 history and P2P's
seed, so those results do not automatically apply to the integrated transition.
The explicit bounded-window serial API can reproduce the historical 31 s seed;
its recorded first-eligible geometric-zone selection is diagnostic and leaves
ownership unresolved. It never weakens native initialization guards.

### Observed geometry and timing

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

For A1/A2, the measured plane/edge constraint first uses the dominant current
surface query. A relief face can dominate that query while the originally
selected material remains visible on another component. If the dominant query
cannot satisfy the existing local constraint, the adapter expands once to all
components measured in the same current SAM3/RGB-D capture. It then reruns the
same finite conditioning, complete native-pair residual, root-depth consistency,
material identity and ambiguity checks. Missing support remains unavailable;
no native pose, controller, error bound or admission threshold is rewritten.

The compatible dominant-plane path retains its existing cost. Additional geometry
work belongs to provider publication latency and does not fabricate a fresh capture.
`local_surface_query` and the cumulative `local_surface_query_expansions` appear
in the existing tracking diagnostics and native terminal perception record, outside
policy observations. CPU regressions cover nominal geometry, relief on both sides
and absent material. They establish software behavior; the changed native D3
condition still requires its own full-cycle qualification in the
[[../experiments/act-a1-complete-cycle-collection|collection campaign]].

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
remain absent. See [[../experiments/p2p-observed-geometry-and-sdf-jacobian|the separate intervention results]].
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

Workers remain isolated from Isaac and from one another; packed arrays and
whitelisted observations cross IPC. One complete request is in flight and one
latest acquisition may wait; a newer waiting capture replaces the older one.
The coordinator processes only dispatched captures, retaining their original
IDs/times. Queue replacements, skipped capture IDs, queue waiting, snapshot
copies, SAM3, geometry, P2P and publication/adapter latency remain explicit.
Late results publish only for their own acquisition/generation. A frame gap does
not synthesize missing frames or silently reset/reselect an object. Support
still expires at 150 ms. Sampled simultaneous GPU/RAM residency includes both
workers and coordinator GPU/RSS, including the live renderer; allocator/high-watermark
peaks remain separate. A supplied provider clock/`now` must share capture origin;
acquisition times remain immutable. Complete native service and polling/publication
wait are separate. Continuous `point2pose-operational` and stationary live retain
processed-only quality and explicit skips; see
[[../experiments/p2p-operational-selection-and-light-peaks|operational measurement]].

Reset kills both workers, joins the old request thread and clears SAM3 temporal
state/IDs, P2P references/keyframes/TSDF, static/current maps, selections, queues
and published outputs. Prepared sessions reload before new acquisition; invalid
input closes the session and defers reload to the next valid capture, whose
latency includes restart startup. Old requests cannot contaminate the new session.
A process failure latches unavailable until reset; it is never retried for a better
result. Original finite failures and support times remain visible.

Operational replay samples one explicitly supplied pilot recording at stride three
(20 Hz); full and independently observable denominators remain distinct. Live
observer checks run in fresh Isaac processes with a parked arm. Synthetic input
visibility faults do not validate physical occluders. Setup, execution success and
accuracy are separate from useful availability and release qualification.

### Serial offline evaluation

The [[../experiments/p2p-temporal-renewal-admission|temporal renewal diagnostic]]
compares independent observed body/camera motion on chronological saved candidates.
Its historical evaluator source remains at `24a2aa7`; the rejected implementation
and its exclusive tests are retired. It was rejected before runtime activation because
the current waiting budget cannot separate body/static hypotheses and useful
support is depleted. The persistent runtime and selected recipe remain unchanged.

`scripts/perception.py point2pose-offline` calls the **same PerceptionSession**
serially for every original 60 Hz row, including the final observation. There is
no replaceable queue or latency-based input selection. SAM3 starts at the first
scheduled capture; normal initialization follows the 25 s inspection. The API
accepts an explicit `capture_window_s`: a window starting after inspection can
reproduce the historical 31 s seed without earlier SAM history. The CLI requires one explicit `--episode` and starts exactly one attempt.
Single inputs retain recording validity, frozen-corpus identity, non-test split
and asset/condition checks before worker startup. No other episode is enumerated.
The old four-recording/twelve-attempt campaign, live matrix and concurrent smoke
alias are retired; their source remains at `24a2aa7`. No scientific acceptance
threshold or formal qualification requirement changes.

Truth motion belongs only to scoring. Seed-relative error is alignment, not
absolute initialization/ownership accuracy. Strict initialization failure stops
that attempt and retains all terminal scheduled rows as unprocessed. Saved
historical diagnostics that continued after an initialization refusal remain
readable at their recorded revision; the maintained runtime does not bypass that
guard. Native acceptance, measured geometric support, semantic presence and
material verification remain distinct claims.

Incremental `frames.jsonl` retains finite lost poses, invalid poses, missing
initialization and terminal unprocessed rows. Native tracking and measured-support
integration have separate coverage, conditional errors, loss/recovery timelines
and full/after-seed/observable denominators. Observability uses truth-projected
seed geometry and measured depth independently of tracker acceptance; it remains
unknown before a seed. Error distributions exclude the zero seed row, while
coverage retains it. Startup and request/IPC/native compute latency are separate;
the 150 ms deadline never affects offline results. No local contact state is
published and all official qualification/contact flags remain false.

SIGINT closes the workers, retains the completed prefix and counts the interrupted
request and remaining scheduled rows as unprocessed. It starts no further attempt.
Failed processes retain their partial results. Worker constructors also close a
process interrupted during startup. Historical campaign outcomes and unstarted
slots remain in [[../experiments/b1-perception-findings|B1 perception findings]].

For a bounded diagnostic, `offline_episode(..., capture_window_s=(4.0, 10.0),
registration_diagnostics=True)` processes only that inclusive acquisition window
and preserves original HDF5 row/frame IDs, including interrupted suffix counts.
An explicit bounded window initializes automatic candidates inside that window,
even after the static inspection schedule ends; the frozen recipe is unchanged.
The optional native `registration.jsonl` observes registration inputs, hypotheses,
inliers/residuals, SDF poses before/after refinement, frontend decisions and graph
updates before publication. Observers copy intermediate values and return the
original native results unchanged; they do not alter thresholds, models or
tracking decisions. Tracing is observational, retains strict initialization and
receives no truth.
`resources.jsonl` inventories live tracker query/features/causal CUDA storages,
retained frame masks, every candidate's active/historical references, keyframes,
LM graph variables/factors and TSDF buffers. Unique storage is counted separately
from allocator reservation and temporary peaks. Published source points retain
the current post-graph landmarks for all selected IDs, including same-frame
retirements. Renewal events and active query IDs make compaction auditable.
It also saves all extracted object-track pixels, visibility/uncertainty values,
map coordinates/IDs and synchronized per-frame masks, including excluded
references. Saved SAM2-era traces remain readable.
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

The bounded A1/A2 pilot now implements explicit local admission independently
of global hinge qualification. Its finite material selection, retained static
view witnesses, separated frame/contact uncertainty and common observer/IO
recipe checks are described in [[b1-policy-pilot|the pilot contract]]. Missing
relevant collision-space/response support remains a refusal. The global
operational-v1 and historical complete-state qualification metrics are unchanged.
