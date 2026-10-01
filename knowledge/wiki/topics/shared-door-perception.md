# Shared Door Perception

Phase 6.0 remains **unqualified**. The custom DINOv2 estimators and their training,
feature-cache and comparison commands are retired. The diagnostic provider uses
GroundingDINO + SAM 3, explicit calibrated RGB-D/multiview geometry and DINOv3
patch features for associations. The provider and full-state evaluator are
**implemented as a prototype**, with no qualified release or dynamic validation.
[[experiments/b1-perception-findings|Perception findings]] records the measured
train/development gap, corrected component screening and limitations.

The approved October 1 successor is specified in
[[../implementation_phases/phase-6-0-operational-perception-and-contact|the operational perception/contact plan]].
6.0A contracts, numerical admission rules and consumer compatibility are
implemented. 6.0B static object memory, geometric queries and four bounded CUDA
scan diagnoses are complete, with unresolved leaf/frame identity and unobserved
physical hinge axes. Causal operational field production
and measured torque/load monitoring remain planned. Runtime estimates still follow
the legacy full-state contract; static evidence does not qualify them.

## Recording and storage

`recording/b1.py` and `recording/b1_runtime.py` retain synchronized metric RGB-D,
valid-depth masks, arm/neck proprioception, intrinsics and optical camera poses.
`b1.rgbd.v1` stores N+1 observations including reset/terminal, N applied commands,
separate annotations, calibration and a factual outcome. Observation selection
excludes asset identity, split, expert angle, phase and simulator labels.

`scripts/collect_perception.py` uses the frozen train/development corpus and one
fresh Isaac process per episode, sequentially on CUDA. A worker failure stops
admission. `--resume` skips only validated complete episodes; existing output is
never silently overwritten. Qualification records are not changed. The collector
is retained for future authorized work; no collection was started by cleanup.

Raw data remain under `datasets/b1/perception/`: 50 episodes each in
`engineering-v1` and `engineering-v2`, two complete `inspection-pilot-01` episodes,
and the separate `inspection-tallest-01` diagnostic. The first campaign lacks the
later scan views; do not silently combine it with the refreshed recipe. Engineering
recordings are not the final matched policy dataset. The inspection-only diagnostic
has no expert hold/release and is not admitted as a manipulation episode.

`recording.b1.episode_paths` checks complete campaign membership, recorded corpus
identities, conditions, duplicates and train/development isolation without loading
visual features. The frozen family assignment remains authoritative. Five
interrupted original payloads were explicitly removed; their metadata/calibration
and failure evidence remain in `outputs/b1/perception/evidence/cleanup.json`.

Selected weights and auxiliary resources live in `models/perception/`; historical
results live in `outputs/b1/perception/evidence/`. Their READMEs and local cleanup
record describe relocations. Removed ignored payloads cannot be recovered from Git.

## Inspection and calibration

`configs/perception_inspection.json` retains the common 25-second bounded scan,
seven sample times and simulated 10-degree upward camera mount adjustment. The
arm holds its parked pose while the neck scans; contact, door motion, tool drift
and tracking checks can stop inspection. Physical neck limits are unchanged.
This is a simulated mounting proposal, not validated hardware calibration.

The same inspection recipe is recorded with observations and bound to future
policy releases. `load_inspection`/`validate_inspection` check its trajectory.
The runtime mount transform and recorded calibration stay aligned. RGB-D depth
is ideal rendered optical-axis depth, not a reproduced real ZED stereo-error model.

## Model-independent policy boundary

`perception.contracts.DoorEstimate` describes timestamp, validity/reason, confidence,
world hinge frame, panel rotation, signed articulation, dimensions and world contact
pose. Positions are meters; rotations are matrices; articulation is radians.
`B1Observer` combines a provider's static/recent visual encoding with current nine
joint positions and nine velocities. It shares the same path for raw preparation
and live execution, with causal frame/time checks, reset and freshness rejection.

A provider exposes `binding`, a finite NumPy `encoding`, `reset()` and `update(sensor)` returning
`DoorEstimate`. It receives only recorded sensor keys and owns device conversion,
inference and observation-dependent validity. Between inference ticks, only a
fresh qualified cached estimate may authorize policy execution, with current
proprioception. Operational providers also declare `generation`; resets prevent
cross-generation reuse. `PolicyObservation.features_available` is distinct from
profile-specific `valid`: finite features can describe provisional/unqualified
geometry without authorizing a command. No qualified production provider, fallback or
registry is supplied yet; the prototype and numerical fixtures do not qualify it.

`b1.perception.release.v2` declares named SHA256 artifact identities,
`config.visual_dims` (static/recent widths), `warmup_s`, `max_gap_s`, inspection,
and mandatory true `offline_passed`, `dynamic_passed`, `frozen` flags. The binding
is immutable JSON. `verify_artifacts(paths)` verifies exact names and file bytes
before a future provider loads them. The provider must bind that verified recipe;
release declarations themselves are not evidence of qualification. Release v1 and
the model-specific loader are retired; no qualified v2 release has been produced.
Release v2 retains `legacy-full-state` and rejects operational reinterpretation.
`PrototypeRecipe` permits an explicit diagnostic `geometry_profile`; operational
release compatibility remains a 6.0H deliverable.

Policy observation/action schemas and matched data retain their existing semantics.
No B1 policy dataset or trained policy artifact requires conversion. ACT/Diffusion,
A1–A4 adapters, normalization and stop-only execution remain maintained components,
but final raw/live equivalence and physical integration await the qualified provider.

## Operational contracts — 6.0A implemented

`DoorEstimate.operational` adds observed leaf identity, separate hinge hypotheses,
closed reference, signed-angle/panel support and an identified local/world contact
selection. `FieldSupport` declares acquisition, availability, actual support time,
episode generation and conservative position/rotation bounds. Missing bounds remain
missing. Static evidence has no age-only expiry; dynamic support expires within
150 ms and prediction cannot refresh it. The legacy fields and `valid` retain their
complete-state meaning.

`validate_geometry` explicitly dispatches `legacy-full-state` or `operational-v1`.
Operational support checks require a unique supported state; aggregate accuracy
and release qualification still require independent offline/dynamic evidence.
`admit_action` can separately admit a diagnostic provisional state against every
credible hypothesis for the same physical patch and world trajectory. It sums
translation bounds, rigid rotational travel, robot error, relative motion over
latency/action duration and stop travel against both finger footprints and relevant
collision clearance. Each rotation source uses its own declared maximum lever over
the relevant footprint/sweep; the hinge and angle levers cannot underbound the
observed contact offset. Clearances must cover the continuous sweep and stop envelope.
An unloaded declaration additionally needs a positive observed separation bound
over that envelope after uncertainty; a Boolean alone cannot exclude contact.
Only declared required dimensions enter this decision. Producers must establish
these bounds; no operational measurement recipe has been supplied yet.

Static geometric bounds and clearance queries are implemented in 6.0B; causal motion
support remains 6.0C work and physical leaf-response bounds remain unmeasured. Robot
error, feedback timeout, residual/load uncertainty, gains and stop travel need
6.0D measurements. Bounded diagnostic duration/displacement/speed are supplied by
the shared 6.0D/F recipe before execution. Missing necessary parameters refuse
admission; optional dimensions or unrelated unknown space do not.

Contact selection orders producer-verified reachable candidates by minimum footprint
clearance, resolved-hinge distance and stable selection ID. Target changes declare a
predecessor. `MaterialContactReference` belongs only to the evaluator; independent
evaluator material-surface correspondence and dual-profile scoring remain 6.0E work.
`RobotFeedback` and `admit_load` declare a separate torque/device channel and require
fresh feedback, known signal semantics, bounded robot-only residual/load evidence
and verified contact/stop behavior. They supply neither a torque sensor nor a load
model. `PurdueIO.robot_feedback()` and `recording.b1.robot_feedback_at()` explicitly
report absent torque for the current runtime/v1 recordings, including when recorded
commands contain effort values. The observed monitor continues to block loaded
execution; even supplied torque cannot bypass its unimplemented load/stop model.

Observers, adapter/runner, dataset compilation and the observed monitor dispatch
the same explicit geometry profile. Operational adapters require the admitted
action, source, schedule and world trajectory, preserve A4's admitted frame through
the segment and latch on incompatible identity/contact/reference updates. Provisional
admission is diagnostic-only. Legacy data/features/action schemas remain unchanged;
the legacy evaluator and CLI refuse operational scoring before creating output or
loading models. Independent scoring remains 6.0E work. 6.0A ran no model campaign
or physics; the bounded 6.0B model diagnosis is described below.

## Static object scan — 6.0B implementation

`configs/perception_geometry.json` explicitly selects `scan_fusion="object-v1"`.
Recipes without that key retain the legacy plane-selection path. `Surface` retains
disconnected measured components, dense extrema, supported edges and packed
per-observation membership with calibrated poses, capture/availability times and
observed/clipped/unobserved edge status. Registered overlapping proposals share
geometric storage; unresolved associations retain competing references instead of
creating another independent object on every result. This storage consolidation
does not establish leaf ownership. Plane extraction re-samples the remaining dense
support after each fit instead of limiting a semantic mask to three planes. This
preserves smaller measured faces/connectors without lowering `min_points` or fit
tolerances. A partial-view edge cannot contract measured material bounds; fusion
removes a perimeter claim contradicted by observed support beyond that edge while
preserving the original observation record. Enclosed SAM omissions recover only
valid measured depth inliers on a seeded plane; missing/off-plane depth and the
outer semantic silhouette are not filled. Seam association checks every component
pair with the same captured frame and mask index, including repeated proposals;
choosing only the first per-frame observation previously discarded measured seams.

`GeometryProvider.scan_state` exposes candidate leaf objects, attached surfaces,
surrounding support and unresolved associations. Relief faces require an observed
internal connecting seam; parallelism, proximity, common color or a semantic mask
alone cannot attach a face. Fixed surrounding support requires exclusion by observed
separating borders in every retained object alternative; one measured side can
exclude support beyond that side without requiring a bottom/top border. Other
missing borders remain unknown. This is geometric surrounding support, not proof
of temporal fixed/leaf identity. A unique object candidate must have
supported boundaries, and unresolved associations prevent selection. No area winner,
handedness label, nominal dimensions or universal bottom-frame rule resolves ambiguity.

`StaticHingeCandidate` preserves observed panel-edge alternatives without promoting
them to physical axes. The hardware fit excludes measured leaf-plane support so
it cannot dominate the arc fit. Separated, well-conditioned cylinder arcs can yield an
explicit `HingeHypothesis` with position/angular bounds and a calibrated floor
intersection. This remains a predominantly vertical-axis model. Bounds use the
declared metric-depth tolerance, fit sensitivity and observed separation; hardware
calibration and real stereo noise are not qualified. Hardware evidence retains its
own latest acquisition and availability, independently of the panel observation.

Observed `ContactPatch` proposals retain material surface IDs, full pose and
`FieldSupport`. `footprint_support` checks both entire finite distal contact
covers against observed rasters, including holes and image clipping. Repeated
vertices on a line fail as `degenerate_finger_face`; rasterizing a line cannot certify
a finite contact face. The current closed collision extrema are lines, so their
historical patch counts do not establish full-face support. Queries now use the
projected, clipped mesh bands documented in
[[purdue-b1-robot-and-contact|the robot contact contract]], preserving the extrema
and calibration. This supplies a finite geometric cover; actual load/contact,
compliance and model-error bounds remain unverified before interaction.
The query does not replace missing support with a convex panel hull. `ObservedSpace.query` accepts a
caller-supplied covering-ball representation of the relevant continuous sweep and
stop envelope. Calibrated depth rays distinguish observed free cover, occupied
support and unknown space; unrelated unknown regions are not queried. Its returned
clearance includes the sides of the certified ray volume, rather than only the
gap to background depth. Support is static and timestamped. The approximation
assumes the recorded ideal
depth/calibration model and resolved pixel support, not arbitrary thin unseen objects.

These queries supply geometric evidence only. The caller must still cover the full
robot/tool/leaf volume and uncertainty; no stop travel, articulation response, load
bound, reachability result or command is invented. `DoorEstimate.operational` remains
unproduced pending 6.0C; complete-state validators, 150 ms dynamic freshness and release
flags are unchanged. Tracker transforms preserve surface IDs, original observation
references, measured edges and dense extents without rewriting their capture times.

`scripts/perception.py diagnose-scan --output outputs/b1/perception/NEW_SCAN` consumes
the two train pilots, both nominal/light, chronologically through 25 s. Pending
static results release at their actual completion without reading later manipulation
RGB-D or synthesizing an observation. It writes candidate reprojections, observed
geometry, finite-footprint checks and explicit uncertainty/ambiguity reports in a
fresh directory. No annotations enter this diagnostic. When ownership remains
unresolved, one conditional native SAM3 forward-video comparison uses exactly the
captured RGB frames, automatic GroundingDINO boxes and the shared `door` concept.
The box-only historical recipe produced raw masks but native hotstart removed the
unmatched track; a box on one frame did not provide continuing detector matches.
The worker now records the initial prompt mask count/object IDs separately from
propagation. Video confirmation can use later frames: all comparison geometry is retrospective and does not replace the
causal provider or backdate an accepted runtime estimate. Models stay frozen on CUDA,
and the video worker starts only after the image worker closes.

The initial four successful image scans in `outputs/b1/perception/operational-scan-02/`
all retain multiple candidates and unresolved support; none uniquely assigns fixed
surfaces or supports a physical hinge axis. The separate successful video comparison
in `operational-scan-video-01/` retained zero masks with the box-only recipe.
`operational-scan-review-01/` joins that initial evidence and local volume probes.
The subsequent `operational-scan-diagnosis-01/` re-audit recovers measured support
and explains native mask suppression; neither recovered pixels nor the positive
video prefix establish leaf identity or admit interaction. Historical finite-face
counts are invalidated by the degenerate collision extrema described above. See
[[../experiments/b1-perception-findings|the measured scan results]] and
[[../implementation_phases/phase-6-0-operational-perception-and-contact|the 6.0C handoff]].

`operational-contact-readiness-01/` separately verifies two interior material
anchors, nominal/light, with finite mesh-band covers, raw SAM membership, valid
depth and multiview registration. Human review confirms their local leaf role;
neither this review nor fusion proves subsequent rigid/mobile membership. The white
patch's latest full-cover observation is about 13 s, despite later observations of
other parts of its plane. Consumers must retain local support times and reacquire
the patch before loaded use. Endpoint CUDA IK and a small free-space ball are
initialization diagnostics, not continuous collision/control qualification.
Current operational admission still requires supported hinge hypotheses; an explicit
pre-articulation Cartesian diagnostic path is planned in C/D, without changing
adapters or substituting an edge for the axis.

## Geometric prototype — legacy inference

`GeometryProvider` and `CueEngine` share capture/completion events in replay and
live use. `ModelWorker` sends only RGB bytes to frozen CUDA models in an isolated
process. GroundingDINO uses `door.` with 0.30/0.25 thresholds; native SAM 3 uses
positive normalized center/size box prompts and confidence 0.5. DINOv3 excludes
CLS/register tokens and retains the letterbox-to-pixel mapping. Its descriptors
propose identity and mutual matches; calibrated depth provides metric scale.

Observations across the seven-pose scan are fused causally in the calibrated world
frame. RANSAC/SVD planes, measured silhouettes and visible side surfaces support
geometry. Competing surfaces remain separate. Rank/descriptor/reprojection checks
can reject panel/jamb/wall ambiguity; this heuristic is not a verified semantic
part classifier. Width/height spans remain diagnostic until all four extent edges
have interior-image metric evidence. Thickness requires an observed side face;
an unverified parallel wall cannot supply it. No nominal dimensions are filled.
Dense plane-compatible boundary support is retained before interior sampling and
area-weighted quantiles. Certified edge lines remain separate from mere observed
extent. Their world points survive fusion and are projected into the updated basis.
Registered depth overlap can associate differently colored regions and reject
offset parallel support. It does not certify object identity: different relief
faces of one leaf may remain separate, while nearby fixed support may contaminate
the selected surface. DINO descriptors are computed per face,
and a bounded multiview anchor bank retains later-view features.
Morphological opening removes thin image support before retaining dense extents;
it does not guarantee that all floor/frame contamination is removed. Being inside
the segmentation mask or close to a fitted plane is insufficient.

Visible cylindrical hinge arcs can propose a floor-anchored axis. Mutual DINO
matches and robust rigid RGB-D alignment can refine it through `(I-R)h=t`, rejecting
insufficient rotation and ill-conditioned fits. The floor origin is the existing
calibrated robot-world z=0; panel orientation comes from the closed scan. The
prototype approximates a vertical revolute door with predominantly planar faces.
Contact uses the common fraction 0.295 and height 1.09 m on a locally measured
surface, including its full normal. Neither teacher contact nor door coordinates
enter inference. In the implemented September path, a hidden hinge keeps the scan
invalid and active probing is unavailable. The successor plan defines separate,
bounded provisional admission rather than bypassing this validator.
Patch centers are proposals, not exact physical correspondences. After the scan,
`PixelMotionTracker` uses sparse subpixel RGB flow from OpenCV already supplied by
the workstation. Forward/backward consistency, valid interpolated depth without
discontinuities and rigid consensus verify fixed observed points. Loss clears pixel
history. Points can be replenished only from a successfully verified current pose,
mapping their observed depth back to the original reference and accumulating anchor
uncertainty per anchor. Surviving original references remain intact; replenishment
adds points rather than replacing the reference bank. Excess uncertainty rejects a complete state. Current-depth support
still validates each propagated surface. Recorded
motion can diagnose/refine a hinge, but it does not authorize live motion from an
unresolved scan.

Semantic requests target 5 Hz from the first observation, including transitions
between the seven held scan poses. The worker permits one in-flight request and discards old generations after
reset/loss. Replay releases results at capture plus measured worker latency,
including process transport. Live uses a background worker. Current RGB-D
reprojection validates intermediate geometry; it cannot refresh a state by copying
its timestamp. Nonmonotonic observations reset history. Acquisition, competing
identity, missing RGB-D, discontinuous motion, unsupported dimensions and uncertain
fits are explicit rejection causes. A complete consumer state must be fresh within
150 ms, finite, physically shaped and internally consistent with signed angle.

`configs/perception_geometry.json` is one diagnostic recipe, represented by
`b1.perception.prototype.v1`. It cannot satisfy `PerceptionBinding` release flags.
`scripts/perception.py smoke` checks all three frozen models; `evaluate --pilot`
checks left `door-2738468b94d74c5f` and right `animated-door-1-88abf40`, both
nominal/light. Full `evaluate` consumes all 50 engineering-v2 episodes in HDF5 row
order, keeping frame counters separate. The evaluator alone reads annotations,
phase and metadata. Missing/rejected estimates remain in the gate denominator.
Reports include all finite, rejected and accepted error quantiles, overlapping
rejection causes, per-door/condition/phase rates, worker latency and support recovery.
PNG/NPZ diagnostics preserve masks and competing/selected observed surfaces.

The September full-state recipe requires every train/development door to pass
offline before dynamics. The approved successor has explicit pilot and release
admission stages in the operational plan; those stages are not implemented yet.
`ObservedControlChecks`/`ObservedPurdueSafety` provide a prototype-only monitor
using observations, robot limits and FK. The existing runner clears pending actions
and latches stops on invalid estimates, requiring explicit reset to resume. The
prototype monitor cannot certify force/load from RGB-D and reports
`force_feedback_unavailable` before contact/push/hold. Legacy `PurdueSafety` remains
for other maintained consumers; it must not guide this prototype. No simulator
execution, physical force check or hardware safety is established by these tests.

## Current and approved acceptance boundaries

### Control requirements versus qualification

A short geometric door-motion command needs a reliable hinge axis, closed reference,
opening angle, local contact position/normal and observed robot state. Total leaf
height is not intrinsically required to compute that local motion. However, a local
patch alone does not establish the moving collision volume, authorized contact
surface or force/load safety. The current provider/recipe still require the complete
legacy `DoorEstimate`; the implemented operational interfaces do not repair inference.

The teacher and September diagnostic contact is at 0.295 of total width and 1.09 m
world height; reproducing it requires width. It is not a mandatory policy target.
The approved successor diagnostic controller chooses a reachable observed patch
with footprint clearance. It neither rewrites teacher labels nor substitutes a
point for a policy. Total dimensions remain required by the existing full-state
profile, while the new operational profile requires them only where an action
depends on them. Human review resolved the displayed
pilot regions: A/C are fixed frame, B is the leaf bottom; other doors may have a
bottom frame. These diagnostic labels never enter inference. Corrected pilot replay
still fails complete-state gates; extended evaluation remains stopped for object
association and hinge/tracking correction. See
[[experiments/b1-perception-findings|the retained evidence]].

### Legacy full-state gates — implemented and unchanged

Per train/development door, during contact/push/hold: valid coverage at least 95%,
contact-position p95 at most 0.01 m and orientation p95 at most 5 degrees. Preserve
complete-state checks: hinge origin, dimensions and local/world contact position
within 1 cm; hinge, local/world contact and panel rotation plus wrapped signed angle
within 5 degrees. Confidence must accept accurate states and reject missing or
incorrect states; accepted-state precision must reach 95% per door, jointly across
all complete-state limits. Empty accepted sets fail. Accepted p95 must satisfy every
limit; finite rejected states are also reported. Report coverage
and errors separately, with worst per-door results for both handednesses.

Fit learned quantities/normalization on train only; use development for selection
without changing gates or the frozen identity split. Never use sealed-test
qualification evidence as training or model-selection input. Plane-only component
success is not complete geometry qualification. This profile and its previous
failures remain reportable; it is not silently reinterpreted as the new operational
profile.

### Operational-v1 — interfaces implemented, scoring/qualification pending

The canonical [[../implementation_phases/phase-6-0-operational-perception-and-contact|6.0 protocol]]
defines the required hinge/angle/local contact state, per-field lifetime, provisional
action checks, chosen-contact evaluation and staged offline/dynamic gates. Required
state retains 1 cm / 5 degrees and per-door 95% coverage/precision; provisional
actions are not qualified estimates. Full dimensions remain separate diagnostics.
Action margins additionally include uncertainty, footprint, latency and stopping
travel. Unknown space outside the relevant swept volume is not a universal veto.

Existing `b1.rgbd.v1` recordings have no torque. They can validate geometry and
causal replay, not the new loaded-control path. The timestamped robot-feedback
interface serves the common monitor only; acquisition remains 6.0D work. Both gate
groups and explicit release/profile compatibility are required before a qualified operational provider;
do not manufacture existing `PerceptionBinding` flags. Phase 6.1 will integrate the
qualified path across all eight cells without changing A4 semantics.

## Retained collection interface

For a separately authorized future collection, from the repository root:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/collect_perception.py \
  --output datasets/b1/perception/NEW_CAMPAIGN \
  --inspection configs/perception_inspection.json --device cuda:0
```

Choose a fresh output; use `--resume` only for that same campaign. Optional
`--asset-id`, `--condition` and `--smoke` bound the selection. `--inspection-only`
requires `--inspection` and produces a diagnostic without expert manipulation.
The diagnostic smoke/evaluate CLI is described above. There is no maintained
estimator training or launch CLI.
