# System Architecture

The maintained runtime is `AlexDoor-DoorPush-Purdue-v0`: fixed-base Purdue
Alex003, WSG32/UMI v1, measured pedestal and head ZED RGB-D. Real-door expert
qualification has a shared executor and one common zero-yaw mounting/ready
configuration, validated by two complete GPU cycles each on a right and a left
door. Routine qualification changes only the asset ID. Shared RGB-D perception
preparation is implemented; estimator development/dynamic qualification remains
open. B1 ACT/Diffusion software integration is available, with CUDA model checks
and final observed-geometry rollout validation still pending.

## Runtime and Data Boundaries

`DoorPushPurdueEnv` owns scene composition, reset, seven-joint A1, full-pose A2
and explicitly framed A3. The external Alex package owns robot assets, physical
limits, mimic constraints, collision filters, pedestal and camera factories.
The consumer adds gravity compensation; simulation truth never replaces an action.

`purdue_contacts.py` classifies raw contacts against distal-finger geometry and
exact partner actors. Normal force, forbidden contacts and separation are
**diagnostics**, separate from policy observations; tangential force is not measured.

`env.capture.sample` provides copied, synchronized RGB, metric optical-axis depth,
valid-depth mask, seven-arm/two-neck proprioception, timestamps and frame IDs.
The Gym policy tensor contains the 18 joint positions/velocities. Phase 6 owns
image encoding, histories and the observed-only learning interface. The separate
`perception` package now provides frozen DINOv2 features, a four-sample causal
estimator, train/development loaders and training/evaluation entry points. See
[[topics/shared-door-perception|Shared Door Perception]] for the pretraining boundary.

See [[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]] and
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]].

The separate B1 policy path uses a shared frozen RGB-D encoding plus current
proprioception, explicit A1-A4 tensors, and family-specific normalization.
`B1Runner` connects the policy or labeled matched-replay source to `PurdueIO`.
Only its stop monitor reads door truth; action adapters consume observed geometry.
This is independent software preparation, not a completed Phase 6.1 or generated
policy dataset. See [[topics/learned-policy-stack|Learned Policy Stack]].

## Preparation and Storage

`assets/doors/b1/corpus.json` freezes Phase 5 asset membership, geometry families,
train/development/test assignments, setup and expert references. The numerical
`qualification.corpus.load_corpus` validates the tracked records; the CLI
`verify_door_corpus.py` optionally also verifies local evidence. This metadata
selects assets for Phase 6 recording; it is separate from episode splits
and cannot enter policy observations. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Frozen corpus]].

`scripts/prepare_doors.py` handles inspection, normalization, static/visual/isolated
physics checks and promotion. `DoorInspectionEnv` remains necessary for this B1
path. It does not execute the robot expert.
Its isolated physics scene includes the task floor, so floor-binding leaves cannot
pass solely because the hinge rotates in an otherwise empty scene. Older prepared
records predate this check; diagnosed clearance repairs retain original payloads
and require affected static/physics checks and a fresh expert pair.

`scripts/qualify_door.py` loads a prepared door into the same Purdue environment
and reuses the controlled probe. Prepared geometry supplies the material-point
trajectory, local surface orientation and physical bounds; the common
robot/controller settings stay frozen. Tool orientation follows the collidable
surface normal at the prescribed fraction/height, with projected vertical as up.
The executor checks initial pedestal interference, requires two valid controlled
cycles for an expert angle, and publishes a compact result in `prepared.json`.
The optional `--diagnostic` runs one cycle and writes cache evidence only; it
cannot replace or publish a formal expert reference.
It also checks the actual distal support footprints for raised leaf features
incompatible with the prescribed pose; evidenced initial exclusions have no expert
angle and need no unsafe dynamic pair.
All detailed qualification evidence stays in the verification cache, never in
learning data. The same sustained-window measurement serves synthetic and real
doors; a valid final hold/release is required independently of the maximum angle.
The geometry adapter accepts scalar and array collider-component tags from both
published preparation layouts. The expert reserves local tracking margin for
holding and follows the moving material point while removing opening lead, with
bounded normal contact support during push/hold and tangential servo-bias compensation. The full
holding transition completes before the last 0.5 s is assessed. Actual validity
always uses the prescribed material point, not the compensated command. After
valid holding, release withdraws along the fingers before rotating toward an
achieved pose, avoiding lateral scraping against raised relief. These
simulator-truth checks belong only to the scripted expert; they do not extend learned-policy observations. See
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] for the
common controller and unchanged validity limits.

The geometric visibility diagnostic uses rendered panel/frame triangles;
collision hulls remain authoritative only for physical contact and clearance.
Prepared-door samples are cached in the closed frame and rotate with the leaf;
frame samples cover the contact-height band. Triangle intersections use bounded
ray batches while preserving all candidates and final sample selection. The depth
test retains occlusion rejection. Saved RGB-D samples have measured camera poses/intrinsics for offline
inspection. These geometry and camera diagnostics never become policy inputs.

`qualification/contracts.py` owns shared source admission limits and geometry
validation; preparation and verification use the same bounds. Promotion stages
records and payloads together, preserves source-relative dependencies and writes
`prepared.json` last. Failed publication restores the candidate for retry.

Each promoted door has tracked `candidate.json`, `recipe.json`, `prepared.json`
and local `source/` and `prepared/` payloads. Paths resolve from the door folder;
no accepted asset depends on a temporary attempt. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] for the
contract, frozen corpus, rights scopes and completed qualification.

Runtime caches and verification reports belong under `~/.cache/alexdoor-xas/`.
Perception engineering recordings and frozen features use ignored `datasets/`
payloads; backbone weights, preparation reports and later learned runs use `outputs/`.

## Reusable Algorithms

Action math, recording, numerical dataset loaders/export, split/normalization
utilities and ACT/Diffusion tensor training
remain available as components. Their integration with B1 is not implemented.
Policies consume a caller-supplied observation function; they do not read door
truth from the simulator. See [[topics/episode-and-dataset-contracts|Data Components]]
and [[topics/learned-policy-stack|Learned Policy Stack]].

B0 calibration, manifests, generation, training/evaluation orchestration, datasets
and D0–D4 scenes were retired. Historical conclusions remain in the wiki and
source history in Git. The workstation GPU is authoritative for Isaac; no command
in this repository controls physical hardware.

The workstation preflight checks declared Python dependencies, CUDA, the pinned
Isaac installation and external Alex assets. Missing or broken imports and asset
resolution errors produce a failure summary. Ordinary Python dependencies are
packaged; Isaac, Alex, PyTorch, Warp and CUDA remain external runtime components.
