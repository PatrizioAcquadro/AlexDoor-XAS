# System Architecture

The maintained runtime is `AlexDoor-DoorPush-Purdue-v0`: fixed-base Purdue
Alex003, WSG32/UMI v1, measured pedestal and head ZED RGB-D. Real-door expert
qualification and learned observation integration remain future work.

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
image encoding, histories and the observed-only learning interface.

See [[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]] and
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]].

## Preparation and Storage

`scripts/prepare_doors.py` handles inspection, normalization, static/visual/isolated
physics checks and promotion. `DoorInspectionEnv` remains necessary for this B1
path. It does not execute the robot expert.

Each promoted door has tracked `candidate.json`, `recipe.json`, `prepared.json`
and local `source/` and `prepared/` payloads. Paths resolve from the door folder;
no accepted asset depends on a temporary attempt. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] for the
contract, corpus, rights scopes and pending qualification.

Runtime caches and verification reports belong under `~/.cache/alexdoor-xas/`.
Future datasets and learned runs use ignored `datasets/` and `outputs/` payloads.

## Reusable Algorithms

Actions, adapters, the scripted state machine, recording, numerical dataset
loaders/export, split/normalization utilities and ACT/Diffusion tensor training
remain available as components. Their integration with B1 is not implemented.
Policies consume a caller-supplied observation function; they do not read door
truth from the simulator. See [[topics/episode-and-dataset-contracts|Data Components]]
and [[topics/learned-policy-stack|Learned Policy Stack]].

B0 calibration, manifests, generation, training/evaluation orchestration, datasets
and D0–D4 scenes were retired. Historical conclusions remain in the wiki and
source history in Git. The workstation GPU is authoritative for Isaac; no command
in this repository controls physical hardware.
