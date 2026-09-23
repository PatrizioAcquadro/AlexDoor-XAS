# System Architecture

The registered runtime is `AlexDoor-DoorPush-Purdue-v0`: one fixed-base Purdue
Alex003 with WSG32/UMI v1, measured pedestal and head ZED. It commissions control
and sensing on synthetic collidable fixtures, with a frozen synthetic-door probe.
Real-asset expert qualification, learned observation encoding and dataset
integration remain later phases.

## Runtime Boundary

`DoorPushPurdueEnv` owns scene composition, reset, seven-joint A1 addressing,
full-pose A2 execution and explicitly framed A3 conversion. The external Alex
package owns assets, PD profiles, physical limits, mimic constraints, collision
filters, pedestal and ZED factories. The consumer adds model gravity compensation;
no simulator door state restores or replaces a requested action.

`purdue_contacts.py` resolves imported rigid owners and classifies raw contact
points against distal-finger geometry and exact partner actors. Its reports are
diagnostics, separate from observed inputs. They include normal forces only,
forbidden contacts, and separation; no friction-inclusive force claim is made.

`recording/rgbd.py` supplies copied, synchronized RGB, metric optical-axis depth,
valid-depth mask and seven-arm/two-neck proprioception. `env.capture.sample` is
the acquisition interface; the Gym `policy` tensor currently exposes only the
18 position/velocity values. Phase 6 owns their learned-model integration.

See [[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]] and
[[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] for geometry,
commissioning defaults and evidence.

## Historical Data and Models

The dataset loaders, `phase2.v1/v2` readers, matched A1–A4 export structures,
normalization, ACT/Diffusion models and checkpoint formats remain readable.
Offline training on existing B0 data remains available. Historical calibration
and manifest validation are retained for interpreting those contracts.

B0 simulator execution and its robot-specific controller/asset loader were
retired. Full generation and learned evaluation commands fail explicitly before
Isaac startup; they do not silently execute B0 checkpoints on Purdue. Generic
scripted/data/adapter components remain for the planned migration and their
software contracts, without claiming a working B1 learning workflow.

## Preparation and Storage

The B1 preparation path takes an explicit source/recipe through component inspection,
canonical USD authoring and independent static/GPU checks. It uses the Phase 4
nominal dynamics and baked convex geometry, including collidable handles and a
clear frame opening. Prepared assets expose opening/hinge/panel transforms for
the next qualification phase; preparation does not execute or retune the robot probe.

`scripts/prepare_doors.py` owns the local workflow. Candidate metadata, license
evidence and recipes under `assets/doors/b1/<id>/` are versioned; source snapshots,
generated attempts and the prepared pointer are ignored. Source/geometry duplicates
and source-bound local review are checked before promotion to `ready_for_5.1`.
The pointer also records whether the asset is redistributable or local-only;
technical readiness does not grant distribution rights.
See [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] for the
interface, commands, supported formats, approximations and measured readiness.

Legacy door preparation remains independent of robot execution. Its physics
inspection now instantiates only the door; static checks and preparation keep
their documented B0 geometry limitations. Neither establishes B1 qualification.

Datasets and learned-policy runs remain untouched. Canonical D0–D4 USD layers
and historical results retain their B0 identity. Verification reports and images
belong in `~/.cache/alexdoor-xas/verification/`, not in the tracked output tree.

## Deployment Boundary

The workstation GPU is authoritative for Isaac integration. No repository
command controls physical hardware. There is no added dependency, robot copy,
cluster orchestration or external Alex/Isaac Lab modification.

## Version Notes

- 2026-09-23 — Added B1 asset preparation and isolated GPU validation, separated from real-door expert qualification.
- 2026-09-22 — Replaced B0 execution with Purdue commissioning and separated observed capture from diagnostic truth and historical learning interfaces.
- 2026-08-13 — Documented the maintained B0 data and learned-policy path.
