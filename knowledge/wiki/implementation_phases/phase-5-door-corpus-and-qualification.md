# Phase 5 — Door Corpus and Qualification

Subphase 5.0 is complete for **32 prepared doors**: 29 redistributable, two
local-only and one private/noncommercial. The common expert executor is implemented.
The first full campaign produced 15 qualified, 15 out-of-domain and two unresolved
outcomes. A user-authorized recovery corrected shared control defects and
revised one common placement/contact before any split. Six recovery pairs and
four successful representative single cycles supported the corrections. Formal
paired requalification then completed seven doors under the final controller:
six qualified and `psx-worn-20d5505` was published `out_of_domain` at 17.92°.
The campaign stopped at that result; **25 doors still need fresh pairs**. Their
retained 21 qualified and four out-of-domain statuses predate the final revision.
Subphase 5.1 still owns the final 24-door, 12/4/8 identity split; Phase 6 generates
demonstrations. See [[decisions/visuoproprioceptive-generalization-benchmark|B1 Design]].

## Subphase 5.0 — Intake and Preparation

### Published Asset Contract

Each `assets/doors/b1/<id>/` contains three tracked records:

| Record | Canonical responsibility |
|---|---|
| `candidate.json` | Identity, source URL/selected part, author/attribution, license/scope, source path/fingerprint and relevant exceptions. |
| `recipe.json` | Accepted normalization, component ownership, physics/collider parameters and concise reasons for approximations. |
| `prepared.json` | Relative USD path, dimensions/transforms, handedness, geometry fingerprint, acquired preparation results, mechanical limit and published expert status. |

Local `source/` retains the used source and necessary dependencies. Local
`prepared/` contains the final USD, materials/textures and front/rear previews.
Asset payload paths are relative to the door folder. Expert evidence locations
are machine-local verification-cache paths, documented under 5.1 below. Explicit
`source_dependencies` in
recipes resolve relative to the recipe file. Payloads and previews stay outside Git;
copy the complete folder when moving an asset. License scope lives only in the
candidate record; technical readiness does not grant redistribution rights.

Promotion copies the accepted result into that structure. Numbered `attempts/`
are disposable workspaces, separate from published source/payloads. No published
asset depends on them. Published folders cannot be mutated by preparation commands.
Source copies preserve relative glTF/OBJ dependencies and localize USD asset paths.
Promotion stages every payload and record, then publishes `prepared.json` last.
On a publication error it restores prior records and removes its partial outputs,
so the same reviewed attempt can be retried.

The 2026-09-24 cleanup carried forward all acquired results without new real-door
normalization, static/visual checks or physics runs. Final USD/material/texture
and preview bytes are unchanged. Four source copies required localized texture
references; their original geometry and external downloads were preserved.
The source fingerprint identifies the originally reviewed bytes, before path
localization. Source and geometry fingerprints remain useful duplicate checks.

### New Candidate Workflow

1. Review one user-supplied URL: source/author, asset-specific terms, dependencies,
   selected part, available format, full-size single-leaf geometry and duplicates.
   Return a concrete download/reject/unresolved decision before local preparation.
2. The user downloads and unpacks the source, preserving sidecar layout. Keep
   original downloads outside the repository; there is no automatic collector.
3. Inspect locally and create a recipe. Use shared preparation tools for repairs;
   source-specific transforms and component selections belong in the recipe.
4. Normalize, run static checks, review both previews and run one short isolated
   GPU functional check. Diagnose/repeat only checks affected by an actual defect.
5. Record source-bound license/dependency/duplicate/visual review and promote.
   Promotion means `ready_for_5.1`, never expert qualification.

Use the normalization attempt printed by the command for subsequent gates;
inspection has a separate attempt. The pending candidate record needs source and
license evidence plus passing `license_scope_review`, `custom_terms_review` and
`duplicate_review`. Before promotion provide `local_dependency_review`,
`local_duplicate_review`, `local_visual_review` and `reviewed_source_sha256` from
`inspect.json`. Explain fingerprint collisions in `duplicate_resolution`.
Admission evidence is compacted into the canonical records at promotion.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py review \
  --candidate assets/doors/b1/<id>/candidate.json
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py inspect \
  --asset-id <id> --source /path/to/door.glb --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py normalize \
  --asset-id <id> --source /path/to/door.glb \
  --recipe assets/doors/b1/<id>/recipe.json --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py static \
  --attempt assets/doors/b1/<id>/attempts/<number> --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py preview \
  --attempt assets/doors/b1/<id>/attempts/<number> --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py physics \
  --attempt assets/doors/b1/<id>/attempts/<number> --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py promote \
  --attempt assets/doors/b1/<id>/attempts/<number> \
  --candidate assets/doors/b1/<id>/candidate.json
```

### Admission and Approximations

Use original handedness and unique geometry, with separable leaf/frame. Mirrors,
recolors and repeated source parts are not independent identities. Related pack
parts require distinct `source_part` and geometry. Exclude sliding/double doors,
cabinets, gates and fantasy geometry. Target leaf dimensions are width 0.65–1.20 m,
height 1.80–2.40 m, thickness 0.025–0.10 m, at most 250,000 visual triangles and
4K textures. Supported conversion prefers USD/USDZ, GLB/glTF, Blend/FBX, then OBJ.

The task starts **closed and already unlatched**. Preserve frame, leaf and handle
collision. Use the common Phase 4 nominal dynamics with derived inertia; this
controls geometry variation, not source-specific real dynamics. Apply the frozen
robot base/tool/contact/probe unchanged. An obstructed contact point does not
permit per-door retuning or collision removal.

| Recipe choice | Meaning and limit |
|---|---|
| `scale`, `rotation`, `translation_m`, `opening_center_source` | Proper rotation and positive uniform scaling of inspected coordinates; opening center maps to floor origin. USD/FBX conversion uses glTF Y-up meters. |
| `components`, `component_splits` | Assign complete components or reviewed whole-face subsets to Frame/Panel/Handle; retain source vertices, UVs and material ownership. |
| `shell_backings` | Opt-in flat rear and boundary walls for front-only reliefs; thickness/rear appearance are inferred, front UVs reused. No folded/non-manifold result. |
| `frame_fits` | Reviewed adaptation of an existing licensed frame aperture, preserving profiles on untouched axes; never reshape the leaf through this option. |
| `moving_translation_m`, `moving_scale` | Move visuals/colliders together; uniform moving fit at most 2%, with explicit center and `clearance_review`. Update dimensions/inertia consistently. |
| `hinge_m`, `hinge_axis_components` | Infer a plausible axis where absent, or validate separated modeled barrels; never infer physical manufacturer accuracy. |
| `leaf_components`, `clear_aperture_m` | Separate leaf dimensions/inertia from attached hardware; declare actual centered passage covering at least 90% of the leaf envelope. |
| `unlatched_components` | Disable only reviewed separate locking-part collision. Visuals remain; leaf/frame/handle collisions cannot be excluded. |
| `hinge_contact_exclusions`, `hinge_jamb_components` | Filter explicit local internal bearing pairs/cells within 10 cm of the hinge; retain all other contacts, including robot contacts. |
| `colliders`, `collider_groups` | Bake hulls/decompositions or reviewed partitions; group material-separated surfaces of one solid without filling a frame opening or crossing rigid bodies. |

An overlapping slab can use a reviewed surface mount instead of forced in-frame
fitting. Any mounting, backing or frame adaptation is a simulation assumption,
recorded in that door's recipe. The geometric sweep sets the mechanical stop;
it does not insert an arbitrary 90° stop or the 45° expert-admission threshold.
An initial intersection fails; no demonstrated stop within 270° is unresolved.

Static checks cover physical ownership, dimensions, mass/inertia, clear opening,
visual/collider agreement and collision sweep. The isolated GPU scene now includes
the same floor as the robot task. Earlier preparation omitted it and could accept
a leaf that binds against the floor during robot execution. The GPU run uses three
resets, passive holds and controlled opening: reset angle 0.1°, speed 0.01 rad/s,
passive drift 0.25°, frame translation 0.1 mm/rotation 0.1°, anchor error 1 mm,
penetration 2 mm and stop agreement 1°. Harmless contact alone is not a failure;
missing/overflowed measurements cannot pass. This establishes preparation only.

Conversion preserves the supported glTF/PreviewSurface path; arbitrary source
shaders and animation are not guaranteed. Rectangular-opening checks, convex
collision approximations and a 512-component limit remain explicit boundaries.

### Rights Scopes

- **R — redistributable:** reviewed CC0/CC BY 4.0 source and dependencies.
- **L — local_only:** reviewed Sketchfab Free Standard, within the downloading
  licensee's authorized workspace; no shared geometry/textures.
- **P — private_noncommercial:** the specifically user-approved CC BY-NC-ND door,
  for private preparation/tests only; no commercial use or adapted-material sharing.

Restricted doors' rendered datasets or derived products need a separate rights
review before release. Extra author/dependency terms still apply. `NoAI` and
`CreatedWithAI` have different meanings; assess the actual program output and
source terms, not merely whether a model uses diffusion. Unclear rights remain
unresolved before inspection. Per-source evidence and exceptions live in the JSON.

## Prepared Pool

All entries below retain acquired preparation passes. Expert status is recorded per door
in `prepared.json`; the table describes preparation, not expert admission.
The stop is geometric, not `theta_expert_d`. The prison door's frame approximation was corrected from a false 23.1° stop to
a measured 47.0° stop; its previous expert pair is superseded. Phase 5.1 owns
selection and reachable-domain decisions.

| Asset ID | Hinge | Mechanical stop | Scope |
|---|---|---:|---|
| `animated-door-1-88abf40` | right | 90.7° | R |
| `animated-door-2-88abf40` | right | 90.7° | R |
| `animated-door-3-88abf40` | right | 91.8° | R |
| `door-2738468b94d74c5f` | left | 113.1° | R |
| `door-5035d7977155` | left | 194.4° | R |
| `door-adf292f437f2` | left | 176.0° | R |
| `door-door-metal-b21ec273` | right | 59.8° | R |
| `door-prison-metal-old-45306a46` | left | 47.0° | R |
| `door-with-doorframe-c29da62c` | left | 161.2° | L |
| `door-with-frame-2f2f149f` | right | 188.0° | R |
| `interior-wood-d1-32707dc` | left | 54.3° | R |
| `modern-door-2fb8d024` | right | 93.0° | L |
| `psx-bathroom-20d5505` | left | 121.3° | R |
| `psx-front-001-ee7d5c6` | left | 84.4° | R |
| `psx-front-002-ee7d5c6` | left | 166.2° | R |
| `psx-front-005-ee7d5c6` | right | 94.1° | R |
| `psx-front-008-ee7d5c6` | left | 121.2° | R |
| `psx-front-20d5505` | left | 121.3° | R |
| `psx-industrial-001-4f5561b` | left | 91.0° | R |
| `psx-industrial-002-4f5561b` | left | 91.1° | R |
| `psx-industrial-003-4f5561b` | left | 91.0° | R |
| `psx-industrial-004-4f5561b` | left | 91.0° | R |
| `psx-interior-wood-002-02e442c` | left | 121.2° | R |
| `psx-interior-wood-003-02e442c` | left | 121.2° | R |
| `psx-interior-wood-005-02e442c` | left | 121.2° | R |
| `psx-interior-wood-006-02e442c` | left | 121.2° | R |
| `psx-interior-wood-007-02e442c` | left | 121.2° | R |
| `psx-interior-wood-008-02e442c` | left | 121.2° | R |
| `psx-wooden-001-20d5505` | left | 121.4° | R |
| `psx-wooden-009-20d5505` | left | 121.2° | R |
| `psx-worn-20d5505` | left | 121.2° | R |
| `void-frame-studio-animated-classic-door-08bdf51b` | left | 91.0° | P |

### Excluded or Unresolved Intake History

Only the accepted payloads remain locally. Git retains source records and the
attempt narrative; original downloads remain outside this checkout.

| Source selection | Disposition | Git reference |
|---|---|---|
| `psx-front-006-ee7d5c6` | Exact duplicate of an earlier prepared door. | `4d1044b` |
| `psx-front-007-ee7d5c6` | In-frame intersection; alternative mount looked detached. Unresolved, never promoted. | `4d1044b` |
| `psx-industrial-005-4f5561b` | Geometry variant of industrial `003`; not an independent identity. | `381c727` |
| `psx-interior-wood-001-02e442c` and `009` | Exact duplicates of earlier prepared doors. | `15701da` |
| Interior PSX `004`; Hawtor D2 | Repeated geometry/appearance variants. | `e765f55`, `0212d96` |
| Double-leaf pack selections | Outside the single-leaf task. | `345e37b`, `49abf8c` |

## Subphase 5.1 — Expert Qualification and Final Split

### Operational Command

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/qualify_door.py \
  --asset-id <id> --device cuda:0 --headless
```

The command loads the published USD and the current common setup, acquires RGB-D
and contact diagnostics, and runs two complete cycles. It exposes no per-door
controller tuning. `--rerun` explicitly allows a justified recheck of an already
qualified door. Unresolved doors may be rerun after diagnosis; every invocation
creates a new evidence directory under `~/.cache/alexdoor-xas/verification/expert/`.
The report records the previous evidence directory. Earlier failed pairs remain
available and cannot be mixed into a new pair.

For fixing a known failure, add `--diagnostic` to run exactly one complete cycle
with the same preflight, setup and evidence. It never writes `prepared.json` or
creates `theta_expert_d`. Read `trials[0]` for the measured outcome; the report
remains `unresolved` with `diagnostic_only: true` because no pair was collected.
A zero exit code means a valid single cycle at or above 45 degrees; otherwise
read the report for an exclusion, a lower angle or an execution failure. A successful diagnostic is evidence for a correction,
not formal qualification. Run a complete pair separately when authorized.

`prepared.json` retains its preparation fields and stores a structured
`expert_qualification`: status, reason, expert angle in degrees when defined,
compact trial results, repeat difference and evidence location. Unvisited doors
retain `not_run`. Full traces, raw contacts, RGB/depth samples, effective setup and
input records remain in the cache, outside learning datasets. Only a complete
report is published; concurrent record edits are not overwritten.

A positive-volume intersection between the closed door and the fixed pedestal
establishes initial setup infeasibility before movement. This uses the composed
convex colliders and a 2 mm penetration tolerance, not overlapping AABBs alone.
It records `out_of_domain`, the intersecting geometry and **no expert angle**;
the preparation pass remains valid. No unsafe pair is needed to establish that
this closed starting configuration is impossible. Other execution failures remain
`unresolved` unless a loaded asset explicitly contradicts its structural contract
(`invalid_asset`). A missing file or simulator failure is not proof of bad geometry.
The loader accepts both published collider tags, scalar `b1:sourceComponent` and
array `b1:sourceComponents`, matching preparation verification. The earlier
`door-with-frame-2f2f149f` rejection was a loader compatibility defect; its
existing collision meshes loaded correctly after this fix.

The surface trajectory follows the actual leaf colliders at the frozen fraction
and height, with tool +X into the local collidable surface and +Z aligned with
projected vertical. Actual frame/handle
bounds replace synthetic dimensions. Clearance remains a conservative diagnostic;
raw contact ownership determines contact validity. Visibility samples actual
surfaces and remains a geometric diagnostic, not learned-perception qualification.

The initial contact check intersects each actual distal support footprint with
the convex leaf surfaces, including narrow features between support vertices.
A surface protruding ahead of the prescribed fingertip pose by more than the
10 mm position budget establishes `prescribed_footprint_obstructed`, an
`out_of_domain` geometric result without an expert angle. The PSX front-005
has a real wood strip 22.1 mm ahead of the glass under one finger, independently
confirmed on the original visual mesh. Moving the contact point or removing the
strip is not permitted during routine qualification. The authorized common
contact revision below avoids the strip while preserving it.

### Procedure for Subsequent Doors

1. Select a prepared identity awaiting qualification or an explicitly deferred
   post-fix recheck. Run the command without `--diagnostic` for a formal pair;
   add `--rerun` when its retained historical status is already `qualified`.
2. Read `report.json`; a nonzero shell exit alone does not distinguish exclusion
   from a runtime error. `qualified` requires the valid pair below. `out_of_domain`
   means an evidenced geometric exclusion or a valid reference below 45 degrees.
3. Inspect representative images, sustained-contact windows, limiting cause and
   release. Retain the automatically published summary and all failed evidence.
4. Escalate `unresolved` or contradictory evidence for diagnosis. Do not change
   the robot pose, contact point, thresholds, colliders or controller to obtain a pass.
5. After a general fix, rerun the entire affected pair. Do not choose favorable
   repetitions. Commit only the relevant tracked result and documentation changes.

### Common Setup and Corrective Qualification

The user-authorized setup keeps **robot and pedestal together at zero yaw**, at
**X = -0.437 m, Y = 0.380 m**, with the shared right-arm ready and left-arm parked
posture in `configs/purdue_synthetic_probe.json`. Contact is now at 0.295 of leaf
width from the actual hinge and 1.09 m height. Head pose, controller gains, push
speed and validity thresholds are unchanged. See
[[phase-4-robot-and-task-configuration|Phase 4]] for the full effective controller.

The paused campaign exposed correctable integration/preparation defects:

- The loader omitted older scalar collider-component tags. Both published metadata
  layouts are now accepted; `door-with-frame` was not a defective source asset.
- The formerly floor-flush `door-with-frame` leaf bound against the task floor.
  The isolated preparation check now includes that floor and reproduces the failure.
  A documented 2 mm upward moving-assembly fit passes static and GPU checks while
  retaining the 188-degree stop. All visual/collision surfaces move together, with
  consistent COM/panel-center updates and unchanged dimensions, hinge axis and
  prescribed robot contact point.
  The same defect was reproduced and repaired on then-pending
  `interior-wood-d1-32707dc`; its revised geometric stop is 54.3 degrees. Its
  expert status was `not_run` at this repair stage. Both original and repaired
  payloads are archived.
- The PSX `front-005` recipe omitted component 7, the existing collidable glazing,
  from leaf selection. Adding it alongside component 8 preserves all USD bytes,
  dimensions, inertia and the prescribed contact point. Its real wood strip still
  protrudes 22.1 mm under one fingertip ahead of the glass; the new support-footprint
  preflight identifies this incompatible prescribed pose before movement.
  The old 0.40-width contact was geometrically excluded without an expert angle;
  the recovery uses one new common fraction for every door and retains the strip.
- The prison-door frame collider filled an empty hinge-corner region and imposed
  a false 23.1-degree stop. Repartitioning the same surfaces at the measured
  sill/header boundaries (Z = 0.041961/2.045949 m) gives a 47.0-degree stop.
  Static and isolated GPU checks pass. Original visual mesh sections independently
  confirm actual leaf/frame intersection at 47.2 degrees; further opening would
  require changing the modeled geometry or hinge. No source surface, pivot, mass,
  material, unlatched state or other collision was changed.
- Abrupt removal of opening lead, or merely slowing a fixed angular target, lost
  contact with inertially coasting doors. The shared expert now checks projected
  local holding reach, follows the moving material point while smoothly removing
  lead over 3 s, and provides bounded normal support if load declines. A complete
  valid 0.5 s final hold and safe release remain separate mandatory conditions.

### Recovery of the Full Prepared Pool

The first full campaign is retained in Git at `cf73eb0` and in each record's
previous evidence chain: 15 qualified, 15 out of domain and two unresolved.
The seven initial pedestal exclusions were real frame intersections, including
independent visual-surface checks for the two large left doors. The common
placement moves robot/pedestal 5.7 cm back and 8 cm laterally from the first full
campaign. Contact moves from 0.40 to 0.295 of width and from 1.00 to 1.09 m height
for every door. This clears the composed pedestal geometry and prescribed finger
footprint of all 32 assets while providing a compromise between left/right arm
reach and wrist/body clearance. Zero yaw, ready/parked joints, head, gains,
nominal speed, collision geometry and admission/validity thresholds are retained.
Local multistart IK and footprint screening support the candidate; they do not
replace dynamic qualification or prove global reachability.

The recovery corrects shared control defects: a push reference that could
lag behind a coasting panel; persistent tangential servo bias before contact;
and a final hold that could end after its first valid window instead of completing
the configured 3 s transition. Holding reach is checked periodically even when
present tracking is accurate, reserving space for inertial motion. The final
0.5 s of the complete transition must be valid. Actual errors always refer to the
prescribed material point, never the compensated command. See
[[phase-4-robot-and-task-configuration|Phase 4]] for the controller.

A further PSX front-005 diagnostic found a constant 3.55 mm normal offset on
sloped relief. Treating every collected surface as a flat slab incorrectly used
that geometric offset as tracking drift. The shared adapter now derives the
contact orientation from the actual convex surface at the prescribed point;
fraction, height and position remain fixed. The support-footprint preflight
uses the same orientation and checks protrusions along its actual approach axis.
No surface or threshold is modified. Single-cycle diagnostics verified this
correction; complete pairs must use the current common configuration.

The front-001 leaf metadata also omitted its existing glazing component 3.
Including it with component 10 preserves USD bytes, leaf bounds, inertia and the
prescribed point. Its subsequent full pair exposed a different problem: the low
contact required wrist/body interpenetration in all 18 locally solved precontact
configurations examined against the actual URDF collision meshes. Rerouting the
approach alone did not resolve it. A higher common contact and lateral placement
clear this collision in the diagnostic run. The right frame-door constrains the
remaining reach; its latest diagnostic sustains 45.09 degrees with all 3 s of
holding valid, no forbidden contacts and safe release. The completed recovery
pairs below use this common compromise; local IK screens are not proof of global
infeasibility. Missing initial fingertip support now raises a diagnostic
error before motion instead of running known-invalid cycles. A diagnostic void-frame
cycle recovered controlled contact, but the latest pair stops below admission and
led to the further diagnosis below.

Complete pairs with the revised setup recovered front-001, industrial-001,
the right frame-door and the largest left door. Front-005 then exposed a release
path defect: rotating while retreating scraped a finger side against its retained
wood relief after valid pushing/holding. Both cycles reproduced the forbidden
contact. Release now withdraws along the measured finger axis before rotating
toward the previously achieved pose, within the same 3 s budget. A follow-up isolated a velocity-limiting defect: clipping each pose-control
joint step separately could create lateral hand drift from nullspace centering.
Pose control now scales the bounded target step uniformly; numerical regression
reproduces the old defect and verifies the correction. Contact rules, geometry
and the admission threshold are unchanged. The corrected release is
verified by a complete GPU diagnostic at 59.71 degrees, with all 180 holding
ticks valid, no forbidden contact and safe release. Opening/holding reproduce
the failed case exactly; only release changes. Uniform scaling preserves the commanded
joint-step direction; it does not guarantee a straight measured hand path or
smaller transient error during free release. The release endpoint, separation and
contact conditions are independently checked.

Six complete pairs finished before the user switched recovery to single-cycle
diagnostics. Their traces, contacts and representative hold/release images were
audited without another simulation:

| Recovered door | Sustained expert angle (degrees) |
|---|---:|
| `psx-front-005-ee7d5c6` | 59.71 |
| `door-with-frame-2f2f149f` | 45.09 |
| `psx-industrial-001-4f5561b` | 46.99 |
| `door-2738468b94d74c5f` | 57.84 |
| `psx-front-001-ee7d5c6` | 60.64 |
| `door-prison-metal-old-45306a46` | 47.00 |

These pairs used `a46ed1a`; evidence paths remain in their prepared records.
Void-frame's subsequent pair remained below admission at 32.45 degrees,
triggered by the tracking guard. The user then limited recovery to **single-cycle
representatives of distinct failures**, preceded by static checks. No complete
campaign followed that small correction. The six recovered doors and 15
baseline-qualified doors were not rerun during the fix campaign. Formal pairs
under the final shared code were deferred at that stage; a single diagnostic
never creates an expert reference or overwrites a published outcome.

The void-frame diagnosis separated two common defects. The old guard stopped on
a stable 2.53 mm material-point bias although the future pose was locally
reachable and the validity budget remained 10 mm. The revised guard reserves
2.5 mm before that budget and anticipates sustained growing error, rather than
stopping on a small stable offset. A further run exposed brief loss of normal
load: push now uses the same bounded support as hold before declaring a declining
load stop. The 0.25 s detachment failure, 10 mm/5 degree validity, force limits and
complete final holding/release conditions remain unchanged. See Phase 4 for the
exact common controller; no door-specific settings were introduced. The corrected
void-frame single cycle sustains **66.92 degrees**, completes holding, has no
actual forbidden contacts and releases with 22.2 mm normal separation. Its
unpublished diagnostic is
`~/.cache/alexdoor-xas/verification/expert/void-frame-studio-animated-classic-door-08bdf51b/20260925T202136.654999Z/`.
The old 32.45-degree paired record is deliberately retained until a separately
authorized formal rerun.

`door-with-doorframe-c29da62c` subsequently completes a single cycle at **54.73
degrees**, and `door-5035d7977155` at **56.70 degrees**, with valid final hold and
release. These are unpublished diagnostics, not new expert references. The first
checks another low-load stop; the second represents the former pedestal
intersections. Reports are under their expert-cache directories at
`20260925T202545.111364Z` and `20260925T202912.129189Z`, respectively.

The current static screen clears pedestal/contact geometry on all 32 doors.
The three remaining industrial doors also have local IK solutions at 45–47
degrees; this is not dynamic validation. Only industrial-002 is selected for a
further dynamic diagnostic, which completes at **47.20 degrees** with valid hold
and release (`expert/psx-industrial-002-4f5561b/20260925T203313.100427Z/`).
Industrial-003/004 and front-002, front-20d5505,
wooden-001, wooden-009 and worn-20d5505 were deferred to systematic requalification
at that recovery stage. Routine successes use the saved
cycle diagnostics; detailed image/raw-contact audits target failures and the
representative evidence needed to validate a correction. Display angles to two
decimals; admission still uses unrounded measurements and the report status.

Earlier pairs are retained in each record's evidence chain; no repetitions are
mixed across runs. Diagnostics and independent audits remain under
`~/.cache/alexdoor-xas/verification/corpus-recovery-20260925/`.

### Formal Requalification Started 2026-09-25

The first seven doors were run once each, in the prescribed order, with two
complete cycles per invocation on `cuda:0`. Reports are under
`~/.cache/alexdoor-xas/verification/expert/<id>/<timestamp>/report.json`;
the exact paths are published in each `prepared.json`. The result records were
committed at `cdd3204`. The lower sustained angle is the published reference;
the table rounds only for display.

| Door | Hand | New status | `theta_expert_d` | Report timestamp |
|---|---|---|---:|---|
| `psx-industrial-003-4f5561b` | left | `qualified` | 48.03° | `20260925T204643.280842Z` |
| `psx-industrial-004-4f5561b` | left | `qualified` | 47.92° | `20260925T205324.978919Z` |
| `psx-front-002-ee7d5c6` | left | `qualified` | 59.10° | `20260925T205830.755088Z` |
| `psx-front-20d5505` | left | `qualified` | 64.89° | `20260925T210606.477452Z` |
| `psx-wooden-001-20d5505` | left | `qualified` | 63.84° | `20260925T211254.703702Z` |
| `psx-wooden-009-20d5505` | left | `qualified` | 65.39° | `20260925T211949.936782Z` |
| `psx-worn-20d5505` | left | `out_of_domain` | 17.92° | `20260925T212634.130486Z` |

All seven reports agree with their published records. Each pair has two valid
holds and releases, consistent limiting causes and at most 0.10° sustained-angle
spread. The six qualified references meet the unrounded 45° gate. The seventh
pair consistently stops under the gate with `safety_stop / tracking_margin`.
At the push stop, orientation error was 0.04510 rad against the controller's
0.04363 rad preventive guard; the material-point error was 0.00515 m against
its 0.00750 m reserve. Contact stayed loaded and valid, local hold-endpoint IK
error was negligible, and no actual forbidden contact was recorded. The final
hold sustained 17.92° and released safely in both cycles. Preparation reports a
121.2° mechanical limit. This is a nominal controller/setup-domain exclusion,
not proof that the asset is defective or that 45° is physically unreachable.
Whether the early orientation guard reflects an unavoidable task constraint or
a controller limitation remains unresolved. No setting or threshold was changed.

Sampled geometric RGB-D visibility failed for both cycles of
`psx-industrial-004-4f5561b`, `psx-front-20d5505` and
`psx-wooden-009-20d5505`. These are separate perception limitations, not
expert-admission failures. The other four had no failed sampled frames.
The campaign stopped after the first non-qualified result; the remaining 25
doors were not invoked under the final controller. No corpus split or learning
data was created.

The 15 previously deferred rechecks are the three `animated-door-*`,
`modern-door-2fb8d024`,
`door-door-metal-b21ec273`, `door-adf292f437f2`, `interior-wood-d1-32707dc`,
`psx-bathroom-20d5505`, `psx-front-008-ee7d5c6`, and the six
`psx-interior-wood-*` identities (002, 003, 005, 006, 007, 008). Keep their current
records/evidence intact and run the same operational command with `--rerun` later.
Their historical `qualified` status does not certify the revised setup. Do not
change code, assets or controller settings during those rechecks; diagnose any
failure before proceeding.

The pool contains 25 left and seven right identities. Even if every current door
qualifies, at least five additional qualifying right identities are needed for
the planned 12/12 balance. No final corpus, train/development/test split, new asset
downloads or learning data were introduced. Modern-door remains `local_only`.

RGB-D capture is operational, but geometric visibility fails in several qualified
trials, including insufficient sampled frame points. It is a diagnostic, not an
expert-admission gate, and does not establish learned-perception readiness. No
synthetic sweep was repeated; historical synthetic results belong to the earlier
45-degree setup. Runtime qualification remains nominal simulated evidence, not
hardware safety, statistical robustness or a global optimum.

#### Implementation

For each prepared door, run the frozen probe to its valid controlled limit,
including approach, sustained contact, hold, and safe release. Continue past
45 degrees. Record sustained angle, force/contact validity, stop reason, and
joint margins. Repeat once from the same reset.

Require valid controlled completion, consistent limiting causes, and sustained
maxima within 2 degrees. Diagnose inconsistent pairs without favorable best-of-
many selection; after resolving the cause, rerun the prescribed pair. Set
`theta_expert_d` to the lower of its two valid maxima. This is a conservative
practical reference, not a statistical percentile or global optimum.

Admit only doors with `theta_expert_d >= 45 deg`. Exclude lower-angle candidates
as outside the reachable benchmark domain even when their assets are valid.
A stall or timeout without an evidenced limit remains unresolved, not an automatic
asset rejection. Ask for another candidate URL when a door is conclusively excluded.

After 24 doors pass, freeze the manifest and 12/4/8 train/development/test split,
with left/right counts 6/6, 2/2, and 4/4. Group related source geometry to prevent
family leakage. Record the common setup/probe and each door's expert result.

#### Key Decisions

- Distinguish invalid asset, valid-but-out-of-domain asset, and unresolved probe
  failure. Never replace a qualified door because a learned policy performs poorly.
- Test qualification traces/images are isolated evaluation evidence, never
  training data for perception, gaze, normalization, policies, or model selection.
- No common primary angle, adaptive qualification count, bootstrap, or percentile
  over identical trials. The 45-degree rule is nominal admission only.
- The final corpus contains assets, provenance, references, and splits. It does
  not yet contain the Phase 6 matched demonstration dataset.

#### Problems / Limitations

Complete with 24 eligible nominally reachable doors, stable expert references,
and the frozen split. Conclusions apply to this qualified domain; neither the
corpus size nor repeated trials prove broad coverage. The 2026-09-25 user-authorized recovery revises the common setup before any split.
After corpus/split freeze, do not retune it from held-out assets.

## Maintained Entry Points

- `scripts/prepare_doors.py` and `qualification/`: shared intake and promotion.
- `scripts/qualify_door.py`: frozen real-door probe, paired decision and per-door result.
- `scripts/verify_door_preparation.py`: synthetic format/negative/physics checks,
  run only when affected infrastructure changes justify them.
- `DoorInspectionEnv`: robot-independent door measurement, still used by B1.

Historical infrastructure results are recorded at `d12f0b2`; accepted intake
through `9c16e3d`. Preparation-pool cleanup is recorded at `7a1ffb1`.
