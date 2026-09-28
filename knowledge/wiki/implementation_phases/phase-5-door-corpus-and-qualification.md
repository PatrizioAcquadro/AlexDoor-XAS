# Phase 5 — Door Corpus and Qualification

Subphase 5.0 is complete for **32 prepared doors**: 29 redistributable, two
local-only and one private/noncommercial. The common expert executor is implemented.
The first full campaign produced 15 qualified, 15 out-of-domain and two unresolved
outcomes. A user-authorized recovery corrected shared control defects and
revised one common placement/contact before any split. Six recovery pairs and
four successful representative single cycles supported the corrections. Formal
paired requalification now has **32 published qualified doors: 25 left and seven
right**. The first six pairs precede the angular-guard correction; the seventh,
`psx-worn-20d5505`, qualified after it. All seven precede fresh-process trial
isolation. The other 25 pairs used fresh Isaac processes for both cycles, including
the repaired `void-frame` pair. The 24 pending pairs after that repair each
received one formal invocation and qualified. These results are not yet one
unchanged-implementation corpus: remeasure the earlier seven before final
selection. RGB-D sampling was corrected and checked at three poses each on
three affected doors, without repeating their qualification cycles.
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
and contact diagnostics, and runs two complete cycles. Each cycle starts in a
fresh Isaac process and physics scene loaded from the same prepared asset and
common setup. Joint
position/velocity resets alone do not isolate PhysX contact history between
reference trials. It exposes no per-door controller tuning. `--rerun` explicitly allows a justified recheck of an already
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
input records remain in the cache, outside learning datasets. New runs also save
`camera.json`, indexed by image tick, with measured camera pose and intrinsics
for reproducing visibility checks against the saved depth. Only a complete
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

All 32 doors have published paired results, with two complete cycles per
invocation on `cuda:0`. Reports are under
`~/.cache/alexdoor-xas/verification/expert/<id>/<timestamp>/report.json`;
the exact paths are published in each `prepared.json`. The first seven invocations
were committed at `cdd3204`; the corrected worn pair supersedes its earlier
published outcome without removing that historical evidence. The lower sustained
angle is the published reference; the table rounds only for display.

| Door | Hand | New status | `theta_expert_d` | Report timestamp |
|---|---|---|---:|---|
| `psx-industrial-003-4f5561b` | left | `qualified` | 48.03° | `20260925T204643.280842Z` |
| `psx-industrial-004-4f5561b` | left | `qualified` | 47.92° | `20260925T205324.978919Z` |
| `psx-front-002-ee7d5c6` | left | `qualified` | 59.10° | `20260925T205830.755088Z` |
| `psx-front-20d5505` | left | `qualified` | 64.89° | `20260925T210606.477452Z` |
| `psx-wooden-001-20d5505` | left | `qualified` | 63.84° | `20260925T211254.703702Z` |
| `psx-wooden-009-20d5505` | left | `qualified` | 65.39° | `20260925T211949.936782Z` |
| `psx-worn-20d5505` | left | `qualified` | 67.42° | `20260926T060507.800125Z` |
| `void-frame-studio-animated-classic-door-08bdf51b` | left | `qualified` | 66.92° | `20260928T173830.453850Z` |
| `door-with-doorframe-c29da62c` | left | `qualified` | 54.73° | `20260928T175236.896147Z` |
| `door-5035d7977155` | left | `qualified` | 56.70° | `20260928T175916.343847Z` |
| `psx-industrial-002-4f5561b` | left | `qualified` | 47.20° | `20260928T180557.417781Z` |
| `animated-door-1-88abf40` | right | `qualified` | 61.39° | `20260928T181142.560713Z` |
| `animated-door-2-88abf40` | right | `qualified` | 61.48° | `20260928T181637.948391Z` |
| `animated-door-3-88abf40` | right | `qualified` | 61.78° | `20260928T182144.459603Z` |
| `door-2738468b94d74c5f` | left | `qualified` | 57.84° | `20260928T182721.985793Z` |
| `door-adf292f437f2` | left | `qualified` | 58.88° | `20260928T183349.176113Z` |
| `door-door-metal-b21ec273` | right | `qualified` | 59.80° | `20260928T184039.310056Z` |
| `door-prison-metal-old-45306a46` | left | `qualified` | 47.00° | `20260928T184604.720361Z` |
| `door-with-frame-2f2f149f` | right | `qualified` | 45.09° | `20260928T185323.405174Z` |
| `interior-wood-d1-32707dc` | left | `qualified` | 54.30° | `20260928T185804.702758Z` |
| `modern-door-2fb8d024` | right | `qualified` | 54.41° | `20260928T190404.772206Z` |
| `psx-bathroom-20d5505` | left | `qualified` | 62.66° | `20260928T190858.524911Z` |
| `psx-front-001-ee7d5c6` | left | `qualified` | 60.64° | `20260928T191639.445074Z` |
| `psx-front-005-ee7d5c6` | right | `qualified` | 59.71° | `20260928T192411.520178Z` |
| `psx-front-008-ee7d5c6` | left | `qualified` | 60.45° | `20260928T193000.882108Z` |
| `psx-industrial-001-4f5561b` | left | `qualified` | 46.99° | `20260928T193623.002470Z` |
| `psx-interior-wood-002-02e442c` | left | `qualified` | 62.68° | `20260928T194222.735949Z` |
| `psx-interior-wood-003-02e442c` | left | `qualified` | 62.62° | `20260928T194935.898468Z` |
| `psx-interior-wood-005-02e442c` | left | `qualified` | 62.11° | `20260928T195646.540133Z` |
| `psx-interior-wood-006-02e442c` | left | `qualified` | 62.71° | `20260928T200430.841160Z` |
| `psx-interior-wood-007-02e442c` | left | `qualified` | 62.81° | `20260928T201145.437578Z` |
| `psx-interior-wood-008-02e442c` | left | `qualified` | 62.46° | `20260928T201907.191003Z` |

All 32 current reports agree with their published records. Each pair has two
valid holds and releases, consistent limiting causes and at most 0.10°
sustained-angle spread. All 32 qualified references meet the unrounded 45°
gate. Geometric visibility separately warned on `door-adf292f437f2` (556 of
3,949 frames per cycle) and `door-door-metal-b21ec273` (all 3,453 frames per
cycle), and on `psx-front-008-ee7d5c6` (844 of 3,895 frames per cycle).
Representative RGB frames for the metal door show a visible panel and
robot hand; these warnings do not invalidate expert qualification and require
separate visibility evaluation. The first seven reports predate process
isolation; earlier full-trajectory visibility warnings also remain on
`psx-industrial-004-4f5561b`, `psx-front-20d5505` and
`psx-wooden-009-20d5505`. Worn's superseded pair consistently stopped under
the gate with `safety_stop / tracking_margin`.
At the push stop, orientation error was 0.04510 rad against the controller's
0.04363 rad preventive guard; the material-point error was 0.00515 m against
its 0.00750 m reserve. Contact stayed loaded and valid, local hold-endpoint IK
error was negligible, and no actual forbidden contact was recorded. The final
hold sustained 17.92° and released safely in both cycles. Preparation reports a
121.2° mechanical limit. This is a nominal controller/setup-domain exclusion,
not proof that the asset is defective or that 45° is physically unreachable.
The targeted follow-up identified a premature controller stop: the angular error
grew only about 0.026 degrees/s near that guard. The common guard now forecasts
sustained material-orientation error over the hold duration, preserving the
5-degree validity limit and the local hold-reach check. One diagnostic RTX 4090
diagnostic, `psx-worn-20d5505/20260926T012738.474735Z`, sustained **67.42 degrees**
with all 180 hold samples valid, safe release and no actual forbidden contacts.
The final stop came from the local hold-endpoint position reserve, not the old
half-budget angular comparison. The 17.92-degree pair remains intact as historical
evidence. The subsequent formal pair,
`psx-worn-20d5505/20260926T060507.800125Z`, repeated 67.42 degrees in both cycles,
completed valid holds and releases, passed the corrected RGB-D check on all 4,445
sampled frames per cycle and published `qualified` with zero repeat spread.

The previous sampled geometric RGB-D diagnostic failed for both cycles of
`psx-industrial-004-4f5561b`, `psx-front-20d5505` and
`psx-wooden-009-20d5505`. Targeted diagnosis found two sampling defects: convex
collision points can sit ahead of recessed visible surfaces, and sparse frame
facet centroids miss the visible height band. The diagnostic now intersects the
actual rendered triangles for the same 25 panel and eight contact-surround
locations, and samples the visual frame at seven heights around contact, capped
at 40 points. The existing depth tolerance, occlusion check and minimum counts
(6 panel, 2 surround, 2 frame) are unchanged. Physical contacts, assets, camera
pose and common setup are unchanged.

Three RTX 4090 pose snapshots per affected door all passed the corrected check.
The front and wooden hold/release snapshots reproduce failures with the old
proxy on the same RGB-D, while seven surround points pass with the corrected
surface. Evidence, overlays and the snapshot script are under
`~/.cache/alexdoor-xas/verification/visibility-targeted/20260926-fix/`.
These nine snapshots establish the targeted fix, not visibility over every frame
of a new trajectory. The corrected worn pair subsequently exercised the revised
diagnostic across its complete trajectories. At that point, the remaining 25 doors had no
complete new pair under the revised controller. No corpus split or learning data
was created.

Repeated `void-frame` starts were interrupted by memory exhaustion. The last
interrupted report, `expert/void-frame-studio-animated-classic-door-08bdf51b/20260928T164437.649868Z/report.json`
under the verification cache, records `unresolved / incomplete_execution` with zero
trials; that interruption did not change the historical published record. The user journal records
an OOM kill in the desktop app scope at 2026-09-28 12:45:27 EDT. The user service
cgroup has a 59.70 GiB peak since boot and one OOM kill; its current memory limits
are unlimited. Kernel OOM victim details are not readable by the current account.

A read-only reproduction on the saved scene identifies the oversized allocation
in `front_mesh_points`, introduced by `51742ca`: 6,965 lateral frame coordinates
at seven heights produce 48,755 rays against 73,735 nondegenerate triangles.
The first broadcast array alone requires 53.57 GiB; each following scalar matrix
requires another 26.78 GiB. The 40-point selection happens only after this work.
A child process with a restricted address space reproduced the exact allocation
error at 227.33 MiB peak RSS, without Kit or qualification cycles. The four doors
used for the corrected visibility checks have much simpler frames; their
corresponding arrays require less than 0.1 MiB. Evidence and the safe reproducer
are in `~/.cache/alexdoor-xas/verification/memory-diagnosis/20260928-void-frame/`.

The authorized correction batches rays by triangle count, with at most one
million ray/triangle pairs per batch under the prepared-mesh limit. All candidates,
tolerances, nearest hits, ordering and final point selection are preserved.
Capping candidates before intersection would change the sample selection; a new
spatial index adds unnecessary tolerance and maintenance risk for this cached
calculation. The full saved-scene sample calculation completed in 54.76 seconds
including reference comparisons, with 264.64 MiB process peak RSS. All 360 checked
intersections exactly match the prior implementation, including every selected
frame point; group sizes remain 25/8/40 and the cache is reused. A subprocess
regression also checks hits, misses, edges and nearest surfaces with only 128 MiB
additional virtual memory available. `validation.json` and the validation script
are beside the diagnosis evidence.

Those checks preceded a fresh complete formal pair on `cuda:0`. That
`void-frame` report was `unresolved / inconsistent_or_unresolved_pair`, matching
its then-published summary. Both cycles passed valid hold and release, with
`safety_stop / tracking_margin`, but sustained angles were 66.92° and 34.23°:
the unrounded 32.690094745097674° spread exceeds the 2° limit. That pair has no
`theta_expert_d`. Geometric visibility passed all 4,081 and 1,684 checked frames.

The first push stopped at 65.63° when predicted hold-endpoint position error
reached 4.21 mm against the 2.50 mm reserve, while measured material error was
0.78 mm. The second stopped at 30.97° with predicted material error 8.17 mm
against the 7.50 mm position margin, while measured error was 3.98 mm. Both
stops occurred with valid loaded contact; targeted contact records show no
forbidden contact. The second push began faster and with a larger contact impulse
than the first. This pair did not establish an asset defect. Its evidence remains
under `~/.cache/alexdoor-xas/verification/expert/void-frame-studio-animated-classic-door-08bdf51b/20260928T170246.207228Z/`.
The campaign stopped with 24 doors still awaiting fresh pairs.

A targeted reset diagnosis reproduced the divergence after one complete cycle.
The first 531 recorded physical states match exactly. At 8.85 s, the same contact
geometry produces 0.717 N in the first cycle and 36.844 N after the reused-scene
reset. Instrumented robot/door joint positions, velocities, root poses, command
targets and gravity compensation agree before this impulse; joint states diverge
on the next sample. Two shortened 10 s cycles instead match exactly. This
isolates a dependence on the preceding simulation history, not a different
commanded initial pose. The specific internal PhysX cache responsible is not
identified. NVIDIA documents persistent internal contact state as a
[simulation-resume limitation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html).

A scene-only reconstruction produced two identical physical trajectories at
66.92 degrees (`20260928T172510.288301Z`), but image inspection exposed displaced
robot visual geometry in the second scene. That intermediate pair remains in
the evidence chain; its RGB-D must not be treated as a clean rendering reference.

The qualification runner now restarts the Isaac process before the second cycle,
isolating physics and rendering state together. Both cycles stay in one evidence
directory and only the completed pair publishes. Internal continuation requires
the same saved input records/setup, exactly one matching completed trial and no
existing second attempt; it cannot retry an interrupted second cycle. The controller,
geometry, material properties, prescribed pose and all acceptance thresholds
remain unchanged. This isolates nominal reference trials; it does not establish
history-independent resets for a reused training environment. Diagnostic scripts,
state arrays and original-trace comparisons are preserved under
`~/.cache/alexdoor-xas/verification/reset-diagnosis/20260928-void-frame/`.

The final fresh-process pair (`20260928T173830.453850Z`) is **qualified at
66.91685227985398 degrees**, with zero spread. Both 4,081-step physical traces
match each other and the original first cycle exactly. Every one of the 180 hold
samples is valid in each trial, releases pass, and no forbidden contact is
recorded. Geometric visibility passes all frames; representative hold/release
images show coherent robot geometry after process isolation. The report and
published record agree, with preparation fields unchanged. The seven other newer
references predate process isolation and were not rerun in this targeted repair.
At that point, 24 pairs remained pending; those later completed as reported above.
Corpus selection and learning work remain separate.

The 15 previously deferred rechecks were the three `animated-door-*`,
`modern-door-2fb8d024`,
`door-door-metal-b21ec273`, `door-adf292f437f2`, `interior-wood-d1-32707dc`,
`psx-bathroom-20d5505`, `psx-front-008-ee7d5c6`, and the six
`psx-interior-wood-*` identities (002, 003, 005, 006, 007, 008). All 15 have now
qualified with `--rerun` and fresh-process pairs; their historical evidence is
preserved. The earlier `qualified` status alone did not certify the revised setup.

The pool contains 25 left and seven right identities. Even if every current door
qualifies, at least five additional qualifying right identities are needed for
the planned 12/12 balance. No final corpus, train/development/test split, new asset
downloads or learning data were introduced. Modern-door remains `local_only`.

RGB-D capture is operational and the diagnosed sampling defects are corrected.
Full-trajectory checks of the revised diagnostic remain pending on the first
seven published pairs. It is not an expert-admission gate and does not establish
learned-perception readiness. No
synthetic sweep was repeated; historical synthetic results belong to the earlier
45-degree setup. Runtime qualification remains nominal simulated evidence, not
hardware safety, statistical robustness or a global optimum.

#### Implementation

For each prepared door, run the frozen probe to its valid controlled limit,
including approach, sustained contact, hold, and safe release. Continue past
45 degrees. Record sustained angle, force/contact validity, stop reason, and
joint margins. Repeat once from the same initial conditions in a fresh Isaac process.

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
