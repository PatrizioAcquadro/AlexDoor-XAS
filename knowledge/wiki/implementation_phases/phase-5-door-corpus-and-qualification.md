# Phase 5 — Door Corpus and Qualification

Subphase 5.0 is complete for **32 prepared doors**: 29 redistributable, two
local-only and one private/noncommercial. The shared expert command and left/right
pilot are implemented and GPU-verified. Corrective qualification resolved the
paused cohort, and routine qualification has now processed every remaining
prepared door on the same RTX 4090 setup. The pool has **15 qualified, 15 out
of domain, two unresolved, and no unvisited doors**. Subphase 5.1 still owns
the final 24-door, 12/4/8 identity split; Phase 6 generates demonstrations.
See [[decisions/visuoproprioceptive-generalization-benchmark|B1 Design]].

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
and height, retaining tool +X into the panel and +Z upward. Actual frame/handle
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
strip is not permitted. The other seven corrective-cohort footprints pass.

### Procedure for Subsequent Doors

1. Select one prepared identity with pending expert status and run the command.
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
**X = -0.380 m, Y = 0.300 m**, with the shared right-arm ready and left-arm parked
posture in `configs/purdue_synthetic_probe.json`. Contact remains at 0.40 of leaf
width from the actual hinge and 1.00 m height. Head pose, controller gains, push
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
  preflight identifies this incompatible prescribed pose before movement. No point
  relocation or strip removal is permitted, so the geometric exclusion has no
  expert angle.
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

The first pilot references (right 59.33077°, left 46.36825°) and the six paused
campaign outcomes remain historical evidence. The table below contains the fresh
corrective results from the same common implementation. Each dynamic result uses
exactly two complete GPU cycles from reset; no repetitions are mixed across runs.
Evidence directories are relative to
`~/.cache/alexdoor-xas/verification/expert/<asset-id>/`.

| Door | Hinge | Current status | `theta_expert_d` | Consistent limit | Evidence directory |
|---|---|---|---:|---|---|
| `modern-door-2fb8d024` | right | `qualified` | 49.87672° | `tracking_margin` | `20260924T223701.060503Z` |
| `door-adf292f437f2` | left | `qualified` | 46.36825° | `tracking_margin` | `20260924T224236.598587Z` |
| `animated-door-1-88abf40` | right | `qualified` | 58.58486° | `tracking_margin` | `20260924T224851.407233Z` |
| `animated-door-2-88abf40` | right | `qualified` | 58.78583° | `tracking_margin` | `20260924T225459.300835Z` |
| `door-with-frame-2f2f149f` | right | `out_of_domain` | 42.58362° | `tracking_margin` | `20260924T231212.802123Z` |
| `psx-front-005-ee7d5c6` | right | `out_of_domain` | — | `prescribed_footprint_obstructed` | `20260924T232605.585663Z` |
| `door-prison-metal-old-45306a46` | left | `out_of_domain` | 44.91557° | `tracking_margin` | `20260924T232615.566689Z` |
| `door-door-metal-b21ec273` | right | `qualified` | 59.33077° | `mechanical_stop` | `20260924T233327.320423Z` |

The prison-door expert result is **44.91557 degrees**, 0.08443 degrees below the
inclusive 45-degree admission threshold. It stops at the common measured 2.5 mm
tracking guard, not at the repaired 47-degree mechanical stop. A local IK solve
can reach the latter pose; this reference does not prove global robot
unreachability. A later transient angle above 45 degrees is not a valid sustained
reference. Likewise, the frame-door result is the common baseline's 42.58362-degree
limit, not its isolated 188-degree geometric range. Neither value is rounded up,
and no door-specific controller changes are used to obtain admission.

An independent audit reconstructs contact loading from raw substeps, verifies
sustained/final-hold windows and release separation, and checks for actual
forbidden contacts. Representative hold/release images were inspected. The audit,
frame-interference proof, and archived original/repaired payloads are under
`~/.cache/alexdoor-xas/verification/qualification-repair-20260924/`.
The consolidated trace/contact review is `expert-audit.json` with its `audit.py`.
Failed and intermediate pairs remain in their original directories, connected by
`previous_evidence`; the old 22.63268-degree prison reference is superseded.
Identical paired values satisfy the prescribed repeat check, not statistical
robustness or global optimality. Modern-door remains `local_only`.

Two earlier setup exclusions remain valid: `door-2738468b94d74c5f` and
`door-5035d7977155` have actual frame surfaces penetrating the fixed pedestal.
The review checked source-derived visual triangles inside the pedestal, not just
convex envelopes. Recovering them would require a new common placement or an
asset geometry change. They retain `out_of_domain` with no expert angle; another
unsafe dynamic run is unnecessary for that unchanged initial geometry.

### Routine Qualification of the Remaining Prepared Doors

The routine campaign started from `main` at `1b7b515b` with 22 `not_run`
records and no working-tree changes. Each dynamic door used one complete
two-cycle GPU invocation with the frozen common probe. The following table
records the published outcomes; directory names are relative to
`~/.cache/alexdoor-xas/verification/expert/<asset-id>/`. Earlier results above
were not rerun or changed.

| Door | Hinge | Status | `theta_expert_d` | Limiting cause or exclusion | Evidence directory |
|---|---|---|---:|---|---|
| `animated-door-3-88abf40` | right | `qualified` | 59.13405° | `tracking_margin` | `20260924T235228.554584Z` |
| `door-with-doorframe-c29da62c` | left | `out_of_domain` | 38.95732° | `declining_contact_load` | `20260924T235916.871403Z` |
| `interior-wood-d1-32707dc` | left | `qualified` | 48.05294° | `tracking_margin` | `20260925T000522.606847Z` |
| `psx-bathroom-20d5505` | left | `qualified` | 54.25710° | `tracking_margin` | `20260925T001226.351545Z` |
| `psx-front-001-ee7d5c6` | left | `unresolved` | — | Actual forbidden jaw/panel contact; invalid cycles | `20260925T001844.264863Z` |
| `psx-front-002-ee7d5c6` | left | `out_of_domain` | — | Initial frame/pedestal intersection | `20260925T002719.102141Z` |
| `psx-front-008-ee7d5c6` | left | `qualified` | 52.16840° | `tracking_margin` | `20260925T002809.838259Z` |
| `psx-front-20d5505` | left | `out_of_domain` | — | Initial frame/pedestal intersection | `20260925T003446.883037Z` |
| `psx-industrial-001-4f5561b` | left | `out_of_domain` | 35.08331° | `tracking_margin` | `20260925T003547.797551Z` |
| `psx-industrial-002-4f5561b` | left | `out_of_domain` | 36.85799° | `tracking_margin` | `20260925T004116.689550Z` |
| `psx-industrial-003-4f5561b` | left | `out_of_domain` | 37.98466° | `tracking_margin` | `20260925T004559.838766Z` |
| `psx-industrial-004-4f5561b` | left | `out_of_domain` | 37.80632° | `tracking_margin` | `20260925T005103.214811Z` |
| `psx-interior-wood-002-02e442c` | left | `qualified` | 54.23718° | `tracking_margin` | `20260925T005640.609678Z` |
| `psx-interior-wood-003-02e442c` | left | `qualified` | 54.25234° | `tracking_margin` | `20260925T010332.710364Z` |
| `psx-interior-wood-005-02e442c` | left | `qualified` | 54.25481° | `tracking_margin` | `20260925T011011.206935Z` |
| `psx-interior-wood-006-02e442c` | left | `qualified` | 54.22672° | `tracking_margin` | `20260925T011727.292206Z` |
| `psx-interior-wood-007-02e442c` | left | `qualified` | 54.24174° | `tracking_margin` | `20260925T012549.709852Z` |
| `psx-interior-wood-008-02e442c` | left | `qualified` | 54.22830° | `tracking_margin` | `20260925T013212.411223Z` |
| `psx-wooden-001-20d5505` | left | `out_of_domain` | — | Initial frame/pedestal intersection | `20260925T013917.941867Z` |
| `psx-wooden-009-20d5505` | left | `out_of_domain` | — | Initial frame/pedestal intersection | `20260925T014002.639890Z` |
| `psx-worn-20d5505` | left | `out_of_domain` | — | Initial frame/pedestal intersection | `20260925T014035.250761Z` |
| `void-frame-studio-animated-classic-door-08bdf51b` | left | `unresolved` | — | `lost_contact` in both invalid cycles | `20260925T014119.153120Z` |

All ten new qualified doors and five new below-threshold references have
two valid cycles, matching limiting causes, zero sustained-angle spread,
valid final holding and release. Their `theta_expert_d` values are the lower
sustained angles, not transient maxima. The five new frame/pedestal exclusions
have composed-collider penetration evidence at the fixed initial setup and no
expert angle; their nonzero command exits do not indicate infrastructure failure.
The two unresolved doors retain their complete failed pairs. `psx-front-001`
has positive-force forbidden jaw/panel contact in both cycles. `void-frame`
loses contact after only a short push, with no final hold; the second cycle has
no sustained angle. Neither result establishes an asset defect or admission.
These cases need targeted diagnosis before any fresh complete pair.

The 32-door pool now contains **10 qualified, 13 out of domain and two
unresolved left doors**, plus **five qualified and two out-of-domain right
doors**. All 32 have an outcome. The right side still needs at least seven
additional qualifiable identities to reach the planned 12/12 balance; no final
corpus or train/development/test split has been selected. No new assets were
downloaded, and qualification evidence remains outside learning datasets.

RGB-D capture is operational, but geometric visibility fails in several qualified
trials, including insufficient sampled frame points. It is a diagnostic, not an
expert-admission gate, and does not establish learned-perception readiness. No
camera retuning, learning data or split was introduced. No synthetic sweep was
repeated; historical synthetic results belong to the earlier 45-degree setup.

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
corpus size nor repeated trials prove broad coverage. Never retune Phase 4 from
collected assets.

## Maintained Entry Points

- `scripts/prepare_doors.py` and `qualification/`: shared intake and promotion.
- `scripts/qualify_door.py`: frozen real-door probe, paired decision and per-door result.
- `scripts/verify_door_preparation.py`: synthetic format/negative/physics checks,
  run only when affected infrastructure changes justify them.
- `DoorInspectionEnv`: robot-independent door measurement, still used by B1.

Historical infrastructure results are recorded at `d12f0b2`; accepted intake
through `9c16e3d`. Preparation-pool cleanup is recorded at `7a1ffb1`.
