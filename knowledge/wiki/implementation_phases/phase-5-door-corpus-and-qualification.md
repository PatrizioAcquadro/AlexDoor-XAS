# Phase 5 — Door Corpus and Qualification

Subphase 5.0 is complete for **32 prepared doors**: 29 redistributable, two
local-only and one private/noncommercial. No door has expert qualification.
Subphase 5.1 selects the final 24-door, 12/4/8 identity split; Phase 6 generates
demonstrations. See [[decisions/visuoproprioceptive-generalization-benchmark|B1 Design]].

## Subphase 5.0 — Intake and Preparation

### Published Asset Contract

Each `assets/doors/b1/<id>/` contains three tracked records:

| Record | Canonical responsibility |
|---|---|
| `candidate.json` | Identity, source URL/selected part, author/attribution, license/scope, source path/fingerprint and relevant exceptions. |
| `recipe.json` | Accepted normalization, component ownership, physics/collider parameters and concise reasons for approximations. |
| `prepared.json` | Relative USD path, dimensions/transforms, handedness, geometry fingerprint, acquired preparation results, mechanical limit and pending expert status. |

Local `source/` retains the used source and necessary dependencies. Local
`prepared/` contains the final USD, materials/textures and front/rear previews.
All paths are relative to the door folder. Explicit `source_dependencies` in
recipes resolve relative to the recipe file. Payloads and previews stay outside Git;
copy the complete folder when moving an asset. License scope lives only in the
candidate record; technical readiness does not grant redistribution rights.

Promotion copies the accepted result into that structure. Numbered `attempts/`
are disposable workspaces, separate from published source/payloads. No published
asset depends on them. Published folders cannot be mutated by preparation commands.

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
visual/collider agreement and collision sweep. The isolated GPU run uses three
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

All entries below retain acquired preparation passes and pending expert status.
The stop is geometric, not `theta_expert_d`. In particular the prison door's
23.1° limit is preserved; 5.1 owns selection and reachable-domain decisions.

| Asset ID | Hinge | Mechanical stop | Scope |
|---|---|---:|---|
| `animated-door-1-88abf40` | right | 90.7° | R |
| `animated-door-2-88abf40` | right | 90.7° | R |
| `animated-door-3-88abf40` | right | 91.8° | R |
| `door-2738468b94d74c5f` | left | 113.1° | R |
| `door-5035d7977155` | left | 194.4° | R |
| `door-adf292f437f2` | left | 176.0° | R |
| `door-door-metal-b21ec273` | right | 59.8° | R |
| `door-prison-metal-old-45306a46` | left | 23.1° | R |
| `door-with-doorframe-c29da62c` | left | 161.2° | L |
| `door-with-frame-2f2f149f` | right | 188.0° | R |
| `interior-wood-d1-32707dc` | left | 54.2° | R |
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
- `scripts/verify_door_preparation.py`: synthetic format/negative/physics checks,
  run only when affected infrastructure changes justify them.
- `DoorInspectionEnv`: robot-independent door measurement, still used by B1.

Historical infrastructure results are recorded at `d12f0b2`; accepted intake
through `9c16e3d`. Preparation-pool cleanup is recorded at `7a1ffb1`.
