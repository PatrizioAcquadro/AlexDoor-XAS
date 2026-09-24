# Phase 5 — Door Corpus and Qualification

Subphase 5.0 is complete for **32 prepared doors**: 29 redistributable, two
local-only and one private/noncommercial. The expert command is implemented.
The common zero-yaw placement and ready posture qualify one right and one
left door on the RTX 4090. The shared infrastructure and left/right pilot are
complete. The subsequent-door campaign is paused after six additional identities
pending loader compatibility and final-hold diagnosis; 22 remain unvisited.
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
existing collision meshes must be requalified, not rebuilt.

The surface trajectory follows the actual leaf colliders at the frozen fraction
and height, retaining tool +X into the panel and +Z upward. Actual frame/handle
bounds replace synthetic dimensions. Clearance remains a conservative diagnostic;
raw contact ownership determines contact validity. Visibility samples actual
surfaces and remains a geometric diagnostic, not learned-perception qualification.

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

### Pilot and Common-Pose Revision

The user-authorized revision keeps **robot and pedestal together at zero yaw**,
with common floor position **X = -0.380 m, Y = 0.300 m** and a revised right-arm
ready/left-arm parked posture. Relative to the first zero-yaw attempt, the common
mount moves 2 cm forward and 7.5 cm left. The parked arm clears the frame; the
ready posture and lateral placement clear the approaching wrist from the robot
body. See [[phase-4-robot-and-task-configuration|Phase 4]] and
`configs/purdue_synthetic_probe.json` for the single effective setup.

Bounded kinematic/collision checks guided this revision; complete GPU cycles on
collected doors establish its result. Contact height/fraction, head pose, gains,
speeds, timing and validity thresholds remain unchanged. No asset geometry or
collision was removed. The earlier shared contact-guard correction still rejects
actual touching/penetration or nonzero normal force, while retaining separated,
zero-force PhysX candidates as diagnostics.

The final setup qualifies both agreed substitutes on 2026-09-24:

| Door | Hinge | Trial 1 | Trial 2 | `theta_expert_d` | Consistent limit |
|---|---|---:|---:|---:|---|
| `door-door-metal-b21ec273` | right | 59.33077° | 59.33077° | 59.33077° | Mechanical stop |
| `door-adf292f437f2` | left | 46.36825° | 46.36825° | 46.36825° | Tracking margin |

Both pairs complete valid final holding and release. In these four trials the
final-hold angle equals the maximum sustained angle; an independent trace audit
reproduces both measurements and finds no actual forbidden contact. Identical
paired values establish the prescribed repeat check, not a statistical robustness
claim. The left reference is above 45 degrees with a modest 1.37-degree margin.

The compact audit and links to complete paired evidence are at
`~/.cache/alexdoor-xas/verification/common-approach-pilot-20260924/report.json`.
Each pair retains its effective setup, input records, raw contacts, time traces
and RGB-D. Representative images were inspected. All preparation fields and
source/prepared assets are preserved; only expert summaries change.

Earlier attempts remain linked through `previous_evidence`. The original left
candidate (`door-2738468b94d74c5f`) has a deep frame that intersects the pedestal
at the revised placement; this is setup exclusion, not an invalid asset. The
original right candidate (`modern-door-2fb8d024`, still `local_only`) retains its
older unresolved pair: one of two trials failed final holding. That result used
the earlier zero-yaw setup and has not been rechecked with the final placement;
it is not a current expert reference. The substitute left's earlier valid
44.23-degree pair is also superseded, not combined with the new pair.

The following records describe the paused campaign before corrective reruns. The subsequent-door campaign
attempted six more identities on 2026-09-24 and was then paused at user request.
The following paths are relative to
`~/.cache/alexdoor-xas/verification/expert/<asset-id>/`; each contains the
complete `report.json`, captured input records and effective `setup.json`.

| Door | Status at campaign pause | `theta_expert_d` | Evidence directory | Finding |
|---|---|---:|---|---|
| `animated-door-1-88abf40` | `unresolved` | — | `20260924T212508.154992Z` | Two cycles reached 57.03° sustained, but final hold was invalid (`tracking_margin`); release completed. |
| `animated-door-2-88abf40` | `unresolved` | — | `20260924T213207.125877Z` | Two cycles reached 56.95° sustained, but final hold was invalid (`tracking_margin`); release completed. |
| `door-with-frame-2f2f149f` | `invalid_asset` | — | `20260924T213853.832105Z` | Loader reported missing leaf or frame collision geometry before cycles. This published classification needs loader compatibility diagnosis before attributing the fault to the asset. |
| `psx-front-005-ee7d5c6` | `unresolved` | — | `20260924T213927.306714Z` | Prescribed contact point found no collidable leaf surface before cycles; cause is unconfirmed. |
| `door-5035d7977155` | `out_of_domain` | — | `20260924T213949.754385Z` | Closed frame intersects the pedestal lower base; no cycles or expert angle. |
| `door-prison-metal-old-45306a46` | `out_of_domain` | 22.63268° | `20260924T214021.597645Z` | Two valid cycles, zero spread, mechanical stop, valid final hold and release; below 45°. |

Representative approach, push, hold and release images were reviewed for the
three executed pairs, along with focused traces and contacts. The animated pairs
had valid sustained push windows but insufficient valid final holding; their
reported push maxima are not expert angles. No actual forbidden contact was found
in those reviewed traces. The prison-door pair had valid holding, release and no
actual forbidden contact. The pedestal exclusion has an initial-scene image and
composed-collider intersection evidence. The two pre-cycle errors have reports
and error traces; they are not evidence of an expert limit. All attempt evidence
is retained outside learning datasets.

At campaign pause the 32 prepared doors had **two qualified, three out of domain,
four unresolved, one published `invalid_asset`, and 22 `not_run`**. By handedness:
left 1 qualified / 3 out of domain / 21 not run; right 1 qualified / 4 unresolved /
1 published `invalid_asset` / 1 not run. The right-door balance still needs at
least five additional qualifiable identities. Diagnose the loader compatibility
problem and the repeated invalid final hold before resuming; no further door
commands are part of this paused campaign. Any future common setup change
requires fresh complete pairs for affected references; compare archived
`setup.json` before reusing an earlier result.

The prison-door stop was traced to false convex volume at the frame's lower
hinge corner. At 23.3 degrees the colliders intersected while the actual visual
surface sections remained separate. Repartitioning the same frame at its measured
sill/header boundaries (Z = 0.041961/2.045949 m) moves the geometric stop to 47.0
degrees, independently reached by the isolated GPU physics check. No source
surface, pivot, mass, material, unlatched state or other collision was changed.
The former payload/records and the staged static/physics evidence are retained at
`~/.cache/alexdoor-xas/verification/qualification-repair-20260924/prison/`.
The former 22.63268-degree expert reference is superseded; a new complete pair
is required on the corrected collision geometry.

Reviewed RGB shows the pushing hand and panel, but geometric visibility does
not pass throughout the qualified pilot or animated-door trials, including
insufficient sampled frame points. The prison-door pair passes this geometric
diagnostic. Neither result establishes learned-perception readiness. No camera
retuning, learning data or split was introduced. No synthetic sweep was repeated;
the historical synthetic results belong to the original 45-degree setup.

The pool contains 25 left and seven right doors. At least five additional
qualifiable right identities are needed for the final 12/12 balance, potentially
more after actual exclusions. The final corpus selection and split remain pending.

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
