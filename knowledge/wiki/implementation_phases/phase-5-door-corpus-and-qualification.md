# Phase 5 — Door Corpus and Qualification

> Subphase 5.0 infrastructure is implemented and verified on the RTX 4090.
> Fourteen real doors pass preparation and are ready for 5.1: eleven redistributable, two local-only and one private/noncommercial. Expert qualification remains planned.

## Objective

Produce the definitive licensed B1 door corpus, per-door expert references, and
12/4/8 identity split. The corpus is the set of usable environments; training
demonstrations are generated later in Phase 6. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 5.0 — Preparation Infrastructure and Incremental Door Intake

#### Implementation

`scripts/prepare_doors.py` provides one local workflow: `review`, `inspect`,
`normalize`, `static`, `physics`, `preview`, and `promote`. Reusable implementations
live under `src/alexdoor_xas/qualification/`. The retained legacy commands do not
admit B1 assets. No candidate-specific normalization script is needed.

**Preparation.** Inspection inventories source/dependency checksums, connected mesh
components, materials, textures and complexity. Each inspection or normalization
creates a new numbered attempt and copies its input files. Sources and accepted
attempts are preserved. Geometry fingerprints flag possible duplicates; local
review must resolve them, and identical source payloads cannot be promoted twice.

**Welded static surfaces.** Connectivity is not physical ownership: a graphics
asset can join a modeled U-shaped frame and leaf at their common contour. Inspect
that contour and the cross-section before declaring the frame absent. Optional
`component_splits` selects whole faces with all vertices inside reviewed `bounds`
(`x`, `y`, `z` intervals in inspected coordinates). The remainder retains its
component index; the selected surface is appended. This preserves source positions,
UVs and materials without importing a frame. It does not authorize inventing a
frame where no frame geometry is modeled.

For a front-only leaf relief, optional `shell_backings` specifies `component`,
`axis`, an exterior rear `plane` in inspected coordinates, and `review`. It preserves
the front and adds a flat projected rear plus boundary walls. Thickness and rear
appearance are explicit inferred preparation choices; rear UVs repeat the front.
Folded/opposing surfaces and non-manifold or non-closed results are rejected.
Splits run before backings, then the usual transforms, collider generation and
gates run. `components` uses the resulting indices; inspection/fingerprints still
refer to the preserved original. This is opt-in preparation, not an automatic
repair or evidence of the author's intended working mechanism.

**Reviewed frame adaptation.** For an explicitly approved composite assembly,
`frame_fits` can adapt an existing licensed frame to a differently sized leaf.
Each record identifies a Frame `component`, inspected-coordinate `axes` mapping
`[[source_lo, source_hi], [target_lo, target_hi]]`, and a provenance/geometry
`review`. Vertices outside the source opening interval translate with its edge;
only connecting spans resize, preserving jamb/header profile thickness and depth
on untouched axes. Source faces and UVs remain; the leaf cannot use this operation.
This is an authored assembly change, not recovered source dimensions or a way to
relax motion/contact checks. Keep the frame source and adaptation in the recipe.

A JSON recipe specifies `handedness`, positive uniform `scale`, proper `rotation`,
`translation_m`, `opening_center_source`, `hinge_m`, `dimensions_m`, and the complete
`components` assignment to `Frame`, `Panel`, and optional `Handle`. It also records
`modifications`. Rotation/scale/translation act on the **inspected coordinates**;
USD/FBX pass through glTF Y-up meters. The opening center maps to the floor origin.
Optional `moving_translation_m` shifts Panel and Handle together relative to Frame
when the source's closed leaf is slightly miscentered. It moves their visuals and
colliders together; the recorded hinge coordinates describe the corrected pose.

For a graphics asset with no operating clearance, `moving_scale` may uniformly
reduce Panel and Handle together by at most 2% about `moving_scale_center_m`
(canonical coordinates before `moving_translation_m`). Default is 1. A nontrivial
repair requires `clearance_review`, documenting measured interference and final
gaps; update leaf dimensions/inertia consistently. This bounded fitting repair
preserves proportions, visuals/collider alignment and attached hardware. It is
not a reconstruction of manufacturer dimensions. Infer a plausible hinge from
the opening face and jamb edge when no hinge is modeled; record the assumption.
A mid-thickness pivot is not a mandatory default. Larger remodeling needs a
separate scope decision; missing hardware measurements alone do not reject an asset.
The 2% allowance replaces the initial conservative 1% fitting bound after the
Theocritus source showed a leaf wider than the jamb passage even at 1%. It is
a permitted geometry edit, not a relaxed collision/physics acceptance tolerance.
Use the smallest justified repair and check the sweep, not only the closed pose.
Choose the scale center to preserve relevant hinge-face alignment; distribute
clearance with a measured rigid translation if the thick latch edge needs it.
The hinge-edge guard retains its 10 mm tolerance against either the fitted edge
or the corresponding edge before clearance scaling, so a fixed source hinge
is not incorrectly rejected just because the leaf was fitted around it.

An overlapping slab need not fit *inside* the jamb aperture: a reviewed mounting
may place its closed back face just in front of the jamb face, with an inferred
back-face hinge, so the slab covers the opening. Use the existing rigid
`moving_translation_m`, retaining all visible geometry and collision. Record
that mounting as a simulation inference, verify the actual clear passage using
`clear_aperture_m`, inspect both sides, and check the complete sweep/runtime.
Do not assume width excess alone demands shrinking the slab. The common 2%
moving-fit limit remains unchanged; mounting choice never waives collision gates.
When two distinct moving hinge barrels visibly establish an axis, optional
`hinge_axis_components` and `hinge_axis_review` check their compact horizontal
bounds, centers and vertical separation instead of comparing the axis to an
outer barrel edge. The axis must remain on the declared hinge side and within
10 mm of each barrel center. This handles a barrel radius larger than the edge
guard's proxy while leaving that guard unchanged for other doors.

**Common task state: closed and already unlatched.** The robot pushes the leaf; it
does not operate a lock. Source latch/lock bolts are not a reason to reject an
otherwise useful door. Optional `unlatched_components` lists those separate moving
locking parts; `unlatched_review` identifies them and records the reason. Their
visuals remain attached but no collider is generated. This explicit abstraction
avoids modeling lock internals or retracting the visual mesh. It applies consistently
to all candidates, not only this door. It does not represent a functioning lock.

The selection cannot include measured leaf components, any frame component, or
handles. All other components remain collidable. USD colliders retain their source
component IDs, and static checks require exact coverage of the nonexcluded parts.
Bounds checks compare physical geometry against the corresponding visual subset.
The visible bolt can remain extended; its locking function is deliberately absent.
Future tasks involving handle/lock operation must use a different preparation.

**Internal hinge contacts.** Graphics assets can use overlapping solid pins and
barrels without modeling bearing bores. The ideal revolute joint already provides
that bearing constraint; these internal intersections do not by themselves reject
a push-door asset. Optional `hinge_contact_exclusions` lists explicit
`[moving_component, fixed_component]` pairs with a `hinge_contact_review` identifying
the interfaces and approximation. Only non-leaf, non-latch Panel hardware paired
with Frame hardware is allowed; all participating collider vertices must lie
within 10 cm of the hinge axis and each must retain its own component ID.
The exclusions become USD `FilteredPairsAPI` relationships between individual
colliders and are applied identically in the geometric sweep. Static verification
requires exactly the reviewed relationships. All hardware still collides with the
robot and unfiltered geometry; leaf/frame and handle/frame contacts remain active.
Never disable whole-body self-collision to fix a bearing. Check the actual pin axis,
opening side and collider inflation before adding an interface exception. Detailed
bearing mechanics, wear and lock operation are outside this preparation model.
Optional `colliders` maps component indices (JSON strings) to `auto`, `convexHull`,
or `convexDecomposition`. Default `auto` uses the installed PhysX cooker, baking
individual hulls so clearance checks and simulation use the same collision shapes.
Collider-only vertices are welded and decompositions use shrink wrapping.

Connected/material-separated meshes are not necessarily separate physical solids.
Optional `collider_groups` contains records with `components` (indices within one
rigid body), `approximation` (`convexHull` or `convexDecomposition`), and `review`.
Cook those surfaces together when they describe one solid; retain their individual
visuals and source IDs on the shared collider. This avoids artificial thickness
from independently cooking zero-volume skins. Groups cannot overlap, include
excluded latch parts, cross body boundaries or duplicate individual collider
recipes. A leaf's common convex envelope may fill shallow decorative grooves;
document that approximation. Never group the frame with the leaf or fill a frame
opening. Static coverage accepts both legacy single IDs and grouped ID arrays.

For doors with projecting hardware, optional `leaf_components` selects a nonempty
subset of `Panel` for leaf dimensions, nominal inertia and center of mass. All
remaining panel components stay attached and collidable. Visual nodes retain their
source component indices so static checks measure the selected leaf in USD.
The hinge-edge check uses the complete moving assembly, including hinge fittings.

For rebated frames, optional `clear_aperture_m` is `[ [y_min, z_min], [y_max, z_max] ]`
in canonical meters. It describes the actual passage through the full frame depth,
not the larger leaf envelope. It must be centered, inside that envelope and cover
at least 90% of its width and height; this conservative recipe guard prevents
shrinking a declared aperture to conceal an obstruction. Unsupported apertures
need review, not automatic asset rejection.

A component collider can instead specify `{"partitions": {"x": [...], "z": [...]}}`.
The planes clip the original triangle surfaces into cells before convex cooking,
retaining narrow rebates without voxel expansion. Coordinates are canonical meters;
choose cuts at the actual concavities and inspect the result. Cuts alone do not
prove surface fidelity. Coplanar faces belong only to their material side; vertices
within 1 micrometer of a cut are snapped for USD floating-point precision. At most
32 cuts per axis and 256 occupied cells are supported. Additional cuts can avoid
elongated hulls; they do not modify the visual source.

Supported inputs are `.usd`, `.usda`, `.usdc`, `.usdz`, `.glb`, `.gltf`, `.fbx`,
and `.obj`. USD cubes are tessellated in a derived layer; other analytic primitive
types remain unresolved. Blend requires prior export. Multi-file inputs retain
buffers, material files and textures. FBX recipes must explicitly list
`source_dependencies`, relative to the source file; use `[]` only after confirming
textures are embedded. `inspect --dependency PATH` inventories additional sidecars.
The normalized USD bundle resolves its dependencies locally. A missing dependency
or conversion limitation is unresolved rather than evidence of an unsuitable door.

The canonical default prim is `/Door`, with `Frame`, `Panel`, optional `Handle`,
`Hinge`, and fixed attachments. It uses meters, Z-up, zero closed angle and positive
opening for either handedness without reflecting geometry. The report records
opening/hinge/panel-center transforms. Nominal panel mass is 25 kg, hinge damping
4 Nm s/rad, friction 0.5 and restitution zero. Panel inertia follows its cuboid
reference; handle/frame inertia uses their measured bounds and 0.5/50 kg nominal
masses. These are benchmark approximations, not reconstructed hardware dynamics.

**Checks.** Static validation measures dependencies, geometry, visual/collision
bounds, nominal mass/inertia/materials, articulation and enabled collisions. The
frame must leave the rectangular opening clear. Panel and attached handle hulls
are swept against frame hulls at 0.1-degree increments; the first intersection
minus 0.2 degrees defines the upper joint limit. Initial intersection fails; no
demonstrated stop within 270 degrees remains unresolved. SAT intersection uses a
1-micrometer numerical tolerance, separate from runtime contact tolerances.
Failed mechanical checks retain `collision.json`, `collision_components.json` and
`source_intersections.json`. The last file witnesses proper edge/triangle crossings
on the original transformed surfaces, independent of collider approximations.
Crossings require an ownership review; an empty list does not prove clearance
because containment and coplanar contact are excluded. No arbitrary 90-degree
limit or qualification-angle threshold is inserted.

The GPU gate uses an isolated door at 120 Hz physics / 60 Hz commands. It measures
three closed resets, passive drift, frame pose, hinge-anchor stability, finite
states, raw contacts/separations and torque-controlled opening to the measured
stop. Tolerances retain Phase 4 reset angle 0.1 degrees, reset speed 0.01 rad/s,
passive drift 0.25 degrees, frame translation 0.1 mm and stop agreement 1 degree.
Additional bounds are 0.1-degree frame rotation, 1 mm hinge-anchor error and 2 mm
penetration, matching the authored 2 mm contact-offset scale. Contact forces and
separations are recorded, but a contact alone does not fail readiness: rubbing or
bearing contact can be harmless. Opening progress to the geometric limit, reset,
stability and penetration determine the functional result. Contact buffer overflow
or unavailable measurements cannot pass. Broad-phase bounds prune separated hull
pairs during the geometric sweep without changing its collision criterion.

Each command reports `pass`, `fail` or `unresolved` with a reason/category.
`preview` captures front/rear RGB through the installed Isaac Lab camera renderer;
its capture success still requires human/assistant appearance review. `promote`
requires matching source review, local license/dependency/duplicate/visual review,
and current passing normalization/static/physics files. It writes only
`ready_for_5.1`, never expert qualification or corpus completion. Accepted attempts
cannot be overwritten by rerunning a command.

**License scope.** `distribution_scope` is `redistributable` for CC0/CC BY 4.0
sources and dependencies, or `local_only` for reviewed Sketchfab Free Standard
sources and their packaged dependencies. The source-specific, user-approved
`private_noncommercial` scope additionally admits CC BY-NC-ND 4.0 for private,
noncommercial preparation and tests. It does not permit commercial use or sharing
adapted material. Existing CC records without the field remain redistributable.
Local-only means processing in the downloading licensee's authorized workspace,
not permission to share files with other users. Source payloads, normalized
geometry and textures from restricted doors cannot enter shared asset packages.
Their rendered datasets and other derived artifacts require a separate rights
review before release. The preparation and expert criteria do not change;
`prepared.json` records the scope beside technical readiness. The legacy Phase 4
license gate remains CC-only.
Free Standard alone does not clear extra source terms or third-party dependencies
for B1 training and evaluation. An explicit failed source review returns `fail`;
missing evidence remains `unresolved`. Stop before `inspect` while source rights
remain unresolved or fail. Sketchfab `NoAI` restricts use with programs designed
to generate new content; `CreatedWithAI` separately labels how an asset was made.
Review the actual program's output and any added author terms before deciding
whether `NoAI` applies. A diffusion architecture alone does not settle that
question.

**Usage.** Keep reviewed `candidate.json`, `recipe.json` and license evidence under
`assets/doors/b1/<asset-id>/` in Git. Generated `attempts/` and `prepared.json` are
ignored. `review` requires asset/source IDs, source URL, author, license and its
asset-specific evidence, attribution, retrieval date, door type, selected format
(including the dot), and passing license-scope/custom-terms/duplicate reviews.
Unknown remote geometry may remain unset. Before promotion, record passing
`local_dependency_review`, `local_duplicate_review`, `local_visual_review` and the
`reviewed_source_sha256` from `inspect.json`; explain any fingerprint collision in
`duplicate_resolution`. License evidence must cover every dependency under its
declared distribution scope.
For a multi-door source pack, identify each selected assembly with `source_part`.
The duplicate-source check permits the same source URL/UID only when both records
name distinct, nonempty parts; the local geometry fingerprints must still be
distinct. A whole-pack candidate or repeated part remains a duplicate.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py review \
  --candidate assets/doors/b1/<id>/candidate.json
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py inspect \
  --asset-id <id> --source /path/to/download/door.glb --viz none --device cuda:0
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/prepare_doors.py normalize \
  --asset-id <id> --source /path/to/download/door.glb \
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

Use the **normalization** attempt printed by the command for subsequent gates;
inspection has its own separate attempt. Unpack downloaded archives manually and
preserve their relative dependency layout. No network collection/downloader runs.

**Sequential intake starts after readiness.** First provide the user with
[Sketchfab](https://sketchfab.com/features/free-3d-models) and the admission filters
below, including license, downloadable format, full-size single-leaf geometry,
separable frame/panel, dimensions, triangle/texture limits and duplicate exclusions.
The user chooses a candidate URL. For each candidate, repeat the download/source
link and specific warnings before the user downloads anything:


1. The model reviews source, license scope, format, visible geometry, available
   dimensions/complexity, and duplicates. Reject clearly unsuitable candidates
   before download; identify properties that require the local payload.
2. If the remote review is acceptable, the user downloads the asset manually
   and supplies its local path. No automatic download or broad asset search is needed.
3. The model uses the existing preparation commands and an asset recipe to
   normalize the payload and run static/physics checks. Inspect unknown properties
   locally before accepting them. Report the result and concrete rejection or
   unresolved reason.
4. Mark each technically valid door ready for Subphase 5.1. As an initial working
   batch, prepare **3–4 real doors in total** with distinct geometry and, where
   available, both original handednesses before the first robot checks. This is
   a workflow recommendation, not a new admission gate. Do not wait for all 24:
   early 5.1 checks can expose shared issues before more intake work accumulates.

Do not write a new standalone normalization pipeline for every door or modify
robot/contact rules to make a candidate pass. Fix genuine shared-tool defects
in the shared infrastructure and recheck affected results. Ordinary differences
in scale, source prims, pivot, or allowed separation belong in the asset recipe.

**Proportionate per-door workflow.** Use standard preparation, one visual review
and one short isolated GPU functional run. That run checks three resets with brief
passive holds and one controlled opening. Diagnose further only when the result
shows a problem relevant to robot contact, motion or stability. Ordinary source
modeling details outside the task are handled through explicit common recipes;
they are not grounds for repeated geometric audits or automatic rejection. Repeat
only checks affected by an actual correction. Do not rerun the format matrix,
unrelated software suite or synthetic regressions for routine intake. Separate
infrastructure development cost from per-door processing.

**Next-candidate operator handoff.** Follow the commands above for one user-supplied
URL/local payload at a time; remain within 5.0 unless 5.1 is explicitly requested.

1. Review the individual source page and dependency license scope. Return its
   link, `DOWNLOAD` or `REJECT`, recommended available format, known facts and
   local unknowns. If source/license evidence is insufficient, report `unresolved`
   and what evidence is missing. Missing online dimensions are not a rejection.
   If the payload is already downloaded, continue locally without another download.
2. Create a distinct candidate record and run `inspect`. Use its component
   inventory and geometry to identify frame, leaf, attached hardware and hinge.
   Inspect original views only if ownership or orientation is ambiguous. A welded
   frame/leaf requires contour review before rejection; use the documented surface
   split/backing operations only when existing geometry supports that interpretation.
3. Write the smallest adequate recipe and run `normalize`. The source of recipe
   coordinates is the converted inventory (USD/FBX conversion uses glTF Y-up
   meters), not assumed raw-file units. Determine scale, handedness, pivot and
   component indices for this asset. Before shrinking a wider-than-aperture slab,
   review whether an explicit surface mount is appropriate; width excess alone
   does not establish unusability. Use `leaf_components`, `clear_aperture_m`
   and collider partitions only where its geometry needs them. Explicitly identify
   any latch/lock bolt collision exclusions through `unlatched_components` and
   `unlatched_review`; retain leaf, frame and handle collisions.
4. On the normalization attempt, run `static`, then `preview`; actually inspect
   the front/rear images for orientation, materials, alignment and missing parts.
   A successful render alone is not a visual pass. Run `physics` on RTX 4090
   (`--device cuda:0`), requesting runtime/cache access if the sandbox blocks it.
5. Complete the source, dependency, duplicate and visual reviews with this asset's
   evidence. Promote only after the required results pass. Record a concise
   `preparation-review.json` with dimensions, handedness, modifications, outcome,
   evidence paths and remaining limits. Preserve source and prior attempts;
   never rerun commands into an accepted attempt.

Use `assets/doors/b1/door-with-frame-2f2f149f/` as the worked example. Its accepted
attempt is `000007`; older failures are superseded diagnostics. Reuse the record
structure and common task model, **not** its scale, component IDs, partition cuts,
188-degree limit, license evidence, checksums or passing review values.

When a gate fails, first read its reason and implicated component pair. Distinguish
an asset defect from incorrect ownership/pivot/collider approximation or a converter
limitation. A latch intersecting its strike plate is handled by the reviewed
closed-unlatched rule, not by discarding the door. A collider filling a frame rebate
calls for a geometry-preserving recipe correction, not removal of frame collision.
Do not blindly vary parameters, force a 90-degree stop, disable required collisions
or weaken thresholds to obtain a pass. Fix routine preparation issues, then repeat
only affected checks. If a task-relevant obstruction or unsupported conversion
remains, report the concrete blocker as `fail` or `unresolved`; do not start a broad
audit. Change shared code only for a demonstrated tool defect, with a focused
regression. The 45-degree criterion and frozen robot/expert run belong to 5.1.

Before stopping on planar-skin or immediate-opening failures, check whether mesh
fragments actually form one solid, whether the inferred hinge lies on the opening
face, and whether the graphics model lacks operating clearance. Use reviewed
collider groups and bounded uniform clearance repair where justified. Requiring
measured real hinge hardware for every graphics asset is outside 5.0's scope.
When visible hinge knuckles exist, infer the axis and swing side from them before
choosing canonical handedness. A proper 180-degree scene rotation may be needed
to make opening positive toward +X; it does not mirror or create a new door.
Do not assume that removing closed-pose overlap proves clearance during opening.

**First real door (2026-09-23).** The user supplied
[Door with frame by witnessk](https://sketchfab.com/3d-models/door-with-frame-2f2f149f3ec44d658a02c1f924dfa449)
and `~/Downloads/Door_with_frame.usdz`. Official source and embedded metadata agree
on CC BY 4.0. The asset has 10,154 triangles, 22 components and two embedded 2K
textures. Uniform scale 0.72 gives a 0.891 × 2.097 × 0.042 m right-hinged leaf.

Initial attempts exposed shared leaf/aperture/collider defects, which were corrected.
The source hierarchy also identifies component 15 as the fixed strike plate. A
literal rigid-lock preparation then found original latch/strike intersections.
That finding remains valid, but its earlier interpretation as grounds to discard
the candidate is superseded by the user-approved closed-unlatched task model.
The source does not need to be a mechanically complete, functioning lock assembly.

The final preparation (`attempts/000007`) omits collision only for latch/lock bolts
2 and 4. All original visuals, leaf/frame/strike geometry and collidable handles
are preserved. Normalization, static component coverage, front/rear RTX appearance
review and isolated RTX 4090 physics pass. Measured opening is 188.00001 degrees
against a 188-degree geometric limit, with zero reset error, zero frame drift,
zero reported penetration and 1.08-micrometer maximum hinge error. No contact
samples were reported in this functional run. The candidate is promoted to
`ready_for_5.1`; this does not establish robot reachability or expert qualification.

The task-state change was checked with 14 focused preparation tests and this one
real-door GPU run; no unrelated regressions or format runs were repeated. Earlier
rebate corrections retain their 29-test and single synthetic GPU evidence.
The Phase 4 common robot/contact/neck setup is unchanged.

Candidate records are versioned under `assets/doors/b1/door-with-frame-2f2f149f/`.
`preparation-review.json` records current readiness; ignored `attempts/` preserve
payloads, earlier outcomes and current reports/images. `prepared.json` points to
the accepted attempt, whose evidence is protected from overwrite.

**Next source and USDZ review (2026-09-23).** The user supplied
[modern door by Ahmed sayed](https://sketchfab.com/3d-models/modern-door-2fb8d02419b84d628cf9a4ac85360cec)
and `~/Downloads/modern_door.usdz`. The page lists Free Standard, original GLB and
converted USDZ/glTF/GLB downloads, about 3.6k triangles, and shows one wooden leaf,
frame and long handle. Embedded USDZ metadata independently says `SKETCHFAB Standard`.
The earlier CC-only `review` rejected this source before inspection. After the
user approved local-only intake, that rejection is superseded as a policy outcome;
`review` now passes with `distribution_scope=local_only`. Local USDZ inspection
finds 3,622 triangles, 55 components, four embedded 2K textures and a measured
0.998 × 2.165 × 0.061 m right-hinged leaf. The U-shaped frame's default convex
approximation filled the opening; a surface-preserving frame partition resolves
that recipe issue. The converted USDZ also separates a hinge-edge visual skin into
two triangles. The original closed surfaces overlap the frame by about 0.10 mm,
and PhysX expands that skin's standalone collider several millimeters into the
jamb. Translation aligns the main leaf but cannot make this skin cook faithfully;
convex-hull and partition recipes retain the failure. Those USDZ attempts remain
historical unresolved evidence; the successful GLB repair below supersedes the
candidate-level blocker. Original downloads and numbered attempts are preserved.

The user then supplied `~/Downloads/modern_door.glb` (5,486,088 bytes, SHA-256
`a457aba383e1e3fdc5863a4af7752d31db0a299be7f856054fce7fee24170ec5`).
Its embedded Sketchfab author, source and Standard license match the page; all
four JPEG dependencies are internal. The size suggests the displayed converted
2K GLB option, though the file does not identify its download option. Inspection
finds the same 55 components and 3,622 triangles as the USDZ; their bounds differ
only by a source-coordinate offset within conversion precision. The revised GLB
recipe recenters that offset. Its planar hinge-edge skin still crosses the frame;
the supported convex hull intersects the jamb, while a surface partition has no
volume. A temporary volumetric proxy for only that skin cleared the closed pose
but exposed a separate immediate leaf/frame collision on opening. The leaf almost
exactly fills the jamb opening; a mid-thickness pivot was inferred without modeled
hinge hardware. The temporary proxy code was reverted. Those findings describe
the original recipe, not proof that the candidate cannot be prepared.

**Modern-door repair (2026-09-23).** Source-section review found that the planar
skin's standalone cooked hull extends about 10.8 mm beyond its true side bound.
The 52 Panel fragments describe surfaces of one solid leaf, so the revised recipe
cooks their common convex envelope. This preserves all visual meshes and separate
handle colliders, while filling decorative face grooves up to 6.35 mm. Frame
partitions remain in place. Uniformly scaling Panel and Handle together by 0.996
about the leaf center provides about 2 mm side clearance (1.895 mm at the skewed
skin), with 4.28 mm top clearance. The inferred hinge is on the opening-face corner
at the original right-hand edge. These are explicit benchmark approximations;
requiring manufacturer hinge measurements was unnecessarily restrictive for 5.0.

Attempt `000012` passes normalization, static coverage/material/physics-property
checks, front/rear RTX visual review and one isolated RTX 4090 functional run.
The prepared leaf measures 0.994 × 2.157 × 0.061 m. It reaches 93.0000 degrees
against the geometry-derived 93-degree limit, with three exact resets, negligible
passive drift, zero frame drift and zero reported penetration; maximum hinge error
is 1.12 micrometers. No contact samples were reported. Promotion records
`ready_for_5.1` and `distribution_scope=local_only`. Twenty focused preparation
tests cover grouped collision ownership, common moving transforms and the zero-gap
failure/repair. No unrelated suite, format matrix or synthetic GPU runs were needed.
The original GLB/USDZ and attempts `000001`–`000011` remain untouched. No robot
reachability or expert qualification is claimed.

**Third supplied source (2026-09-23).** The user supplied
[Door With Doorframe by Theocritus](https://sketchfab.com/3d-models/door-with-doorframe-c29da62ca4e34dbb9801f98a4b5d3382)
and both `~/Downloads/door_with_doorframe.glb` and
`~/Downloads/Door_With_Doorframe.usdz`. The page shows one wood leaf, frame and
lever, 4.7k triangles, Free Standard, available original/converted GLB and USDZ,
and an explicit **NoAI** notice. The GLB metadata confirms author, source and
Standard license; both downloaded files remain untouched. The nine embedded GLB
images include materials named Poliigon wood and metal. Their provenance and
separate AI/ML rights were not demonstrated. [Sketchfab's NoAI policy](https://help.sketchfab.com/en/articles/16152133-generative-ai-policies-tagging-and-noai-protection)
and [terms, section 15](https://sketchfab.com/terms) bar datasets, development and
inputs for programs designed to generate new content. B1 ACT/Diffusion policies
output robot actions, not generated media or 3D content; NoAI alone therefore does
not establish a conflict for this limited use. `CreatedWithAI` is the separate
creation label. [Poliigon's licensing guidance](https://help.poliigon.com/it/articles/8749749-utilizzo-degli-asset-e-licenze)
excludes ML/AI use of its assets under ordinary terms. The names suggest a
third-party dependency but do not prove its licensing chain. Downloading the model
does not clear that gap. The user authorized this door for local preparation. A
geometry-identical local GLB retains the original geometry buffer, removes all
nine source images and uses independently authored flat PBR materials. Its source
`review` passes; inspection attempt `000002` finds 4,722 triangles, 22 connected
components and no textures. Original GLB/USDZ downloads remain untouched. The
inspected USDZ has the same 22 components and triangle count, with corresponding
vertices within 0.1 µm of the selected GLB.

The initial recipe uses global uniform scale 0.85 and fixed-jamb knuckles in Frame.
Attempt `000007` demonstrates genuine closed-pose leaf/frame intersections after
1% moving reduction. The uncorrected leaf is 0.89716 m wide versus a 0.88355 m
narrow jamb passage: even after 1% fitting it is 4.64 mm too wide. Thus translation
alone cannot fix that pose. The initial 1% fitting bound was a conservative editing
limit, not a physics tolerance; reviewed fitting now permits up to 2%.

The repaired recipe uses 2% uniform Panel/Handle scaling about the original
hinge-facing plane, preserving depth alignment with the source knuckles. A 0.99 mm
translation toward the hinge leaves 0.68 mm hinge-side and 3.66 mm latch-side gaps;
the thick latch edge needs the asymmetric clearance during its sweep. The axis is
measured from the twelve knuckle bounds. A proper 180-degree rotation corrects
the earlier swing-side assumption: the door is left-handed in the canonical
positive-opening convention, without reflection. A 2% change alone would still
stop near closed; both fitting and the hinge-side interpretation matter.

Attempt `000009` passes normalization, static checks and actual front/rear RTX
visual review. Its leaf measures 0.879 × 2.097 × 0.080 m. One isolated RTX 4090
run reaches 161.20003 degrees against the geometry-derived 161.2-degree stop,
with three exact resets, 0.000196-degree passive drift, zero frame drift,
1.02-micrometer maximum hinge error and zero reported penetration. It is promoted
as `ready_for_5.1`, `local_only`. Twenty-five focused preparation tests pass;
no collision/physics tolerance, nominal physical parameter or frozen robot setup
was changed. All prior attempts and both originals are preserved.

The selected GLB has six independently authored flat PBR materials, zero images
and zero textures. Both views retain leaf relief, frame molding, handles and
hinge hardware. That simple appearance is adequate for preparation; Phase 6.2
can vary approved materials with Replicator. Randomization does not replace the
present geometry/visual review or clear rights to excluded original textures.

**Fourth supplied source review (2026-09-23).** The user supplied
[Void Frame Studio – Animated Classic Door by VOID FRAME STUDIO](https://sketchfab.com/3d-models/void-frame-studio-animated-classic-door-08bdf51b9e5f4301be5de3a28c8a36ff)
and downloaded the original GLB archive and converted USDZ. The page shows one
classic paneled leaf, frame, lever and hinge, with opening/handle animation and
192.7k triangles. Original GLB, converted USDZ, glTF and GLB are available. The
GLB archive has one file, five named mesh nodes, one animation, two materials and
no image or external-file dependency. Exact size, handedness, geometry separation
and physics suitability were not inspected.

The page and download modal explicitly label the source **CC BY-NC-ND 4.0**, not
Sketchfab Free Standard. [CC's license text](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)
permits private noncommercial adaptations but prohibits sharing adapted material.
The initial 5.0 `review` returned `fail` because the previous scope did not admit
that license. The user then authorized private, noncommercial preparation of this
source. The revised `private_noncommercial` scope passes source review, without
changing the author's license or permitting distribution of a normalized asset.
Candidate-specific evidence is in
`assets/doors/b1/void-frame-studio-animated-classic-door-08bdf51b/`. The three
prepared doors and both downloaded originals remain intact.

The original GLB and converted USDZ both fail inspection on source triangles with
zero or negligible area. A private local GLB copy retains the source vertex,
material, node and animation data and omits 7,787 triangles of area at most
2e-14 m²; 184,865 remain. Inspection attempt `000003` passes with 70 components,
zero textures and a fingerprint distinct from the three prepared doors. The
separate original-format inspection attempts remain as diagnostics.

The source animation identifies a left-side hinge and opening direction under a
proper Y-up to Z-up rotation. At closed pose, source-surface crossings showed the
leaf and latch-side hardware entering fixed strike and hinge parts. Uniform 2%
Panel/Handle fitting and a 3.1 mm shift toward the latch resolve those crossings;
the canonical origin is recentered on the fitted leaf. The frame collider is
partitioned at jamb/header levels, and a narrower central aperture acknowledges
fixed hinge and strike hardware while keeping every frame collision. Several
planar latch-side handle-base/lock-casing fragments cooked as separate thick
colliders despite no remaining source crossings; a reviewed common solid collider
retains their collision. Both lever handles and the leaf remain collidable. Only
separate latch tongue component 69 loses collision in the closed-unlatched state;
its visual mesh remains. Earlier normalization attempts document those defects.

Attempt `000008` passes normalization, static coverage and inspected front/rear RTX
previews. The white paneled leaf, molded frame, three hinge positions and handles
on both sides are visible and aligned. The leaf measures 0.819 × 2.283 × 0.035 m.
One isolated `cuda:0` RTX 4090 run reaches 91.0000 degrees against the
geometry-derived 91-degree limit, with three exact resets, 0.000202-degree
passive drift, zero frame drift, 1.25-micrometer maximum hinge error and zero
reported penetration. Promotion records `ready_for_5.1` and
`private_noncommercial`; no robot/expert result is implied. A frame hull emitted
a PhysX oblong-shape CPU-collision fallback warning during preview cooking; the
rigid-door GPU functional check passed. No particle/deformable contact is claimed.
All prior attempts and both downloaded originals remain untouched.

**Fifth supplied source review (2026-09-23).**
[Door by DJMaesen](https://sketchfab.com/3d-models/door-2738468b94d74c5f827e7e5df7be8359)
shows one dark leaf, complete frame, right-side hinges and a left-side lever.
The page and download modal specify CC BY 4.0, 590 triangles and 2K maps;
original FBX plus converted USDZ, glTF and GLB are available. The newly
downloaded `Door (1).usdz` contains five embedded maps and matching author,
source and license metadata, with no external dependency. Inspection attempt
`000002` passes with 13 components and a geometry fingerprint distinct from
the four prepared doors. At uniform 0.75 scale the original leaf is 0.964 ×
2.080 × 0.051 m. The initially proposed right-handed push orientation was
corrected from the visible hinge hardware below.

The source leaf and frame opening share side/top coordinates. A 2% uniform
moving-assembly fit and a 0.8 mm latchward shift give a 0.945 × 2.038 × 0.050 m
leaf. Attempt `000007` recorded genuine internal hinge surface crossings, but
that alone did not establish a functional obstruction: the graphics model uses
solid overlapping pins/barrels even before fitting.

The corrected recipe models the bearing with the existing revolute joint and
filters only internal hardware pairs 6/8, 6/11, 7/9 and 7/12. Hinges retain external
collision; the leaf, frame and handles retain all contacts. Fixed plates 8/9 use
conservative convex hulls, avoiding inflation from decomposing their thin meshes.
The common fixed-pin XY center sets the axis, correcting an approximately 11 mm
lateral pivot error. A proper 180-degree Z rotation selects the actual opening
side: the prepared door is **left-handed**, not mirrored. No further shrinking
or acceptance-tolerance relaxation is used. The fit leaves approximately 10 mm
lateral offset between graphical moving/fixed hinge parts; the ideal bearing is
an explicit task-level approximation, not a reconstruction of working hardware.

Attempt `000008` passes normalization and static checks; front/rear RTX captures
were visually reviewed, retaining the dark textured leaf, frame, lever and hinge
hardware. The separate latch remains visible without collision. The opening limit
from the final cooked geometry is 113.1 degrees. One isolated RTX 4090 run reaches
113.099996 degrees with three exact resets, 0.000191-degree passive drift, zero
frame drift/rotation, 1.02-micrometer hinge-anchor error and zero reported
penetration in active contacts. Attempt `000008` is promoted for 5.1 under CC BY
4.0; five distinct doors are now prepared, with expert qualification still pending.
The focused preparation suite passes 27 tests, including rejection of leaf/handle
filtering, retained leaf/handle obstructions and a nonlocal hardware exception;
Ruff passes. The four earlier accepted assets were not changed or rerun.
Original downloads and all earlier attempts remain preserved in
`assets/doors/b1/door-2738468b94d74c5f/`. This repair changes the reusable preparation
recipe, not the frozen Phase 4 setup or the robot/expert gate in 5.1.

**Hoschu source review (2026-09-23).**
[Door by hoschu](https://sketchfab.com/3d-models/door-adf292f437f24151918a3b16ecef52d2)
shows a purple four-panel leaf with a gold lever. The page and USDZ metadata
agree on hoschu, source URL and CC BY 4.0. Original FBX and converted USDZ,
glTF and GLB are offered. The user-supplied `Door (2).usdz` packages five 1K
maps without external dependencies. Inspection attempt `000001` passes local
inventory: 1,577 triangles, four connected components and a fingerprint
distinct from the five accepted doors. Its unscaled converted envelope is
1.415 m wide and 3.000 m high before the preparation below.

A second review supersedes the initial connectivity-based rejection. Component 2
is a static open shell containing the central relief **and modeled frame**: two
full-height side strips and a header outside the relief contour. The original
source is not a complete solid leaf. Source-space selection within
Y `[-1.493612, 1.391966]` and Z `[-0.622471, 0.622471]` separates 296 central faces
from the 70 frame faces without moving or deleting any source triangle. The
handle/plate components 0, 1 and 3 remain attached to the moving leaf.

The appended leaf component 4 receives a planar rear at inspected X=0.01 and
boundary walls. Its front geometry/UVs remain intact; the rear is an inferred
flat surface repeating the front texture. The frame retains its source profile;
partitioned convex colliders supply its solid volume without closing the opening.
Uniform scale 0.75 and 0.4% clearance fitting give a 0.930 × 2.156 × 0.053 m leaf,
about 1.87 mm clearance at each jamb, and 1,857 prepared visual triangles.
A left hinge is inferred opposite the source handle at the opening-face corner;
no mirrored geometry or internal hinge-contact exception is needed.

Attempt `000002` passes normalization and static checks, including a 176-degree
geometric stop. Front/rear RTX captures were viewed: purple frame and relief,
gold source lever, fitted gaps and the inferred flat rear are adequate for the
push task. One isolated RTX 4090 run reaches 176.000000 degrees, with three exact
resets, 0.000193-degree passive drift, zero frame drift/rotation, 1.11-micrometer
hinge-anchor error and zero reported penetration. Attempt `000002` is promoted
under CC BY 4.0; six distinct doors are prepared, with expert qualification still
pending. The 29 focused preparation tests and Ruff pass. The source download and
initial inspection remain preserved; the previous five prepared assets are unchanged.

**Nikolayy source review (2026-09-23).**
[Door by Nikolayy](https://sketchfab.com/3d-models/door-5035d79771554b0a8daca21bae3ce062)
is a rusty single leaf with upper glazed panels, a lever, kick plate and separate
frame. The actual page and download modal show CC BY 4.0 and offer original FBX
plus converted USDZ, glTF and GLB. The user-supplied `Door (3).usdz` contains four
embedded 1K maps, no external dependencies and matching author/license/source
metadata. Its inspection attempt `000001` has 988 triangles, 11 components and a
fingerprint distinct from the six earlier accepted doors. Source animation is not
the common push control path; its zero pose supplies a closed leaf.

Components 5 and 9 are the modeled leaf and U-frame; 1–4 are paired handles,
6–8 moving hinge hardware, 0 a moving lock-edge plate and 10 a fixed strike-side
plate. No standalone bolt requires a collision exclusion. After 0.70 uniform
scale and proper rotation, a reviewed 2% uniform moving-assembly fit and 18 mm
opening-face shift clear real rebate/hinge interference. Frame collider partitions
follow its measured narrow rear opening; all frame, leaf, handle, hinge and plate
collisions remain. The two original hinge barrels locate the ideal joint axis.
Default collider cooking missed 4–6 mm of leaf skin, so a single convex leaf
collider preserves the outer contact envelope while filling decorative recesses.
The fitted leaf is 0.929 × 1.958 × 0.071 m and left-handed in the canonical push
convention. These are explicit simulation approximations, not recovered hardware
dimensions.

Attempt `000007` passes normalization and static checks. Front/rear RTX images
were viewed: frame, leaf, both handles, hinges, glazing and kick plate are present
and aligned. One isolated RTX 4090 run reaches the 194.4-degree geometric limit,
with three exact resets, 0.000193-degree passive drift, stable frame, 1.07-micrometer
hinge-anchor error and zero reported penetration. It is promoted under CC BY 4.0;
seven distinct doors are ready for 5.1. The focused 28-test preparation suite and
Ruff pass. The original download and all prior attempts remain preserved; robot
and expert qualification remain pending.

**Mehdi Shahsavan prison-door review (2026-09-23).**
[door _Prison Door_metal_old -12MB](https://sketchfab.com/3d-models/door--prison-door-metal-old-12mb-45306a46c95b44ca8369f89c6648648c)
shows a barred metal leaf, paired levers and a separate frame. The page and download
modal show CC BY 4.0; original FBX and converted USDZ, glTF and GLB are offered.
The supplied USDZ packages four 2K maps without external dependencies. Its USD
metadata records the same author account, license and source URL. The asset contains
two spatially separate copies of this one-leaf assembly, one open for display and
one closed. The original USDZ and first inspection are preserved; a local USDC
selection omits only the open display copy and keeps every closed mesh, UV and map.
Inspection `000002` reports 5,974 triangles, 20 components and a fingerprint
distinct from the seven earlier accepted doors.

In the selected assembly, component 14 is the leaf, 19 the frame, 6 and 8 the
moving hinge barrels, and 0–3/15–18 the paired handle hardware. Component 10 is
the projecting latch tongue: its visual remains but only its collider is omitted
for the common closed-unlatched task. Leaf, frame, handles, bars, hinges and edge
plates retain collision. A 0.75 overall scale followed by a reviewed 2% uniform
moving fit resolves original edge/hinge crossings. Frame collider partitions at
the measured aperture preserve its corner rebates. Default cooking omitted 5.3 mm
of lower leaf skin; one convex leaf envelope restores full contact coverage while
filling decorative window recesses. The prepared left-handed leaf is
0.902 × 1.960 × 0.058 m.

Attempt `000006` passes normalization and static checks. Both RTX views were
inspected: one red/black leaf aligns with its frame, barred window, lower hatch
and both levers. One isolated RTX 4090 run reaches the **23.1-degree geometric
limit** with three exact resets, 0.000195-degree passive drift, zero frame drift,
1.03-micrometer hinge-anchor error and zero reported penetration. The lower
leaf/frame corner limits this prepared model; additional leaf-collider partition
did not increase the limit. This short isolated result passes 5.0 and is promoted
under CC BY 4.0 as the eighth door. The small opening is recorded for 5.1; no
robot run or 45-degree qualification criterion was applied. Original/derived
sources and prior attempts remain preserved.

**Mehdi Shahsavan metal-door review (2026-09-23).**
[door door metal](https://sketchfab.com/3d-models/door-door-metal-b21ec273c1a342568ab9f9ac14291c5b)
shows a single dark metal leaf in a U-frame with a narrow upper window,
horizontal bar and lower kick plate. The page and download modal show CC BY 4.0,
the same author as the prison door, and original OBJ plus converted USDZ, glTF
and GLB. The supplied USDZ embeds four 2K maps and matching author/license/source
metadata. Inspection `000002` reports 3,338 triangles, three connected components
and a fingerprint distinct from all eight earlier accepted doors.

Source component 1 joins the modeled frame and leaf at their contour. A reviewed
whole-face split keeps 30 frame faces and selects 66 leaf faces without moving
source vertices or altering UVs. Component 0 is the bar; component 2 is the narrow
window plane. All remain collidable, and no separate latch/bolt is modeled. Uniform
0.75 scale yields a full-size leaf; a 0.4% uniform moving fit creates about 1.86 mm
of side clearance from the originally touching jambs. The fixed frame retains
partitioned collision around its aperture; the solid leaf collider fills the
decorative window for contact while retaining the original visual. The ideal hinge
is inferred at the negative-X source jamb, mapping to a right-handed canonical
push. The leaf measures 0.926 × 2.150 × 0.054 m.

Attempt `000004` passes normalization/static checks and viewed front/rear RTX
previews: one face is plain and the other retains the textured window, bar and
kick plate. One isolated RTX 4090 run reaches the **59.8-degree geometric stop**
with three exact resets, fixed frame, about 1.03-micrometer hinge-anchor error
and zero reported penetration. It is promoted under CC BY 4.0 as the ninth door.
The source USDZ and diagnostic attempts remain. The inferred hinge and collision
approximations are recorded; no robot run or 45-degree qualification criterion
was applied.

**Icevanilla PSX door-pack review (2026-09-23).**
[Low-Poly PSX Style Essential Doors Pack](https://sketchfab.com/3d-models/low-poly-psx-style-essential-doors-pack-20d55059056044b885a5267c2de1ec18)
is by Icevanilla (@vanillao03). The page and download modal show CC Attribution
(CC BY 4.0), credit required and commercial use allowed. The page offers original
FBX and converted USDZ, glTF and GLB. The supplied USDZ has matching author,
license and source metadata, one packaged 4096 × 1024 texture, and six named
door/frame displays. The first closet display has two leaves and is excluded.
Five single-leaf assemblies are selected into separate local USDC sources, each
retaining its matching modeled frame, original surfaces, UVs and local copy of
the packaged texture. The original USDZ is unchanged. The five source geometry
fingerprints differ from each other and all nine previously prepared doors;
in particular, the two brown wooden doors have different panel geometry, not
just a changed material. `source_part` identifies each assembly so the shared
source URL does not mask a real duplicate.

| Selected door | Source triangles | Prepared leaf W × H × T (m) | Attempt | Isolated geometric stop |
| --- | ---: | --- | --- | ---: |
| Bathroom, white vented | 400 | 0.757 × 1.809 × 0.086 | `000006` | 121.3° |
| Wooden 001, one large panel | 280 | 0.757 × 1.809 × 0.086 | `000005` | 121.4° |
| Wooden 009, two panels | 340 | 0.757 × 1.809 × 0.086 | `000005` | 121.2° |
| Worn, slatted/paneled | 452 | 0.757 × 1.809 × 0.086 | `000005` | 121.2° |
| Front, three upper panes | 482 | 0.815 × 1.947 × 0.092 | `000005` | 121.3° |

All five map to the canonical left-handed push orientation without reflection.
Four shorter source leaves use 1.015 uniform assembly scale to remain above the
1.80 m height minimum; the front door uses 1.0. The modeled leaves initially
overlap the jamb/header slightly. An initial 0.4% moving fit still caused an
early opposite-jamb collision and a 1.7-degree geometric stop. A reviewed 1.3%
uniform fit of each moving leaf and its hardware, with measured 2.1–2.4 mm width
centering, yields about 5.9–6.3 mm side clearance; the inferred hinge stays at
the hinge-side outer face. Frame collision is partitioned around the aperture
with cuts 2 mm inside the inner jamb/header boundary, so the central frame hull
does not fill the opening. These are declared simulation approximations, not
measurements of manufacturer hardware. Leaf, frame, handles and the front-door
window retain collision; no separate projecting latch/bolt is modeled. Each
door starts closed and already unlatched.

Each selected source was inspected, then its final normalization attempt passed
static checks. Both RTX 4090 preview images per door were viewed: the respective
panels, frames, two-sided hardware, bathroom vent and front windows are visible
without missing parts or misalignment. One short isolated RTX 4090 `cuda:0`
physics run per door reached its geometry-derived stop with three exact resets,
zero frame drift and zero reported penetration. All five were promoted under
CC BY 4.0. Individual `candidate.json`, `source-evidence.json`, `recipe.json`
and `preparation-review.json` records contain provenance, source/selection hashes,
component choices, failed earlier attempts and evidence paths. No robot/expert
run or 45-degree criterion was used; the inferred axes and prepared stops remain
to be evaluated in 5.1.

**Icevanilla industrial PSX pack review (2026-09-23).**
[Low-Poly PSX Style Industrial Metal Doors Pack](https://sketchfab.com/3d-models/low-poly-psx-style-industrial-metal-doors-pack-4f5561bbc441414587e3650ceae5593a)
is by Icevanilla (@vanillao03), with CC Attribution/CC BY 4.0 on the page and
download modal. It offers original FBX and converted USDZ, glTF and GLB; the
downloaded USDZ contains matching author/license/source metadata and one
1024 × 256 packaged atlas. The pack has five single-leaf groups, `001`–`005`,
and a separate two-leaf `007` display excluded at review. All five selected
groups pass source/license and dependency inspection at 140–400 triangles per
original group; none matches a fingerprint among the fourteen prepared doors.
Distinct source-part names alone do not settle geometric duplication.

The initial three intersection findings were real for their recessed closed-pose
recipes; the conclusion that shrink beyond 2% was necessary was too restrictive.
A 3.6% diagnostic fit for `003` cleared the closed pose but stopped at 0.8 degrees.
That diagnostic is preserved and not promoted. The final mount instead places
the closed slab's back face 2 mm in front of the measured jamb face, with 2 mm
floor clearance and an inferred left back-face hinge. It is an explicit
surface-mounted simulation assembly, not a claim about the author's intended
mechanism. All frame, slab and handle collisions remain enabled; the 2% moving
fit limit and every static/physics tolerance are unchanged.

| Group | Final preparation | Attempt | Dimensions W × H × T (m) | Geometric stop |
| --- | --- | --- | --- | --- |
| `001` | Reuse the same-pack source frame already selected; mount the original three-panel slab in front of it, overall scale 0.965, no moving shrink. | `000008` | 0.841 × 1.816 × 0.100 | 91.0° |
| `002` | User-approved reuse of the pack frame, with aperture spans adapted to the larger ornate leaf and jamb/header profiles retained. Original slab shape retained; overall scale 0.895, no moving shrink. | `000005` | 0.913 × 1.902 × 0.100 | 91.1° |
| `003` | Original plain slab, knobs and frame; surface-mounted at overall scale 0.965, no moving shrink. | `000007` | 0.841 × 1.816 × 0.100 | 91.0° |
| `004` | Original detailed slab, levers, frame and projecting lintel retained. Surface mount plus the existing 2% uniform moving fit clears the lintel; overall scale 0.98. | `000005` | 0.837 × 1.808 × 0.099 | 91.0° |
| `005` | Geometry variant of `003`, not an independent corpus identity. | inspected `000002` | — | not rerun |

For `002`, the adapted rectangular frame sits behind the source arched upper
outline. It is a documented reconstructed assembly from the same licensed pack;
no new door surface, borrowed external texture or double-door component is used.
Front/rear previews for all four prepared assemblies were actually viewed and
retain the source metal atlas and distinct slab details. Static checks pass.
One isolated RTX 4090 run per distinct repaired door reaches its geometric stop
(91.0, 91.1, 91.0 and 91.0 degrees for `001`–`004`). Each passes three exact resets,
zero frame drift/rotation and zero reported penetration; maximum passive drift is
0.000200 degrees and maximum hinge-anchor error is below 0.96 micrometers. All
four are promoted as redistributable CC BY assets, bringing the prepared count
to **eighteen**. The 32 focused preparation tests, Ruff and wiki checks pass.
No physics rerun was performed on earlier accepted assets or duplicate `005`.

The `005` duplicate conclusion is independently confirmed by aligning each
component: symmetric nearest-vertex error is below 0.02 micrometers for the slab,
0.003 micrometers for the knobs and 1.91 micrometers for the frame. Relative to
the slab, the frame shifts about 10 mm, and the knobs shift 10.88 mm horizontally
and 9.22 mm vertically. These placements and the different atlas region do not
make another independent geometry. Its source remains available as an appearance
variant, without promotion, a second physics run or a separate train/test identity.

The original USDZ, source selections and all diagnostic attempts are preserved;
the fourteen earlier accepted assets are unchanged. Source/candidate/preparation
records retain each part's provenance and the specific inferred changes. Robot
reachability, expert rollouts and the 45-degree qualification remain in 5.1.

#### Key Decisions

- Target 24 accepted unique identities, 12 left- and 12 right-hinged, with no
  unused reserve payload. The user controls the sequential URL intake.
- Accept CC0/CC BY 4.0 and dependencies for redistributable assets; admit reviewed
  Sketchfab Free Standard sources only as local-only and the specifically approved
  CC BY-NC-ND source as private/noncommercial. Hold unclear terms as unresolved
  and reject demonstrated incompatibility; retain attribution and license evidence.
  No restricted geometry or textures enter shared packages.
- Use full-size single-leaf interior/exterior/industrial push doors, separable
  panel/frame, prepared closed and already unlatched. Exclude gates, cabinets, sliding/double
  doors, and fantasy geometry. Mirrors/recolors are not independent identities.
- Retain width 0.65–1.20 m, height 1.80–2.40 m, thickness 0.025–0.10 m, at most
  250,000 visual triangles and 4K textures. Prefer USD/USDZ, then GLB/glTF,
  Blend/FBX, and OBJ when supported conversion avoids substantial remodeling.
- Permit format conversion, uniform scaling (including reviewed moving-assembly
  clearance repair), simple panel/frame separation, inferred pivot correction,
  materials, grouped solid colliders, articulation and reviewed disengaged-latch
  collision exclusions. Preserve handedness
  and meaningful geometry; do not create reflected action frames.
- Use the Phase 4 nominal physics template with consistently derived inertia.
  This isolates geometry rather than reconstructing every original door's dynamics.
  Mechanical limits follow geometry/clearance, not desired benchmark scores.
- Apply the common base pose, tool, contact fraction/height, and probe unchanged.
  Handle obstruction is not permission to move the contact point per asset or
  disable collisions. Distinguish invalid geometry from a bad normalization recipe.

#### Problems / Limitations

Preparation is verified on the documented fixtures and fourteen real doors in the
closed-unlatched state. No real door has expert qualification yet. Passing
these checks does not establish robot reachability. Subphase 5.1 owns the frozen
expert probe, per-door reference and 24-door split. Missing URLs are expected input.

The Ahmed sayed candidate's earlier license rejection and preparation blockers
are superseded by local-only scope and the reviewed GLB repair. The initial batch
has grown to fourteen technically ready doors; eleven are redistributable.
Void Frame Studio's initial source-license failure and geometry
attempt failures are superseded by its private/noncommercial admission and
attempt `000008`. Inferred pivots and clearance fitting do not reconstruct real
hardware; the nominal benchmark still requires the frozen 5.1 robot probe.

Conversion uses a glTF/PreviewSurface material path. Texture preservation is tested,
but arbitrary shaders, animations and all source-format features are not guaranteed;
appearance and FBX sidecar/license completeness require local review. Rectangular
opening checks, convex collision approximations, a 270-degree search domain and a
512-component processing limit are explicit boundaries; unsupported cases remain
unresolved without silently changing admission rules.

The initial USD converter exported centimeter layers whose references rendered
100 times too large. This is fixed by explicit meter export and an independent
visual/collider bounds check. Synthetic USD cubes needed explicit tessellation.
The installed headless runtime requires its Isaac Lab camera renderer; the working
preview uses that path rather than relying on an unpumped raw render product.

Validation covers the four Phase 4 synthetics, all eight supported suffixes with
textures, external glTF/OBJ dependencies, and concave PhysX decomposition. Negative
fixtures detect missing dependencies, zero mass, negative inertia, absent handle
collisions, a blocked opening, wrong limits, wrong visual scale, closed
interpenetration and degenerate geometry. A GPU-injected obstacle is detected
through loaded raw contact during opening. Software tests cover reflections,
invalid recipes, duplicate identity and preservation/stale-evidence behavior.

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

## Artifacts

Infrastructure evidence: `~/.cache/alexdoor-xas/verification/door-preparation-50/`.
The four `*/attempts/000001/prepared/` folders contain normalized USDs, static and
GPU reports, numerical traces and raw-contact summaries. All four reach 183.2
degrees; maximum passive drift is 0.000209 degrees, hinge-anchor error is below
1.20 micrometers, fixed-frame drift is zero and measured penetration is zero.
These are isolated synthetic simulation results, not physical-robot accuracy.

`formats/formats.json` records eight successful textured format paths;
`conversion-final/` rechecks the final explicit-sidecar inventory contract.
`frame-recheck/` verifies proper hinge transforms and the positive joint axis for
both handednesses; right hinges expose a proper 180-degree X rotation.
`invalid/invalid.json` records negative cases and concave decomposition;
`gpu-obstruction/` records the expected physical rejection. FBX front/rear preview
images are retained under `formats/fbx/prepared/` and were visually inspected.
That infrastructure run produced no real candidate payload, expert reference,
split or learned dataset. Fourteen real candidate records now have prepared
attempts under `assets/doors/b1/`; their accepted pointers and individual
preparation reviews carry the current per-door evidence.

All 350 software tests, Ruff and wiki link/index checks pass. The public CLI was
also exercised through review, normalization, static/GPU checks and promotion with
an explicitly simulated review record in `/tmp/b1-cli-proof/`; this is command
verification only. A subsequent rerun preserved the prepared pointer and reports.

Reproduce infrastructure validation in a new output directory:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_preparation.py \
  --formats --invalid --physics --output /path/to/new/verification \
  --viz none --device cuda:0
```

## Files

- `scripts/prepare_doors.py` — sequential local preparation and review commands.
- `scripts/verify_door_preparation.py` — synthetic conversion/negative/GPU verification.
- `src/alexdoor_xas/qualification/` — recipes, conversion, collision geometry, gates and preview.
- `src/alexdoor_xas/envs/door_task/door_inspection.py` — isolated GPU environment with optional contacts.
- `tests/test_preparation.py` — admission, recipe, preservation and evidence contracts.
- `tests/test_mesh_preparation.py` — source-face/UV preservation and closed backing checks.
