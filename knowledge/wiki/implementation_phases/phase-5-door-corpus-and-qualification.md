# Phase 5 — Door Corpus and Qualification

> Subphase 5.0 infrastructure is implemented and verified on the RTX 4090.
> The first real door passes preparation in the common closed-unlatched state and is ready for 5.1. Expert qualification remains planned.

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

A JSON recipe specifies `handedness`, positive uniform `scale`, proper `rotation`,
`translation_m`, `opening_center_source`, `hinge_m`, `dimensions_m`, and the complete
`components` assignment to `Frame`, `Panel`, and optional `Handle`. It also records
`modifications`. Rotation/scale/translation act on the **inspected coordinates**;
USD/FBX pass through glTF Y-up meters. The opening center maps to the floor origin.

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
Optional `colliders` maps component indices (JSON strings) to `auto`, `convexHull`,
or `convexDecomposition`. Default `auto` uses the installed PhysX cooker, baking
individual hulls so clearance checks and simulation use the same collision shapes.
Collider-only vertices are welded and decompositions use shrink wrapping.

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

**Usage.** Keep reviewed `candidate.json`, `recipe.json` and license evidence under
`assets/doors/b1/<asset-id>/` in Git. Generated `attempts/` and `prepared.json` are
ignored. `review` requires asset/source IDs, source URL, author, license and its
asset-specific evidence, attribution, retrieval date, door type, selected format
(including the dot), and passing license-scope/custom-terms/duplicate reviews.
Unknown remote geometry may remain unset. Before promotion, record passing
`local_dependency_review`, `local_duplicate_review`, `local_visual_review` and the
`reviewed_source_sha256` from `inspect.json`; explain any fingerprint collision in
`duplicate_resolution`. License evidence must cover the redistributed dependencies.

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
   Inspect original views only if ownership or orientation is ambiguous.
3. Write the smallest adequate recipe and run `normalize`. The source of recipe
   coordinates is the converted inventory (USD/FBX conversion uses glTF Y-up
   meters), not assumed raw-file units. Determine scale, handedness, pivot and
   component indices for this asset. Use `leaf_components`, `clear_aperture_m`
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

#### Key Decisions

- Target 24 accepted unique identities, 12 left- and 12 right-hinged, with no
  unused reserve payload. The user controls the sequential URL intake.
- Accept only CC0 or CC BY 4.0 covering the asset and redistributed dependencies.
  Reject unclear or incompatible terms; retain attribution and license evidence.
- Use full-size single-leaf interior/exterior/industrial push doors, separable
  panel/frame, prepared closed and already unlatched. Exclude gates, cabinets, sliding/double
  doors, and fantasy geometry. Mirrors/recolors are not independent identities.
- Retain width 0.65–1.20 m, height 1.80–2.40 m, thickness 0.025–0.10 m, at most
  250,000 visual triangles and 4K textures. Prefer USD/USDZ, then GLB/glTF,
  Blend/FBX, and OBJ when supported conversion avoids substantial remodeling.
- Permit format conversion, uniform scaling, simple panel/frame separation,
  pivot correction, materials, colliders, articulation and reviewed disengaged-latch
  collision exclusions. Preserve handedness
  and meaningful geometry; do not create reflected action frames.
- Use the Phase 4 nominal physics template with consistently derived inertia.
  This isolates geometry rather than reconstructing every original door's dynamics.
  Mechanical limits follow geometry/clearance, not desired benchmark scores.
- Apply the common base pose, tool, contact fraction/height, and probe unchanged.
  Handle obstruction is not permission to move the contact point per asset or
  disable collisions. Distinguish invalid geometry from a bad normalization recipe.

#### Problems / Limitations

Preparation is verified on the documented fixtures and one real door in the
closed-unlatched state. No real door has expert qualification yet. Passing
these checks does not establish robot reachability. Subphase 5.1 owns the frozen
expert probe, per-door reference and 24-door split. Missing URLs are expected input.

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
No real candidate payload, expert reference, split or learned dataset was produced.

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
