# Phase 5 — Door Corpus and Qualification

> Subphase 5.0 implementation in progress. Preparation contracts and canonical
> normalization are implemented; infrastructure readiness requires the complete
> static/GPU validation and operational entry points below. No real candidate admitted.

## Objective

Produce the definitive licensed B1 door corpus, per-door expert references, and
12/4/8 identity split. The corpus is the set of usable environments; training
demonstrations are generated later in Phase 6. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 5.0 — Preparation Infrastructure and Incremental Door Intake

#### Implementation

The reusable preparation core is implemented in `qualification/preparation.py`,
`qualification/prepare_usd.py`, and `qualification/convex_geometry.py`. It uses
explicit component recipes, preserves source snapshots and separate attempts,
and bakes PhysX convex hulls before geometric opening checks. The four Phase 4
synthetics have traversed normalization; textured USD/USDZ, GLB/glTF, FBX and OBJ
fixtures have traversed conversion. Full infrastructure closeout is still pending.

**First, deliver and validate the reusable infrastructure before requesting or
processing the first candidate URL.** Reuse suitable code retained by the earlier
cleanup. Implement a complete local path from a source asset and a small explicit
normalization recipe to a canonical USD, static/physics results, and a concise
candidate record. The preparation path must cover:

- Source/dependency inventory, dimensions, complexity, separable frame/panel, and
  duplicate identity checks; fields for URL, author, license evidence, attribution,
  modifications, and release checksums.
- Conversion for the explicitly supported input formats and canonical normalization:
  meters, Z-up, closed angle zero, positive opening convention, panel/frame/hinge
  identities, and both center-of-opening and hinge/panel transforms.
- Colliders, one vertical hinge, mass/inertia, nominal physics, and mechanical
  opening limits; keep handles attached/collidable and the frame opening clear.
- Static checks for dependencies, geometry, units, articulation, colliders, and
  positive mass properties. GPU physics checks for fixed frame, stable hinge,
  closed reset, passive drift, interpenetration, numerical validity, and unobstructed
  collision-consistent opening.
- An explicit pass/fail/unresolved report, saved normalization parameters, and
  rerun behavior that preserves the source and already accepted candidates.

Validate this path with the existing synthetic fixtures and deliberately invalid
cases appropriate to its contracts. Prove the physical checks in the supported
GPU runtime. State the supported formats and conversion limits; do not build a
universal format framework or collection service. The infrastructure milestone
ends with usable entry points, concise usage instructions, and readiness evidence.
It can be delivered before any real door URL exists; the assistant then waits
for the user's first candidate. Do not claim that the corpus is already complete.

**After infrastructure readiness, process one user-provided URL at a time:**

1. The model reviews source, license scope, format, visible geometry, available
   dimensions/complexity, and duplicates. Reject clearly unsuitable candidates
   before download; identify properties that require the local payload.
2. If the remote review is acceptable, the user downloads the asset manually
   and supplies its local path. No automatic download or broad asset search is needed.
3. The model uses the existing preparation commands and an asset recipe to
   normalize the payload and run static/physics checks. Inspect unknown properties
   locally before accepting them. Report the result and concrete rejection or
   unresolved reason.
4. Pass each technically valid door to Subphase 5.1's expert check without
   waiting for the other candidates. Continue with the next user-provided URL.

Do not write a new standalone normalization pipeline for every door or modify
robot/contact rules to make a candidate pass. Fix genuine shared-tool defects
in the shared infrastructure and recheck affected results. Ordinary differences
in scale, source prims, pivot, or allowed separation belong in the asset recipe.

#### Key Decisions

- Target 24 accepted unique identities, 12 left- and 12 right-hinged, with no
  unused reserve payload. The user controls the sequential URL intake.
- Accept only CC0 or CC BY 4.0 covering the asset and redistributed dependencies.
  Reject unclear or incompatible terms; retain attribution and license evidence.
- Use full-size single-leaf interior/exterior/industrial push doors, separable
  panel/frame, and no latch operation. Exclude gates, cabinets, sliding/double
  doors, and fantasy geometry. Mirrors/recolors are not independent identities.
- Retain width 0.65–1.20 m, height 1.80–2.40 m, thickness 0.025–0.10 m, at most
  250,000 visual triangles and 4K textures. Prefer USD/USDZ, then GLB/glTF,
  Blend/FBX, and OBJ when supported conversion avoids substantial remodeling.
- Permit format conversion, uniform scaling, simple panel/frame separation,
  pivot correction, materials, colliders, and articulation. Preserve handedness
  and meaningful geometry; do not create reflected action frames.
- Use the Phase 4 nominal physics template with consistently derived inertia.
  This isolates geometry rather than reconstructing every original door's dynamics.
  Mechanical limits follow geometry/clearance, not desired benchmark scores.
- Apply the common base pose, tool, contact fraction/height, and probe unchanged.
  Handle obstruction is not permission to move the contact point per asset or
  disable collisions. Distinguish invalid geometry from a bad normalization recipe.

#### Problems / Limitations

Infrastructure readiness and asset readiness are separate outcomes inside 5.0.
Only after the first is demonstrated does sequential real-asset intake start.
Passing static/physics checks does not prove robot reachability; 5.1 supplies
that result. Work through the two subphases per candidate; freeze the split only
when 24 doors pass both. Missing URLs are expected user input, not a tool failure.
Insufficient eligible candidates must not silently relax admission criteria.

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

Future outputs: first the reusable preparation/checking tools and their readiness
evidence; then one candidate result per URL, normalized corpus, exclusions,
licensed manifest, expert references, and frozen split. No learned dataset or
training result is produced here.

## Files

Expected surfaces: reusable normalization/qualification under `src/alexdoor_xas/`
and `scripts/`, normalization recipes and manifest, scene loading under
`src/alexdoor_xas/assets/` and `src/alexdoor_xas/envs/`. The earlier cleanup
identifies reusable legacy `phase4_1` code; new tooling is implemented only when
Phase 5.0 is explicitly undertaken.
