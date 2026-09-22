# Phase 5 — Door Corpus and Qualification

> Planned. Requires the common setup and probe from Phase 4. No assets were
> downloaded, normalized, or qualified by the 2026-09-22 documentation revision.

## Objective

Produce the definitive licensed B1 door-asset corpus, per-door expert references,
and disjoint 12/4/8 identity split. This is the environment corpus; recorded
training demonstrations are a separate Phase 6 deliverable. Follow
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Benchmark Design]].

## Subphase 5.0 — Candidate Review and Manual Acquisition

#### Implementation

The user provides candidate URLs. Review each source, license scope, available
format, visible geometry, dimensions/complexity when available, and duplicates
before the user downloads it manually. Assess local-only properties after the
download. Collect 24 accepted identities, 12 left- and 12 right-hinged, without
an unused reserve payload or automatic download workflow.

#### Key Decisions

- Accept only CC0 or CC BY 4.0 covering the model and redistributed dependencies.
  Record URL, author, license evidence, attribution, retrieval date, modifications,
  identifier, and checksums needed for release.
- Use full-size, single-leaf interior/exterior/industrial push doors with separable
  panel and frame, one vertical revolute hinge, and no latch operation. Exclude
  gates, cabinets, sliding/double doors, and fantasy geometry.
- Retain the declared envelope: width 0.65–1.20 m, height 1.80–2.40 m, thickness
  0.025–0.10 m, at most 250,000 visual triangles and 4K textures.
- Prefer USD/USDZ, then GLB/glTF, Blend/FBX, and OBJ. Conversion must not require
  substantial remodeling. Recolors, mirrors, and base-mesh variants are not
  independent identities. Identify related geometry families before splitting.

#### Problems / Limitations

Twenty-four doors is a bounded initial corpus, not proof of broad door coverage.
Report geometric diversity and rejection reasons. Lack of eligible candidates
is a collection blocker, not permission to silently relax the contract.

## Subphase 5.1 — Canonical Normalization and Static/Physics Gates

#### Implementation

Normalize to meters, Z-up, closed angle zero, a positive opening-angle convention,
canonical panel/frame/hinge identities, and local redistributable dependencies.
Preserve physical handedness; do not introduce reflected action frames. Provide
both the center-of-opening placement reference and hinge/panel transforms.

Allow uniform scaling, format conversion, simple panel/frame separation, pivot
correction, materials, colliders, and articulation. Preserve meaningful geometry.
Keep handles rigidly attached and collidable, with no handle actuation or latch.
Keep a real frame collision opening rather than a hull that fills the doorway.

Use the nominal physics template frozen on synthetics in Phase 4, deriving
geometry-dependent inertia consistently. A common mass/friction/damping template
isolates geometry; it does not reconstruct each original door's real dynamics.
Declare mechanically admissible opening limits from the normalized geometry and
collision clearance, not from the desired benchmark score.

Static gates check dependencies, units, geometry, articulation, colliders, and
positive mass properties. GPU physics gates check a fixed frame, stable hinge,
closed reset, passive drift, interpenetration, numerical validity, and collision-
consistent opening. Do not disable handle/frame collisions to pass.

#### Key Decisions

- Apply the frozen base pose, contact fraction/height, tool orientation, and probe
  mechanically to every asset. Width changes the metric contact location through
  the same fraction; it does not authorize a new fraction or height.
- If a handle obstructs that contact footprint, do not move the point just for
  that asset. Report an out-of-domain geometry unless the normalization itself
  is demonstrably wrong.
- Substantial remodeling, unresolvable licensing, or asset-specific unstable
  physics makes a candidate invalid. A runtime/probe bug requires diagnosis.

#### Problems / Limitations

Passing static/physics gates does not prove robot reachability. That is measured
in Subphase 5.2 with the already frozen setup.

## Subphase 5.2 — Expert Reference, Admission, and Split Freeze

#### Implementation

Run the frozen probe to its valid controlled limit on each door, including
approach, sustained contact, hold, and safe release. Do not stop upon crossing
45 degrees. Record the sustained angle, force/contact validity, stop reason,
and relevant joint margins. Repeat once from the same reset as a lightweight
repeatability check.

The pair must have valid controlled completion and a consistent limiting cause,
with sustained maxima within 2 degrees. Diagnose inconsistent pairs instead of
selecting the favorable rollout. If extra runs are needed for diagnosis, do not
turn them into a best-of-many reference search. Once the cause is resolved,
rerun the prescribed pair and retain the cause/result record.

Set `theta_expert_d` to the lower of the two valid maximum sustained angles.
This conservative deterministic reference avoids rewarding numerical overshoot;
it is not a rollout percentile or a certified global optimum. Admit only doors
with `theta_expert_d >= 45 deg`. Exclude lower-angle candidates even when their
assets are technically valid, labeling them outside the reachable benchmark
domain rather than corrupt assets. A stall or exhausted horizon without a
justified limit leaves qualification unresolved, not an automatic replacement.

After 24 doors pass, freeze a manifest and train/development/test split of 12/4/8,
with left/right counts 6/6, 2/2, and 4/4. Group related source geometry so no
family leaks across splits; balance coarse geometry without selecting identities
using learned-policy scores. Record the frozen setup/probe reference and each
door's expert result alongside release provenance.

#### Key Decisions

- Distinguish invalid asset, valid-but-out-of-domain asset, and unresolved probe
  failure. Replace rejected candidates only under these predefined criteria.
- Never replace a qualified door because a learned policy performs poorly.
- Test qualification traces/images remain isolated evaluation evidence. They
  cannot train perception, gaze, normalization, policies, or model selection.
- There is no dataset-wide primary angle, adaptive qualification rollout count,
  bootstrap selection, or percentile over identical deterministic trials.
- The corpus contains USD assets, provenance, qualification, and split membership.
  The demonstration dataset contains synchronized observations/actions from
  accepted training-door episodes and is produced later in Phase 6.

#### Problems / Limitations

Complete only with 24 legally/technically eligible and nominally reachable doors,
their stable expert references, and the frozen disjoint split. The admission
criterion conditions scientific conclusions on this qualified domain. Asset
geometry must never retune the Phase 4 base or contact rule.

## Artifacts

Future outputs: normalized corpus, license/attribution manifest, exclusion report,
static/physics evidence, expert references, and 12/4/8 split. No learned dataset
or final training result is an output of this phase.

## Files

Expected surfaces: reusable door normalization/qualification capabilities under
`src/alexdoor_xas/` and `scripts/`, manifest and normalization recipes, canonical
scene loading under `src/alexdoor_xas/assets/` and `src/alexdoor_xas/envs/`.
The pre-Phase-4 cleanup determines which existing `phase4_1` tools are reusable.
