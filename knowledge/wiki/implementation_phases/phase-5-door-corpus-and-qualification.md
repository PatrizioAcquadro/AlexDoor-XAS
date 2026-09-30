# Phase 5 — Door Corpus and Qualification

**Phase 5 is complete as of 2026-09-28.** The frozen corpus contains all **32
qualified doors: 25 left and seven right**, with two fresh-process GPU trials
per door under the same common setup. Rights scopes remain 29 redistributable,
two local-only and one private/noncommercial.

The tracked `assets/doors/b1/corpus.json` fixes the 19/6/7 train/development/test
membership, 12 reviewed geometry families spanning 15 sources, common setup and
expert references. Qualification records are pinned at `76658aa`; the shared
control recovery and interrupted attempts below remain historical evidence.
No new simulation or asset modification was needed for corpus freeze. Phase 6
owns observed-only perception and the demonstration dataset. See
[[decisions/visuoproprioceptive-generalization-benchmark|B1 Design]].

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

### Qualification history and final boundary

The initial campaign (`cf73eb0`) reported 15 qualified, 15 out of domain and two
unresolved. The authorized common recovery corrected placement, coasting contact,
tracking bias, support, final hold and release without weakening the 45-degree
admission gate. Earlier exclusions and individual recovery diagnostics remain
historical, never interchangeable with the final paired references.

Bounded rendered-surface ray batches fixed excessive visibility memory allocation
(`243c2ef`) without changing criteria. Same-process reset retained contact history
on void-frame: fresh Isaac processes (`efe1325`) isolated both physics and rendered
state. Its final reference is 66.92 degrees, with zero paired spread. Worn's old
17.92-degree exclusion was superseded by a complete 67.42-degree reference after
the shared angular guard correction. Interrupted industrial-004 execution (code
143) remains incomplete evidence; the separate fresh pair at `76658aa` closes it.
All 32 final pairs are qualified under the common protocol, before split freeze.

The pool contains 25 left and seven right identities. The approved corpus revision
retains all eligible doors, replacing the former 24-door/12-per-hand proposal.
Additional intake requires a specific coverage gap. Per-door angles, evidence runs,
geometry and scope remain in the records and frozen corpus; original trials and
attempts in the verification cache were not deleted by cleanup.

#### Frozen corpus and identity split

`assets/doors/b1/corpus.json` (`b1.corpus.v1`) is the canonical asset-level split.
All 32 qualified identities occur exactly once. Membership was chosen from
source/geometry relationships and handedness before learned-model results;
expert angle or policy performance did not determine membership. This replaces
the earlier source-URL-only feasibility example, which was never frozen.

| Partition | Complete reviewed families | Left | Right | Total |
|---|---|---:|---:|---:|
| Train | PSX residential: essential + interior wood + front (15), animated doors (3), `door-2738468b94d74c5f` | 15 | 4 | 19 |
| Development | PSX industrial (4), `modern-door-2fb8d024`, `void-frame-studio-animated-classic-door-08bdf51b` | 5 | 1 | 6 |
| Test | Both Mehdi Shahsavan metal-door sources (2), `door-5035d7977155`, `door-adf292f437f2`, `door-with-doorframe-c29da62c`, `door-with-frame-2f2f149f`, `interior-wood-d1-32707dc` | 5 | 2 | 7 |

The final review inspected all 32 prepared front previews, source records and
composed rendered Panel/Frame/Handle meshes. It compared centered vertex sets
in canonical axes, normalized by height, using bidirectional nearest-neighbor
maximum distance and X/Y reflections. This detects reuse despite float noise,
translation, uniform scale and handedness; it is a screening method, not a
proof against every remeshed or differently rotated derivative. Whole-asset
fingerprints alone had missed cross-pack component reuse:

- The essential, interior-wood and front packs reuse residential frames. For
  example, front-002 versus essential front differs by only `2.60e-7` of frame
  height in the vertex-set screen. Their three source groups are therefore one
  15-door train family. Keeping the front pack in test would leak shared geometry.
- Bathroom versus interior-wood-008 also matches panel vertices within `1.25e-7`
  of panel height, but the handles differ. Preserve both qualified assemblies
  in the same family; panel reuse alone does not make them mere recolors.
- No additional whole-assembly match appeared within `1e-5` of assembly height
  in this screen. The industrial pack has different frame/panel construction.
  Similar raised-panel forms from independent authors were reviewed as generic
  resemblance, with distinct relief/frame/hardware. Both metal-door sources by
  Mehdi Shahsavan are conservatively kept together in test.

The review scripts, component comparisons and preview sheets are retained at
`~/.cache/alexdoor-xas/verification/corpus-freeze/20260928-phase5-closeout/`.
The manifest records the family rationale, each identity's handedness, existing
geometry fingerprint, usage scope, exact expert angle and evidence run. A digest
of each canonical three-record bundle and of the common setup detects silent
changes to the freeze; the qualification revision identifies the executable
baseline. These bindings do not package ignored USD payloads or external runtimes.

`qualification.corpus.load_corpus` and `scripts/verify_door_corpus.py` validate
full membership, family/source separation, both handednesses, frozen records,
counts and valid paired expert summaries. The optional local-evidence check
also compares qualification inputs/setup, detailed results, release-ended traces
and RGB/depth/camera frame inventories. It checks inventory rather than decoding
images or rerunning physics. The freeze audit passed all 64 trials and 7,850
saved RGB-D pairs; previously acquired image/physics validation remains applicable.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_corpus.py \
  --evidence-root ~/.cache/alexdoor-xas/verification/expert
```

Both hands occur in each partition, but development has only one right-hand
identity and train is dominated by a single 15-door residential family. Test has
six reviewed families across seven doors. Report per-door and handedness results;
this corpus does not establish broad population coverage. No additional intake
is required for the defined Phase 5 gate. Local-only and private restrictions
remain in force, including private preparation/tests only for the VOID FRAME
asset; freeze grants no new learning-use or redistribution permission.

This is an asset identity split, distinct from the retained numerical episode
split utilities. Phase 6 must take membership from this manifest, keep test
qualification evidence out of learning/tuning, and generate matched training
episodes only from the assigned training identities. No demonstrations or
learned-perception claims are produced by Phase 5 closeout.

### Visibility diagnostic limits

Offline replay at `cd71bca` reproduced the original counts on all 750 saved RGB-D
frames of six complete cycles (22,594 trace rows). Reports and overlays remain in
`~/.cache/alexdoor-xas/verification/visibility-targeted/20260928-offline-review/`.
Frame visibility fails transiently on `door-adf292f437f2` because the 40-point cap
underrepresents a visible strip, and throughout the metal-door pair because sampled
frame surfaces are initially occluded/out of view. On `psx-front-008-ee7d5c6`, panel
warnings begin around 51.4 degrees; raised members self-occlude fixed surface samples.
Contact-surround checks pass. Alternative surfaces may still be visible in RGB.

These warnings do not justify geometry changes or invalidate expert admission.
All seven targeted rechecks, including completed industrial-004, passed the revised
full-trajectory diagnostic. Qualification traces cannot become learning/tuning data;
observed perception usability remains a separate Phase 6 gate. The older synthetic
results describe the 45-degree setup and are not requalification of the current pose.

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

The completed freeze implements the following contract: a manifest containing
all eligible qualified identities and a disjoint train/development/test split
covering that corpus. Do not discard qualified doors to reach 24 or equal
left/right counts. Choose counts after auditing source families and handedness;
represent both handednesses in each partition and keep related source geometry
together to prevent family leakage. Request additional identities only when a
specific coverage gap makes them necessary, and justify the minimum number.
Record partition counts, coverage limits, the common setup/probe and each door's
expert result. The frozen manifest and reviewed membership are documented above.

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

Completed with all eligible qualified identities included, stable expert references
under the same implementation, and the frozen split. Conclusions apply to this
qualified domain; neither corpus size nor repeated trials prove broad coverage.
The 2026-09-25 user-authorized recovery revises the common setup before any split.
After corpus/split freeze, do not retune it from held-out assets.

## Maintained Entry Points

- `scripts/prepare_doors.py` and `qualification/`: shared intake and promotion.
- `scripts/qualify_door.py`: frozen real-door probe, paired decision and per-door result.
- `scripts/verify_door_corpus.py`: frozen asset membership, family separation and reference checks.
- `scripts/verify_door_preparation.py`: synthetic format/negative/physics checks,
  run only when affected infrastructure changes justify them.
- `DoorInspectionEnv`: robot-independent door measurement, still used by B1.

Historical infrastructure results are recorded at `d12f0b2`; accepted intake
through `9c16e3d`. Preparation-pool cleanup is recorded at `7a1ffb1`.
