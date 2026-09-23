# Project Status

Current as of 2026-09-23. Subphases 4.0 and 4.1 are implemented and GPU-verified.
Subphase 5.0 preparation infrastructure is implemented and verified; real-door
intake has started. Five doors pass static, visual and isolated GPU checks
in the common closed-unlatched state and are ready for Subphase 5.1. See
[[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]]
for commands, supported formats, admission rules and evidence.

The fifth source, `door-2738468b94d74c5f` (DJMaesen, CC BY 4.0), is now
prepared in attempt `000008`. Its internal pin/barrel mesh overlaps are represented
by the ideal revolute joint with explicit contact-pair filters; external hinge,
leaf, frame and handle collisions remain enabled. Correcting the measured pin
axis, opening side and inflated plate hulls resolves the previous blocker without
further shrinking or relaxed physics tolerances. The prepared 0.945 × 2.038 ×
0.050 m leaf is left-handed in the canonical push convention. Static and viewed
RTX previews pass; one RTX 4090 run reaches 113.1 degrees with exact resets,
0.000191-degree passive drift, zero frame drift and no reported penetration.
The detailed bearing is an approximation; robot/expert qualification remains pending.

The fourth supplied source, `void-frame-studio-animated-classic-door-08bdf51b`,
is CC BY-NC-ND 4.0. Its initial source-review failure is superseded by a
user-approved `private_noncommercial` scope: no commercial use or sharing adapted
assets. Both original downloads have degenerate triangles; a local GLB copy omits
only 7,787 negligible-area triangles and passes inspection with 184,865 triangles.
After reviewed 2% moving-assembly fitting, 3.1 mm translation and grouped
hardware collision, attempt `000008` passes static and viewed front/rear previews.
On RTX 4090 it reaches the geometric 91-degree limit with three exact resets,
zero frame drift and zero reported penetration. The 0.819 × 2.283 × 0.035 m leaf
is left-handed in the canonical push convention. It is ready for 5.1 within its
private/noncommercial scope; both downloads and earlier attempts are preserved.

The next supplied source, `modern-door-2fb8d024`, has a Sketchfab Free Standard
license. Its earlier CC-only rejection is superseded by the approved local-only
intake scope. The GLB now passes preparation in attempt 000012 after grouped
solid-leaf cooking, a documented 0.4% uniform moving-assembly reduction for side
clearance, and an inferred opening-face hinge. Isolated RTX 4090 physics reaches
the geometry-derived 93-degree limit with stable resets/frame and no reported
penetration. Robot/expert qualification remains pending. Local-only source and
normalized assets must remain
with the authorized licensee and outside shared asset packages.

The third supplied source, `door-with-doorframe-c29da62c`, now passes local technical
preparation. Sketchfab's NoAI restriction concerns programs designed to generate
new content; B1 policies output robot actions, so that label alone does not
establish incompatibility. The original downloads contain Poliigon-named textures
with unverified ML rights. A geometry-identical local GLB removes all nine source
images and uses independently authored flat materials; the original downloads
remain untouched. Source review and inspection pass for this local-only derivative.
The earlier 1% recipe genuinely overlaps the frame. Attempt `000009` resolves it
with reviewed 2% uniform fitting, 0.99 mm hinge-side translation and a corrected
hinge-side canonical rotation. The leaf is 0.879 × 2.097 × 0.080 m and left-handed
in the canonical push convention. Static and visual checks pass; RTX 4090 reaches
161.2 degrees with stable resets/frame and no reported penetration. It is promoted
as local-only; it was the third ready door, with 5.1 still unrun. Flat local
materials suffice for preparation; later Replicator variation must use permitted
materials. Physics tolerances and the frozen setup are unchanged.

The common setup is frozen in `configs/purdue_synthetic_probe.json`: all four
exact-width synthetic doors exceed 45 degrees through sustained contact and safe
release. Paired minima are 66.15/77.91 degrees for left/right 0.65 m doors and
46.35/48.07 degrees for left/right 1.20 m doors. Repeats agree exactly and retain
the same safety-stop cause. The fixed head view passes all eight full cycles.
Code and tests define executable behavior.

## Current Runtime

The sole registered robot environment is `AlexDoor-DoorPush-Purdue-v0`: fixed-base
Purdue Alex003, measured pedestal, WSG32/UMI v1, seven right-arm joints and head
ZED X Mini Wide. It supports the original panel/frame/handle commissioning fixtures and optional
articulated synthetic doors for common-setup qualification. It is not a learned
training task.

- A1 addresses all seven joints; A2 actuates tool translation and rotation;
  A3 transforms an explicitly supplied frame into A2.
- Gravity compensation uses the model, with external PD gains and physical limits.
- Raw contact diagnostics resolve imported rigid owners and distal-finger points.
  Exact panel partners are authorized; other contact surfaces/partners are rejected.
  Force reports include normal components only.
- RGB-D at 960 × 600 is captured with arm/neck proprioception, time and frame IDs.
  Valid depth contains no segmentation or simulator object labels. The Gym policy
  tensor remains proprioception-only until Phase 6; capture is reusable separately.

See [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] and
[[topics/purdue-b1-robot-and-contact|Purdue Robot and Contact Contract]] for
interfaces, numerical defaults, verification and approximation limits.

## Migration Boundary

B0 robot execution, calibrated scripted preset and simulator asset loader are
retired. Existing datasets, checkpoint loading, model training and historical
results remain available. Calibration/manifests needed to interpret the historical
offline contracts are retained; they are not Purdue configuration.

Full generation and learned-policy evaluation commands now fail clearly before
simulator startup or output creation. B0 checkpoints are never reinterpreted as
Purdue-compatible models. Generic data/adapter components remain tested for their
software contracts, without claiming B1 learned execution.

## Verification

The dependency/CUDA preflight passes on the RTX 4090. The last full software run
passed 350 tests, including historical data/model contracts. The subsequent rebate
correction passed 29 focused preparation/qualification tests, Ruff, wiki-link/index
checks and one complete synthetic GPU regression. The subsequent closed-unlatched
change passed 14 focused preparation tests and one real-door GPU run; unrelated
checks were not repeated.
The modern-door repair passes 20 focused preparation tests, static/visual checks
and one RTX 4090 functional run. Shared recipes now support reviewed solid-surface
collider groups and bounded uniform clearance fitting; the frozen setup is unchanged.
GPU checks pass three stable resets, seven-joint/full-pose control, loaded distal
contacts and forbidden-contact detection, and synchronized metric head RGB-D.
The local pose targets remain within 0.0204 mm / 0.00209 degrees during their
0.5 s holds; this is synthetic commissioning evidence, not hardware accuracy.

The complete post-integration regression passes in
`~/.cache/alexdoor-xas/verification/purdue-after-41/`.
The original 4.0 run is retained in the sibling `purdue-final/` directory.
Targeted final checks in sibling `purdue-final-rgbd/` and `purdue-final-contacts/`
verify reset renderer settling and force direction. Numerical traces, raw contact
records and representative images are retained. The separate D0 door-only GPU
smoke also passes; it establishes no B1 corpus qualification.

`~/.cache/alexdoor-xas/verification/door-preparation-50/` adds four normalized
synthetic doors with passing static/GPU reset, drift, hinge and opening checks.
All eight supported input suffixes pass textured conversion checks; malformed
assets and an injected physical obstruction are detected. Front/rear textured
previews were rendered and inspected. A later real-door run in
`assets/doors/b1/door-with-frame-2f2f149f/attempts/000007/` passes technical readiness
under the closed-unlatched policy. No expert reference is qualified.

The launcher still warns that `setup_conda_env.sh` is absent. Its actual Python,
Isaac provenance, Alex assets, ZED dependency and CUDA preflight pass. No runtime
installation or driver modification was made.

## Maintained Entry Points

- `scripts/check_env.py` — supported runtime and Purdue/pedestal/ZED preflight.
- `scripts/verify_purdue_runtime.py --viz none --device cuda:0` — complete operational gate.
- `scripts/screen_synthetic_setup.py` — GPU kinematic candidate screening only.
- `scripts/verify_synthetic_setup.py` — synthetic physics, controlled probe and candidate search;
  see Phase 4 for arguments and the frozen setup/qualification boundary.
- `scripts/verify_benchmark_scene.py` and `scripts/verify_adapters.py` — route to that same gate.
- `scripts/verify_dataset_interface.py` — retained historical dataset interface checks.
- `scripts/train_policy.py` — offline training on existing supported data.
- `scripts/prepare_doors.py` — B1 review, inspect, normalize, static, physics, preview and promote.
- `scripts/verify_door_preparation.py` — synthetic infrastructure and format verification.
- `scripts/prepare_phase4_1_assets.py`, `scripts/normalize_phase4_1_door.py`,
  `scripts/verify_phase4_1_doors.py` — retained legacy preparation, with the limits below.

`eval_policy.py`, `run_scripted_baseline.py`, `verify_policy_rollout.py` and
`verify_scripted_baseline.py` are explicit migration stops, not available B1 workflows.

## Legacy Preparation and Storage

Preparation still assumes the B0 anchor, simple panel/jamb/header collision boxes,
non-colliding handles, 0–90-degree stops and the previous mass/damping template.
Its physics inspection now instantiates only the door: it measures reset, drift
and applied-torque response, not collision-consistent robot qualification.
The new Phase 5.0 workflow replaces these assumptions for B1 preparation; the
legacy commands remain separate and cannot admit B1 candidates.
The ignored worklist and all local assets were preserved.

Historical D0–D4 layers, datasets and policy outputs are unchanged. Verification
reports and images live in the project cache; no generated dataset, corpus,
training run or media was added to the tracked output tree.

## Next Phases

The common floor/contact/neck setup and expert protocol are frozen on synthetics.
The first candidate, `door-with-frame-2f2f149f`, is technically ready for 5.1 after
explicitly omitting latch/lock bolt collisions for the common closed-unlatched task.
It reaches 188 degrees in isolated GPU physics with stable resets/frame and no
reported penetration. The leaf, frame and handles retain their collisions.
The initial batch of five distinct real doors is prepared. Start the unchanged
common robot/expert protocol in 5.1 only when requested; the next-candidate
[[implementation_phases/phase-5-door-corpus-and-qualification|handoff]] remains
available for further sequential intake. Expert qualification, the final corpus
and split are still pending.
Phase 6 integrates observations and all learned A1–A4 paths, then a demonstration
pilot. Phase 7 owns training and generalization evaluation.

There is no active gaze, wrist-camera requirement, corpus download, physical
robot control, hardware-safety claim or sim-to-real qualification in 4.0–4.1.

## Historical Results

The completed scale study used 550 matched A2/A3 episodes across D0-D4 with nested N50, N100, N250, and N500 training views. Sixteen ACT/Diffusion x A2/A3 x data-size cells were evaluated over 576 rollouts. Every rollout succeeded, so the benchmark did not select a policy family, representation, or dataset size.

One ACT-A3-N50 rollout at seed 112 produced a reproducible 219.95 N one-tick peak. Two +/-1 mm door-position perturbations reduced the peak, but the original cell remains `REVIEW_REQUIRED`.

These are historical scientific conclusions, not active workflows. See [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] and [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]. Git retains removed implementation and evidence files.

## Version Notes

- 2026-09-23 — Repaired DJMaesen preparation with reviewed internal hinge contact pairs, a measured pin axis and corrected opening side; static, viewed previews and one RTX 4090 run pass at 113.1 degrees. Five doors are ready for 5.1; expert qualification remains pending.

- 2026-09-23 — Reviewed DJMaesen's CC BY door and its USDZ; inspection passes, but the maximum fit and a small shift leave true internal hinge crossings. The candidate is held before static/visual/GPU gates; four doors remain ready for 5.1.
- 2026-09-23 — Prepared Void Frame Studio's door under private/noncommercial scope; local removal of negligible-area triangles, 2% fitting and reviewed hardware colliders pass static, visual and one RTX 4090 run. Four doors are ready for 5.1.
- 2026-09-23 — Recovered Theocritus door with reviewed 2% fitting and corrected hinge-side mounting; 25 focused tests and one RTX 4090 run pass. Three doors are ready for 5.1, including two local-only.
- 2026-09-23 — Prepared a texture-free local derivative of Theocritus's door; source and inspection pass, but normalization remains blocked by a genuine leaf/frame overlap at the current 1% fitting limit.
- 2026-09-23 — Corrected Theocritus source review to `unresolved`: NoAI alone does not show a conflict with action policies, while Poliigon-named texture rights remain unverified.
- 2026-09-23 — Initially rejected Theocritus's NoAI-marked door before preparation; that source decision is superseded by the correction above.

- 2026-09-23 — Inspected the supplied modern-door GLB; matching surface geometry and a separate opening obstruction leave normalization unresolved without a supported hinge/clearance correction.

- 2026-09-23 — Added local-only Sketchfab Free Standard intake for internal B1 study while keeping CC-only assets as the redistributable class; the modern-door USDZ has an unresolved closed-pose collider issue.

- 2026-09-23 — Rejected the Ahmed sayed modern-door source at the Phase 5.0 license gate; kept the local download intact and recorded no technical preparation claim.

- 2026-09-23 — Adopted closed-unlatched preparation and proportionate checks; the first real door now passes and is ready for 5.1.

- 2026-09-23 — Corrected rebated-frame preparation and verified the first candidate has an original latch/strike intersection; retained it without promotion.

- 2026-09-23 — Inspected the first user-supplied real door; recorded source evidence and an unresolved preparation outcome without claiming physical qualification.
- 2026-09-23 — Verified Subphase 5.0 preparation infrastructure, conversion paths and static/GPU gates; awaiting the first candidate URL.
- 2026-09-22 — Implemented Purdue operational integration and retired B0 execution, preserving historical readers and isolated door preparation.
- 2026-09-22 — Completed preparatory cleanup and revised the B1 nine-subphase plan.
- 2026-08-13 — Recorded the historical B0 system and approved held-out-door study.
