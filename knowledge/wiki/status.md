# Project Status

Current as of 2026-09-22; preparatory cleanup, software tests, and dependency/CUDA preflight only, with no fresh simulation validation. Code, configurations, and deterministic tests define executable behavior. This page describes only the maintained repository; completed research is summarized separately as historical evidence.

## Current B0 System

AlexDoor-XAS maintains one simulation workflow:

`scripted Alex V2 door push -> matched v2_pose A1-A4 exports -> A2/A3 ACT or Diffusion -> adapter-v1 -> closed-loop evaluation`

- The simulator runtime is the fixed-base IHMC Alex V2 torso in the single registered `AlexDoor-DoorPush-AlexV2-v0` environment.
- D0-D4 are the maintained B0 door poses. The runtime uses one calibrated six-joint, position-only right-arm controller and exact-door raw PhysX contact sensing.
- New recordings use `phase2.v2`. Existing `phase2.v1` episodes and legacy A4 records remain readable, but the repository does not write them.
- A1 is export-only. A2 and A3 support scripted execution and state-only learned policies. A4 is exported and adapter-executable but has no learned policy.
- ACT and Diffusion share dataset, configuration, checkpoint, run-allocation, and closed-loop reporting primitives while retaining separate models and training logic.

See [[topics/system-architecture|System Architecture]] for the current data flow and [[topics/alex-v2-benchmark|Alex V2 Benchmark]] for the simulator contract.

## Approved B1 Study and Execution Order

B1 compares A1-A4 x ACT/Diffusion on held-out push-door identities using fixed-base
Purdue Alex + WSG32/UMI v1, seven right-arm joints, head ZED RGB-D/proprioception,
and progress relative to a frozen expert. Its main score includes invalid policy
rollouts as zero valid progress. A 45-degree expert opening is asset admission
only; no shared opening angle stops the controller or ranks policies.

The approved plan has nine subphases (2/2/3/2):

1. [[implementation_phases/phase-4-robot-and-task-configuration|Phase 4]] — 4.0 makes Purdue Alex003 with its measured pedestal, WSG, pose control, and RGB-D operational; 4.1 selects the common synthetic-qualified pose/contact setup and checks visibility.
2. [[implementation_phases/phase-5-door-corpus-and-qualification|Phase 5]] — 5.0 first builds and validates preparation/checking tools, then processes one user-provided URL at a time; 5.1 qualifies each prepared door and freezes the 24-door 12/4/8 split.
3. [[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]] — 6.0 supplies observations/perception; 6.1 completes all learned A1-A4 paths; 6.2 passes a small pilot before producing final demonstrations.
4. [[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7]] — 7.0 trains the matrix; 7.1 evaluates/analyzes ID/GEO first, then the retained stress tiers.

There is no mandatory backbone comparison. Gaze, Mimic, and teleoperation are
implemented only for a demonstrated need. Public-release packaging is a final
checklist, not an additional research subphase. The Phase 5.0 infrastructure can
be delivered before any URL is supplied; corpus completion still requires actual
assets to pass preparation and expert qualification. No infrastructure has been
implemented by this documentation update.

The [[decisions/visuoproprioceptive-generalization-benchmark|scientific decision]]
owns metrics and information boundaries. The
[[topics/purdue-b1-robot-and-contact|robot/contact contract]] records local source
facts and planned choices. No revised phase has been implemented or executed.

## Pre-B1 Cleanup and Phase 4 Entry

Audited `73ff306`, `6d74789`, and `30b2e46` against the approved plan at
`5ccf95c`. The tracked working tree was clean; `main` was the only local branch.
The revised Phase 4–7 pages and scientific requirements are preserved.

| Local change | Audit outcome |
|---|---|
| `73ff306` old acquisition plan | Already superseded by the approved documentation; no revert. |
| `6d74789` door tooling | Keep reusable preparation and measurements; remove the old expert/corpus protocol. |
| `30b2e46` Alex contract | Keep package discovery/install path, asset root, URDF identity, and runtime fingerprint. They match the installed external Alex package and current B0 consumers. |

Retained capabilities:

- `src/alexdoor_xas/door_qualification.py` retains dimensions/inertia, bounded
  connected-component discovery and seam welding, geometry duplicate screening,
  metadata checks, file checksums, handedness, and raw sustained-angle measurement.
  The latter now requires an explicit sampling window; it does not establish
  controlled-contact validity or a B1 expert reference.
- `scripts/prepare_phase4_1_assets.py` retains only `init`, `gate`, and copy-only
  `ingest`, because the normalization and checking scripts consume its worklist.
  Metadata checks do not independently verify license evidence. The old slot
  format and paths remain local intermediate formats, not a B1 manifest.
- `scripts/normalize_phase4_1_door.py` retains mesh inspection/separation, recipe
  transforms, material conversion, and legacy USD authoring.
- `scripts/verify_phase4_1_doors.py` retains only `static` and `physics`: canonical
  structure checks, reset/drift checks, and isolated torque measurements. The
  torque sweep no longer requires an arbitrary 80-degree opening or reports an
  unmeasured zero penetration. Durations derive from the environment cadence;
  torque hinge state must be finite. Results explicitly identify their limited scope.
- The diagnostic scene override is now `door_scene_usd`; its sole script consumer
  and configuration test were updated. Default D0–D4 behavior is unchanged.
  The existing left/right scripted geometry and its tests remain reusable.

Removed the old nominal/repeatability rollout orchestration, hard-coded
zero-or-three diagnostics, time-aligned curve acceptance, bootstrap/adaptive
`n_qual`, old manifest completion rules/finalizer, and their protocol-specific
tests. Removed destructive rejection cleanup, the unused rejection ledger writer,
source-moving/browser-ingest workaround, and final workspace orchestration.
No retained consumer references these removed APIs or commands.

The B0 controller's 50-degree target and existing success/adapter semantics remain:
current generation, recording, and evaluation consume them. Removing those would
implement part of the B1 migration, outside this cleanup. No new B1 qualifier,
split, expert normalization, or controller was added.

### Cleanup Validation

`ruff check .`, syntax compilation, CLI help/retired-command checks, and
`git diff --check` pass. Two focused pytest runs passed: 50 tests for geometry,
scripted behavior, Alex/environment contracts, and B0 scene assets; then 125 for
updated preparation/source-preservation checks, documentation, data engine,
adapters, recording, and rollout consumers. These runs overlap in preparation
tests; they are not a full-suite or simulation claim. Wiki links/index coverage
pass. No model training, Kit simulation, or Phase 4–7 gate was executed.

### Retained Limits and Next Action

The retained scripts are **legacy preparation components, not Phase 5 readiness**.
Their authoring still assumes the B0 anchor, simple panel/jamb/header collision
boxes, non-colliding handles, 0–90-degree stops, and the old mass/damping template.
The static checks enforce that legacy format. The physics check disables robot
collisions for a door-only measurement; it does not measure penetration or prove
collision-consistent opening, robot reachability, contact validity, or safe release.
Conversion coverage, dependency handling, and preservation of source geometry
still need the Phase 5.0 validation specified in the approved plan.

The existing ignored worklist is empty. Its rejection ledger and all local asset
storage were preserved; no assets were downloaded, normalized, deleted, or simulated.

Dependency/CUDA preflight passes on the host RTX 4090. The sandbox alone initially
hid CUDA; no CPU fallback was used. The launcher still warns about missing
`setup_conda_env.sh`, while its actual Python, pinned Isaac provenance, asset paths,
and Alex module checks pass. No confirmed dependency blocker to beginning Phase 4
was found; simulator startup and physical integration have not been revalidated.

Next is the separately authorized Phase 4.0 integration and GPU validation of
Purdue/WSG, seven-joint pose control, contact ownership, pedestal, and synchronized
head RGB-D. Phase 4.1 then selects the common setup on synthetics. The retained
legacy geometry limitations belong to later Phase 5.0; missing candidate URLs do
not block Phase 4 or the Phase 5 infrastructure milestone.

## Maintained Entry Points

`scripts/check_env.py` checks the supported workstation, Isaac, and external Alex dependencies. The maintained behavior gates are:

- `scripts/verify_benchmark_scene.py`
- `scripts/verify_scripted_baseline.py`
- `scripts/verify_dataset_interface.py`
- `scripts/verify_adapters.py`
- `scripts/verify_policy_rollout.py`

Generation, training, and evaluation use `scripts/run_scripted_baseline.py`, `scripts/train_policy.py`, and `scripts/eval_policy.py`.

## Active Configuration and Storage

The complete active `configs/` surface is `alex_v2_door.json`, `scripted_baseline.yaml`, `act.yaml`, and `diffusion.yaml`.

- `datasets/` stores reusable task/action-space/version exports, shared split files, retained views, and normalization artifacts.
- `outputs/door_scene/` stores exactly the five canonical D0-D4 layers.
- `outputs/door_push_alex_v2/{act,diffusion}/` is reserved for learned-policy runs.
- `outputs/wandb/` exists only when optional W&B tracking is enabled.
- Verification evidence, arbitrary scenes, and scripted-run staging belong under `~/.cache/alexdoor-xas/`.

The repository does not maintain run-specific cluster packages, dataset-construction workspaces, or historical result bundles in the active output tree.

## Historical Results

The completed scale study used 550 matched A2/A3 episodes across D0-D4 with nested N50, N100, N250, and N500 training views. Sixteen ACT/Diffusion x A2/A3 x data-size cells were evaluated over 576 rollouts. Every rollout succeeded, so the benchmark did not select a policy family, representation, or dataset size.

One ACT-A3-N50 rollout at seed 112 produced a reproducible 219.95 N one-tick peak. Two +/-1 mm door-position perturbations reduced the peak, but the original cell remains `REVIEW_REQUIRED`.

These are historical scientific conclusions, not active workflows. See [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] and [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]. Git retains removed implementation and evidence files.

## Retired Surface

The repository no longer maintains surrogate robots, generic door-task layers, sensorless execution, multi-environment runtime support, calibration authoring, scale-dataset construction, cluster/Slurm transfer, pilot and sweep orchestration, smoke or unified matrix runners, or compatibility shims for removed source APIs.

One deterministic fake environment remains for software tests. It mirrors the production state contract but is not a supported simulator runtime or physics result.

## Boundaries

- Revised Phases 4–7 and VLA work have not started; only the audited legacy preparation components described above remain.
- Learned policies are state-only; image and language inputs are absent.
- Results cover one simulated door family and seed-0 training.
- Simulator success and force measurements do not establish hardware safety, sim-to-real readiness, or broader generalization.
- No repository command controls a physical Alex robot.

Implementation of the revised B1 plan has not started. Physical-robot,
sim-to-real, VLA, and later articulated-object work remain separately scoped.

## Version Notes

- 2026-09-22 — Audited the three pre-revision commits, removed the superseded qualification protocol, and retained scoped preparation components and Alex compatibility.

- 2026-09-22 — Reduced the plan to nine subphases and made Phase 5.0 infrastructure readiness precede sequential user-provided URLs.

- 2026-09-22 — Separated current B0 behavior, superseded local tooling, and the approved Purdue/RGB-D Phases 4–7 plan.
- 2026-08-13 — Reconciled the wiki with the simplified current repository and separated maintained behavior from concise historical evidence.
- 2026-08-13 — Recorded the approved visuoproprioceptive held-out-door generalization study without changing current implementation claims.
