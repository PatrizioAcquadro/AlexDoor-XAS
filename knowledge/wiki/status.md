# Project Status

Current as of 2026-09-22; documentation/source inspection only, with no fresh runtime validation. Code, configurations, and deterministic tests define executable behavior. This page describes only the maintained repository; completed research is summarized separately as historical evidence.

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

## Superseded Local Infrastructure and Next Action

At the documentation baseline, `main` was three commits ahead of the local
`origin/main` reference: `73ff306` (old plan), `6d74789` (old Phase 4.1 qualification
tooling), and `30b2e46` (Alex package contract). The tooling includes
`src/alexdoor_xas/door_qualification.py`, the three `phase4_1` preparation,
normalization, and verification scripts, tests, and runtime overrides. Its
existence is not evidence of a completed corpus or the revised robot setup.

Next, separately audit those changes and their consumers against the new plan.
Retain reusable normalization/static/physics/measurement capabilities where
justified; remove or isolate superseded qualification-count/bootstrap and
shared-angle orchestration. Preserve still-needed Alex package compatibility.
Do not blanket-revert the commits, rewrite history, or implement Phase 4 as part
of that cleanup. The 2026-09-22 update changes documentation only.

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

- Revised Phases 4–7 and VLA work have not started; superseded local qualification tooling exists as described above.
- Learned policies are state-only; image and language inputs are absent.
- Results cover one simulated door family and seed-0 training.
- Simulator success and force measurements do not establish hardware safety, sim-to-real readiness, or broader generalization.
- No repository command controls a physical Alex robot.

Implementation of the revised B1 plan has not started. Physical-robot,
sim-to-real, VLA, and later articulated-object work remain separately scoped.

## Version Notes

- 2026-09-22 — Reduced the plan to nine subphases and made Phase 5.0 infrastructure readiness precede sequential user-provided URLs.

- 2026-09-22 — Separated current B0 behavior, superseded local tooling, and the approved Purdue/RGB-D Phases 4–7 plan.
- 2026-08-13 — Reconciled the wiki with the simplified current repository and separated maintained behavior from concise historical evidence.
- 2026-08-13 — Recorded the approved visuoproprioceptive held-out-door generalization study without changing current implementation claims.
