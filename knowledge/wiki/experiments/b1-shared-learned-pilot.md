# Shared Learned A1/A2 Door-Pushing Pilot

Baseline `060a0e8af9bb7e8249af9ebdf5aa6db15928ea91`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
This is a **bounded, incomplete pilot**, not completed end-to-end evidence. See
[[b1-approach-controller-compatibility|the latest controller/reference results]]
and [[../topics/b1-policy-pilot|existing observation/admission contracts]].

## Scope and dependencies

Fixed-base Alex003 pushes a closed, unlatched panel. The 19/6/7 corpus split stays
unchanged and test identities remain sealed. The two existing train identities
and existing development identity remain selected. New-controller demonstrations
must be newly recorded, with one shared RGB-D/proprioceptive recipe for ACT and
Diffusion across A1/A2. Teacher geometry is supervision; independent evaluator
truth never enters policy or operational admission. No production collection,
20-run campaign, physical robot trial, RL or separate perception frontend.

Execution order is reference feasibility -> load/fault/stop and observed local
support -> teacher/new data -> four learned runs and independent physical panel.
Encoder/data-reuse work can proceed independently of robot-contact readiness.
A1/A2 require the local action contract, not a hinge unused by their conversion.
An essential absent observed provider remains a blocker; missing bounds or
qualification flags must not be fabricated. Synchronized simulation may be slower
than wall time; physics continues after stops.

## Finite pilot ceilings fixed before implementation

- Reference: one candidate replacing the demonstrated infeasible Cartesian path;
  at most one correction justified by a demonstrated cause. Each candidate has
  two full train approaches with endpoint holds/stops and two moving stops.
- Dynamic/contact prerequisites: at most 12 new bounded physical cases, selected
  before their execution from the candidate path/domain; existing valid tests reused.
- Collection: at most six new complete episodes (up to two per selected identity),
  all failed/interrupted attempts retained; maximum 300 GiB additional storage.
- Training: 3,000 updates per four model/representation cells, seed 17; at most
  one common evidence-justified correction with 1,500 further updates per cell.
  This is a ceiling of 18,000 updates, not a mandatory allocation.
  The local cost trial plus remote compatibility trial charge 164 synthetic
  updates to this ceiling, leaving at most 17,836 for subsequent real-data work; it produces no pilot checkpoint. Checkpoints
  selected only by development evidence; no development normalization/updates.
- Evaluation: three exact initial conditions on each of the two pilot train
  doors, four cells, 24 outcomes. Conditions/physical rules/expert references
  freeze before any learned evaluation; ranges must follow train/development
  feasibility evidence. Every refusal stays in the denominator. Bug fixes require
  transparent affected-comparison reevaluation, never favorable seed retries.
- Local CUDA workload ceiling: three GPU-hours excluding shared-GPU waiting;
  Gilbreth at most two brief Slurm jobs and one allocated GPU-hour total. If any
  ceiling is exhausted, preserve verified work and state that the goal is incomplete.

The original three camera-enabled teacher recordings took 301.85/373.00/312.40 s
wall time and the historical pilot tree occupies 88 GiB including prepared data
and models. Six episodes therefore extrapolate to roughly 35 minutes collection
and 176 GiB with the old storage layout, before new qualification/evaluation.
These are planning estimates, not new measurements or permission to skip gates.
Old small-model updates cost about 6-8 ms after warmup; a learned visual encoder
requires a new throughput/memory measurement before committing to its full budget.
There is approximately 1.6 TiB free on the workstation at initial inspection.

## Acceptance and current state

Goal completion requires admitted learned commands for all four cells, loadable
checkpoints and at least one identical checkpoint completing a valid episode on
both train doors within the frozen panel: sustained controlled opening >=80% of
each expert reference, valid holding/release, budget and physical rules. Report
all 24 outcomes and partial progress. This is pilot functionality, not the future
>=80% unseen-door reliability claim. Essential A3/A4 recording omissions must be
resolved; reproducible load/visualize/record/evaluate commands, remote checkpoint
roundtrip, measured costs and the proposed production/20-run/evaluation budget
must be delivered before completion. Current state: **incomplete**. Both reference
candidates and the subsequently authorized IK cycle were rejected. A further
explicitly authorized initial-placement study now passes both full train
approaches and all four unloaded cases with the original solver. Contact/load
and observed-admission prerequisites remain incomplete. No new physical
demonstrations, real-data training or learned command is admitted. Essential observed support remains unresolved; remote access and
the A100-to-workstation checkpoint return are verified. No new trajectory is
installed in the controller, teacher or policy adapters.

Local evidence root: `outputs/evidence/b1-shared-learned-pilot-20261016/`.
The original objective and completed previous TODO are preserved there. `TODO.md`
is the implementation checklist and remains ignored, consistent with repository
instructions. Changes are made only in the requested worktree; no push.

## Bounded reference outcome

The frozen controller, old source initial states, endpoint task, 60/120 Hz clocks,
physical limits and original criteria are retained. This tests references in
unloaded robot-only simulation; no door is spawned and any contact aborts.

Candidate 1 maps a quintic joint segment to Cartesian FK goals. Endpoint IK is
seeded once from the saved initial joints. Movement time is at least 4.2 s and
at least `1.875 * maximum joint travel / (0.8 * 0.5 rad/s)`, followed by the
original 1.8 s dwell. The 20% target-speed reserve is a planning choice, not a
measured-speed guarantee. Two full trajectories, endpoint holds/stops and two
moving stops at peak planned joint speed complete on CUDA.

| Candidate 1 | Right train door | Left train door |
|---|---:|---:|
| Approach command ticks | 360 | 507 |
| Position p95 / maximum, mm | 0.883 / 0.897 | 11.941 / 12.849 |
| Rotation p95 / maximum, degrees | 0.246 / 0.249 | 3.222 / 3.468 |
| Measured peak joint speed, rad/s | 0.190 | 0.465 |
| Rate-scaled / position-clipped targets | 0 / 0 | 0 / 184 |
| Planned minimum joint margin, rad | 0.266 | 0 |
| Moving-stop travel / rotation | 3.373 mm / 0.373 deg | 5.067 mm / 0.797 deg |
| Full-path 10 mm / 5 degree criteria | Pass | **Fail** |

All native feedback is finite/available at both substeps; holds and continuing-
physics stops pass, without effort clipping. The original residual calibration
still refuses outside its domain: 57/720 right and 55/1,014 left approach samples
are available. No calibration/domain or contact qualification is promoted.
Six-second timing alone was not the remaining cause: the left FK-generated path
still reaches the wrist boundary even without target-rate scaling.

Candidate 2 corrects that demonstrated endpoint boundary by adding a 0.05 rad
planning margin, above the old 0.04798 rad maximum servo discrepancy. The same
endpoint, seed, controller and criteria remain. The right full/moving cases
complete; left is refused **before actuation** because the bounded 100-iteration
local endpoint solve retains 9.351 mm / 2.221 deg error, beyond the unchanged
1 mm / 0.1 degree planning screen. This does not prove global unreachability.
No third endpoint seed, new gain, threshold relaxation or favorable retry runs.
The rejected planner and its margin option are removed from active source after
preserving local sources, completed traces, refusals and protocol.

The two-candidate reference ceiling is exhausted. The subsequent one-cycle IK
authorization and its rejection are described below; neither study establishes
a feasible continuous left path at the original criteria. No new
placement range is invented from these failures; the varied-pose panel remains
unfrozen and unexecuted, rather than counted as 24 evaluated failures.

`reference-candidate-1/` retains four complete cases and comparison; candidate 2
retains two complete right cases plus the left planning refusal and exact numerical
error. The first planner reconstruction is explicitly marked non-byte-identical;
its creation script is retained. The second source is preserved separately. The
initial analysis-only missing-static-screen error and benchmark import failure
are infrastructure diagnostics, not new physical/model outcomes. Neither triggers
a repetition of completed physical cases or the passing 128-check suite.

## A3/A4 recording reuse audit

All three historical episodes are read: 14,655 commands including inspection,
10,155 manipulation commands and 41,458,642,501 raw-file bytes. Exact causal
pre-command joins, N+1 observations, nine-joint ordering, 120 Hz feedback times,
RGB-D/intrinsics/optical depth conventions, fixed root and neck calibration,
applied joint/neck/effort targets, measured q/dq/ddq/J/torque, phases and terminal
observations are present. Evaluator hinge/angle/contact references are separately
stored as supervision, never appended to policy features.

Numerical reconstruction using **supervisor-only** annotated frames succeeds on
the two train and one development episodes. A3 position/rotation maxima are
1.74e-18 m / 1.41e-16 rad. A4 fitting yields 2,993 / 3,685 / 2,924 segments for
3,217 / 3,835 / 3,103 commands, with maximum reconstruction errors below
0.978 mm / 0.265 degrees at the unchanged 1 mm / 0.5 degree tolerances.
This verifies retained supervision inputs, not causal observed geometry or compact
A4 control: 9,602 segments cover 10,155 commands, so granularity must be considered
when the matched protocol is designed. No final matched dataset is published.

Essential unresolved production dependencies are concrete:

- The existing SAM3/P2P session supplies diagnostic `DoorEstimate.local` candidate
  poses and support. `Point2PoseTracker.local_state` can retain ambiguous ownership
  and incomplete bounds. No active producer constructs `LocalActionSupport` over
  the **actual** controller-preview continuous sweep and stopping envelope, or
  connects `admission_source` with measured material response and robot/load bounds.
- A1/A2 require that local support, not a hinge. A3/A4 additionally need observed
  object/hinge support at the first command. A stationary closed scan does not
  establish a physical pin; the admitted pre-articulation bootstrap is absent.
- Exact operational reuse needs provider recipe plus causal snapshots, selection,
  support/publication times and common-initialization evidence. Privileged retained
  hinge annotations cannot fill these inputs; no future observation is backdated.
- Final matched export currently requires qualified/frozen release-v2 with
  `legacy-full-state`; the new pilot is `operational-v1`. An operational release
  contract needs explicit evidence and integration before production, never a flag.
- Historical observations/actions cannot become new-controller data by relabeling.
  New load-domain/stop verification and new complete physical recordings are needed.

Only the separate perception tree at `99fa48e` was read. It contains provider,
contracts/admission and observed-articulation code but not this branch's local
admission module; its hinge estimator leaves complete position bounds unmeasured.
No changes to that worktree, second frontend or perception optimization are made.

Local audit commands, from the required worktree, use the workstation interpreter:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p outputs/evidence/b1-shared-learned-pilot-20261016/audit_reuse.py
/home/pacquadr/IsaacLab/isaaclab.sh -p outputs/evidence/b1-shared-learned-pilot-20261016/derive_supervision.py
```

These are read-only numerical audits of historical files. Their JSON reports and
supervision-only arrays remain in the local evidence root; no raw data is edited.

## Remote access and versioned execution

The initial SSH attempt fails public-key authentication with the old configured
agent socket absent; its diagnostic is preserved. The user configures a new key,
and the same connection then succeeds without that socket. One immutable-source/
synthetic-data Slurm job and its workstation CUDA checkpoint return complete;
see the measured-cost section. No remote task/production dataset or extended
campaign is claimed. Isaac remains on the workstation.

## Learned encoder software milestone

`rgbd-conv-v1` is implemented in preparation, both model families, batch units and
checkpoint inference. [[../topics/b1-policy-pilot|The canonical recipe]] gives the
architecture and causal/missing-data behavior. All four new family/representation
paths pass visual-gradient, checkpoint and deterministic prediction checks inside
128 targeted CUDA checks. No real demonstration optimizer update or admitted
learned physical command has occurred; there is no new trained pilot checkpoint.
The historical four grid-policy refusals remain evidence, not new-encoder outcomes.

| New recipe cell | Input/gradient/checkpoint path | Genuine pilot training | Physical rollout |
|---|---|---|---|
| ACT A1 | Pass | Not started | Not started |
| ACT A2 | Pass | Not started | Not started |
| Diffusion A1 | Pass | Not started | Not started |
| Diffusion A2 | Pass | Not started | Not started |

No varied-pose panel is evaluated before its conditions are fixed. Neither zero
executed evaluations nor a synthetic cost trial is reported as 0/24 task failures.

With the required three shared GPU leases and the 180-second idle protocol, use:

```bash
PYTHONPATH="$PWD/src" /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/benchmark_policy_pilot.py --recipe rgbd-conv-v1 --output outputs/evidence/b1-shared-learned-pilot-20261016/new-model-cost.json
```

This creates a **new** benchmark report; the accepted report and all failures stay
unchanged. It is model/software cost measurement, never an admitted rollout.
Future new-controller preparation explicitly uses
`scripts/pilot_offline.py --recipe rgbd-conv-v1` with its existing required root,
mode and protocol arguments. The old collection/training budget in
`configs/b1_policy_pilot.json` is historical; do not use its 96-update limit as the
new learning protocol. No production collection command is authorized by this note.

Remote authentication is restored; the immutable Slurm/checkpoint verification
now passes, as detailed below. No campaign is started.

## Measured costs and proposed campaign budget

Local synthetic training measures model/action loss, gradient clipping and AdamW,
with batch 32 and horizon 12. After five warmup updates, 20 updates/cell measure
median 6.56/6.74 ms ACT A1/A2 and 5.63/5.63 ms Diffusion; PyTorch allocation peaks
are about 220/140 MiB. All four first-convolution weights change under the actual
action loss. There are 100 such local updates and no real-data pilot updates.
This excludes data I/O, EMA, development selection, checkpoint I/O and simulation.

Gilbreth job `11964913` uses account `euge`, `a100-80gb`, normal QoS, one A100
80 GB, four CPUs and 8 GiB requested RAM. It completes in 62 seconds allocated
(0.0172 GPU-hours), with a 54-second Python step and 573,580 KiB step peak RSS.
Python 3.11.15, PyTorch 2.12.1+cu126 and Diffusers 0.39.0 reuse the personal pilot
environment; an isolated venv adds existing project dependency SciPy 1.17.1,
without changing the original environment. No Isaac modules are imported.

The immutable Python-source archive is exactly commit `a54ef52`; its seeded
synthetic fixture, inspection recipe and robot identity are frozen read-only in
`/depot/euge/data/pacquadr/alexdoor-xas/b1-shared-20261016-a54ef52/`. Remote work
does not read a changing checkout. Sixteen action-loss updates per four cells
change CNN weights and save four 9.07-9.96 MB **infrastructure-only** checkpoints.
Batch-8 median update cost is 15.36/15.31 ms ACT and 12.94/12.93 ms Diffusion.
These are synthetic compatibility/cost checks, not door demonstrations or trained
pilot models; they consume another 64 updates from the declared ceiling.

One concrete **proposal**, not a frozen production protocol, is 19 train doors x
six verified conditions x two repetitions (228 episodes), plus six development
doors x six conditions (36 episodes). Which poses/appearance/dynamics constitute
those conditions remains evidence-dependent and unselected. All 20 eventual runs
share physical train episodes/input recipe, use train-only statistics, five seeds
per ACT/Diffusion x A1/A2 cell, and at most 100,000 optimizer updates/run. This
budget does not claim that those updates achieve the reliability target.

| Resource | Extrapolated work | Proposed ceiling/reserve |
|---|---:|---:|
| Production collection, 264 episodes | 24.1 workstation hours from historical 329 s/episode | Included in 200 workstation hours overall |
| Production raw observations | 3.65 TB from measured 13.82 GB/episode | Archive and transfer plan required |
| 20 trainings x 100k updates | RTX batch-32 model-only 3.41 GPU-hours; A100 batch-8 model-only 7.85 GPU-hours | 64 A100 GPU-hours including unmeasured batch-32 I/O/EMA/selection overhead; <=8 concurrent allocated GPUs |
| Future extended evaluation | 20 checkpoints x 7 sealed doors x 3 poses x 2 repetitions = 840 episodes; 76.8 historical workstation hours | Remaining share of 200 workstation hours, including support/review overhead |
| Evaluation full raw layout | 11.61 TB if retaining historical per-tick RGB-D | 20 TB total archival allowance, subject to retention design |
| Checkpoints/logs | Measured approximately 9-10 MB per policy checkpoint | 10 GB reserve; optimizer/resumption storage must be included |

The seven final-test doors are **not opened** to create this arithmetic; future
nominal/varied-pose grouping and exact panel/seed lists require protocol freeze.
All compute-hour and storage figures beyond the small trials are extrapolations.
No monetary cost is measured and no paid service is purchased. Actual learning
or causal geometry failure may require redesign rather than more training time.

The workstation has about 1.6 TiB free and cannot hold the proposed raw campaign.
Current dense float64 concatenated image vectors would add about 0.824 TB before
in-memory record copies; duplicating raw files in each export roughly doubles
raw storage. Before production, use a verified lazy/deduplicated image path and
review immutable archival/transfer capacity. The bounded pilot ceiling remains
six new episodes / 300 GiB; no production data or 20-run job is launched.

`campaign-budget-proposal.json` retains exact arithmetic, measured inputs,
assumptions and prerequisites. Further physical execution needs dynamic/load/stop
qualification of the later accepted nominal pilot placement, plus local observed
sweep/response support. Only after these succeed may the new-controller teacher and exact varied-condition panel be
frozen and genuine pilot learning begin.

## Returned checkpoints and completion boundary

All four returned infrastructure checkpoints load through `B1Policy.from_checkpoint`
on workstation PyTorch 2.10.0+cu128 / RTX 4090. With TF32 disabled on both hosts,
DDIM 10 steps and sampling seed 23, maximum prediction differences are
4.18e-7 / 2.54e-7 ACT A1/A2 and 8.95e-7 / 7.16e-7 Diffusion A1/A2, inside the
predeclared absolute/relative 1e-4 FP32 cross-runtime tolerance. This checks one
fixed query per checkpoint; it does not establish training equivalence across
PyTorch versions, physical rollout success or generalization.

The files are in local `gilbreth/returned/` under this experiment's evidence root:
`act-A1_joint_delta.pt`, `act-A2_ee_delta.pt`, `diffusion-A1_joint_delta.pt`,
`diffusion-A2_ee_delta.pt`. Metadata explicitly marks them `synthetic=true` and
`pilot_training=false`. They are **not trained door-pushing pilot checkpoints**.
The exact source/data, job, load/prediction code and comparisons are preserved.

After acquiring the shared GPU leases, the local load/parity command is:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p outputs/evidence/b1-shared-learned-pilot-20261016/checkpoint-return/checks.py --repo "$PWD" --output /tmp/b1-checkpoint-parity-new
```

It uses the declared returned artifacts and requires CUDA. Existing accepted
reports remain unchanged. `pilot-report.json` summarizes every model path,
physical reference trial, refusal, A3/A4 audit, costs and missing goal criteria.
Original teacher recordings remain usable historical evidence at
`outputs/generated/b1-policy-pilot/act-a1a2-pilot/`; no new-controller recording,
learned episode/video, complete learned evaluation command or learned task result
is fabricated. The rejected reference/IK cycles remain closed; the later
placement study resolves only unloaded approach feasibility. The goal is
**incomplete**.

## Authorized continuation: one constrained IK cycle

The user explicitly authorizes one additional IK cycle after the reference stage:
independent numerical comparison followed by at most four robot-only Isaac cases
within the existing three local GPU-hours. The original six-second Cartesian
references, gains, damping, physical limits and acceptance criteria stay fixed.
[[b1-joint-limit-aware-ik|Numerical checks pass; the physical correction is rejected]]: complete left translation p95 improves 28.783 -> 21.363 mm and
rotation p95 7.873 -> 2.607 deg, while peak actual speed is 0.529 rad/s. Translation
and speed still fail 10 mm / 0.51 rad/s. Right traces are identical to baseline;
all stops pass, native feedback is complete and residual-domain refusals remain.

Exactly four new cases complete without interference. Rejected code/tests and
all evidence are preserved locally, then the active solver is restored. This
one-cycle ceiling is exhausted. The later separately authorized placement study
below resolves unloaded feasibility with the original solver; it does not
retroactively accept this correction or establish a learned result.

## Authorized nominal-placement result

One later user-authorized audit derives a single common Y translation from the
three selected train/development precontact endpoints, without a placement sweep
or another IK/gain change. [[b1-initial-placement-pilot|The numerical and physical study]] passes both train approaches, endpoint holds/stops and moving
stops at floor (-0.437, 0.482327, 0). Measured approach p95 is 2.186/5.070 mm and
0.763/1.701 degrees, actual peak speed 0.319/0.277 rad/s; moving stops travel
3.434/3.668 mm. All original criteria pass. Native feedback covers all 3,384
samples, but old-calibration estimates remain available on only 128 samples.

The prospective `configs/b1_shared_pilot_setup.json` changes only common floor Y;
`pilot_physics.py --setup <path>` explicitly selects and records it. The original
corpus setup stays intact. This is unloaded robot-only feasibility at one nominal
point, not door/frame/camera/contact qualification or a placement range. No new
demonstration or learned outcome is claimed. The next dependency is the existing
12-case-ceiling dynamic/contact prerequisite phase, followed by observed support
and new demonstrations; the successful four-case placement study is closed.
