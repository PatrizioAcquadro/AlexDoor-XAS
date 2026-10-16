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
  A separate synthetic cost trial charges 100 updates to this ceiling, leaving
  at most 17,900 for subsequent real-data work; it produces no pilot checkpoint. Checkpoints
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
must be delivered before completion. Current state: **incomplete**. Two finite reference candidates are exhausted
without a valid left approach. No new physical demonstrations, real-data training,
or learned command is admitted. Essential observed support and remote access also
remain unresolved. The code rejects both proposed teacher-reference changes; no
new trajectory is installed in the controller, teacher or policy adapters.

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

The two-candidate reference ceiling is exhausted. Further physical progress
requires a separately bounded continuous constrained-IK/redundancy or reference
strategy, with complete-path validation at the original criteria. No new
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

## Remote access

`ssh -o BatchMode=yes -o ConnectTimeout=10 gilbreth-purdue ...` fails public-key
authentication. The configured `~/.ssh/gilbreth-agent.sock` is absent. No Slurm
job is submitted, no remote resource is consumed and no remote source/data version
is claimed verified. The user is asked to unlock the configured agent or identify
an authorized connection. Immutable source/data transfer and checkpoint roundtrip
remain required after access is restored; Isaac stays on the workstation.

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

Remote authentication was restored by the user after the initial failed attempt.
The same `gilbreth-purdue` connection now succeeds without the old agent socket;
brief immutable-code/data Slurm verification is in progress, with no campaign.
