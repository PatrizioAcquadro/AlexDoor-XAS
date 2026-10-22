# ACT A1 first learned completion

Status: authorized implementation and training; no learned completion established.
Baseline `7024110`, branch `PatrizioAcquadro/b1-policy-pilot`, clean tracked tree.
The objective is one saved temporal ACT A1 checkpoint completing acquisition,
opening, required hold, release and stabilization in both opening directions.
[[act-a1-diverse-collection|Qualified collection]] remains closed;
its original failures and narrower usable scopes remain preserved.

## Fixed experiment

Start with the existing temporal ACT configuration: 4 s genuine causal history,
240×384 RGB-D, measured q/dq, supported simulated joint-output feedback and
acknowledged applied targets; chunk 12, execute 6, no visual feature reuse.
Physics remains the qualified 240 Hz with 60 Hz control and 10 Hz RGB-D.
Teacher phase/action-role labels may select training windows and score execution;
no evaluator, teacher action or simulator truth enters model observations.
Injected commands remain in history and outside target chunks; valid correction
commands retain supervision. Fit normalization only on the admitted selected train
union, with source ancestry/split isolation. Development never fits statistics or
weights; final test remains sealed.

Sampling will balance doors, then phase/event/recovery strata within each door,
then qualified source and real eligible start. Every item preserves its genuine
history and contiguous twelve-step targets. Long push sequences cannot gain mass
merely by their duration. Historical controls retain only their admitted scope.

First pass: short CUDA learning pilot, saved checkpoint reload and genuine native
learned execution through the existing safety/admission/controller/stop path.
Only after that path operates, launch independent seeds 9101/9102/9103, initially
two Gilbreth A100-80GB jobs and one Blackwell job, same model/training recipe.
No architecture/action-space comparison or broad parameter sweep. Checkpoints
are selected primarily by actual completion and task progress, with physical
violations, operational refusals, runtime errors and policy failures separate.
Imitation loss is diagnostic. Extend promising runs explicitly after evidence.

## Predeclared evaluations

Full training conditions: nominal animated-door-1-88abf40 right opening, seed 7100;
nominal door-2738468b94d74c5f left opening, seed 23. Use unchanged common setup,
inspection, task requirements, operational recipe and qualified 240 Hz physics.
Both must complete from fresh normal initial conditions with the SAME saved file.
Initial evaluations include full starts and qualified intermediate/recovery starts
with actual reconstruction and fresh causal pre-roll, never fabricated history.
No replay, teacher correction or scripted manipulation can count as learned success.

After two-direction completion, run four full normal-start development conditions:
modern-door-2fb8d024 and psx-industrial-001-4f5561b, each at nominal and existing
qualified high-contact pose. Freeze exact setup/seed files before first execution.
Report all four outcomes irrespective of success; no development success gate.
Every allocated attempt, including interrupted/refused/failed attempts, stays in
the denominator. A changed implementation may justify a separately identified
new attempt; no favorable retry may replace an earlier outcome.

## Resources and stopping

Continue existing cumulative limits: local GPU 40 h; aggregate remote GPU 32 h;
200,000 optimizer updates; retained evidence 1 TiB. Initial recorded local charge
28.0718414411 h, remote 3,503 s, update upper bound 120,364; 977,600,261,507 bytes
of local evidence (910.4611924915 GiB). Verify live usage before reservations.
Sum concurrent remote allocations across hosts; count startup/shutdown/failures.
Keep at least 60 s workload and 12 GiB evidence headroom; one local simulator at
a time and preserve unrelated workloads. Previous collection-only attempt and
increment budgets remain recorded; this task does not recollect the campaign.
Do not duplicate the whole prepared corpus locally. Reserve evaluation capacity
before extending training. Stop on demonstrated completion, an authorized limit,
or a demonstrated blocker that available means cannot resolve. Falling loss and
partial task progress are not completion.

## Results

The local seed9100 pilot completed128 optimizer updates in133.977 s of Python
process time,141.019 s charged allocation including startup and tail. Checkpoints
at64/128 updates reload with identical predictions. The18 focused numerical
regressions pass. Both full-start pilot evaluations failed before contact: right1019 learned
commands, left1 command, controlled opening0 degrees in both. Both stop with
`admitted_unloaded_separation_exhausted`, without physical violations; each2s stop
and separate4s continuation passes. These original0/2 outcomes remain preserved.

The native tracker includes the same linear aging in pose and normal-position
bounds. The unloaded receipt also charged explicit relative motion, then credited
only one prepaid interval, duplicating the identified normal aging. The correction
credits only positive growth of the existing identified temporal primitive, bounded
by explicit relative motion. New registration/calibration growth, missing/invalid
attribution and absent prepaid interval retain refusal. No physical bound, margin,
model/controller parameter or success threshold changes.137 focused regressions
pass. A single predeclared120-command left prefix with unchanged pilot checkpoint
executes all120 commands and passes2+4s stops, with zero contact/violations. It is a
separate accounting diagnostic, not a favorable replacement or full-task success.


The launcher now explicitly accepts `--temporal-act`. Native measured feedback,
RGB-D and acknowledged delivered targets feed the existing temporal runner;
admission, control and continuing-physics stop stay active. Evaluation declares
four native substeps at240Hz and additionally requires the qualified ordinary
stabilization after release. Legacy evaluation keeps its previous two-sample and
release semantics; it cannot be compared as an identical completion definition.

The admitted index contains112 train sources/155,944 windows and eight development
sources/23,705 windows, with disjoint canonical source groups. The HDF5 metadata
uses `family`, not `source_group`; the first index attempt rejected missing groups.
The corrected index uses the canonical qualification handoff group assignments.
No data or original split was changed. Exact panel files are in
`outputs/evidence/act-a1-first-completion-20261022/evaluation-protocol.json`.


## Resumable training check

The resumed training loader keeps RNG byte states on CPU; model and optimizer
`load_state_dict` move their tensors to the active GPU. Loading the entire resume
file onto CUDA would make `torch.set_rng_state` reject the CPU RNG state. An actual
CUDA continuation from128 to129 updates passes, with all saved AdamW counters at129,
112 unchanged train normalization identities and identical checkpoint reload. This
one additional engineering update is charged and is not a new task evaluation.

## Initial training placement

All 120 selected prepared stores (112 train, eight development) were copied
losslessly and verified on both remote hosts. Gilbreth jobs `12005026` and
`12005027` run independent seeds 9101/9102 on A100 80 GB GPUs, initially targeting
1,024 updates with checkpoint 512. Their measured full update times are about
2.6–2.8 s, including input loading. The local pilot measured about 0.8–1.0 s.

An unrelated Blackwell workload acquired its existing GPU lock before launch.
Seed 9103 waited on that lock for 900 s without a GPU allocation; the lock remained
held and the wait ended without a training start. The permitted placement
adaptation uses the idle local RTX 4090 for seed 9103 with identical data, model,
normalization, optimizer and sampling. Its first 512 updates precede native task
evaluation, then the same optimizer/RNG can continue to the initial 1,024 target.
The RNG-loading fix does not alter the initial learning recipe.

Only newly created Blackwell input replicas were retired after verifying no
training/queue process remained. Original qualified stores and the byte-verified
Gilbreth copy remain; all results, failures, code and transfer records are kept.
This frees 38,684,943,269 bytes under the cumulative retained-storage cap. No
training cost or prior outcome is refunded. Exact placement, commands and costs
are in `outputs/evidence/act-a1-first-completion-20261022/`.

## Initial 1,024-update tranche

Seeds 9101/9102 complete 1,024 updates on A100, taking 2,854.289/2,812.590 s of
Python process time. Slurm extern lifetimes are 2,897/2,856 s; charge the maximum
whole allocation plus five seconds each, totaling 5,763 remote GPU seconds.
Seed 9103 completes 512+512 updates on RTX 4090 in 377.119+356.060 s of process
time, preserving AdamW/RNG state. All checkpoints retain the same model, robot
identity and 112-source training-only statistics. Final test remains sealed.

At checkpoint 512, seed 9103 has 0/2 full nominal completions: right 2,062 commands
then prospective position/stop support refusal; left 467 commands then verified
load-domain refusal. Neither acquires contact; both independent2+4s stops pass
without physical violations. The right future shoulder-Z interval exceeds the
main empirical minimum by20.7microradians; this is not evidence of physical danger.
The policy approach is also slow and differs from the nominal teacher path. No
calibration or physical threshold is enlarged to accept this checkpoint.

A fixed saved-input diagnostic evaluates96 qualified intermediate source states
across approach/contact/push/hold/release/stabilization/transitions/recovery on all
three1024 checkpoints (288 CUDA queries,zero updates). Original causal histories
and eligibility masks remain; future labels are excluded from policy queries.
This measures action errors only,not simulated phase completion. Overall action
magnitudes are not uniformly collapsed; release errors remain comparatively large.
The physical comparison uses fixed seed order9101/9102/9103 on both predeclared
normal starts and selects primarily by actual task completion/progress.

## Evaluation termination

Temporal A1 has no end token. Native evaluation now uses an external task-complete
predicate after the real action, observation, feature and safety checks. It can
end an episode only after the independent unchanged opening/hold/release criteria
and ordinary stabilization pass; it never supplies an action or policy input.
The result distinguishes evaluated completion from predicted termination, and
the independent safety stop remains required. No completed phase or task had been
observed when this interface was corrected. All previous refusals are preserved.
150 focused regressions pass, including safety priority and no completion before
a real policy step. An obsolete unit-test error-message assertion was aligned
with the previously qualified120/240Hz runtime; its rejection behavior is unchanged.


## Separate loaded-phase diagnostics

`pilot_physics.py rollout --temporal-act --intermediate-state` accepts an existing
qualified matching training state. Its source must be in the checkpoint's selected
training IDs. The maintained reconstruction restores joints, velocities, targets,
door motion and task gains, and repeats the independent loaded start-stop gate
before reconstructing the recording start. The native producer then resets its
history and acquires one second of fresh held-pose observations, requiring sustained
loaded support and unchanged physical limits. No source history is imported.
The existing partial-history masks represent unavailable pre-start observations.

This mode is explicitly an intermediate-phase diagnostic: the source opening is
initialization, and overall `passed` is always false for full-task accounting.
`intermediate_phase_success` describes only the subsequent learned continuation,
with accepted termination and independent stop continuations required. Physical
rows and initialization audits remain inspectable. Teacher/simulator information
is confined to reconstruction and qualification; after handoff, ACT and the
unchanged observed admission/controller generate all manipulation commands.

The 42 related numerical tests pass, including source rejection, reset ordering,
fresh observation acquisition, missing loaded support and stop continuations.
Native loaded-phase applicability remains to be measured; this implementation
alone is not evidence of later-phase or recovery success.


## Full initial checkpoint panel

All three1,024-update checkpoints complete both fixed normal-start attempts:
0/6 full successes, zero controlled opening. Every2+4s independent stop passes;
no physical violation is recorded. These six failures remain separate from the
prior128/512-update attempts and the120-command accounting diagnostic.

| Seed | Direction | ACT commands | Terminal reason |
|---|---|---:|---|
| 9101 | right | 570 | `weak_position_requires_contact_free_motion_and_stop` |
| 9101 | left | 791 | `observed_contact_orientation` |
| 9102 | right | 192 | `load_estimation_failed: outside_verified_load_envelope` |
| 9102 | left | 136 | `load_estimation_failed: outside_verified_load_envelope` |
| 9103 | right | 311 | `observed_contact_orientation` |
| 9103 | left | 5,383 | `load_estimation_failed: outside_verified_load_envelope` |

Seed9101left reaches8.878N native peak and1.409degrees raw door motion, but never
satisfies controlled contact/opening. This is not learned task completion. Its
observed contact orientation exceeds the unchanged combined requirement. Seed9103
left remains contact-free for5,383 commands; terminal measured shoulderX exits
an empirical load position domain by8.70microradians. That numerical exit is an
operational refusal, not a physical violation. Its preceding slow approach also
fails to reach the prescribed material point; the refused domain alone does not
establish an otherwise successful policy.

The three checkpoints tie on the declared completion/minimum/summed controlled
opening criteria. Fixed seed order breaks the tie: only9101continues, to4,096
updates with checkpoints2,048/4,096, preserving optimizer/RNG and the exact data,
normalization, sampler and model. The successful actual resume migrates from
Gilbreth PyTorch2.12.1 to local PyTorch2.10; initial continued updates execute on
CUDA. Blackwell still has an unrelated compute workload. The local continuation
reserves at mostone GPU hour plusfive further local evaluation hours before launch.

A deterministic sampler audit counts32,768 sampled windows/21,087 distinct source
starts in seed9101's initial tranche. Every phase/event stratum receives roughly
one ninth of windows; the two nominal approach sources contribute40/76 windows.
This early exposure is not convergence evidence or proof of missing data. At the
first live query, seven arm positions, validity masks and image ages match the
corresponding teacher state exactly; small rendered and feedback differences
remain. Later behavior, not first-query equality, determines task performance.


The loaded-source preflight confirms both nominal IDs are in the112-source train
fit and their saved normal stiffness/damping match the qualified runtime exactly.
Startup rejects differing saved motion gains before reconstruction; it cannot
silently replace the authoritative normal controller. The43 related numerical
regressions pass. This configuration check still does not qualify native loaded
reconstruction, phase performance or recovery behavior.


Reference replay is explicitly ineligible for learned full-task accounting, as are
loaded intermediate starts. The attempt inventory also retains allocated trials
whose reports are unavailable, separately from completed physical outcomes; they
cannot disappear from attempted denominators.20 related evaluator/stop regressions
pass after this reporting-boundary correction.


## Loaded initialization revision

The first loaded right diagnostic executes zero ACT commands: the new harness
replaced restored preload targets by measured q, immediately unloading the panel.
Its source restoration/start-stop check passes, but fresh pre-roll has zero force.
The left source alternates loaded/unloaded native intervals during its six reset
steps; an instantaneous last-tick force check refuses before any observation.
The empty rollout then raises an incomplete-transition error, obscuring the
startup reason. Both allocated attempts and all raw conditioning records remain.
These outcomes do not measure ACT's later-phase capability.

The corrected initializer preserves restored controller state and runs the
maintained material-following source controller only during genuine pre-roll.
Its requested poses, actual targets, sensor rows and feedback are retained with
`learned=false`; it ends before ACT dispatch. No old history is attached. Loaded
start qualification uses the unchanged0.1s causal mean,0.02N minimum and0.1mm
current native gap criterion across the six real conditioning steps, matching
physical task contact semantics. Native per-substep force/gap values are now
retained alongside the original maximum-force diagnostic. Missing support and
excess gap still refuse; physical limits, gains and stop requirements are fixed.

An empty failed initialization can record its explicit outcome while keeping the
HDF5 recording incomplete and observation-free. It cannot claim success or discard
existing samples. Runner results are also saved before transition finalization,
so a recording failure cannot conceal the original execution reason.63 related
regressions pass. Revision2 prospectively allocates exactly one right and one left
loaded diagnostic with the SAME4,096 checkpoint and SAME source states, preserving
both original failed/unavailable attempts. No revised native outcome is assumed.


Revision2 passes both fresh loaded pre-rolls and independent stops, but dispatches
zero ACT commands: the observed geometry engine still waits for the common5–25s
inspection captures while the loaded pre-roll ends at1s. Thus no explicit observed
contact selection exists at handoff. Both outcomes remain operational setup
refusals, not ACT phase failures. The new producer preserves five actual captures
at0/0.2/0.4/0.6/0.8s for this loaded-only acquisition. The independent recorded
workspace retains its original capture schedule and all calibrated limits;
normal-start acquisition is unchanged. Once observed selection exists, actual
native feedback primes the existing operational force window. No teacher geometry
or force is substituted for observed operational evidence.64 related regressions
pass. Revision3 fixes exactly two further cells, right then left, same checkpoint
and source states, with every earlier failed/unavailable attempt retained.


## Bounded nominal-transition curriculum

At4,096 updates seed9101 remains0/2 from full normal starts. Right reaches40.1mm
material error before the unchanged3s contact-acquisition budget, compared with
149.5mm at1,024; left instead terminates at249.8mm after462 commands. Both stops
pass without physical violations. The512/1,024/4,096 failures remain preserved.
The original seed9103 continuation reaches2,048 on A100 (job12006518,2,832s whole
allocation charge); fresh full evaluation is waiting for the unrelated local GPU
workload. No checkpoint has completed the task.

One prospective intervention addresses sparse nominal initiation exposure while
the simulator GPU is unavailable: resume seed9101 from4,096 to6,144 on Gilbreth.
Half of each32-item batch uses the two original complete nominal sources equally,
then uniformly chooses initiation (first1s of admitted decisions), approach,
contact, transitions or release. The other half keeps the original four-door,
phase/source-balanced112-source sampler, including valid recovery supervision.
Every item retains its genuine4s history and contiguous12 target rows. Narrow
short-task sources cannot be declared complete focal cycles. The original sampler
remains the default, with no extra RNG calls in that path.

Normalization remains the original selected112-source train fit. No label,
observation, model, optimizer, learning rate, physical/runtime parameter or final
panel changes. `--focus-episode` records explicit source IDs and recipe
`primary-nominal-critical-v2` in the run metadata. This is one bounded sampling
hypothesis, not a parameter search or new collection.13 related numerical tests
pass. Select or extend using fresh full task progress/completion on both fixed
normal conditions; imitation loss remains diagnostic. If the same file completes
both, run all four predefined development conditions and report every outcome.


A fixed zero-update image intervention uses the two qualified source histories at
25.4s with identical four-second arm history. For each immutable checkpoint, only
the door images change while the robot tokens stay fixed. Teacher actions remain
outside the queries. Seed9101/4,096 and seed9103/2,048 both respond to images, but
produce about0.00038/0.00037rad L2 image contrast versus0.00290rad teacher contrast
across six actions. Initial action norms are16–26% of teacher norms; source-left
cosines are0.35/0.50. This saved-input diagnostic supports initiation weighting; it does
not establish deployed causal success or justify action rescaling. Eight queries
execute in1.717s within the existing owned A100 allocation12006811, zero updates
and no additional GPU reservation. The whole job charge includes this shared step.


An exact first-query numerical check uses the recorded right/left native causal
inputs, the immutable4,096 weights, original normalization and matching backend
flags. A100/PyTorch2.12.1 predictions differ from recorded RTX4090/PyTorch2.10
outputs by at most7.31e-8rad, within the existing1e-7 round-trip tolerance. This
rules out a material network-forward discrepancy for these two inputs; it does
not prove later closed-loop equivalence. Two queries take1.806s within the same
owned allocation, zero optimizer updates and no additional GPU allocation.

The frozen96 genuine training histories are also queried at4,096, retaining
future labels only in the diagnostic caller. Contact/hold predictions align
closely with source actions, while transitions have the weakest mean direction
agreement. Left approach and recovery cosine means remain0.61/0.45, compared with
right0.97/0.97. These action diagnostics delimit demonstrated fitting; they do
not count as physical recovery or phase completion and do not select checkpoints.
All96 results are retained in `frozen-phase-step4096.json`;32.559s shares the same
owned A100 allocation. Fresh simulator results remain required.
