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
