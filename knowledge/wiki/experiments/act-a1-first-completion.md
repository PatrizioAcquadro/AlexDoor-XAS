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
regressions pass. Native full-start right evaluation is active; more than188
actual learned commands have executed with no physical violations so far. This
establishes dispatch progress only, not task completion. Final outcome is pending.

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

