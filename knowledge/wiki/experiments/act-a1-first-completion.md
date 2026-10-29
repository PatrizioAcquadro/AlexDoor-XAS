# ACT A1 First-Completion Experiment

## Question and outcome

Could one temporal ACT A1 checkpoint complete opening, loaded hold, intentional
release and ordinary stabilization on both fresh nominal directions, then the
four frozen development cases? **The authorized experiment ends with the goal
unmet:** principal seed 9101 reaches the 75k ceiling; offline selection fixes
checkpoint 61k; the final pair is 0/2 and cumulative learned completion is 0/22.
Development remains gated and final test sealed. The October 25, 2026 UTC closeout
pauses the heartbeat. Current authority is stated in [[status]].

Both final recordings and independent stops pass, without physical violations,
teacher actions or valid learned contact. A healthy stop is not task completion.
No alternate checkpoint is retried on the task after the failed final pair.
Earlier native failures, interruptions and resource charges remain retained.

## Data, training and selection

The admitted union has 114 train sources/168,178 overlapping windows and eight
development sources/23,705 windows. The original 112-source train-only
normalization is frozen. The two additions are qualified continuous cycles;
older nominal/gentle/slow-tail/handoff candidates retain their own narrower
admitted scopes rather than silently becoming full demonstrations. The later
[[act-a1-complete-demonstration-audit|complete-source audit]] distinguishes original
admission from current full-cycle proof: 15 complete-phase sources, seven
continuous and eight with masked pre-release losses, with only two complete
matched current observed references. No active manifest or checkpoint is rewritten.

The model consumes genuine 4 s causal history, 240×384 RGB-D, measured q/dq,
verified simulated joint-output feedback and acknowledged applied targets.
Physics is 240 Hz, control 60 Hz and RGB-D 10 Hz; chunk 12/execute 6, with visual
reuse disabled. Teacher/evaluator phase labels may select or score windows but
never enter observations. Injected actions remain history and are excluded from
target chunks. Sampling balances four doors, then available phase/transition/
recovery strata, sources and actual eligible starts; long pushes gain no mass
from duration. There is no half-batch nominal focus in substantial base training.
The maintained implementation is in [[topics/act-a1-observations-and-time]].

The two-direction operational reference prerequisite passes separately: right
4,108 commands/50.236 degrees and left 8,126 commands/52.564 degrees, with complete
hold/release/stabilization, exact source-command parity and independent 2+4 s
stops. These are zero-learned-command references, never policy successes.

Training continues the same model/optimizer/RNG through bounded local/A100 blocks.
Allocation boundaries at 16k/30k/45k/60k do not authorize native evaluation. Saves
occur every 1k updates; fixed balanced train/development diagnostics preserve RNG
and do not update weights. The declared offline plateau requires 15k updates
without 2% improvement in any of fixed train error, development error or actual
rolling loss, using three-checkpoint medians after at least 30k main updates.
Nonfinite values stop immediately; persistent divergence/overfitting require
separate diagnosis. No native task trial occurs during substantial training.

All 69 principal checkpoints remain finite, source-compatible and normalization-
compatible. The 75k state preserves all 128 optimizer states and Python/NumPy/
CPU/CUDA RNG; recorded rolling losses match their preceding 256 real updates.
The full history supports a ceiling stop, not stagnation: useful improvements
remain at 72k/73k.

| Three-checkpoint median | 58k–60k | 73k–75k |
|---|---:|---:|
| Fixed training L1 | 0.063728 | 0.057593 |
| Fixed development L1 | 0.277977 | 0.272103 |
| Actual rolling optimizer loss | 0.083115 | 0.076369 |

Offline selection ranks complete three-checkpoint windows by development median,
then train and rolling-loss medians for ties, and chooses the center checkpoint.
The best window is 60k/61k/62k: 0.059429 train, 0.271096 development and 0.081865
rolling loss. `base-training-finalization.json` verifies that every allocation
has ended, main step is 75k, reason is `maximum_main_updates`, the chosen checkpoint
is 61k and all seven diagnostic histories participate. Completion still requires
that same checkpoint's real task behavior; the offline score is not a substitute.

## Final fresh nominal pair

| Case | Learned commands | Controlled opening | Terminal reason | Independent stop |
|---|---:|---:|---|---|
| Right | 428 | 0 degrees | Contact-acquisition budget exhausted | Pass, 2+4 s |
| Left | 239 | 0 degrees | Weak position support requires certified contact-free motion and stopping | Pass, 2+4 s |

Both use only finite 12×7 predictions from checkpoint 61k. Actual command counts,
60 Hz dispatch, causal acquisition/availability and first/middle/final RGB-D
checks pass. Peak native contact force is zero; neither achieves hold, release or
ordinary task stabilization. The final images support the observed acquisition
failure. They do not identify the architecture or thresholds as its cause.
Each failed task returns 1 without timeout and releases its lease; charged
allocation is 284.977 s right and 232.351 s left.

All 61 native attempts remain: 22 learned, 14 teacher and 25 diagnostic/reference
attempts, with missing historical reports explicit. Earlier 52.295-degree opening
and valid hold without release remains partial capability, not success or a
replacement for the final 0/2. Subsequent [[act-a1-failure-diagnosis|causal diagnosis]]
reproduces saved actions and separates acquisition/support mechanisms. A staged
left start maintains learned contact for 10 s/+10.471 degrees, but adds no complete
normal-start success. [[approach-contact-corrections|Corrective qualification]] and
[[act-a1-complete-cycle-collection|later collection]] are separate experiments.

## Retained failures, costs and evidence

A local continuation segfaults after 2,207 logged updates, with no result file or
identified native cause. Accounting charges 2,208 updates, including one possible
unlogged update; up to 208 discarded updates may be repeated from the saved 52k
resume, and remain expenditure. The failed allocation charges 1,601.456 GPU seconds.
Finite losses, successful CPU data reads and later successful continuation do not
identify or exclude a GPU/runtime cause. Failed-run uncertainty remains in the
cumulative upper count.

Final experiment accounting is **42.5080 local/31.8231 remote GPUh, 198,773 cumulative
updates and 994.27 GiB retained**. The then-current ceilings were 48/32 GPUh,
200k updates and the explicitly raised 1.1 TiB aggregate storage ceiling. Storage
counts distinct local inodes plus every remote replica/output; historical
conservative path sums remain alongside it. These are dated closeout figures,
not fresh budgets for later campaigns.

An earlier incomplete remote inventory left a short possible 1 TiB exceedance
whose exact transient peak is unavailable. Later authorized cleanup verified
58.627 GiB remote and 3.383 GiB local redundant caches against retained copies
before deletion; original raw data, compact readers and every outcome remain.
Historical costs are not refunded. No uninterrupted cap-compliance claim is made
for that interval, and no such payload cleanup occurs in the present maintenance.

Evidence remains under `outputs/evidence/act-a1-first-completion-20261022/`:
`base-75000-offline-review.json`, `base-training-finalization.json`,
`base-training-evaluation-policy.json`, `base-final-pair-review.json`,
`base-seed-9101-step-61000-panel.json`, `base-local-60000-crash-review.json`,
`failed-training-update-reconciliations.json` and `base-cleanup-summary.json`.
Their immutable protocols, native recordings, missing outcomes, original failed
allocations and source snapshots remain authoritative. The complete source/attempt
narrative is available from Git at `4d21837`; ordinary chronology is not repeated here.
