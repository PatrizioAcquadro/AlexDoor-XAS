# ACT A1 complete-demonstration audit

Completed October 26, 2026, from `4ee7c2f` on
`PatrizioAcquadro/b1-policy-pilot`, in the existing `23f1` worktree.
This is an offline evidence audit. No simulation, policy inference, training,
collection, development payload or sealed-test access occurred. Runtime code,
thresholds, checkpoints, raw recordings and active selections are unchanged.

## Decision supported by the evidence

The inventory contains **129 train sources, of which 15 span the complete phase
sequence**, distributed **4/5/3/3** across the four training doors. The selected
61k model's principal training used 114 sources, including these same fifteen.
The fifteen later approach/contact corrections have not been trained.

The fifteen complete-phase recordings are useful, but they do not represent
fifteen independently qualified demonstrations of current executable completion:

- **Two:** retain the full episode as the strongest available executable reference
  supervision. Both have continuous native contact and complete matched observed
  references at 240 Hz, including hold, release, ordinary stabilization and stops.
- **Five:** retain the full original as a qualification candidate. Native replay
  finds continuous contact; current operational/loaded-entry compatibility is
  incomplete. Missing qualification is not a demonstrated defect.
- **Eight:** retain useful admitted segments, without treating the entire episode
  as a continuous full-task positive. Native replay demonstrates pre-release load
  losses. The existing masks already exclude the loss commands and crossing
  chunks; these sources are not wholly invalid or evidence of unmasked bad labels.

These are **proposed future dispositions**, not selection changes or retroactive
edits to historical outcomes. Every original, negative result and denominator stays.
Learned completion remains **0/22 in the current campaign**, including the selected
checkpoint's **0/2 normal starts**. Reference execution is not learned completion.

## Evidence and method

The entry point is
`outputs/evidence/act-a1-first-completion-20261022/recovery-guard-20261026/`.
The audit's local payloads are in its `complete-demonstration-audit/` subdirectory:

- `recording-audit.json`: every episode identity, prepared/raw/report path, setup,
  phase counts, masks, source clocks, force events, probabilities and comparisons.
- `native-contact-audit.json`: replay of saved `teacher/contacts.json` forces and
  gaps through the unchanged `ContactLoad` and `ContactContinuity`, checked against
  `teacher/trace.json` and the raw HDF5 annotations. All fifteen agree exactly on
  loaded/valid status and native force peaks. The two explicit recent continuity
  records are reproduced exactly.
- `temporal-validation.json`: the maintained numeric reader reconstructs all
  fifteen eligible 12-command window lists exactly, reading zero image frames.
- `context-audit.json`: train-only source ancestry, sampling lineage and saved
  historical/current comparison evidence.
- `proposed-dispositions.json`: per-source recommendations and unresolved gates;
  it is not a training manifest. The three `complete_demo_*_audit.py` scripts are
  read-only analysis companions; only their new audit outputs are written.

Common inspection is excluded from contact-continuity replay: its `contact_valid`
annotation describes inspection validity, not loaded contact. Native sampling is
two physics samples per control interval at 120 Hz and four at 240 Hz. Native
peak force, the causal 100 ms load filter, and the operational observed force upper
bound are different quantities. No threshold or force/gap interpretation changes.

Physical endpoint checks use the saved measured traces as well as reports. All
fifteen achieve their original opening requirement, retain 180 valid loaded hold
intervals (3 s), physically withdraw, and execute 120 stabilization intervals
(2 s). Final-window separation is **20.11–27.74 mm**; maximum arm speed is at most
**0.00702 rad/s**, door speed **0.000129 rad/s**, and tool travel **0.0237 mm**,
within the existing respective bounds. Native prerelease forbidden-contact
sample count is zero. These outcomes do not erase an earlier continuity loss or
establish current operational admission. A phase label or `passed` flag alone
was not used as proof.

## Disposition of every complete-phase recording

IDs map to exact UUIDs and raw/prepared paths in `recording-audit.json`.
Each linked ID opens its original source report. Angles are source controlled
opening; force is native peak, not observed load. Loss times are raw episode
times. `2` and `2+4` mean recorded terminal continuing-physics stop evidence in
seconds; they do not certify every earlier loaded state. All fifteen have
compatible temporal contracts and exact applied-minus-measured A1 labels.

| ID / source | Physics; opening; peak | Native continuity | Stop; windows | Proposed disposition and specific uncertainty |
|---|---|---|---|---|
| [C01](../../../outputs/evidence/act-a1-teacher-temporal-20261019/A-right-attempt-2/report.json) — original right nominal | 120 Hz; 51.749°; 9.698 N | 7 losses; first 64.183 s | 2; 7,537 | Retain valid approach/contact, loaded push and hold/release/tail segments. Exclude a continuous-full-task claim; 14 ambiguous push commands and crossing chunks are already masked. Current 240 Hz response and operational support require qualification. |
| [C02](../../../outputs/evidence/act-a1-teacher-temporal-20261019/B-left-attempt-1/report.json) — original left nominal | 120 Hz; 52.551°; 7.683 N | No loss | 2; 7,996 | Retain full as a qualification candidate. Native continuity is positive evidence. Loaded trajectory/stop support and response under current 240 Hz execution are missing, not proven failures. |
| [C03](../../../outputs/evidence/act-a1-four-door-20261019/N-left/report.json) — interior left nominal | 120 Hz; 54.371°; 8.582 N | No loss | 2; 9,517 | Retain full as a qualification candidate. No complete current observed reference is established for this door; old physical/temporal success does not supply that gate. |
| [C04](../../../outputs/evidence/act-a1-four-door-20261019/N-right/report.json) — front right nominal | 120 Hz; 51.281°; 10.056 N | No loss | 2; 7,891 | Retain full as a qualification candidate, including useful low-load segments. One contact interval has a native peak above 10 N; this alone does not reject it. Current observed loaded-entry and 240 Hz/stop compatibility remain unresolved. |
| [C05](../../../outputs/evidence/act-a1-posture-20261019/P-left-attempt-1/report.json) — left shoulder-Y −0.05 rad | 120 Hz; 52.615°; 7.684 N | No loss | 2; 7,729 | Retain full as a qualification candidate. It provides the only different initial arm configuration, but needs matched current response, position/load and stop qualification for that posture. |
| [C06](../../../outputs/evidence/act-a1-diverse-20261019/train-1-complete-pose-high/report.json) — right high | 120 Hz; 52.885°; 9.907 N | 1 loss at 51.217 s | 2; 7,980 | Retain valid segments; do not promote the whole episode to continuous completion. The two invalid push commands and crossing chunks are already excluded. Current-runtime compatibility is also unqualified. |
| [C07](../../../outputs/evidence/act-a1-diverse-20261019/train-1-complete-pose-low/report.json) — right low | 240 Hz; 49.956°; 11.548 N | No loss | 2+4; 6,937 | Retain full as a qualification candidate. A saved 360-command unloaded reference prefix matches this source exactly; full observed entry/opening/stop support is not thereby established. One high native contact peak needs observed-load qualification, not automatic exclusion. |
| [C08](../../../outputs/evidence/act-a1-diverse-20261019/train-2-complete-pose-high-gap-step/report.json) — left high, saved gap-step variant | 240 Hz; 53.537°; 14.960 N | 1 loss at 34.700 s | 2+4; 6,645 | Retain valid segments; exclude a continuous-full-task claim. Early push unloads despite later successful hold/release. Inspect loaded-entry compatibility and the gap-catch-up controller difference before future selection. |
| [C09](../../../outputs/evidence/act-a1-diverse-20261019/train-2-complete-pose-low/report.json) — left low | 240 Hz; 50.952°; 11.957 N | 1 loss at 34.683 s | 2+4; 6,384 | Retain valid segments. Early contact→push continuity is defective for the full task; higher native entry force is a separate operational question. |
| [C10](../../../outputs/evidence/act-a1-diverse-20261019/train-3-complete-pose-high/report.json) — front right high | 240 Hz; 52.862°; 20.600 N | 10 losses; first 41.500 s | 2+4; 7,003 | Retain valid bounded segments and all negative evidence. Repeated unloading makes this the weakest full-continuity source; masks exclude its 37 invalid push commands and crossing chunks. Current load/entry support is unqualified. |
| [C11](../../../outputs/evidence/act-a1-diverse-20261019/train-3-complete-pose-low/report.json) — front right low | 240 Hz; 49.845°; 17.702 N | 1 loss at 51.417 s | 2+4; 7,091 | Retain valid segments; full-cycle continuity is disproven by the saved trace. Complete current operational entry and support remain unqualified. |
| [C12](../../../outputs/evidence/act-a1-diverse-20261019/train-4-complete-pose-high/report.json) — interior left high | 240 Hz; 55.953°; 17.289 N | 1 loss at 34.667 s | 2+4; 8,427 | Retain valid segments. Early push unloads. Later opening reaches its recorded kinematic endpoint before valid hold/release; this is not a new physical violation or permission to ignore the earlier loss. |
| [C13](../../../outputs/evidence/act-a1-diverse-20261019/train-4-complete-pose-low/report.json) — interior left low | 240 Hz; 52.844°; 17.116 N | 1 loss at 34.700 s | 2+4; 8,472 | Retain valid segments; exclude a continuous-full-task claim. Early push unloads and current operational loaded-entry support is missing. |
| [C14](../../../outputs/evidence/act-a1-first-completion-20261022/tracked-support240-left-attempt-1/report.json) — left nominal, tracked support | 240 Hz; 52.540°; 5.095 N | No loss; explicit gate reproduced | 2+4; 8,126 | Retain full within its qualified domain. All 8,126 A1 commands have exact source/reference request, delta, joint and tool-position parity; the complete observed reference succeeds with zero learned commands. Preserve the original wrapper/finalization incident. |
| [C15](../../../outputs/evidence/act-a1-first-completion-20261022/damping-support240-right-attempt-1/report.json) — right nominal, damping-matched pressure | 240 Hz; 50.220°; 6.862 N | No loss; explicit gate reproduced | 2+4; 4,108 | Retain full within its qualified domain. All 4,108 A1 commands have exact source/reference parity and complete observed execution. Preserve the earlier deadline-limited reference as a separate negative outcome. |

For C14/C15, the saved full-reference operational load streams reach respectively
**5.291/7.058 N upper estimates**, with model/stop verification present throughout;
both full references and their 2+4 s stops pass. This supports execution of the
named trajectories under the qualified 240 Hz recipe family. It is not a new
replay of the latest guard implementation or proof of autonomous recovery.
See `base-reference-review.json`, `qualified-tracked-support240-left-full-parity.json`
and `base-right-reference-parity.json` in the parent campaign directory.

The eight above-10 N native peaks each occur in one recorded contact control
interval. All are below the unchanged 80 N physical limit. Six coincide with
sources that have later continuity losses, while C04/C07 remain continuous;
C01/C06 lose contact despite peaks below 10 N. Neither a native peak below 10 N
nor a passing unloaded terminal stop proves the observed operational 10 N domain
through loaded execution. Conversely, a larger native peak alone does not prove
an operational refusal. The saved nominal240 reference pair did demonstrate
observed refusals at first touch (13.067/15.248 N upper), but those are different
recordings and must not be substituted for these eight sources.

No whole original should be deleted. For future selection, retain already valid
segments with their genuine histories; keep holes, refusals and losses excluded
from positive targets. Do not join across a hole, convert cleanup into recovery,
or use a post-loss continuation as proof that the preceding full cycle passed.
The audit does not certify every retained segment against present operational
guards. That is the remaining compatibility gate, not a reason to discard it now.

## Effective variety and what the learner receives

There are fifteen distinct raw recordings but only **13 configured door/height/
initial-arm combinations**. C14/C15 repeat the original nominal geometry with
changed runtime/teacher control. Four doors include two opening directions,
three source families and different geometry/inertia; handedness is not a second
manipulator. All executions use the same seven-joint right arm and robot asset.

| Dimension | Complete-source coverage | Limitation |
|---|---|---|
| Robot–door placement | One floor/base pose; one contact fraction, 0.295; fixed neck setup | No deliberate base distance, yaw, lateral placement or contact-fraction spread in complete cycles. |
| Contact height | 1.07/1.09/1.11 m: 4/7/4 recordings | All four high cycles lose contact; only C07 of the four low cycles remains continuous. Six of the seven continuous sources are at nominal height. |
| Initial arm / door state | Fourteen nominal arm configurations; C05 changes shoulder-Y by −0.05 rad. Closed starts, with small saved settling offsets | No varied initial door opening or broad posture/velocity distribution; finite measured offsets are not declared reset diversity. |
| Dynamics | Complete cycles use nominal mass scale and hinge settings. Nine retained active-dynamics records confirm 25 kg panels and hinge damping 4 N m s/rad; geometry-dependent inertia differs | No complete light/resistant cycles. Six older records lack the newer active-dynamics sidecar; nominal setup is evidence, not a retrospective measurement. |
| Observations | One common 25 s inspection motion; lighting scale 1.0 in all fifteen; actual head RGB-D and measured streams | No complete dim/bright, new camera viewpoint or deliberately varied occlusion/uncertainty coverage. Image equality or independent visual noise is not assumed. |
| Runtime | Six complete sources at 120 Hz; nine at 240 Hz. All use 60 Hz control and 10 Hz image capture | Same observation shape does not imply the same force, servo response or stopping dynamics. |
| Phase transitions | All six boundaries retain 11 eligible crossing chunks per recording: 990 boundary-crossing starts in total | This is fifteen traversals per boundary, with shared conditions, not 990 independent transitions or fifteen uninterrupted full successes. |

The nominal approach trajectories are close: C01↔C15 differs by at most
1.217 mrad in measured joints across the six-second approach; C02↔C14 by
1.496 mrad. Their contact control and later trajectories differ materially.
High↔low approach differences reach 0.125–0.134 rad. These are descriptive maximum
differences, not an invented duplicate threshold or proof of identical images.

Shared origins extend beyond the fifteen: **50/114 principal sources are
intermediate starts with explicit source ancestry**. Original nominal right/left
each supply five descendants, front/interior nominal each four, and each high/low
complete source two. These are separately executed reset/conditioning episodes,
not continuations with inherited visual history or independent normal-start
demonstrations. The later two late-left corrections share one saved failure path;
the widest-left correction has a controlled error prefix. See
[[approach-contact-corrections]] for their actual scope and exclusions.

Each training item contains one static image plus 22 sampled recent RGB-D images
(five channels, 240×384), and 241 measured robot rows covering four seconds:
9 q, 9 dq, 7 supported projected joint-output values and 7 prior delivered targets,
with availability/validity masks and clocks. Feedback is simulated projected
joint output, not native contact force or a hardware-verified torque sensor. No
teacher phase, source identity, door truth, action role or outcome enters the policy. A1 supervises the actual
delivered target minus measured pre-command q, with 12 future controls (0.2 s).
Execution consumes six controls (0.1 s) before replanning.

The fifteen yield **111,843 overlapping starts**, 66.50% of the principal
168,178-start inventory. Of these, 93,663 (83.74%) start in push. Adjacent starts
share eleven of twelve target controls and 240 of 241 robot-history positions.
They cannot be treated as independent demonstrations. The reader preserves
genuine history through diagnostic actions but rejects target chunks crossing
ineligible commands. All fifteen numeric window lists and delivered A1 labels
reconstruct exactly; no new image-quality or policy-performance claim is made.

## Sampling exposure, rather than source counts

The principal base sampler chooses a door uniformly, then one of its nine
available phase/transition/recovery strata, then a source containing that stratum,
then an eligible start. Each selection draws four rows with replacement from that
same source/stratum; a batch of 32 therefore contains eight such selections.
For source i, its marginal probability is
`sum_s 1 / (4 * 9 * number_of_sources_for_its_door_and_stratum_s)`.
Within-source starts receive the additional factor `1 / starts_in_that_stratum`.
Transition/recovery strata overlap phases, so their mass is added to those rows.

| Door | Sources | Complete | Expected mass from complete recordings | Per complete source |
|---|---:|---:|---:|---:|
| Animated right | 30 | 4 | 7.824% | 1.956% |
| Original left | 30 | 5 | 8.778% | 1.756% |
| Interior left | 27 | 3 | 7.160% | 2.387% |
| Front right | 27 | 3 | 7.160% | 2.387% |

Thus complete recordings receive **30.923% expected base-sampler mass**, rather
than 15/114 = 13.16% or 66.50% of windows. The seven continuous sources receive
13.952%; the eight with losses receive 16.970% through their valid windows.
C14+C15 together receive **3.712%**, including 1.250 percentage points through
the inspection stratum. Their direct approach/contact draws total only 0.654
percentage points; transition draws can also cover those phases. More long push
frames do not increase a source's phase mass.

Each stratum is chosen with 11.111% mass, but actual decision-phase marginals are
inspect 11.229%, approach 14.446%, contact 16.921%, push 16.463%, hold 14.082%,
release 15.043%, stabilize 11.817%, after overlapping transition/recovery draws.
These describe the decision row; a 12-control chunk can cross a phase boundary.

The 61k checkpoint also has an earlier sampling history: steps 0–4096 used the
112-source broad sampler; 4096–6144 used a half-batch focus on C01/C02 with the
same original union; 6144–61000 used the unfocused 114-source base sampler.
The corresponding expected complete-source shares are 29.089%, 64.544%, and
30.923%. Across that saved lineage the expected share is **31.928%**; C14+C15
contribute **3.338%** because they were absent before 6144. The base portion alone
implies about 542,815 complete-source draws, including 65,152 from C14/C15, among
1,755,392 sampled items. These are **expectations**, not measured exposure counts,
unique states or independent episodes. Progress logs do not instrument source
draws; lost/repeated failed-run updates must not be added to the selected model's
lineage. No model or optimizer was loaded to derive these numbers.

If the unchanged broad sampler were later applied to all 129 sources, the complete
share would be **29.848%**, because new correction sources alter stratum
denominators. This is a counterfactual arithmetic result, not executed training
or an adopted recipe. Normalization remains the original 112-source fit.

## Compatibility conflicts and limits of the data hypothesis

The clearest demonstrated issues are narrower than “bad demonstrations”:

1. **End-state success versus continuity.** Eight complete-phase sources recover
   from loading losses that disqualify uninterrupted full completion. Valid local
   behavior remains useful. Their existing masks show why historical admission
   can be correct for segments while a full-cycle interpretation is incorrect.
2. **Runtime and servo dynamics.** Ten of the 114 principal sources, including six
   complete sources, use 120 Hz physics; 104 use 240 Hz. Saved primary 120→240 Hz
   reference replay showed up to 24.446 mm whole-prefix drift, whereas the matched
   C07 240 Hz unloaded prefix was exact. This demonstrates a response mismatch in
   those specific replays, not universal invalidity of every 120 Hz target.
   Terminal stop profiles also differ: the seven older diverse 240 Hz complete
   sources record `[640,640,320,320,40,80,100]`, versus
   `[2560,1280,640,1280,80,160,100]` for C14/C15. These are stop gains, not proof
   of different ordinary-motion gains.
3. **Contact controller behavior and feedback.** Original entry is linear; C14/C15
   slow the final contact approach and blend the push ramp from measured motion.
   C14 uses native-interval support feedback and pending-gap antiwindup; C15 uses
   guarded filtered feedback and a 0.3 N push request matched to hinge demand.
   Teacher native contact information is not itself a policy observation.
   Successful matched A1 reference execution shows those commands are deliverable;
   it does not show that the learner can choose them from RGB-D/proprioception.
   Similar-looking states from different force/servo histories may call for
   different deltas. This audit does not establish an exact contradictory-input
   label pair or justify exposing evaluator truth to the policy.
4. **Dynamics and reset support.** The wider 114-source set contains 16 light
   (mass scale 0.7) and 16 resistant (1.3) scoped episodes, but no complete cycles
   under either variation. Reset-conditioned late phases may be physically useful
   without covering an autonomous approach→loaded-push→release trajectory.
   Short A/B/C cleanup is already masked according to task scope; its presence is
   not evidence that the policy was taught to release early.
5. **Operational support is state/history dependent.** Fresh inspection and genuine
   incoming history can yield different swept-support results, as the widest-left
   evidence demonstrates. Low source force, individually observed joint extrema
   and terminal stops cannot substitute for prospective position/load/sweep/stop
   support along the actual trajectory.

The evidence supports a **coverage and compatibility hypothesis**: few complete
current-executable examples, narrow start/observation variety and limited direct
sampling of their acquisition behavior can matter. It does not show that those
facts caused the 61k failures, that all original supervision is incompatible, or
that more demonstrations, more updates or different weights will fix them.
The saved diagnosis already establishes learned approach drift under valid
observations; near-contact staging separately exposes right impact/deceleration,
while a supported left start retains contact during early opening. Clock or A1
label corruption is not established by this audit.

## Earlier 52.3° behavior versus current failures

The old result is real partial capability. In `selected-v10`, ACT A1 left/dim
executed **7,171 learned commands**, achieved **52.295° controlled opening**
(99.541% of the same 52.536° matched reference), valid hold and zero contact losses.
It did not attempt release. Its native peak was 7.490 N and its independent
2 s stop passed. Terminal load-domain evidence isolates gripper-Y q above the
then-verified 0.693480 rad maximum by 0.163/0.279 mrad at the last two native
samples; the preceding force upper was only 0.331 N. No threshold is adjusted.

| Factor | Earlier left/dim partial result | Selected 61k normal-start pair |
|---|---|---|
| Policy/data | ACT A1, seed 17, 12k updates; two corrected train episodes, right/dim and left/bright | Temporal ACT A1, seed 9101, 61k selected from a 75k ceiling; eventual 114-source union |
| Allowed observations | Static RGB-D plus three recent 60×96 frames at 0.1 s spacing; current q/dq | Static plus 22 recent 240×384 frames and four seconds of q/dq, supported joint-output feedback and prior delivered targets |
| Runtime | 120 Hz physics; 60 Hz control; chunk 12 / execute 6 | 240 Hz physics; 60 Hz control; chunk 12 / execute 6 |
| Setup | Same left door, nominal geometry/contact height, lighting 0.9 | Left nominal lighting 1.0; right nominal lighting 1.0; same robot and matched opening references |
| Operational contract | Historical calibration/position and contact recipe; terminal gripper-Y load-domain refusal | Current qualified 10 N recipe; right acquisition budget; left future shoulder-Z motion/stop support refusal |
| Outcome | Opening and hold; no release or full success | Right 428 and left 239 learned commands; no valid contact or controlled opening; both 2+4 s stops pass |

The old ACT checkpoint's **other five fixed-panel conditions had zero controlled
opening**, including left/nominal. Therefore the single left/dim result was not
previously demonstrated robust completion, and the new nominal pair is not a
matched rerun of that successful condition. Its six ACT outcomes and the older
24-outcome four-model panel remain separate from the current campaign's 0/22.
Do not combine them into a new success denominator.

Saved current phase diagnostics add relevant evidence: right near acquisition
achieves one valid contact interval but impacts at **13.763 N native / 14.199 N
observed upper**, beyond operational support; left after a reference prefix retains
contact for **600 learned commands / 10 s**, adding **10.471°**. These are mixed
diagnostics, not full learned starts. See [[act-a1-failure-diagnosis]].

Checkpoint, observations, data, lighting, physics, calibration and guards changed
together. Existing evidence does not isolate which change explains the difference,
nor whether a sampling/curation change recovers old capability. A later matched
experiment would need to control those factors and distinguish physical progress
from operational refusal. No historical replay/inference campaign was launched.
The old case is linked through `context-audit.json` and
[[b1-end-to-end-pilot-resume]]; no missing behavior is invented.

## Priorities for the next data-planning discussion

1. **Agree on prospective source scope first.** Preserve C14/C15 as full-reference
   anchors; distinguish the five missing-qualification candidates from the eight
   demonstrated discontinuous originals. Resolve the consequential current
   loaded-entry/runtime questions before counting candidates as executable cycles.
2. **Connect existing recoveries to task completion.** The fifteen untrained
   additions already cover specific posture/velocity, pause and near-contact
   errors. Their qualified contact endpoints do not demonstrate recovery followed
   by continuous opening, hold, intentional release and stabilization. That
   transition is missing; collecting the same contact prefix again may add little.
3. **Prioritize safe acquisition and push handoff.** Needed situations include
   deceleration into contact, appropriate waiting versus active progress, and
   response to changing incoming motion without exhausting support or causing an
   impact. Preserve negative attempted responses; judge contact/opening outcomes
   and appropriate correction, not teacher-joint agreement.
4. **Repair the effective complete-cycle variety on existing doors.** No high
   complete source is continuous, and only one low source is. Initial relative
   distance/orientation, posture, observation uncertainty and dynamics have little
   or no complete current-runtime support. Choose which uncertainty matters for
   the next claim before adding more variations; this audit sets no matrix or quota.
5. **Make a prospective sampler reflect those scopes.** Keep door balance and
   valid transition/correction targets, but separate full compatible anchors,
   qualified partial segments and unresolved candidates. Review nominal onset,
   loaded handoff, hold/release and genuine correction exposure explicitly; global
   “complete” mass overstates direct critical-phase exposure. Group reset descendants
   and close nominal repeats when judging diversity. Log future per-source/phase
   draws so exposure is measured. No numeric reweighting is adopted here.

Saved evidence can establish masks, ancestry, native continuity, source endpoints
and arithmetic exposure. Missing current observed qualification requires a later
bounded reference/stop experiment; effects on autonomous behavior require a
separately approved training and matched learned evaluation. The audit stops here.
