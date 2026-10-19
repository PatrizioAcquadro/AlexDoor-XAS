# ACT A1 Two-Door Diagnostic

The October 19, 2026 bounded diagnostic is complete: **0/2 full-task successes,
two completed operational failures, no interruptions or retries**. This does not
resume the paused scientific panels or authorize training/collection. Baseline
is `PatrizioAcquadro/b1-policy-pilot @ 02fbb83`; executable product behavior since
the speed measurements at `c7c62f8` is unchanged.

The user declared one nominal attempt on each existing training door, right then
left, with simulation seed 7100 and inference seed 23. Both use the same audited
30k ACT A1 checkpoint, `rgbd-conv-v1`, current operational closure recipe,
chunk 12/replan 6 and **9,000 commands available after 1,512 inspection ticks**.
No wall-time estimate truncates either attempt. Physics remains 120 Hz,
control/render 60 Hz and consumed RGB-D 10 Hz. Identical passive prediction
recording is the only addition; policy inputs, weights, controller, physical
criteria and operational settings stay fixed. Evaluator truth never feeds control.

## Measured outcomes

| Result | Right nominal | Left nominal |
|---|---:|---:|
| Executed commands | 6,238 | 368 |
| Process launch through recording flush, stop and exit | 979.767 s | 205.022 s |
| Controlled opening | 45.325 deg | 0 deg |
| Matched 80% requirement | 41.008 deg | 42.029 deg |
| Valid hold | Yes | No |
| Commanded release attempt / completed release | No / no | No / no |
| Contact losses / physical violations | 0 / none | 0 / none; contact never acquired |
| Native force peak | 7.096 N | 0 N |
| Two-second continuing-physics stop | Pass | Pass |

Neither reaches a post-release phase. Stop verification is not commanded release
or post-release stabilization. Both recordings pass clock/causality/role audits,
all executed requests match the actual saved prediction rows, and independent
`PilotEvaluation` replay exactly reproduces both outcomes. The full horizon is
available but existing operational guards stop execution early.

## What causes stopping, and what remains uncertain

Right contact is valid at t 31.3667 s with 4.004 mm material error; the bias remains
3.907 mm at t 34.2 s, versus at most 0.231 mm in the two eligible same-door push
trajectories. This departure from the narrow demonstrated corridor is still
inside the 10 mm physical requirement. The controller follows the submitted
motion; requested/applied target differences are at most 1.20e-7 rad.

At t 129.1667 s, `outside_verified_load_envelope` stops the right case after valid
opening/hold, before release. Measured wrist Z 1.773560405 rad lies 10.014 microradians
below the empirical load-domain lower edge 1.773570418 rad, with zero adopted
allowance on that side. The preceding prediction requests -0.158787 mrad and is
applied as requested. No force, collision or physical joint violation is measured.
The calibrated-domain refusal is real; a physical safety violation or inability
to release later is not established. A future extension needs independent
calibration/stop support, not a favorable policy retry or inference from the
small numerical excess.

Left first departs from an admissible target at t 28.9167 s: wrist Z requests
+1.745061 mrad but the unchanged physical limiter applies +1.552582 mrad. There
are 19/368 clipped requests with maximum 1.055812 mrad correction; the first six
episode actions themselves match to float32 precision. The controller prevents
a physical violation. Clipping is not established as the cause of the later stop.

At t 31.3333 s, the next action is refused by
`weak_position_requires_contact_free_motion_and_stop`. Current separation is
31.999 mm with zero native force. The prospective shoulder-Z motion/stop envelope
extends 12.2 microradians above the main calibration maximum, invoking a positive
contact-free certificate that is unavailable (margin -0.548986 m). That margin
is not measured penetration; current shoulder Z remains within the main domain,
local geometry margins are positive and the actual stop passes. The future
support requirement and its conservatism remain an operational investigation,
separate from learned-action quality and physical scoring.

Both tails are censored by operational requirements. The right corridor drift
and left target clipping support targeted data/transition work, but neither
outcome identifies insufficient temporal information as its cause.

## Data and temporal-information proposal

The saved proposal considers four train identities: retain the existing thin
right and wider left doors; add thick right `psx-front-005-ee7d5c6` and narrower
thick left `psx-interior-wood-002-02e442c`, subject to current-setup qualification.
Added geometry may help initial completion, as well as later generalization.
Nine pairwise posture/robot-door/observation coverage cells per door give 36
normal trajectories; two designed recoveries per door give 44 total training
targets, plus eight independent development motions. Existing records count only
for cells they actually cover. Stage prospective collection attempts, retain
all failures, and budget raw storage/attempts explicitly; these targets exceed
the old remaining demonstration allowance. Every cycle includes transitions,
varied qualifying hold duration and stable post-release behavior.

A four-second causal RGB-D/proprioception/applied-action window is a candidate,
with finite-window temporal convolution/attention before prescribing an RNN.
Current ACT and earlier B1 grid inputs both span 0.2 s of recent images. The
scripted reference instead has 0.1 s force filtering, 0.5 s trends, phase timers,
integrated state and privileged inputs; its 3 s hold is not a 3 s visual history.
Tiny input inequality does not prove robust distinguishability under normal
variability. Test window content, duration and perturbations on independent
development motions, scoring transitions and the first six actions of every
chunk separately from all twelve predictions.

The 68 questionable right+ push samples all fail the 0.1 s filtered-load threshold,
not gap/footprint/material pose. Preserve their original annotations and inspect
native load/command joins before prospective training-only masks or weights;
they are not qualified recovery demonstrations or policy contact features.

Before training, resolve the retained calibration/support questions independently.
Then select on independent development trajectories and full-task behavior;
replace the provisional 5%/+10% relative rules with declared absolute/trajectory
metrics and measured variability. Treat 4k updates as an initial check, with a
separately authorized envelope up to 20k if justified by data exposure/development
progress. Keep all physical requirements and historical results fixed. Final-test
data remain closed; no proposed collection, training or architecture work runs here.

## Evidence and accounting

The [complete report](../../../outputs/evidence/act-a1-two-door-diagnostic-20261019/report.md),
[proposal](../../../outputs/evidence/act-a1-two-door-diagnostic-20261019/proposal.md)
and [summary](../../../outputs/evidence/act-a1-two-door-diagnostic-20261019/summary.json)
retain per-door raw streams, predictions, applied commands, guard measurements,
first-six-action audits and independent scoring. Both original failures are
complete recordings; CLI exit 1 is the failed operational outcome, not a recording
crash. No execution/recording correction or infrastructure retry was needed.

The existing speed closeout is verified without new benchmarks:344.736 s remains
a 1,631-command learned prefix;1,065.262 s is a 7,217-command reference cycle,
with zero learned commands. Seven recorder tests pass. See
[[b1-runtime-speed|speed evidence]] and [[b1-operational-tolerances|calibration limits]].

New allocation is 1,194.789 s including five-second conservative tails per case;
cumulative local allocation is 16.700309/40 h, retained evidence 423.988/1024 GiB.
No remote allocation, training update or demonstration is added, and no owned
GPU workload remains. Prior panel denominators remain unchanged.
