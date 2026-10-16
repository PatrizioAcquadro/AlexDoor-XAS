# Pilot Contact Prerequisites

Baseline `023804c`, branch `PatrizioAcquadro/b1-policy-pilot`, October 16, 2026.
The [[b1-initial-placement-pilot|one common nominal placement]] passes unloaded
approach/hold/stop, but that does not establish complete door pushing. Three
current-teacher diagnostics now demonstrate the remaining physical blocker.

## Fixed protocol and boundaries

Allocate the first three cases from the existing twelve-case prerequisite ceiling
to one complete teacher cycle on each already selected train/development identity,
using fixed 6100/6100/6200 seeds, a fresh Kit process per asset and the accepted
common floor pose. Keep current IK/gains, collisions, contact point, original
teacher logic, physical limits and validity rules. The teacher/evaluator may use
privileged task truth; no policy or operational admission is invoked.

Disable cameras/RGB-D and initialize the neck at the existing inspection's final
target. These are physics-only diagnostics, not collection: no new demonstration
or trained policy result is created. Keep the original corpus expert references
from the historical episode metadata. A valid complete teacher cycle with holding/
release must sustain at least 80% of that preserved reference; never replace that
reference by the new result. Record all robot commands and native feedback at both
120 Hz substeps, full contact traces and actual 60 Hz root endpoints. The mounted
robot remains physically fixed. After release, latch measured joints and continue
physics for two seconds; this is an unloaded post-release observation, not a moving
loaded-contact-stop qualification.

## Complete results

| Selected identity | Preserved expert / required 80%, degrees | Valid sustained progress | Valid hold | Release | Result |
|---|---:|---:|---|---|---|
| Right train `animated-door-1-88abf40` | 61.393 /49.115 | 46.091 (75.1%) | None | Yes | **invalid_hold**, tracking margin |
| Left train `door-2738468b94d74c5f` | 57.839 /46.271 | 35.480 (61.3%) | None | Yes | **lost_contact** |
| Development `psx-industrial-001-4f5561b` | 46.994 /37.595 | 32.625 (69.4%) | None | Yes | **lost_contact** |

No case passes. The raw angle later reaches 66.221 /57.470 /46.902 degrees while
the door coasts without valid controlled contact. Those angles do not replace
valid sustained progress or count as task completion. Development informs the
regime diagnosis only, never weight updates or normalization.

The source's nominal opening reference is 1 degree/s, but it follows a leading
coasting door. Contact/push door speeds reach about 12-14 degrees/s. At loss of
contact, the last push states are not at a joint boundary (minimum margins about
0.364 /0.171 /0.208 rad). The right hold subsequently loses reach. Persistent
positive compression/lead with the current impedance is a concrete next hypothesis
for a shared teacher load/reference correction; no unique causal explanation or
successful correction is claimed without an intervention.

Actual right-arm joint speed peaks at **4.47 rad/s** in right shoulder Z during
hold at 15.558 s; independent position differences confirm 4.470 rad/s. There is
no effort clipping or joint-position violation beyond the original 1e-4 rad
allowance. Left/development peaks are 0.453 /0.462 rad/s. The accepted approach-speed result
does not extend to this full contact/hold/release regime. The target-step cap is
not a measured-speed guarantee. Do not promote the controller/load domain based
on unloaded success or the teacher's eventual release.

The cases contain 57.30 simulated seconds and about 78.66 measured case wall
seconds, excluding the shared-GPU wait, outer launcher and teardown. All 6,876
native feedback samples are retained. Independent FK matches teacher control
endpoint poses below the existing 2 micrometer/1e-5 rad check; clocks and feedback
semantics are verified. Post-release latches remain fixed during two seconds of
physics: maximum travel 0.153 /0.146 /0.131 mm, final speeds below 8e-5 rad/s.
These stops occur after release, not during moving contact. Old residual calibration
and all qualification flags stay unchanged; unavailable estimates remain unavailable.

## Independent observed-contact interface blocker

The existing `surface_support` starts its position budget at the 4 mm plane
tolerance plus measured residual. `ContactBudget` carries that local contribution
into the exported patch pose support. The current observed push/hold monitor
requires `gap_upper <= 0.1 mm` and adds pose uncertainty plus robot error to that
upper bound. Even with perfect nominal alignment and every other uncertainty set
to zero, the current recipe has a 4 mm lower bound, forty times that threshold.
Missing calibration/identity/sweep fields can only cause earlier refusals.

`contact-observability-audit.json` proves this source/config contract gap; it is
not a simulated observation or evidence that every observed-only method is
impossible. Wiring the existing diagnostic provider cannot by itself qualify
loaded policy execution. A measured observed contact-distance/loaded-contact
certificate compatible with the unchanged physical task criterion is indispensable,
along with material identity, actual-controller continuous sweep/stop support,
two-finger/collision clearances and material response/load bounds. No bound,
qualification flag, threshold or evaluator truth is substituted for that evidence.
SAM3/P2P quality/performance work remains separate; the other tree at `99fa48e`
is only inspected and supplies no newer qualified interface.

A separate interface bug is corrected in `8385bbf`: fresh local patches now export
canonical `tracking` rather than the native internal `tracked` state. The temporal
regression reproduces the old mismatch and passes after correction; missing bounds
and ambiguous ownership still refuse. This removes one real integration defect
without claiming that the indispensable producer evidence exists.

## Decision and preservation

Stop this prerequisite allocation under its frozen failure rule. Three of twelve
physical cases are used; **nine remain unspent**, not silently consumed or called
failed evaluations. No alternative setup, controller variant, physical retry,
recollection or learning is started from these failures. The four learned cells
still have zero new physical policy episodes; the planned learned panel is unfrozen
and unexecuted, not 0/24 evaluated failures.

The next concrete decision is one separately bounded shared teacher load/reference
correction using the remaining case allocation, plus resolution of the observed
contact certificate with the maintained perception interface. Preserve task point,
expert references, both train doors, physical rules and independent evaluation.
A new teacher success alone would still not resolve the observed-input blocker.

Local `outputs/evidence/b1-shared-learned-pilot-20261016/contact-prerequisites/`
retains frozen protocols, source snapshots, all complete cases, commands, native
feedback, teacher/contact traces, monitors, independent analysis and `outcome.json`.
The first input attempt fails before environment creation because ignored prepared
USD payloads are absent from this worktree; it is preserved as infrastructure,
with zero physical cases. The successful retry reads only the three existing
primary-checkout asset folders after byte-identical prepared/recipe/candidate
checks. No primary or separate worktree is modified. Kit can return zero after an
initialization exception; explicit per-child completion markers prevent that from
being mistaken for completed physical work.
