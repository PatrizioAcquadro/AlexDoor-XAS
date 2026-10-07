# Selected-Candidate Registration with Preserved Alternatives

One diagnostic schedule is implemented from `main @da804ad`, following
[[../topics/pre-a4-initialization-and-hinge-budget|the shared pre-A4 protocol and uncertainty audit]].
The reference retains graph off, partial batch on, rollback off, five inliers,
4 mm and the vectorized SDF Jacobian. A3/A4 and admission are unchanged.
No acquisition, robot/leaf movement, contact or training is performed.

## Fixed intervention and lifecycle

`selected_registration_only=True` is an opt-in offline diagnostic. The normal
worker/provider defaults are unchanged. It requires graph-off frame-to-map and
`diagnostic_only=True`. Select the original first eligible automatic observed
candidate (native object 0), with original masks, order, seeds and geometry.
Selection is not leaf identity or contact admission. No truth, asset identity or
teacher command crosses the worker's observation whitelist.

Register the selected candidate every acquisition. Register alternatives every
nine acquisitions (150 ms at original 60 Hz). If selected native support fails,
expand to all alternatives in that same frame and continue on the next frame;
recovery retains an already expanded frame. There is no oracle switch, merging
of object maps, identity transfer or model reset. These are observed native gates,
not truth-driven recovery triggers. The audit period is an acquisition cadence,
not a claim that an alternative's publication meets the 150 ms freshness limit.

Every original image still enters shared SAM2, BootsTAPIR and the track table.
Keep all initial maps/TSDFs, queries, historical IDs and original bounded 120-point
renewal. A deferred candidate follows native insufficient-registration handling:
its pose is frozen and unavailable, its current pairs are empty, and map growth,
keyframe sampling/promotion and retirement are suppressed. Explicit
`registration_evaluated=False` distinguishes planned deferral from measured native
failure. No copied pose gains a new support time. Alternatives remain observable
at audits and during expansion, but their dense pose availability is deliberately
reduced. They cannot be described as unchanged backup estimators.

Skipping alternative registration and its unsupported map updates changes the
shared NumPy random stream, candidate/reference history, pending expiry/promotion,
mask filtering and later sampling. The selected estimate may therefore differ on
later frames even though its own gates and registration implementation are fixed.
No independent RNG intervention is combined with the schedule. Deferred native
loss streaks cannot activate a relaxation in this reference: the existing
`pose_jump_guard_relax_when_lost` defaults false. Frame-to-frame/global-graph modes
are refused because their history semantics have not been assessed.

## Comparison protocol

Ignored drivers, protocols, original-frame/candidate rows, native traces, actual
CUDA storage inventories and reports remain under
`outputs/b1/perception/p2p-selected-registration-01/`. Original baselines and
failed trials elsewhere are preserved. The targeted comparison starts at the
original 31 s initialization and processes **every original frame through 41 s**:
601 frames per attempt, both light and nominal, fresh processes and one attempt
per condition/system. It covers static initialization, initial motion and the
previous first numerical hinge times, not just a reset at a favorable event.

Both systems use the current vectorized implementation, identical automatic
initialization and actual IPC. The all-registration reference is also checked
against the preserved partial-only poses/decisions. The pilot progression rule
is fixed before inference: no more than two percentage points lower correct
availability/accepted precision, no more than 150 ms extra maximum correct-pose
gap, and at least 20% median request saving in both conditions. This is an
exploratory progression rule, not a physical qualification threshold or a
statistical confidence claim. If promising, evaluate both complete original
31–78.6167 s sequences from the original seed, retaining tails and all failures.

Correct availability remains accepted integration and relative seed-zone position
within 1 cm/full rotation within 5 degrees. Exclude seed from error/acceptance
counts but retain the complete scheduled denominator. Native loss, inaccurate
acceptance, deliberate deferral, unavailable integration and original-material
recovery are distinct. Compare paired gains/losses, longest gaps, the terminal
window, every candidate's audit availability, reference/cohort/keyframe/TSDF
history, measured CUDA storage and resident peaks, and complete request timing.
Acquisition-time tracking quality is separate from publication age.

Results and adoption decision are pending the fixed CUDA attempts. No new
operational default is adopted while they run.
