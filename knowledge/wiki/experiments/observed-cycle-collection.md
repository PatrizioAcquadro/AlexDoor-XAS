# Observed teacher collection and contact

## Whole-cycle training conditions

The teacher was evaluated on four training doors with nine conditions each. Admission required the observed approach, supported contact, opening, hold, release, ordinary stabilization and independent stopping. The final coordinator receipt records **17/36 qualified conditions** (13 ordinary, four recovery), including two previously qualified sources. D1 reached 8/9, D2 9/9, D3 0/9 and D4 0/9. The 17 sources supply 100,295 overlapping windows. No optimizer updates were performed on this cohort. Unexecuted cells, partial phases and component trials retain their own outcomes.

The final per-condition allocation, qualified sources, negative outcomes and remaining coverage are in `outputs/evidence/act-a1-complete-cycle-20261026/FINAL-HANDOFF-20261108.md` and its JSON companion at revision `aa328df`. The separate earlier 112-cell training/development acquisition used different phase scopes. Its short segments and complete cycles are counted separately in [[policy-pilot-results]].

## D3 material support

The root-anchored observed cycle reached contact and 1.333° of opening. Continued push required bounding a finite finger volume on the moving material. The measured cells and pose uncertainty could not support the whole volume. The independent stopped geometry passed review. A refusal on coverage alone does not establish a collision.

Subsequent saved-state analysis at 10.017° still found 12.882–14.733 mm worst tangential coverage deficits across four ideal layouts. The offline alternative was not adopted. Three fresh prompt alternatives produced zero masks in 12 controls. The complete D3 history and allocation denominators are in `bc8fb95:knowledge/wiki/experiments/d3-observed-cycles.md`. Local inputs, independent reviews and failed controls are under `outputs/evidence/d3-observed-cycles-20261108/{root-anchor-native-review,tangent-rotation-review}/`.

## D4 hold, release and safe stop

The PhysX contact-last nominal opened the door and held it. Separation after withdrawal remained unconfirmed, so the episode did not qualify. The closed review, per-phase measurements and separate stop audit are in `outputs/generated/d4-recontact-resolution-20261112/nominal-closeout/`.

On the first guarded Steel nominal, the tracker’s orientation support failed late in opening. On the second, the actor stopped during push when its load estimate lost a measured planar core. The report records `safety_stop_passed=true`, `task_success=false` and `demonstration_qualified=false`. See `outputs/generated/d4-release-steel-20261113/nominal-observed-01/` and `nominal-observed-02-launch/receipts/report.json` under the same root.

Revision `0f97bd2` used seed reprojections to associate measured planar support on the recorded refusal window. The historical 1,000-frame replay uses reconstructed depth pairs. Native source correspondences, whole-motion admission and a full demonstration require separate evidence. Replay and component records are under `outputs/generated/d4-planar-tracker-20261116/`. These investigations add no qualified matrix cell.
