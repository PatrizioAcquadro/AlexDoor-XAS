# Project status

AlexDoor-XAS runs fixed-base Purdue Alex003 door manipulation in simulation. The frozen B1 corpus contains 32 qualified doors across 12 families, split into 19 training, six development and seven sealed test identities. Standard engineering collection and door qualification use native Steel/Warp. Full D4 attempts use guarded beta2 Steel with RTX RGB-D. Bounded D4 components can also use native Steel/Warp. The source checkout also supports recorded perception diagnostics, observed teacher collection, dataset preparation, ACT/Diffusion training, checkpoint replay and evaluation. [[topics/system-architecture|Workflows and consumers]]

| Question | Measured outcome |
|---|---|
| Can the observed teacher complete whole cycles with variation? | **17/36 training conditions qualified** across four doors. D1 achieved 8/9 and D2 9/9. D3 and D4 each remain at 0/9. [[experiments/observed-cycle-collection]] |
| Can a learned controller finish the task? | The audited temporal ACT-A1 campaign achieved **0/22 full learned completions**, including 0/2 normal starts at the selected 61k checkpoint. [[experiments/policy-pilot-results]] |
| Did guarded D4 Steel execution stop safely? | The second nominal stopped during push when load estimation lost measured planar support. Its safety stop passed. The task and demonstration both failed. [[experiments/observed-cycle-collection]] |

## Open work

D3 needs finite material support over the moving door. D4 needs observed support through push, separation during release, ordinary stabilization and independently verified stopping in a complete cycle. The recorded D4 tracker repair has offline evidence. Its saved follow-up component has no terminal qualification receipt.

General A3/A4 perception and contact admission, a matched Phase 6.2 dataset and the Phase 7 comparison remain open. Teacher and learned campaigns are paused. The sealed test set remains closed. Physical robot execution has not been qualified. [[implementation_phases/phase-6-perception-actions-and-demonstrations|Phase 6]] · [[implementation_phases/phase-7-training-and-generalization-evaluation|Phase 7]]
