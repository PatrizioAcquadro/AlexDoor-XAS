# Datasets

Local reusable episodes and split/normalization products use
`datasets/<task>/<action_space>/<version>/`; payloads stay outside Git.

B0 datasets were removed. B1 perception-engineering collection uses
`scripts/collect_perception.py` and the separate `b1.rgbd.v1` streaming format.
Each episode stores N+1 synchronized RGB-D/proprioceptive observations, N actual
pose/joint commands, separate labels, and a factual terminal result. These
recordings are not the final matched policy dataset.

The collector reads the frozen corpus, rejects test identities, uses fresh Isaac
processes and never updates qualification records. `--resume` skips only complete,
validated episodes. Payloads remain local under `datasets/b1/perception/`.

The retained numerical loaders/exporters are described in
[Episode and Dataset Contracts](../knowledge/wiki/topics/episode-and-dataset-contracts.md).
Final matched B1 policy-dataset generation remains Subphase 6.2 work.

The frozen B1 **asset identity** split is tracked separately in
[`assets/doors/b1/corpus.json`](../assets/doors/b1/corpus.json): 19 train,
six development and seven test doors. Use `qualification.corpus.load_corpus`
to validate it before Phase 6 collection. The numerical episode split helpers
do not replace this family-separated assignment. Qualification evidence stays
in the verification cache and must not be exported as learning demonstrations.

Preserved engineering resources: `b1/perception/engineering-v1` and
`engineering-v2` each contain 50 complete episodes; `inspection-pilot-01` has two.
`inspection-tallest-01` contains one complete inspection diagnostic without expert
hold/release, relocated from outputs. Preserve its diagnostic status. The earlier
campaign lacks the newer inspection; use each recorded calibration/recipe.
Five interrupted HDF5 payloads were explicitly removed after saving metadata,
calibration and failure evidence. Feature caches of the retired estimator were
removed; they are not original observations. Details and path mappings are in
`outputs/b1/perception/evidence/cleanup.json`. No ignored payload is recoverable
from Git. No dataset generation or collection was performed by cleanup.
