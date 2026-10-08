# Datasets

Local B1 recordings and future matched policy datasets stay outside Git.

B1 perception engineering uses the maintained `scripts/collect_perception.py` and
`b1.rgbd.v1` streaming format: N+1 synchronized RGB-D/proprioceptive observations,
N actual commands, separate labels, calibration and a factual terminal result.
The collector validates the frozen corpus, rejects test identities, uses fresh
Isaac CUDA processes and never updates qualification records. `--resume` skips
only complete validated episodes. Collection requires separate authorization.

All **50 `b1/perception/engineering-v2` episodes** remain intact (about 729 GiB).
They support 6.0B and future common train/development qualification; they are not
the final matched policy dataset. Superseded `engineering-v1`, `inspection-pilot-01`
and `inspection-tallest-01` were removed after preserving 53 recording headers,
calibration/metadata and essential results. Earlier interrupted payload removals
remain recorded separately. The inventory is
`outputs/b1/perception/evidence/cleanup.json`; preserved superseded headers are in
`superseded-recording-headers-20261002.json` beside it. Removed ignored recordings
are not recoverable from Git. Cleanup started no collection or training.

The frozen **asset identity** split is tracked in
[`assets/doors/b1/corpus.json`](../assets/doors/b1/corpus.json): 19 train,
six development and seven test doors. `qualification.corpus.load_corpus` validates
it; numerical episode split helpers do not replace its family assignment.
Qualification evidence stays in the verification cache, outside learning data.

B1 dataset export, normalization and sampling are described in
[Episode and Dataset Contracts](../knowledge/wiki/topics/episode-and-dataset-contracts.md).
B0 data are retired. Final matched B1 policy data remain future Subphase 6.2 work.
