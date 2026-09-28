# Datasets

Local reusable episodes and split/normalization products use
`datasets/<task>/<action_space>/<version>/`; payloads stay outside Git.

B0 datasets were removed. B1 collection, observed/truth separation and Replicator
integration remain [Phase 6 work](../knowledge/wiki/implementation_phases/phase-6-perception-actions-and-demonstrations.md).

The retained numerical loaders/exporters are described in
[Episode and Dataset Contracts](../knowledge/wiki/topics/episode-and-dataset-contracts.md).
There is no currently supported end-to-end B1 dataset-generation command.

The frozen B1 **asset identity** split is tracked separately in
[`assets/doors/b1/corpus.json`](../assets/doors/b1/corpus.json): 19 train,
six development and seven test doors. Use `qualification.corpus.load_corpus`
to validate it before Phase 6 collection. The numerical episode split helpers
do not replace this family-separated assignment. Qualification evidence stays
in the verification cache and must not be exported as learning demonstrations.
