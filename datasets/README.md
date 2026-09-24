# Datasets

Local reusable episodes and split/normalization products use
`datasets/<task>/<action_space>/<version>/`; payloads stay outside Git.

B0 datasets were removed. B1 collection, observed/truth separation and Replicator
integration remain [Phase 6 work](../knowledge/wiki/implementation_phases/phase-6-perception-actions-and-demonstrations.md).

The retained numerical loaders/exporters are described in
[Episode and Dataset Contracts](../knowledge/wiki/topics/episode-and-dataset-contracts.md).
There is no currently supported end-to-end B1 dataset-generation command.
