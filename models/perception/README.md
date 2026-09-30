# Selected perception resources

Local weights are preserved for GroundingDINO + SAM 3 with explicit RGB-D/multiview
geometry, and DINOv3 where learned visual features are needed. This is a component
direction, not an implemented or qualified perception pipeline.

| Directory | Upstream model | Weight revision |
|---|---|---|
| `grounding-dino/` | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| `sam3/` | `facebook/sam3`, native `sam3.pt` | `3c879f39826c281e95690f02c7821c4de09afae7` |
| `dinov3/` | `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |

Model directories are ignored by Git. Keep their configuration, preprocessing,
tokenizer and license files with the weights. SAM 3 also retains
`bpe_simple_vocab_16e6.txt.gz` from official source revision
`2345a4ad109ac29c569da749c91d84f10dc08c40`; its native checkpoint is not a
Transformers `model.safetensors` export. GroundingDINO's license and model card
were retrieved from the [official source](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE)
and [pinned model repository](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71).

Weights were checked against the original download digests and moved with their
bytes unchanged. Local hashes, original download reports and path relocations
are in `outputs/b1/perception/evidence/cleanup.json` and its `comparison-01/`
subdirectory. No experiment-specific environment or third-party source checkout
is maintained here. Integration dependencies will be selected with the future
pipeline; none were installed into the shared Isaac runtime by cleanup.
