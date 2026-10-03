# Selected perception resources

Local weights are preserved for GroundingDINO + SAM 3 with explicit RGB-D/multiview
geometry and DINOv3 image features for the maintained 6.0B static scan.
It is not a qualified perception release; Point2Pose integration remains future work.

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
subdirectory. The image worker keeps its native SAM3 source and dependency overlay
under ignored `runtime/`. The shared Isaac NumPy/PyTorch/Transformers installation
is unchanged. Only the worker prepends this overlay; RGB and packed output bytes
cross the process boundary, avoiding NumPy 1/2 ABI and pickle incompatibilities.

## Native worker environment

The workstation uses Isaac Python 3.12.13, shared PyTorch 2.10.0+cu128 and
Transformers 4.57.6. Create the ignored environment with the supported Python:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m venv --without-pip --system-site-packages models/perception/runtime/venv
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install --no-deps \
  --target models/perception/runtime/venv/lib/python3.12/site-packages \
  -r models/perception/runtime-requirements.txt
```

Retain the official source archive at revision
`2345a4ad109ac29c569da749c91d84f10dc08c40` in
`runtime/sam3-2345a4ad109ac29c569da749c91d84f10dc08c40`. Install that source with
`--no-deps --no-build-isolation` into the same target directory using the same
Python. Do not install the upstream dependency bundle into Isaac. Native SAM 3
requires NumPy 1.x here; SciPy/scikit-learn are paired with that ABI. The builder
recognizes `device="cuda"`; the worker explicitly moves and checks all weights on
`cuda:0`, freezes parameters and uses inference mode. No model download occurs
during smoke/scan; missing resources or CUDA cause an error.

`runtime-requirements.txt` contains the image builder's required dependencies.
The retired video path's `decord` dependency was removed from both this list and
the local overlay; the external runtime was not modified.

Smoke all three models with a fresh output:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py smoke \
  --output outputs/b1/perception/NEW_SMOKE
```

This command performs only frozen inference on a train RGB observation. A passed
smoke establishes executable models, not full-state accuracy or control safety.
