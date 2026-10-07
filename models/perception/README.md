# Selected perception resources

Local weights are preserved for GroundingDINO + SAM 3 with explicit RGB-D/multiview
geometry and DINOv3 image features for the maintained 6.0B static scan.
Neither the static scan nor the optional Point2Pose prototype is a qualified perception release.

| Directory | Upstream model | Weight revision |
|---|---|---|
| `grounding-dino/` | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| `sam3/` | `facebook/sam3`, native `sam3.pt` | `3c879f39826c281e95690f02c7821c4de09afae7` |
| `sam3.1/` | `facebook/sam3.1`, native `sam3.1_multiplex.pt` | `daa63191845a41281374e725f4c9e51c7a824460` |
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

## Point2Pose 6.0C worker

The optional CAD-free prototype uses upstream revision
`51856226610df75e5c06e8de545bd27f7c4ba99c` and its paper configuration:
BootsTAPIR, SAM2 large, SuperPoint, clustered SVD registration, keyframe graph and
CUDA TSDF. `scripts/setup_point2pose.py` downloads the pinned sources and official
checkpoints into ignored `point2pose/`, and creates a separate venv with Isaac's
Python. It does not install into Isaac, Alex or the 6.0B overlay.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/setup_point2pose.py
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-smoke \
  --output outputs/b1/perception/NEW_P2P_SMOKE
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-live-smoke \
  --output outputs/b1/perception/NEW_P2P_CONCURRENT
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-replay \
  --output outputs/b1/perception/NEW_P2P_REPLAY
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-offline \
  --output outputs/b1/perception/NEW_P2P_OFFLINE
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-live \
  --output outputs/b1/perception/NEW_P2P_LIVE
```

Live diagnostics launch only one Isaac process at a time. Run them with other
Isaac sessions closed. `point2pose-live --case camera|panel|combined|visibility
--asset PILOT_ID` runs one fresh process. Visibility uses labeled RGB-D input
faults; it does not qualify physical occluders or loaded interaction. Replay
reads all four authorized pilot recordings chronologically at stride three,
retains complete input denominators and separates raw capture accuracy from
fresh useful outputs. The same provider/worker serves replay and live.
`point2pose-offline` instead waits for every native result at the original 60 Hz
capture cadence, including the last frame. It runs four complete pilot episodes
and two extra fresh initialization attempts per episode through one second after
the seed. Native and integration coverage, errors, losses and recoveries remain
separate. Initialization mask rejection is latched per candidate while native
diagnosis continues; operational initialization stays strict. Capture-time truth
is evaluator-only. Startup/latency never apply the 150 ms deadline to this mode.
`frames.jsonl` preserves each observation incrementally, including unavailable
and unprocessed rows. A failed process is never silently restarted. Results are
relative to the immutable seed; zero seed error does not validate absolute
initialization accuracy or object ownership. No qualification/contact flag changes.
Models load before acquisition and report startup separately. Once the synchronized
automatic seed is ready, its stateless 6.0B worker is released to free GPU memory;
an episode reset recreates both workers and clears all temporal state. Frame gaps
invalidate pending semantic requests without repeatedly loading the models.

The workstation worker uses Python 3.12, PyTorch 2.4 CUDA 12.1, NumPy 2.1.3,
Open3D 0.19 and GTSAM 4.3a0; shared Isaac PyTorch/NumPy remain unchanged. The native
SAM2 extension and PyCUDA require a CUDA toolkit and compatible host compiler.
On this host CUDA 12.2 uses GCC 11; GCC 13 was rejected by nvcc. Set `CUDA_HOME`
and `CXX` for another supported toolkit. `sources.json`, `runtime-packages.txt`
and each process's `runtime.json` record local source/package/configuration state.
Missing CUDA for models or TSDF is an explicit failure, never a CPU fallback.

The TSDF adapter derives its extent/radial bound from filtered measured keyframe
geometry and truncation/error padding, keeps 5 mm voxels, rebuilds expanded
volumes through official fusion and enforces available device/host memory.
The pinned CUDA kernel has a one-past-end index guard; the installer applies
`>` to `>=` and preserves the original source. Installer and worker startup also
apply the tracked corrections in `point2pose_patches.py` before native imports:
f2m marks missing/insufficient inlier support or invalid residuals as lost, while
still allowing same-ID registration to recover; an accepted SDF pose is returned
with its residuals/inliers. The unchanged final support gate can fall back to the
pre-SDF cluster pose, recomputing statistics for that pose. Exact zero residual
with sufficient measured support is valid. Pinned-source mismatches fail explicitly;
pre-fix files remain alongside the ignored sources as `.py.before-tracking-fixes`.
`runtime.json` declares the two `native_fixes`; model weights and numeric gates
are unchanged. Calibrated depth limits replace
small-object defaults in all lifting/crop calls. Equivalent dense crops are
cached only within one frame and unnecessary neighborhood gathering is skipped
only when it cannot affect official lifting results. SAM2 retains the authors'
positive prompts. Current pilot diagnostics fail useful availability at the
unchanged 150 ms limit. The prototype remains unqualified; results and limitations
are in the canonical perception findings, not implied by successful setup.

## Unified SAM3 video diagnostic

The same official source/NumPy overlay supports `sam3` and `sam3.1` causal RGB
workers. Keep the SAM3 tokenizer and the selected checkpoint in the directories
above; no runtime download is performed. SAM3.1 uses the multiplex builder with
one-frame grounding, the official PyTorch attention path, no compilation and
unchanged native thresholds. Only arrived RGB/time packets cross worker IPC.

The opt-in `point2pose_offline.offline_episode` argument
`sam3_frontend={"version": "sam3.1", "prompt": "door surface"}` uses the same
frontend for initialization and every subsequent mask. P2P disables its separate
SAM2 segmenter and receives each synchronized mask unchanged; DINOv3 is not loaded.
SAM3.1 additionally accepts `bounded_memory=True` for strictly forward causal
inference. It keeps native image/mask memory and object pointers that future
attention can select, while releasing expired output/input payloads. It does not
support reverse propagation or editing old frames; baseline defaults remain.
The adopted comparison passes `use_key_frame_graph=False`,
`allow_partial_reference_batch=True`, `refit_seed_rollback=False`, and
`performance_controls={"query_chunk_size": 0}` explicitly. Other native parameters,
TAPIR, SuperPoint, measured RGB-D, registration and TSDF are retained. This option
is diagnostic only; existing scan/live/replay defaults and policies are unchanged.
Both original SAM3 openings complete but regress common pose quality/support;
SAM3.1 stops at the preserved TSDF memory guard. Neither replaces the adopted
reference. See the [bounded pilots and original-target evaluation](../../knowledge/wiki/experiments/p2p-sam3-unified-frontend.md).
