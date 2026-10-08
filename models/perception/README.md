# Selected perception resources

Local weights are preserved for GroundingDINO + SAM 3 with explicit RGB-D/multiview
geometry for the maintained 6.0B static scan.
Neither the static scan nor the optional Point2Pose prototype is a qualified perception release.

| Directory | Upstream model | Weight revision |
|---|---|---|
| `grounding-dino/` | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| `sam3/` | `facebook/sam3`, native `sam3.pt` | `3c879f39826c281e95690f02c7821c4de09afae7` |

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

DINOv3 extraction and its unused surface descriptors were removed: maintained
association and contact geometry consume RGB-D support, not learned descriptors.
Its historical revision (`114c1379950215c8b35dfcd4e90a5c251dde0d32`), configuration,
license and download metadata remain in `evidence/cleanup.json`. The retired
SAM3.1 metadata directory was also removed after preserving its contents there.
GroundingDINO remains required for static box prompts and automatic Point2Pose seeds.

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

Smoke both image models with a fresh output:

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

Choose commands deliberately: `point2pose-smoke` uses 12 samples; live diagnostics
launch one fresh Isaac process at a time and require other Isaac sessions closed.
`point2pose-live --case camera|panel|combined|visibility --asset PILOT_ID` limits a
live check to one case. Operational replay samples four recordings at stride three;
`point2pose-offline` processes all original 60 Hz frames plus initialization attempts.
The historical offline campaign is stopped; the commands above document entry
points, not instructions to repeat it. Live commands require no recording directory;
a failed child or missing report produces a nonzero exit status independently of
quality scores. Runtime timing, reset, ground-truth separation
and failure semantics are canonical in
[Shared Door Perception](../../knowledge/wiki/topics/shared-door-perception.md).

The workstation worker uses Python 3.12, PyTorch 2.4 CUDA 12.1, NumPy 2.1.3,
Open3D 0.19 and GTSAM 4.3a0; shared Isaac PyTorch/NumPy remain unchanged. The native
SAM2 extension and PyCUDA require a CUDA toolkit and compatible host compiler.
On this host CUDA 12.2 uses GCC 11; GCC 13 was rejected by nvcc. Set `CUDA_HOME`
and `CXX` for another supported toolkit. `sources.json`, `runtime-packages.txt`
and each process's `runtime.json` record local source/package/configuration state.
Missing CUDA for models or TSDF is an explicit failure, never a CPU fallback.

Installer and worker startup apply `perception/point2pose/patches.py` before native imports;
setup also repairs the pinned TSDF one-past-end index guard (`>` to `>=`). Tracked
patches reproduce current loss/SDF publication, sampling, renewal and vectorized
Jacobian behavior from original sources. Known prior patches migrate idempotently;
unexpected sources fail explicitly. Original `.py.before-tracking-fixes` backups
remain, with `.py.before-retired-variants` preserving the rollback-era installation.
`runtime.json` declares `native_fixes`; setup does not change weights/numeric gates.
Behavioral details, observed TSDF scaling/memory guards and unqualified limits
belong in the perception topic and experiments.

## Selected SAM3 video diagnostic

The selected experimental settings are tracked in
[`configs/point2pose_selected.json`](../../configs/point2pose_selected.json).
They reproduce the saved selected recipe through the existing serial evaluator;
loading this file is explicit and does not change operational defaults. Run from
the repository root with the supported Isaac Python, choosing a fresh output:

```python
import json
from pathlib import Path

from alexdoor_xas.perception.diagnostics.offline import offline_episode
from alexdoor_xas.perception.provider import load_recipe

root = Path.cwd()
selected = json.loads((root / "configs/point2pose_selected.json").read_text())
condition = "light"  # The other preserved condition is "nominal".
report = offline_episode(
    root / f"datasets/b1/perception/engineering-v2/animated-door-1-88abf40/{condition}/episode.hdf5",
    load_recipe(root / "configs/perception_geometry.json", root),
    root / f"outputs/b1/perception/NEW_SELECTED_{condition}",
    root / "models/perception",
    capture_window_s=(31.0, 78.61666666666666),
    registration_diagnostics=True,
    sam3_frontend=selected["sam3_frontend"],
    **selected["point2pose_controls"],
)
assert report["complete"], report["failure"]
```

This is an invocation recipe, not an instruction to repeat a completed experiment.
A bounded 33-capture smoke uses `(31.0, 31.0 + 32 / 60)` instead. Remaining native
settings come from the pinned `eccv_final.yaml` and tracked adapters: TAPIR
full-frame 480/four iterations, SuperPoint, 120 active references, five inliers/
4 mm and vectorized SDF Jacobian. The renewal depth gate is 10 mm/radius 2; seed
selection is unchanged. Original times and evaluator-only ground truth are retained.

The video worker supports SAM3 using the same source/NumPy overlay as the static
image worker. Only newly arrived RGB/time packets enter SAM3; its synchronized
mask enters P2P without a separate SAM2 call. Missing original IDs
produce empty masks, never replacement identities. Bounded history preserves
score-selected useful memories, including old high-quality frames; it requires
reconditioning OFF. The unbounded SAM3 reference remains available explicitly.

SAM3.1, selected-only registration, own-seed refit rollback, SAM2 Small, TAPIR
crop/reduced resolution/iterations and simplified SVD are retired from the active
adapter. Their code is available at `57ad483`; historical launchers, results and
failures remain. SAM3.1 and SAM2 Small checkpoint payloads were removed after
consumer checks; revisions and SHA-256 digests remain in the local cleanup
inventory. SAM3.1 source revision is `daa63191845a41281374e725f4c9e51c7a824460`.
Reproducing retired experiments requires their recorded Git revision and weights.
`performance_controls` now accepts
only `query_chunk_size`; old rollback/selected-registration keywords accept only
false. Unsupported activation fails explicitly. Result and trace fields remain
compatible with saved-data evaluators.

The [selected development](../../knowledge/wiki/experiments/p2p-sam3-selected-development.md)
and [residual diagnosis](../../knowledge/wiki/experiments/p2p-selected-residuals.md)
retain the measured tradeoffs and open material-identity questions. This recipe
is experimental; it does not qualify 150 ms freshness, contact, A3/A4 or policies.
