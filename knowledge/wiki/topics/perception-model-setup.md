# Selected perception resources

The maintained scan, smoke, live/replay and serial evaluator use the common
[`configs/point2pose_selected.json`](../../../configs/point2pose_selected.json).
They share one persistent causal SAM3 worker and optional Point2Pose worker;
[[shared-door-perception|Shared Door Perception]] owns lifecycle, synchronization,
geometry and qualification limits. All model paths are local and ignored.

## SAM3 worker

Use native `models/perception/sam3/sam3.pt` from `facebook/sam3`, weight revision
`3c879f39826c281e95690f02c7821c4de09afae7`, and its
`bpe_simple_vocab_16e6.txt.gz`. Official source revision is
`2345a4ad109ac29c569da749c91d84f10dc08c40`, owned by `sam3.SAM3_REVISION`.
The worker overlay isolates NumPy 1.x from Isaac's ABI. Setup remains:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m venv --without-pip --system-site-packages models/perception/runtime/venv
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install --no-deps \
  --target models/perception/runtime/venv/lib/python3.12/site-packages \
  -r knowledge/wiki/topics/perception-model-setup/runtime-requirements.txt
```

Retain the pinned source archive under
`runtime/sam3-2345a4ad109ac29c569da749c91d84f10dc08c40` and install it into the same
target with `--no-deps --no-build-isolation`, using Isaac Python. Never install the
upstream dependency bundle into Isaac. Weights stay frozen on CUDA; startup
checks availability and parameter placement. Compilation/FlashAttention3 are off,
with official PyTorch attention. Missing local resources fail without downloads
or CPU fallback. SAM3's internal tracker is maintained in full.

GroundingDINO and the box-prompt static worker are retired from maintained
consumers. DINOv3 remains retired. Their historical download/license metadata and
payload relocation records remain in `outputs/evidence/b1/perception/cleanup.json`.
This change preserves existing local weights, recordings and evidence. Historical
launchers use their recorded Git revision; two operative frontends are not maintained.

## Point2Pose worker

The native source remains `tzuyuan/point-to-pose` at
`51856226610df75e5c06e8de545bd27f7c4ba99c`. Setup retains BootsTAPIR, SuperPoint,
registration, GTSAM/keyframe support and CUDA TSDF:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/setup_point2pose.py
```

The [isolated requirements](perception-model-setup/point2pose-requirements.txt)
never install into Isaac, Alex or the SAM3 overlay. The workstation's P2P runtime
uses Python 3.12, PyTorch 2.4/CUDA 12.1, NumPy 2.1.3, Open3D 0.19 and GTSAM 4.3a0.
PyCUDA needs a compatible CUDA toolkit/compiler; this host uses CUDA 12.2/GCC 11.
Set `CUDA_HOME`/`CXX` for another supported toolkit. Missing model/fusion CUDA fails
explicitly. Setup no longer downloads or installs the separate SAM2 source,
extension or checkpoint. Existing historical SAM2 payloads remain untouched.

Startup and setup apply tracked source patches before native imports, preserving
original backups and refusing unexpected source. The pinned segmenter package's
eager SAM2 import is removed; BF16 autocast and Ampere-or-newer TF32 are enabled
explicitly before model construction and through native requests. This preserves
the numerical context formerly supplied by that import. External masks, all-query
TAPIR (`query_chunk_size=0`) and renewal-only filtering do not require
`diagnostic_only`. Initialization remains strict; registration tracing does not
weaken it. Positive query chunking still requires explicit diagnosis. Sources,
installed packages and effective runtime settings remain in local runtime records.

## Supported diagnostics

Choose a fresh output directory. No command admits contact, executes a learned
policy, trains a model or establishes release qualification.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py smoke --output outputs/generated/b1/perception/NEW_SAM3_SMOKE
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py diagnose-scan --output outputs/generated/b1/perception/NEW_SCAN
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-smoke --output outputs/generated/b1/perception/NEW_P2P_SMOKE
scripts/isaac-run.sh scripts/perception.py point2pose-live-smoke --output outputs/generated/b1/perception/NEW_CONCURRENT
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-replay --output outputs/generated/b1/perception/NEW_REPLAY
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-offline --output outputs/generated/b1/perception/NEW_OFFLINE
scripts/isaac-run.sh scripts/perception.py point2pose-live --case camera --asset animated-door-1-88abf40 --output outputs/generated/b1/perception/NEW_LIVE
```

`smoke` processes two distinct synchronized captures with the same SAM3 session.
P2P smoke uses 12 samples at stride three and deliberately starts at its bounded
seed capture; it does not test full inspection history. `--seed` verifies a cached
source acquisition; segmentation is generated by the shared SAM3 session, never
substituted from a cache. Missing/ambiguous/unsupported smoke seeds fail explicitly; continuous providers keep
SAM3 alive while waiting for support. Tracking commands accept `--selection path.json`.
The JSON requires `sam3_id` and `criterion`, plus a session-local `geometric_object`
or explicit `reference_pixel: [u, v]` and optional `after_s`. The same
`select_object`/`select` API applies the request. Choose against the actual displayed
view; alternatives and applied seed capture remain in the results. No first-root
selection is enabled by this option.
Static scan processes the prescribed inspection without P2P. Operational replay
uses stride three and measured availability/replaceable latest inputs. Serial
offline processes every original 60 Hz frame; its four-pilot/three-attempt campaign
is a deliberate command, not automatically resumed work. Live commands use fresh
Isaac processes, require other Isaac sessions closed and do not enumerate recordings.
Live observer cases follow the common inspection before their observer-only motion;
`camera|panel|combined|visibility` remain diagnostic cases, without contact.

## Continuous operational measurement

Use one authorized recording and an explicit observed request; this does not start
training, manipulation or the four-recording/three-attempt serial campaign.

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p scripts/perception.py point2pose-operational --episode datasets/recordings/b1/perception/engineering-v2/animated-door-1-88abf40/light/episode.hdf5 --mode wall-clock --selection outputs/evidence/b1/perception/sam3-operational-20261008/selection.json --output outputs/generated/b1/perception/NEW_CONTINUOUS
scripts/isaac-run.sh scripts/perception.py point2pose-live --case stationary --asset animated-door-1-88abf40 --duration-s 65 --selection outputs/evidence/b1/perception/sam3-operational-20261008/selection.json --output outputs/generated/b1/perception/NEW_STATIONARY_LIVE
```

The example pixel is for its displayed pilot view, not a general selection rule.
`--mode replay` instead uses measured service occupancy on the original acquisition
clock; it does not measure actual wall-clock concurrent ingress. Both retain every
input/publication and skipped ID. The live stationary case covers inspection,
initialization and a sustained tracking interval with rendering active. Other live
observer cases retain their original bounded motion/fault protocol. Post-run truth
scoring stays separate from provider service time. See
[[../experiments/p2p-operational-selection-and-light-peaks|operational results and failures]].

Saved integrated-light diagnosis is read-only and needs no model execution:

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p outputs/evidence/b1/perception/sam3-operational-20261008/diagnose_light.py
```

That evidence-owned script creates a new `light-diagnosis` directory and refuses
an existing destination. Its recorded helpers read the retained frames, traces,
mask stream, prepared evaluator hulls and physical targets. Reuse the completed
saved diagnosis for inspection; do not overwrite it or rerun a model campaign.

## Bounded historical comparison

The same serial coordinator can start after inspection to compare an unchanged
historical 31 s seed. This deliberately omits earlier SAM3 history and records
first-eligible geometric-zone selection while leaving ownership unqualified.

```python
from pathlib import Path
from alexdoor_xas.perception.diagnostics.offline import offline_episode
from alexdoor_xas.perception.provider import load_recipe

root = Path.cwd()
report = offline_episode(
    root / "datasets/recordings/b1/perception/engineering-v2/animated-door-1-88abf40/light/episode.hdf5",
    load_recipe(root / "configs/perception_geometry.json", root),
    root / "outputs/generated/b1/perception/NEW_PREFIX",
    root / "models/perception",
    capture_window_s=(31.0, 31.0 + 32 / 60),
    registration_diagnostics=True,
)
assert report["complete"], report["failure"]
```

A window starting before 25 s retains inspection and initializes only after its
end. Full serial evaluation, operational timing and historical seed comparison
remain separate. Selected historical results, residual failures and retired
variants stay in [[../experiments/p2p-sam3-selected-development|selected development]]
and [[../experiments/p2p-selected-residuals|residual diagnosis]]. SAM3.1, SAM2 Small,
periodic selective registration, rollback and rejected TAPIR/SVD variants remain
retired; historical reconstruction uses its recorded revision and weights.
