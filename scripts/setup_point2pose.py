#!/usr/bin/env python
"""Install the pinned Point2Pose worker locally, never into Isaac's environment."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "upstream": ("tzuyuan/point-to-pose", "51856226610df75e5c06e8de545bd27f7c4ba99c"),
    "sam2": ("Gy920/segment-anything-2-real-time", "e3623cda3f748df999f76db663f08da7a884a820"),
    "tapnet": ("google-deepmind/tapnet", "730cda1c730877cfedbe01bf87fb1cadb78a565d"),
    "lightglue": ("cvg/LightGlue", "eb42fee2d71449efb0aa5c10549752b5d75384d8"),
}
WEIGHTS = {
    "causal_bootstapir_checkpoint.pt": "https://storage.googleapis.com/dm-tapnet/causal_bootstapir_checkpoint.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "superpoint_v1.pth": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
}


def patch_tsdf_index(root):
    source = root / "upstream/point2pose/modules/reconstruction/sdf_builder.py"
    text = source.read_text()
    old = "if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)"
    new = "if (voxel_idx >= vol_dim_x*vol_dim_y*vol_dim_z)"
    if old in text:
        # Zero-based index == voxel count is out of bounds in the pinned kernel.
        source.with_suffix(".py.original").write_text(text)
        source.write_text(text.replace(old, new))
    elif new not in text:
        raise ValueError("Unexpected pinned TSDF kernel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models/perception/point2pose")
    args = parser.parse_args()
    local = args.root.resolve()
    local.mkdir(parents=True, exist_ok=True)
    manifest = local / "sources.json"
    if manifest.exists():
        installed = json.loads(manifest.read_text())
        for name, (repo, revision) in SOURCES.items():
            if installed.get(name) != dict(repository=repo, revision=revision):
                raise ValueError(f"Existing {name} source revision differs; use a fresh --root")
    for name, (repo, revision) in SOURCES.items():
        if (local / name).exists():
            continue
        archive = local / f"{name}.tar.gz"
        urllib.request.urlretrieve(f"https://codeload.github.com/{repo}/tar.gz/{revision}", archive)
        with tarfile.open(archive) as stream:
            prefix = stream.getmembers()[0].name.split("/")[0]
            stream.extractall(local, filter="data")
        (local / prefix).rename(local / name)
    patch_tsdf_index(local)
    from alexdoor_xas.perception.point2pose_patches import patch_tracking

    patch_tracking(local)
    (local / "sources.json").write_text(
        json.dumps(
            {
                name: dict(repository=repo, revision=revision)
                for name, (repo, revision) in SOURCES.items()
            },
            indent=2,
        )
        + "\n"
    )
    python = local / "runtime/bin/python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(local / "runtime")], check=True)
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    env = dict(
        os.environ,
        CUDA_HOME=cuda,
        CUDA_ROOT=cuda,
        PATH=f"{cuda}/bin:{os.environ.get('PATH', '')}",
        CPLUS_INCLUDE_PATH=f"{cuda}/include",
        LIBRARY_PATH=f"{cuda}/lib64",
    )
    # Isaac's launcher exports shared site-packages; the worker must use its own ABI.
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    compiler = os.environ.get("CXX") or shutil.which("g++-11")
    if compiler:
        env.update(CXX=compiler, CUDAHOSTCXX=compiler, NVCC_PREPEND_FLAGS=f"-ccbin {compiler}")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "models/perception/point2pose-requirements.txt"),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(local / "sam2"),
            str(local / "lightglue"),
            str(local / "tapnet"),
        ],
        check=True,
        env=env,
    )
    checkpoints = local / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for name, url in WEIGHTS.items():
        target = checkpoints / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".download")
            urllib.request.urlretrieve(url, temporary)
            temporary.rename(target)
    cache = local / "torch/hub/checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    link = cache / "superpoint_v1.pth"
    if not link.exists():
        link.symlink_to(checkpoints / link.name)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (local / "runtime-packages.txt").write_text(frozen)


if __name__ == "__main__":
    main()
