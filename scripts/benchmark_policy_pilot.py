#!/usr/bin/env python
"""Bounded CUDA sensor-array-to-policy benchmark, random weights; no training/physics."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.act.policy import ActPolicy
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.diffusion.model import DiffusionTransformer
from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy
from alexdoor_xas.policies.encoding import (
    FIXED_UNITS_RECIPE,
    LEARNED_RECIPE,
    RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from alexdoor_xas.policies.observations import B1Observer


def benchmark(iterations=30, recipe=LEARNED_RECIPE):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    binding = pilot_binding(inspection, recipe=recipe)
    visual_recipe = LEARNED_RECIPE if recipe == LEARNED_RECIPE else None
    report = dict(
        device=torch.cuda.get_device_name(),
        obs_dim=binding.obs_dim,
        recipe=recipe,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        scope=(
            "random weights; sensor arrays through encoding, normalization and policy; "
            "no camera/render/geometry/physics"
        ),
        history_frames=3,
        history_period_s=0.1,
        chunk_ticks=12,
        replan_ticks=6,
        paths={},
    )

    def stats(width):
        return NormStats(np.zeros(width), np.ones(width), -np.ones(width), np.ones(width), 1)

    spaces = {
        k: v
        for k, v in ACTION_DIMS.items()
        if recipe != LEARNED_RECIPE or k in ("A1_joint_delta", "A2_ee_delta")
    }
    for space, width in spaces.items():
        norm = DatasetNormStats(
            stats(width), stats(binding.obs_dim), OBS_KEYS, ("synthetic",), space
        )
        for family in ("act", "diffusion"):
            torch.cuda.reset_peak_memory_stats()
            model = (
                ACTModel(
                    binding.obs_dim, width, ActModelCfg(chunk_size=12, visual_recipe=visual_recipe)
                )
                if family == "act"
                else DiffusionTransformer(
                    binding.obs_dim,
                    width,
                    DiffusionModelCfg(horizon=12, visual_recipe=visual_recipe),
                )
            ).cuda()
            policy = (
                ActPolicy(model, norm, device="cuda")
                if family == "act"
                else DiffusionPolicy(
                    model, norm, device="cuda", sampler="ddim", num_inference_steps=10
                )
            )
            observer = B1Observer(None, binding, encoder=RGBDEncoder(binding))
            sensor = dict(
                time_s=0,
                frame=0,
                rgb=torch.full((600, 960, 3), 120, dtype=torch.uint8, device="cuda"),
                depth_m=torch.ones((600, 960, 1), device="cuda"),
                valid_depth=torch.ones((600, 960, 1), dtype=torch.bool, device="cuda"),
                joint_position=np.zeros(9),
                joint_velocity=np.zeros(9),
                camera_world=np.eye(4),
                intrinsics=np.eye(3),
            )
            times = []
            for tick in range(iterations + 5):
                sensor.update(time_s=tick / 10, frame=tick)
                torch.cuda.synchronize()
                begin = perf_counter()
                observation = observer.update(sensor)
                encoded = perf_counter()
                result = policy.predict(observation.features)
                torch.cuda.synchronize()
                end = perf_counter()
                assert np.isfinite(result).all()
                if tick >= 5:
                    times.append(
                        [(encoded - begin) * 1000, (end - encoded) * 1000, (end - begin) * 1000]
                    )
            report["paths"][f"{family}/{space}"] = dict(
                parameters=sum(p.numel() for p in model.parameters()),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                timing_columns=["sensor_copy_encoder_ms", "normalize_policy_output_ms", "total_ms"],
                p50=np.median(times, axis=0).tolist(),
                p95=np.percentile(times, 95, axis=0).tolist(),
                maximum=np.max(times, axis=0).tolist(),
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--recipe", choices=(LEARNED_RECIPE, RECIPE, FIXED_UNITS_RECIPE), default=LEARNED_RECIPE
    )
    args = parser.parse_args()
    result = benchmark(args.iterations, recipe=args.recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
