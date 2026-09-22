#!/usr/bin/env python
"""GPU kinematic candidate screening; never a physics-qualified setup."""

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT

from alexdoor_xas.assets.synthetic_door import CASES, rectangles_overlap
from alexdoor_xas.kinematics.purdue_chain import PurdueChain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--height", type=float, default=1.2)
parser.add_argument("--fraction", type=float, default=0.9)
parser.add_argument("--device", default="cuda:0")
parser.add_argument(
    "--center", type=float, nargs=3, help="Refine floor X/Y/yaw-degrees around this center"
)
args = parser.parse_args()
if not args.device.startswith("cuda") or not torch.cuda.is_available():
    raise RuntimeError("Kinematic screening requires the available GPU")
urdf = (
    REPOSITORY_ROOT
    / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
)
chain = PurdueChain(urdf, args.device)
if args.center is None:
    axes = (np.arange(-0.5, 0.501, 0.1), np.arange(-0.6, 0.801, 0.1), np.arange(-180, 181, 15))
else:
    x, y, yaw_center = args.center
    axes = (
        x + np.arange(-0.10, 0.101, 0.025),
        y + np.arange(-0.15, 0.151, 0.025),
        yaw_center + np.arange(-30, 31, 5),
    )
floors = np.array(list(itertools.product(axes[0], axes[1], np.deg2rad(axes[2]))))
measurements = yaml.safe_load((REPOSITORY_ROOT / "measurements.yaml").read_text())
root_height = measurements["robot"]["right_shoulder_y_world_z_m"]["value"] - float(
    ET.parse(urdf).getroot().find("joint[@name='RIGHT_SHOULDER_Y']/origin").get("xyz").split()[2]
)
# Exact measured lower pedestal box in the floor plane; reject its swept collisions.
base_size = measurements["pedestal"]["parts"]["lower_base"]["size_m"]
pedestal_local = np.array(
    [
        [x, y]
        for x in (-base_size["x"] / 2, base_size["x"] / 2)
        for y in (-base_size["y"] / 2, base_size["y"] / 2)
    ]
)
pedestals = []
for x, y, yaw_value in floors:
    c, ss = np.cos(yaw_value), np.sin(yaw_value)
    pedestals.append(pedestal_local @ np.array([[c, -ss], [ss, c]]).T + [x, y])
base = np.c_[floors[:, :2], np.full(len(floors), root_height)]
yaw = torch.tensor(floors[:, 2], device=args.device, dtype=torch.float32).repeat_interleave(4)
rotation = torch.zeros((len(yaw), 3, 3), device=args.device)
rotation[:, 0, 0] = rotation[:, 1, 1] = yaw.cos()
rotation[:, 1, 0], rotation[:, 0, 1], rotation[:, 2, 2] = yaw.sin(), -yaw.sin(), 1
base = chain.array(base).repeat_interleave(4, 0)
q = chain.array([-0.6, -0.6, 0.2, -1.2, 0, 0, 0]).expand(len(yaw), 7).clone()
frame_clear = []
for pedestal in pedestals:
    for d in CASES:
        jambs = [
            np.array(
                [
                    [x, y + side * (d.width / 2 + 0.045)]
                    for x in (-0.06, 0.06)
                    for y in (-0.04, 0.04)
                ]
            )
            for side in (-1, 1)
        ]
        frame_clear.append(not any(rectangles_overlap(pedestal, jamb) for jamb in jambs))
passed = chain.array(frame_clear).bool()
maximum = torch.full((len(yaw),), -1.0, device=args.device)
margin = torch.ones(len(yaw), device=args.device)
solutions = {}
for degrees in [-1, *range(0, 151, 5)]:
    poses = [
        d.contact_pose(
            np.deg2rad(max(degrees, 0)), args.fraction, args.height, -0.03 if degrees < 0 else 0.0
        )
        for d in CASES
    ]
    positions = chain.array(np.array([p[0] for p in poses])).repeat(len(floors), 1)
    rotations = chain.array(np.array([p[1] for p in poses])).repeat(len(floors), 1, 1)
    target_p = (rotation.transpose(-1, -2) @ (positions - base).unsqueeze(-1)).squeeze(-1)
    target_r = rotation.transpose(-1, -2) @ rotations
    if degrees == -1:
        best = torch.full((len(q),), float("inf"), device=args.device)
        selected = q.clone()
        generator = torch.Generator(device=args.device).manual_seed(4101)
        for seed in range(8):
            initial = (
                q
                if seed == 0
                else chain.limits[:, 0]
                + (0.15 + 0.7 * torch.rand(q.shape, device=args.device, generator=generator))
                * (chain.limits[:, 1] - chain.limits[:, 0])
            )
            trial, pe, re = chain.solve(target_p, target_r, initial, iterations=150)
            joint_margin = (
                torch.minimum(trial - chain.limits[:, 0], chain.limits[:, 1] - trial).min(-1).values
            )
            converged = (pe < 0.001) & (re < 0.01)
            score = torch.where(converged, -1.0 - joint_margin, pe + 0.1 * re)
            better = score < best
            selected[better] = trial[better]
            best = torch.minimum(best, score)
        q = selected
    q, pe, re = chain.solve(target_p, target_r, q, iterations=40)
    valid = (pe < 0.01) & (re < np.deg2rad(5))
    clear = chain.array(
        [
            not rectangles_overlap(pedestal, d.panel_rectangle(np.deg2rad(degrees)))
            for pedestal in pedestals
            for d in CASES
        ]
    ).bool()
    passed &= valid & clear
    maximum[passed] = degrees
    limits = chain.limits
    qm = torch.minimum(q - limits[:, 0], limits[:, 1] - q) / (limits[:, 1] - limits[:, 0])
    margin = torch.where(passed, torch.minimum(margin, qm.min(-1).values), margin)
    if degrees == -1:
        solutions = q.reshape(len(floors), 4, 7).cpu().tolist()
    print(degrees, "all-four remaining", int(passed.reshape(-1, 4).all(-1).sum()), flush=True)
    if not bool(passed.reshape(-1, 4).all(-1).any()):
        break
maximum = maximum.reshape(-1, 4).cpu().numpy()
margin = margin.reshape(-1, 4).cpu().numpy()
order = sorted(range(len(floors)), key=lambda i: (-maximum[i].min(), -margin[i].min(), i))
rows = [
    dict(
        floor_pose=floors[i].tolist(),
        angles_deg=maximum[i].tolist(),
        minimum_deg=float(maximum[i].min()),
        joint_margin=float(margin[i].min()),
        contact_joints=solutions[i],
    )
    for i in order[:40]
]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        dict(
            scope="kinematic_screen_only",
            height=args.height,
            fraction=args.fraction,
            root_height=root_height,
            grid_count=len(floors),
            floor_domain=[[float(a[0]), float(a[-1])] for a in axes],
            resolution=[float(a[1] - a[0]) for a in axes],
            candidates=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(rows[:3]), flush=True)
