#!/usr/bin/env python
"""GPU commissioning gates for Subphase 4.0; synthetic fixtures, never corpus data."""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-cameras", action="store_true")
parser.add_argument("--gate", choices=("all", "contacts", "rgbd"), default="all")
parser.add_argument(
    "--output", type=Path, default=Path.home() / ".cache/alexdoor-xas/verification/purdue"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_cameras
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import apply_delta_pose, compute_pose_error  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from alexdoor_xas.action.frames import ObjectFrame, quat_to_rot_matrix  # noqa: E402
from alexdoor_xas.assets.purdue import PUSH_PARENT, origin_matrix  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv, tensor  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg  # noqa: E402

args.output.mkdir(parents=True, exist_ok=True)
report = {
    "passed": False,
    "gates": {},
    "limits": "Synthetic commissioning; no reachability or hardware claim.",
}


traces = []


def pose_errors(env, goal):
    pe, re = compute_pose_error(*env.tool_pose(), *goal, rot_error_type="axis_angle")
    return float(pe.norm()), float(re.norm())


def hold(env, goal, steps=180, require_clear=True):
    errors = []
    for _ in range(steps):
        env.command_pose(*goal)
        env.step(torch.zeros((1, 6), device=env.device))
        errors.append(pose_errors(env, goal))
        q = tensor(env.robot.data.joint_pos)[:, env.arm_ids]
        traces.append(
            [
                env._time_s,
                *errors[-1],
                float((env.targets[:, env.arm_ids] - q).abs().max()),
                float(env.clamp_excess.max()),
            ]
        )
        if require_clear:
            assert not any(r["forbidden"] for r in env.contact_history), env.contacts.last
    tail = np.array(errors[-30:])
    result = {
        "position_error_m": float(tail[:, 0].max()),
        "rotation_error_deg": float(np.rad2deg(tail[:, 1].max())),
    }
    print("HOLD", result, flush=True)
    assert result["position_error_m"] <= 0.01 and result["rotation_error_deg"] <= 5, result
    return result


def move_fixture(env, name, position, quaternion=(0, 0, 0, 1)):
    obj = env.scene.rigid_objects[name]
    pose = torch.tensor([list(position) + list(quaternion)], device=env.device, dtype=torch.float32)
    obj.write_root_pose_to_sim_index(root_pose=pose)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.device))


def clear_fixtures(env):
    for i, name in enumerate(("panel", "frame", "handle")):
        move_fixture(env, name, (2.0, float(i), 1.0))


def contact_trial(env, name, goal, half_depth, side=False):
    p, q = [v[0].detach().cpu().numpy() for v in goal]
    rotation = quat_to_rot_matrix(q)
    forward = rotation[:, 0]
    if side:
        y_max = max(points[:, 1].max() for points in env.push_geometry.finger_vertices.values())
        p = p - forward * 0.03 + rotation[:, 1] * y_max
        forward = rotation[:, 1]
        q = (Rotation.from_quat(q) * Rotation.from_euler("z", np.pi / 2)).as_quat()
    evidence = []
    for distance in np.linspace(0.005, -0.004, 19):
        move_fixture(env, name, p + forward * (half_depth + distance), q)
        for _ in range(5):
            env.command_pose(*goal)
            env.step(torch.zeros((1, 6), device=env.device))
            evidence.extend(
                c
                for r in env.contact_history
                for c in r["contacts"]
                if c["partner"].endswith("/" + name.capitalize())
            )
        loaded = {c["category"] for c in evidence if abs(c["force_n"]) > 0.05}
        if (name == "panel" and not side and {"negative", "positive"} <= loaded) or (
            (name != "panel" or side) and "forbidden" in loaded
        ):
            break
    assert any(abs(c["force_n"]) > 0.05 for c in evidence), "No loaded contact"
    for contact in evidence:
        assert contact["force_n"] >= -1e-6
        if name == "panel" and not side and contact["force_n"] > 0.05:
            assert np.dot(contact["normal"], forward) < -0.9, contact
        contact["fixture_normal_w"] = (-forward).tolist()
    clear_fixtures(env)
    env.reset()
    hold(env, goal, 180)
    return evidence


def check_jacobian(env):
    root = ET.parse(env.robot.cfg.spawn.asset_path).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    values = dict(
        zip(env.robot.joint_names, tensor(env.robot.data.joint_pos)[0].tolist(), strict=True)
    )
    root_pose = tensor(env.robot.data.root_link_pose_w)[0].cpu().numpy()
    world = np.eye(4)
    world[:3, :3] = quat_to_rot_matrix(root_pose[3:])
    world[:3, 3] = root_pose[:3]

    def forward(link, joints):
        if link not in parents:
            return world
        joint = parents[link]
        transform = origin_matrix(joint.find("origin"))
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * joints[joint.get("name")]).as_matrix()
            transform = transform @ turn
        return forward(joint.find("parent").get("link"), joints) @ transform

    def tool(joints):
        pose = forward(PUSH_PARENT, joints)
        return pose[:3, :3] @ env.push_geometry.translation + pose[:3, 3], pose[:3, :3]

    p0, r0 = tool(values)
    np.testing.assert_allclose(p0, env.tool_pose()[0][0].cpu().numpy(), atol=0.001)
    finite_difference = np.zeros((6, 7))
    for i, index in enumerate(env.arm_ids):
        moved = dict(values)
        moved[env.robot.joint_names[index]] += 1e-5
        p1, r1 = tool(moved)
        finite_difference[:3, i] = (p1 - p0) / 1e-5
        finite_difference[3:, i] = Rotation.from_matrix(r1 @ r0.T).as_rotvec() / 1e-5
    error = float(np.abs(finite_difference - env.tool_jacobian()[0].cpu().numpy()).max())
    assert error < 0.002, error
    return error


def canonical_optical_mount(env):
    stage = env.sim.stage
    head = next(
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(env.robot.cfg.prim_path))
        if p.GetName() == "HEAD_LINK"
    )
    camera = stage.GetPrimAtPath(env.camera.cfg.prim_path)
    zed = stage.GetPrimAtPath(env.camera.cfg.prim_path.split("/base_link/")[0])
    joint = UsdPhysics.FixedJoint(
        stage.GetPrimAtPath(str(zed.GetPath()) + "/AlexPurdueHeadZedXMiniFixedJoint")
    )
    assert str(joint.GetBody0Rel().GetTargets()[0]) == str(head.GetPath())
    mount = np.eye(4)
    mount[:3, 3] = joint.GetLocalPos0Attr().Get()
    q = joint.GetLocalRot0Attr().Get()
    mount[:3, :3] = quat_to_rot_matrix(np.r_[q.GetImaginary(), q.GetReal()])
    zed_authored = np.array(UsdGeom.Xformable(zed).ComputeLocalToWorldTransform(0)).T
    optic_authored = np.array(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)).T
    local = mount @ np.linalg.inv(zed_authored) @ optic_authored
    return local


def optical_error(env):
    local = optical_mount
    index = env.robot.body_names.index("HEAD_LINK")
    live = np.eye(4)
    live[:3, :3] = quat_to_rot_matrix(
        tensor(env.robot.data.body_link_quat_w)[0, index].cpu().numpy()
    )
    live[:3, 3] = tensor(env.robot.data.body_link_pos_w)[0, index].cpu().numpy()
    expected = live @ local
    expected[:3, :3] /= np.linalg.norm(expected[:3, :3], axis=0)
    actual = env.camera.data
    position_error = float(np.linalg.norm(expected[:3, 3] - tensor(actual.pos_w)[0].cpu().numpy()))
    rotation_error = Rotation.from_matrix(
        expected[:3, :3]
        @ np.diag([1, -1, -1])
        @ quat_to_rot_matrix(tensor(actual.quat_w_ros)[0].cpu().numpy()).T
    ).magnitude()
    assert position_error < 0.002 and rotation_error < np.deg2rad(0.5), (
        position_error,
        rotation_error,
    )
    return {"position_m": position_error, "rotation_deg": float(np.rad2deg(rotation_error))}


try:
    cfg = DoorPushPurdueEnvCfg()
    cfg.cameras = not args.no_cameras
    cfg.sim.device = args.device
    env = DoorPushPurdueEnv(cfg)
    optical_mount = canonical_optical_mount(env) if cfg.cameras else None
    zero = torch.zeros((1, 6), device=env.device)
    resets = []
    for _ in range(3 if args.gate == "all" else 1):
        env.reset()
        clear_fixtures(env)
        goal = tuple(x.clone() for x in env.tool_pose())
        metrics = hold(env, goal, 180 if args.gate == "all" else 60)
        q = tensor(env.robot.data.joint_pos)
        for side in ("left", "right"):
            leader = env.robot.joint_names.index(side + "_WSG32_JAW_OPENING")
            follower = env.robot.joint_names.index(side + "_WSG32_JAW_FOLLOWER")
            assert abs(float(q[0, leader])) < 0.001
            assert abs(float(q[0, leader] - q[0, follower])) < 0.001
        shoulder = env.robot.body_names.index("RIGHT_SHOULDER_Y_LINK")
        height = float(tensor(env.robot.data.body_link_pos_w)[0, shoulder, 2])
        assert abs(height - 1.266) < 0.002, height
        metrics["shoulder_height_m"] = height
        resets.append(metrics)
    report["gates"]["assembly_resets"] = resets
    base = tuple(x.clone() for x in env.tool_pose())
    if args.gate == "all":
        report["gates"]["jacobian_max_error"] = check_jacobian(env)
        report["gates"]["control"] = []
        for delta in (
            [0.02, 0, 0, 0, 0, 0],
            [0, 0.02, 0, 0, 0, 0],
            [0, 0, 0.02, 0, 0, 0],
            [0, 0, 0, 0.15, 0, 0],
            [0, 0, 0, 0, 0.15, 0],
            [0, 0, 0, 0, 0, 0.15],
            [0.01, -0.01, 0.01, 0.08, -0.08, 0.08],
        ):
            target = apply_delta_pose(*base, torch.tensor([delta], device=env.device))
            metrics = hold(env, target, 240)
            report["gates"]["control"].append({"delta": delta, **metrics})
            hold(env, base, 120)
        # A1 addresses every joint through the same order; no change to Gym's public shape.
        a1 = []
        for index in range(7):
            before = tensor(env.robot.data.joint_pos)[:, env.arm_ids].clone()
            target = before.clone()
            target[:, index] += 0.08
            for _ in range(180):
                env.step_a1(target - tensor(env.robot.data.joint_pos)[:, env.arm_ids])
            error = float((tensor(env.robot.data.joint_pos)[:, env.arm_ids] - target).abs().max())
            assert error < 0.03, (index, error)
            a1.append(error)
        report["gates"]["a1_errors_rad"] = a1
        hold(env, base, 240)
        frame = ObjectFrame(np.zeros(3), Rotation.from_euler("z", 0.7).as_matrix())
        env.step_a3(np.array([0.0, 0, 0, 0.02, 0.01, 0]), frame)
        assert float(env.pose_error[:, 3:].norm()) > 0.02
        hold(env, base, 60)
    if args.gate in ("all", "contacts"):
        contacts = {}
        for name, half_depth in (("panel", 0.025), ("frame", 0.04), ("handle", 0.04)):
            contacts[name] = contact_trial(env, name, base, half_depth)
            categories = {c["category"] for c in contacts[name]}
            print("CONTACT", name, categories, flush=True)
            if name == "panel":
                assert {"negative", "positive"} <= categories, contacts[name]
                assert "forbidden" not in categories, contacts[name]
            else:
                assert "forbidden" in categories, contacts[name]
        contacts["panel_side"] = contact_trial(env, "panel", base, 0.025, side=True)
        assert any(c["category"] == "forbidden" for c in contacts["panel_side"])
        move_fixture(env, "handle", (0.1265, 0, 0.5))
        pedestal_contacts = []
        for _ in range(15):
            env.command_pose(*base)
            env.step(zero)
            pedestal_contacts.extend(
                c
                for r in env.contact_history
                for c in r["contacts"]
                if "/Pedestal/" in c["actor"] and c["category"] == "forbidden"
            )
        assert pedestal_contacts, "Pedestal contact was not observed"
        contacts["pedestal"] = pedestal_contacts
        clear_fixtures(env)
        report["gates"]["contact"] = contacts
    if cfg.cameras and args.gate in ("all", "rgbd"):
        env.reset()
        clear_fixtures(env)
        goal = tuple(x.clone() for x in env.tool_pose())
        hold(env, goal, 60)
        camera = env.camera.data
        position = tensor(camera.pos_w)[0].cpu().numpy()
        rotation = quat_to_rot_matrix(tensor(camera.quat_w_ros)[0].cpu().numpy())
        panel_rotation = rotation[:, [2, 0, 1]]
        move_fixture(
            env, "panel", position + rotation[:, 2], Rotation.from_matrix(panel_rotation).as_quat()
        )
        hold(env, goal, 5)
        sample = env.capture.sample
        depth = sample.depth_m[0, ..., 0].cpu().numpy()
        rgb = sample.rgb[0].cpu().numpy()
        Image.fromarray(rgb).save(args.output / "left_rgb.png")
        patch = depth[295:305, 475:485]
        depth_error = float(np.median(np.abs(patch - 0.975)))
        color = rgb.astype(float)
        rgb_panel = (color[..., 0] > 1.3 * color[..., 2]) & (color[..., 1] > 1.2 * color[..., 2])
        rgb_panel &= color[..., 0] > 100
        depth_panel = np.abs(depth - 0.975) < 0.01
        alignment_iou = float((rgb_panel & depth_panel).sum() / (rgb_panel | depth_panel).sum())
        assert alignment_iou > 0.90, alignment_iou
        optics_before = optical_error(env)
        assert depth_error < 0.01, depth_error
        assert float(sample.valid_depth.float().mean()) > 0.1
        np.savez_compressed(
            args.output / "rgbd_sample.npz",
            depth=depth,
            valid=sample.valid_depth.cpu().numpy(),
            joints=sample.joint_position.cpu().numpy(),
        )
        old_episode, old_time = sample.episode, sample.time_s
        env.set_neck_target([0.1, 0.0])
        hold(env, goal, 120)
        moved = env.capture.sample
        optics_after = optical_error(env)
        assert not torch.equal(sample.rgb, moved.rgb), "neck motion did not refresh the image"
        assert float(moved.joint_position[0, 7]) > 0.05
        assert moved.time_s > old_time
        Image.fromarray(moved.rgb[0].cpu().numpy()).save(args.output / "left_rgb_neck.png")
        env.reset()
        assert env.capture.sample.episode == old_episode + 1
        assert env.capture.sample.time_s == 0.0
        optics_reset = optical_error(env)
        assert not torch.equal(env.capture.sample.rgb, moved.rgb)
        Image.fromarray(env.capture.sample.rgb[0].cpu().numpy()).save(
            args.output / "left_rgb_reset.png"
        )
        report["gates"]["rgbd"] = {
            "alignment_iou": alignment_iou,
            "optical_before": optics_before,
            "optical_after": optics_after,
            "optical_reset": optics_reset,
            "depth_error_m": depth_error,
            "size": list(rgb.shape),
            "reset_fresh": True,
            "neck_motion_observed": True,
        }
    report["passed"] = args.gate != "all" or not args.no_cameras
    report["scope"] = args.gate
    report["subphase_complete"] = report["passed"] and args.gate == "all"
except Exception as error:
    import traceback

    traceback.print_exc()
    report["error"] = repr(error)
finally:
    np.savez_compressed(
        args.output / "control_traces.npz",
        values=np.array(traces),
        columns=np.array(
            [
                "time_s",
                "position_error_m",
                "rotation_error_rad",
                "joint_tracking_error_rad",
                "target_saturation_rad",
            ]
        ),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", args.output / "report.json", flush=True)
    os._exit(0 if report["passed"] else 1)
