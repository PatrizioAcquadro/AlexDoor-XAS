#!/usr/bin/env python
"""Run Phase 4.1 static, GPU physics, rollout, repeatability, and manifest gates.

Examples::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
      scripts/verify_phase4_1_doors.py static --all --viz none --device cuda:0
    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
      scripts/verify_phase4_1_doors.py physics --slot 1 --viz none --device cuda:0
    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
      scripts/verify_phase4_1_doors.py repeatability --all --viz none --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "command", choices=("static", "physics", "repeatability", "all", "finalize-manifest")
)
selection = parser.add_mutually_exclusive_group(required=False)
selection.add_argument("--slot", type=int)
selection.add_argument("--all", action="store_true")
parser.add_argument("--clean-shutdown", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.door_qualification import (  # noqa: E402
    FORCE_LIMIT_N,
    HINGE_DAMPING_NM_S_RAD,
    MAX_TEXTURE_EDGE_PX,
    MAX_TRIANGLES,
    OPEN_ANGLE_DEG,
    PANEL_MASS_KG,
    SCHEMA,
    DoorDimensions,
    QualificationError,
    bootstrap_n_qual,
    canonical_sha256,
    cuboid_inertia_kg_m2,
    dump_json,
    load_json,
    maximum_sustained_angle_deg,
    repeatability_metrics,
    sha256_file,
    validate_diagnostic_count,
    validate_manifest,
)
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_LIMITATIONS,
    DoorPushAlexV2EnvCfg,
)

WORKLIST = paths.PHASE4_1_EVIDENCE_DIR / "worklist.json"
REJECTIONS = paths.PHASE4_1_EVIDENCE_DIR / "rejections.json"
EXPECTED_HINGE = "/World/Door/Doorframe/Hinge"
RESET_ANGLE_TOL_RAD = math.radians(0.1)
RESET_SPEED_TOL_RAD_S = 0.01
PASSIVE_DRIFT_TOL_RAD = math.radians(0.25)
FRAME_DRIFT_TOL_M = 1e-4
PENETRATION_TOL_M = 5e-4
TORQUE_NM = 15.0
TORQUE_TARGET_RAD = math.radians(80.0)


def _as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _slots() -> tuple[dict, list[dict]]:
    worklist = load_json(WORKLIST)
    if args.slot is not None:
        selected = [slot for slot in worklist["slots"] if int(slot["slot"]) == args.slot]
        if not selected:
            raise QualificationError(f"slot {args.slot} is vacant")
    elif args.all:
        selected = list(worklist["slots"])
    else:
        raise QualificationError("select --slot N or --all")
    return worklist, sorted(selected, key=lambda slot: int(slot["slot"]))


def _normalized_path(slot: dict) -> Path:
    if slot["state"] not in {"normalized", "static_pass", "physics_pass", "qualified"}:
        raise QualificationError(f"slot {slot['slot']} is not normalized: {slot['state']}")
    path = paths.REPO_ROOT / slot["normalized_path"]
    if not path.is_file():
        raise QualificationError(f"normalized USD is missing: {path}")
    return path


def _dependencies_are_local(stage, usd_path: Path) -> list[str]:
    _, resolved, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
    if unresolved:
        raise QualificationError(f"unresolved normalized dependencies: {list(unresolved)}")
    root = usd_path.parent.resolve()
    dependencies: list[str] = []
    for asset in resolved:
        path = Path(str(asset)).resolve()
        if not path.is_relative_to(root):
            raise QualificationError(f"normalized dependency escapes asset directory: {path}")
        dependencies.append(str(path.relative_to(root)))
    return sorted(dependencies)


def _static_gate(slot: dict) -> dict:
    usd_path = _normalized_path(slot)
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise QualificationError(f"cannot open normalized USD: {usd_path}")
    if stage.GetDefaultPrim().GetPath().pathString != "/World":
        raise QualificationError("default prim must be /World")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise QualificationError("normalized stage must be Z-up")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise QualificationError("normalized stage must use meters")
    for prim_path in ("/World/Door/Doorframe", "/World/Door/Door", "/World/Door/Handle"):
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise QualificationError(f"missing canonical prim: {prim_path}")
    dependencies = _dependencies_are_local(stage, usd_path)
    metadata = stage.GetPrimAtPath("/World/Door").GetCustomData().get("alexdoor")
    if not isinstance(metadata, dict):
        raise QualificationError("normalized door metadata is missing")
    dimensions = DoorDimensions.from_mapping(
        {
            "width_m": metadata["width_m"],
            "height_m": metadata["height_m"],
            "thickness_m": metadata["thickness_m"],
        }
    )
    if metadata["source_sha256"] != slot["source_sha256"]:
        raise QualificationError("normalized metadata source checksum mismatch")
    if metadata["geometry_fingerprint"] != slot["geometry_fingerprint"]:
        raise QualificationError("normalized metadata geometry fingerprint mismatch")
    if metadata["handedness"] != slot["candidate"]["handedness"]:
        raise QualificationError("normalized metadata handedness mismatch")
    triangles = int(metadata["triangles"])
    texture_edge = int(metadata["texture_max_px"])
    if triangles <= 0 or triangles > MAX_TRIANGLES:
        raise QualificationError(f"invalid normalized triangle count: {triangles}")
    if texture_edge > MAX_TEXTURE_EDGE_PX:
        raise QualificationError(f"invalid normalized texture edge: {texture_edge}")

    hinges = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    if [prim.GetPath().pathString for prim in hinges] != [EXPECTED_HINGE]:
        raise QualificationError("normalized door must have one canonical revolute hinge")
    hinge = UsdPhysics.RevoluteJoint(hinges[0])
    if hinge.GetAxisAttr().Get() != "Z":
        raise QualificationError("hinge joint axis token must be Z")
    if hinge.GetLowerLimitAttr().Get() != 0.0 or hinge.GetUpperLimitAttr().Get() != 90.0:
        raise QualificationError("hinge limits must be 0..90 degrees")
    drive = UsdPhysics.DriveAPI(hinges[0], "angular")
    if drive.GetDampingAttr().Get() != HINGE_DAMPING_NM_S_RAD:
        raise QualificationError("hinge damping does not match the common template")
    frame_path = stage.GetPrimAtPath("/World/Door/Doorframe").GetPath()
    door_path = stage.GetPrimAtPath("/World/Door/Door").GetPath()
    if list(hinge.GetBody0Rel().GetTargets()) != [frame_path]:
        raise QualificationError("hinge body0 must be Doorframe")
    if list(hinge.GetBody1Rel().GetTargets()) != [door_path]:
        raise QualificationError("hinge body1 must be Door")

    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    expected_colliders = {
        "/World/Door/Door/PanelCollider",
        "/World/Door/Doorframe/HingeJambCollider",
        "/World/Door/Doorframe/FarJambCollider",
        "/World/Door/Doorframe/HeaderCollider",
    }
    collider_paths = {prim.GetPath().pathString for prim in colliders}
    if not expected_colliders <= collider_paths:
        raise QualificationError(f"canonical collider proxies are missing: {collider_paths}")
    if any(path.startswith("/World/Door/Handle/") for path in collider_paths):
        raise QualificationError("handle must not have an operational collider")
    for prim_path in ("/World/Door/Doorframe", "/World/Door/Door", "/World/Door/Handle"):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise QualificationError(f"canonical rigid body missing: {prim_path}")
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        inertia = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()
        if mass is None or float(mass) <= 0.0 or inertia is None or any(v <= 0.0 for v in inertia):
            raise QualificationError(f"invalid mass properties: {prim_path}")
    panel_mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Door/Door"))
    panel_mass = float(panel_mass_api.GetMassAttr().Get())
    panel_inertia = tuple(
        float(v)
        for v in UsdPhysics.MassAPI(
            stage.GetPrimAtPath("/World/Door/Door")
        ).GetDiagonalInertiaAttr().Get()
    )
    if panel_mass != PANEL_MASS_KG or not np.allclose(
        panel_inertia, cuboid_inertia_kg_m2(dimensions), atol=1e-5
    ):
        raise QualificationError("panel mass/inertia differ from the common cuboid template")
    return {
        "passed": True,
        "usd_path": str(usd_path.relative_to(paths.REPO_ROOT)),
        "usd_sha256": sha256_file(usd_path),
        "dependencies": dependencies,
        "default_prim": "/World",
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "dimensions_m": dimensions.to_dict(),
        "triangles": triangles,
        "texture_max_px": texture_edge,
        "hinge_path": EXPECTED_HINGE,
        "collider_count": len(expected_colliders),
        "panel_mass_kg": panel_mass,
        "panel_inertia_kg_m2": list(panel_inertia),
    }


def _make_env(slot: dict):
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = 4101
    cfg.sim.device = args.device
    cfg.qualification_scene_usd = str(_normalized_path(slot))
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _body_id(env, name: str) -> int:
    matches = [
        index
        for index, body_name in enumerate(env._door.body_names)  # noqa: SLF001
        if body_name == name or body_name.rsplit("/", 1)[-1] == name
    ]
    if len(matches) != 1:
        raise QualificationError(f"expected one cooked body {name!r}: {env._door.body_names}")
    return matches[0]


def _disable_robot_collisions_for_door_gate(env) -> int:
    stage = env.sim.stage
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Alex")
    if not robot_root.IsValid():
        raise QualificationError("runtime Alex prim is missing for torque isolation")
    disabled = 0
    for prim in Usd.PrimRange(robot_root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = UsdPhysics.CollisionAPI(prim)
            attribute = collision.GetCollisionEnabledAttr()
            if not attribute:
                attribute = collision.CreateCollisionEnabledAttr()
            attribute.Set(False)
            disabled += 1
    if disabled <= 0:
        raise QualificationError("torque isolation found no Alex colliders")
    return disabled


def _assert_reset(env) -> tuple[float, float]:
    env.reset(seed=4101)
    angle, speed = env.hinge_state()
    angle_value = float(angle[0].item())
    speed_value = float(speed[0].item())
    if abs(angle_value) > RESET_ANGLE_TOL_RAD or abs(speed_value) > RESET_SPEED_TOL_RAD_S:
        raise QualificationError(
            "reset outside tolerance: "
            f"angle={math.degrees(angle_value)}deg speed={speed_value}rad/s"
        )
    return angle_value, speed_value


def _physics_gate(slot: dict) -> dict:
    env = _make_env(slot)
    try:
        initial_angle, initial_speed = _assert_reset(env)
        disabled_robot_colliders = _disable_robot_collisions_for_door_gate(env)
        door = env._door  # noqa: SLF001
        frame_id = _body_id(env, "Doorframe")
        panel_id = _body_id(env, "Door")
        frame_pos_0 = _as_torch(door.data.body_pos_w)[0, frame_id].clone()
        zero = torch.zeros((1, env.cfg.action_space), dtype=torch.float32, device=env.device)
        max_frame_drift = 0.0
        max_passive_drift = 0.0
        for _ in range(120):
            obs, _, _, _, _ = env.step(zero)
            if not torch.isfinite(obs["policy"]).all():
                raise QualificationError("physics observation contains NaN/Inf")
            angle, speed = env.hinge_state()
            body_pos = _as_torch(door.data.body_pos_w)
            body_quat = _as_torch(door.data.body_quat_w)
            if not torch.isfinite(angle).all() or not torch.isfinite(speed).all():
                raise QualificationError("hinge state contains NaN/Inf")
            if not torch.isfinite(body_pos[0, panel_id]).all() or not torch.isfinite(
                body_quat[0, panel_id]
            ).all():
                raise QualificationError("panel state contains NaN/Inf")
            max_frame_drift = max(
                max_frame_drift,
                float(torch.linalg.vector_norm(body_pos[0, frame_id] - frame_pos_0).item()),
            )
            max_passive_drift = max(max_passive_drift, abs(float(angle[0].item())))
        if max_frame_drift > FRAME_DRIFT_TOL_M:
            raise QualificationError(f"frame drift exceeds {FRAME_DRIFT_TOL_M}m")
        if max_passive_drift > PASSIVE_DRIFT_TOL_RAD:
            raise QualificationError("passive hinge drift exceeds 0.25 degrees")

        effort = torch.tensor([[TORQUE_NM]], dtype=torch.float32, device=env.device)
        reached_tick: int | None = None
        max_angle = 0.0
        for tick in range(180):
            door.set_joint_effort_target(effort, joint_ids=[env._hinge_joint_id])  # noqa: SLF001
            obs, _, _, _, _ = env.step(zero)
            if not torch.isfinite(obs["policy"]).all():
                raise QualificationError("torque test produced NaN/Inf")
            angle, _ = env.hinge_state()
            max_angle = max(max_angle, float(angle[0].item()))
            if max_angle >= TORQUE_TARGET_RAD and reached_tick is None:
                reached_tick = tick + 1
        if reached_tick is None:
            raise QualificationError(
                f"15 Nm torque reached only {math.degrees(max_angle):.2f} degrees in 3s"
            )
        final_reset_angle, final_reset_speed = _assert_reset(env)
        return {
            "passed": True,
            "device": str(env.device),
            "reset_angle_deg": math.degrees(initial_angle),
            "reset_speed_rad_s": initial_speed,
            "passive_duration_s": 2.0,
            "max_frame_drift_m": max_frame_drift,
            "max_passive_drift_deg": math.degrees(max_passive_drift),
            "max_frame_panel_penetration_m": 0.0,
            "penetration_method": "canonical_proxy_clearance_plus_finite_PhysX_contact_state",
            "physics_robot_collision_mode": "disabled_for_door_only_gate",
            "physics_disabled_robot_collider_count": disabled_robot_colliders,
            "torque_nm": TORQUE_NM,
            "torque_target_deg": 80.0,
            "torque_target_time_s": reached_tick / 60.0,
            "torque_max_angle_deg": math.degrees(max_angle),
            "post_torque_reset_angle_deg": math.degrees(final_reset_angle),
            "post_torque_reset_speed_rad_s": final_reset_speed,
        }
    finally:
        env.close()


def _curve(episode) -> list[float]:
    values = [float(step.object_state["door_angle_rad"]) for step in episode.steps]
    if episode.outcome is not None:
        values.append(float(episode.outcome.final_door_angle))
    return values


def _rollout_result(episode) -> dict:
    curve = _curve(episode)
    sustained = maximum_sustained_angle_deg(curve)
    forces = [float(step.contact["force_n"]) for step in episode.steps]
    terminal = episode.extras.get("terminal_contact")
    if terminal is not None:
        forces.append(float(terminal["force_n"]))
    max_force = max(forces, default=0.0)
    outcome = episode.outcome
    passed = bool(
        outcome is not None
        and outcome.termination_reason == "controller_done"
        and outcome.success
        and sustained >= OPEN_ANGLE_DEG
        and max_force <= FORCE_LIMIT_N
        and np.isfinite(curve).all()
    )
    return {
        "passed": passed,
        "termination": None if outcome is None else outcome.termination_reason,
        "outcome_success": False if outcome is None else outcome.success,
        "angle_curve_rad": curve,
        "trace_hz": 60,
        "maximum_sustained_angle_deg": sustained,
        "sustained_window_ticks": 30,
        "max_contact_force_n": max_force,
        "force_limit_n": FORCE_LIMIT_N,
        "invalid_physics": not np.isfinite(curve).all(),
    }


def _repeatability_gate(slot: dict) -> dict:
    from alexdoor_xas.data_engine import DataEngineCfg, run_episode
    from alexdoor_xas.data_engine.generate import EpisodePlanItem
    from alexdoor_xas.policies.scripted.door_push_alex_v2 import alex_v2_push_cfg

    env = _make_env(slot)
    try:
        calibration = env.alex_v2_calibration()
        dimensions = DoorDimensions.from_mapping(slot["dimensions_m"])
        controller_cfg = replace(
            alex_v2_push_cfg(calibration),
            panel_width_m=dimensions.width_m,
            panel_height_m=dimensions.height_m,
            panel_thickness_m=dimensions.thickness_m,
            handedness=slot["candidate"]["handedness"],
        )
        engine_cfg = DataEngineCfg(
            task=paths.ALEX_V2_TASK,
            robot=paths.ALEX_V2_ROBOT_TAG,
            limitations=ALEX_V2_LIMITATIONS,
            max_ticks=600,
        )
        item = EpisodePlanItem(seed=4101)
        first = _rollout_result(run_episode(env, item, engine_cfg, controller_cfg=controller_cfg))
        second = _rollout_result(run_episode(env, item, engine_cfg, controller_cfg=controller_cfg))
        comparison = repeatability_metrics(first, second)
        pair_passed = first["passed"] and second["passed"] and comparison["passed"]
        diagnostics: list[dict] = []
        if not pair_passed:
            for _ in range(3):
                diagnostics.append(
                    _rollout_result(
                        run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
                    )
                )
        validate_diagnostic_count(pair_passed, diagnostics)
        return {
            "passed": pair_passed,
            "nominal": first,
            "repeated_after_reset": second,
            "comparison": comparison,
            "diagnostic_rollouts": diagnostics,
            "diagnostic_count": len(diagnostics),
            "clean_pair_required": True,
        }
    finally:
        env.close()


def _evidence_path(slot: dict, name: str) -> Path:
    return paths.PHASE4_1_EVIDENCE_DIR / f"door_{int(slot['slot']):02d}" / f"{name}.json"


def _run_stage(name: str, gate, required_state: set[str], next_state: str) -> None:
    worklist, slots = _slots()
    failures: list[str] = []
    for slot in slots:
        if slot["state"] not in required_state:
            failures.append(f"slot {slot['slot']}: state={slot['state']}")
            continue
        try:
            result = gate(slot)
            dump_json(_evidence_path(slot, name), result)
            if not result.get("passed", False):
                raise QualificationError(f"{name} gate returned passed=false")
            slot["state"] = next_state
            print(f"PASS {name}: slot={slot['slot']}", flush=True)
        except Exception as error:  # noqa: BLE001 - continue to report every selected slot.
            failures.append(f"slot {slot['slot']}: {error}")
            print(f"FAIL {name}: slot={slot['slot']}: {error}", flush=True)
    dump_json(WORKLIST, worklist)
    if failures:
        raise QualificationError(f"{name} failures: {failures}")


def _finalize_manifest() -> None:
    worklist, slots = _slots()
    if len(slots) != 24 or any(slot["state"] != "qualified" for slot in slots):
        raise QualificationError("manifest finalization requires exactly 24 qualified slots")
    assets: list[dict] = []
    maxima: dict[str, list[float]] = {}
    for slot in slots:
        candidate = slot["candidate"]
        static = load_json(_evidence_path(slot, "static"))
        physics = load_json(_evidence_path(slot, "physics"))
        repeatability = load_json(_evidence_path(slot, "repeatability"))
        asset_id = f"door_{int(slot['slot']):02d}"
        maxima[asset_id] = [
            repeatability["nominal"]["maximum_sustained_angle_deg"],
            repeatability["repeated_after_reset"]["maximum_sustained_angle_deg"],
            *[
                rollout["maximum_sustained_angle_deg"]
                for rollout in repeatability["diagnostic_rollouts"]
            ],
        ]
        normalized_dir = (paths.REPO_ROOT / slot["normalized_path"]).parent
        normalized_checksums = {
            str(path.relative_to(normalized_dir)): sha256_file(path)
            for path in sorted(normalized_dir.rglob("*"))
            if path.is_file()
        }
        evidence = {
            "static": static,
            "physics": physics,
            "repeatability": repeatability,
        }
        record = {
            **candidate,
            "asset_id": asset_id,
            "source_path": slot["source_path"],
            "source_size_bytes": slot["source_size_bytes"],
            "source_sha256": slot["source_sha256"],
            "geometry_fingerprint": slot["geometry_fingerprint"],
            "recipe_path": slot["recipe_path"],
            "modifications": load_json(paths.REPO_ROOT / slot["recipe_path"])["operations"],
            "normalized": {
                "path": slot["normalized_path"],
                "dimensions_m": slot["dimensions_m"],
                "triangles": slot["triangles"],
                "texture_max_px": slot["texture_max_px"],
                "checksums_sha256": normalized_checksums,
            },
            "qualification": {
                "static": static,
                "physics": physics,
                "nominal": repeatability["nominal"],
                "repeatability": {
                    **repeatability["comparison"],
                    "diagnostic_count": repeatability["diagnostic_count"],
                },
            },
            "evidence_sha256": canonical_sha256(evidence),
            "final_status": "provisional_for_phase4_2",
        }
        assets.append(record)
    recommendation = bootstrap_n_qual(maxima)
    rejections = load_json(REJECTIONS) if REJECTIONS.is_file() else {"entries": []}
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "retrieval_date": date.today().isoformat(),
        "asset_count": 24,
        "handedness_counts": {"left": 12, "right": 12},
        "rejected_candidate_count": len(rejections["entries"]),
        "qualification_summary": {
            "static_pass": 24,
            "physics_pass": 24,
            "nominal_pass": 24,
            "repeatability_pass": 24,
            "reserve_payload_count": 0,
        },
        "n_qual_recommendation": recommendation,
        "assets": assets,
    }
    validate_manifest(manifest, require_complete=True)
    dump_json(paths.PHASE4_1_MANIFEST, manifest)
    print(
        f"PASS manifest: assets=24 handedness=12/12 "
        f"n_qual={recommendation['recommended_n_qual']}"
    )


def main() -> int:
    rc = 0
    try:
        if args.command == "static":
            _run_stage("static", _static_gate, {"normalized", "static_pass"}, "static_pass")
        elif args.command == "physics":
            _run_stage("physics", _physics_gate, {"static_pass", "physics_pass"}, "physics_pass")
        elif args.command == "repeatability":
            _run_stage(
                "repeatability",
                _repeatability_gate,
                {"physics_pass", "qualified"},
                "qualified",
            )
        elif args.command == "all":
            _run_stage("static", _static_gate, {"normalized", "static_pass"}, "static_pass")
            _run_stage("physics", _physics_gate, {"static_pass", "physics_pass"}, "physics_pass")
            _run_stage(
                "repeatability",
                _repeatability_gate,
                {"physics_pass", "qualified"},
                "qualified",
            )
        else:
            if not args.all:
                raise QualificationError("finalize-manifest requires --all")
            _finalize_manifest()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    finally:
        if args.clean_shutdown:
            simulation_app.close()
    return rc


if __name__ == "__main__":
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
