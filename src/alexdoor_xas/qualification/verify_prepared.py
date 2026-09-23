"""Static structure and isolated GPU physics gates for prepared B1 doors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from alexdoor_xas.door_qualification import DoorDimensions, cuboid_inertia_kg_m2, sha256_file

from .convex_geometry import clear_opening, mechanical_limit
from .preparation import GROUPS, file_inventory, hinge_collision_pairs, require, validate_recipe


def canonical_files(path):
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(path))
    require(not missing, f"Unresolved normalized dependencies: {list(missing)}", category="asset")
    root = path.parent.resolve()
    files = {path.resolve()}
    for raw in [*(layer.realPath for layer in layers), *assets]:
        p = Path(str(raw).split("[", 1)[0]).resolve()
        require(
            p.is_file() and p.is_relative_to(root), f"Non-local dependency: {raw}", category="asset"
        )
        files.add(p)
    return file_inventory(files)


def component_bounds(visual, indices):
    names = {f"component_{i}" for i in indices}
    nodes = [p for p in Usd.PrimRange(visual) if p.GetName() in names]
    require(len(nodes) == len(names), "Selected visual component nodes missing")
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    ranges = [cache.ComputeWorldBound(p).ComputeAlignedRange() for p in nodes]
    return np.array(
        [
            np.min([r.GetMin() for r in ranges], axis=0),
            np.max([r.GetMax() for r in ranges], axis=0),
        ]
    )


def static_check(attempt):
    attempt = Path(attempt).resolve()
    path = attempt / "door.usda"
    recipe = json.loads((attempt / "recipe.json").read_text())
    validate_recipe(recipe, sum(len(v) for v in recipe["components"].values()))
    inventory = canonical_files(path)
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    require(stage is not None, "Cannot open normalized stage", category="asset")
    require(str(stage.GetDefaultPrim().GetPath()) == "/Door", "Default prim must be /Door")
    require(
        UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1,
        "Expected meters and Z-up",
    )
    dimensions = DoorDimensions.from_mapping(recipe["dimensions_m"])
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/Door/PhysicsMaterial"))
    require(
        bool(material)
        and material.GetStaticFrictionAttr().Get() == 0.5
        and material.GetDynamicFrictionAttr().Get() == 0.5
        and material.GetRestitutionAttr().Get() == 0,
        "Nominal contact material changed",
    )
    hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Door/Hinge"))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    require(len(joints) == 1 and bool(hinge), "Expected one canonical hinge")
    require(
        hinge.GetAxisAttr().Get() == "Z" and hinge.GetLowerLimitAttr().Get() == 0,
        "Invalid hinge axis or zero",
    )
    require(
        list(map(str, hinge.GetBody0Rel().GetTargets())) == ["/Door/Frame"]
        and list(map(str, hinge.GetBody1Rel().GetTargets())) == ["/Door/Panel"],
        "Invalid hinge owners",
    )
    require(hinge.GetCollisionEnabledAttr().Get(), "Panel/frame collision must stay enabled")
    require(
        PhysxSchema.PhysxArticulationAPI(stage.GetDefaultPrim())
        .GetEnabledSelfCollisionsAttr()
        .Get(),
        "Articulation self collisions must stay enabled",
    )
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Door/FixFrame"))
    require(
        bool(fixed)
        and not fixed.GetBody0Rel().GetTargets()
        and list(map(str, fixed.GetBody1Rel().GetTargets())) == ["/Door/Frame"],
        "Frame not fixed",
    )
    expected_q = np.array([1, 0, 0, 0] if recipe["handedness"] == "left" else [0, 1, 0, 0])
    for attr in (hinge.GetLocalRot0Attr(), hinge.GetLocalRot1Attr()):
        q = attr.Get()
        require(
            np.allclose(np.r_[q.GetReal(), q.GetImaginary()], expected_q),
            "Hinge positive direction differs from handedness",
        )
    drive = UsdPhysics.DriveAPI(hinge.GetPrim(), "angular")
    require(
        drive.GetDampingAttr().Get() == 4 and drive.GetStiffnessAttr().Get() == 0,
        "Hinge nominal physics changed",
    )
    collision = {name: [] for name in GROUPS}
    collision_components = {name: [] for name in GROUPS}
    collision_paths = {name: [] for name in GROUPS}
    excluded = set(recipe.get("unlatched_components", []))
    visual_bounds = {}
    visual_triangles = 0
    for name in GROUPS:
        body = stage.GetPrimAtPath(f"/Door/{name}")
        if name == "Handle" and not recipe["components"][name]:
            require(not body, "Unexpected handle")
            continue
        require(bool(body) and body.HasAPI(UsdPhysics.RigidBodyAPI), f"Missing rigid {name}")
        mass_api = UsdPhysics.MassAPI(body)
        mass, inertia = mass_api.GetMassAttr().Get(), mass_api.GetDiagonalInertiaAttr().Get()
        require(
            mass is not None
            and inertia is not None
            and np.isfinite([mass, *inertia]).all()
            and mass > 0
            and min(inertia) > 0,
            f"Invalid mass/inertia: {name}",
        )
        require(mass == {"Frame": 50, "Panel": 25, "Handle": 0.5}[name], "Nominal mass changed")
        if name == "Panel":
            require(
                np.allclose(inertia, cuboid_inertia_kg_m2(dimensions), atol=1e-5),
                "Panel inertia differs from nominal geometry model",
            )
        visual = stage.GetPrimAtPath(f"/Door/{name}/Visual")
        require(bool(visual), f"Missing {name} visual")
        bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(visual)
        bounds = bounds.ComputeAlignedRange()
        visual_bounds[name] = np.array([bounds.GetMin(), bounds.GetMax()])
        expected_components = set(recipe["components"][name]) - excluded
        physical_bounds = (
            component_bounds(visual, expected_components) if excluded else visual_bounds[name]
        )
        collider_components = set()
        for prim in Usd.PrimRange(body):
            if prim != body:
                require(not prim.HasAPI(UsdPhysics.RigidBodyAPI), "Nested unintended rigid body")
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get())
                counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
                require(
                    len(points) > 0
                    and np.isfinite(points).all()
                    and len(counts) > 0
                    and counts.min() >= 3
                    and counts.sum() == len(indices)
                    and indices.min() >= 0
                    and indices.max() < len(points),
                    "Invalid mesh",
                )
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    component = prim.GetAttribute("b1:sourceComponent").Get()
                    if component is not None:
                        collider_components.add(component)
                    components = prim.GetAttribute("b1:sourceComponents").Get()
                    if components is not None:
                        collider_components.update(components)
                    binding, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    require(
                        str(binding.GetPath()) == "/Door/PhysicsMaterial",
                        "Collider material changed",
                    )
                    contact = PhysxSchema.PhysxCollisionAPI(prim)
                    require(
                        np.isclose(contact.GetContactOffsetAttr().Get(), 0.002)
                        and contact.GetRestOffsetAttr().Get() == 0,
                        "Collision offsets changed",
                    )
                    require(
                        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                        f"Disabled collider: {prim.GetPath()}",
                    )
                    require(
                        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                        == "convexHull",
                        "Expected baked convex collider",
                    )
                    matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
                    collision[name].append(points @ matrix[:3, :3].T + matrix[:3, 3])
                    collision_components[name].append(
                        list(components) if components is not None else [component]
                    )
                    collision_paths[name].append(str(prim.GetPath()))
                else:
                    visual_triangles += int((counts - 2).sum())
        require(bool(collision[name]), f"Missing collision geometry: {name}")
        if collider_components or excluded:
            require(
                collider_components == expected_components,
                f"Collision coverage differs from task recipe: {name}",
            )
        points = np.concatenate(collision[name])
        require(
            np.allclose(physical_bounds, [points.min(0), points.max(0)], atol=0.002),
            f"{name} visual/collision bounds differ by more than 2 mm",
        )
    panel_bounds = visual_bounds["Panel"]
    if "leaf_components" in recipe:
        panel_bounds = component_bounds(
            stage.GetPrimAtPath("/Door/Panel/Visual"), recipe["leaf_components"]
        )
    require(
        np.allclose(
            panel_bounds[1] - panel_bounds[0],
            [dimensions.thickness_m, dimensions.width_m, dimensions.height_m],
            atol=0.001,
        ),
        "Measured visual dimensions differ from the recipe",
    )
    if recipe["components"]["Handle"]:
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Door/FixHandle"))
        require(
            bool(joint)
            and list(map(str, joint.GetBody0Rel().GetTargets())) == ["/Door/Panel"]
            and list(map(str, joint.GetBody1Rel().GetTargets())) == ["/Door/Handle"],
            "Handle must be fixed to Panel",
        )
    require(0 < visual_triangles <= 250_000, "Visual triangle budget exceeded")
    from PIL import Image

    for item in inventory:
        if Path(item["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr"}:
            with Image.open(item["path"]) as image:
                require(max(image.size) <= 4096, "Normalized texture exceeds 4K")
    clear_opening(collision["Frame"], panel_bounds, recipe.get("clear_aperture_m"))
    excluded_pairs = hinge_collision_pairs(recipe, collision, collision_components)
    expected_filters = {
        frozenset((collision_paths["Panel"][i], collision_paths["Frame"][j]))
        for i, j in excluded_pairs
    }
    actual_filters = {
        frozenset((str(prim.GetPath()), str(target)))
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
    }
    require(
        actual_filters == expected_filters,
        "Collision filters differ from reviewed hinge interfaces",
    )
    limit = mechanical_limit(
        collision, np.asarray(recipe["hinge_m"]), recipe["handedness"], excluded_pairs
    )
    require(
        abs(hinge.GetUpperLimitAttr().Get() - limit) <= 0.11,
        "Mechanical joint limit differs from collision geometry",
    )
    # USD and source records are measured, not accepted on metadata alone.
    for item in json.loads((attempt / "inspect.json").read_text())["files"]:
        snapshot = Path(item.get("snapshot", item["path"]))
        require(
            snapshot.is_file() and sha256_file(snapshot) == item["sha256"],
            "Preserved source changed since inspection",
            category="source",
            status="unresolved",
        )
    return dict(
        status="pass",
        scope="static_asset_validity",
        release_files=inventory,
        mechanical_limit_deg=limit,
        visual_triangles=visual_triangles,
        colliders={k: len(v) for k, v in collision.items()},
    )


def physics_check(attempt, device="cuda:0"):
    import torch
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.envs.door_task.door_inspection import (
        DoorInspectionCfg,
        DoorInspectionEnv,
        tensor,
    )

    require(
        str(device).startswith("cuda") and torch.cuda.is_available(),
        "GPU runtime unavailable",
        category="runtime",
        status="unresolved",
    )
    attempt = Path(attempt).resolve()
    checked = static_check(attempt)
    limit = np.deg2rad(checked["mechanical_limit_deg"])
    recipe = json.loads((attempt / "recipe.json").read_text())
    cfg = DoorInspectionCfg()
    cfg.sim.device = device
    cfg.sense_contacts = True
    cfg.door_scene.spawn.usd_path = str(attempt / "door.usda")
    cfg.door.prim_path = cfg.door_scene.prim_path
    env = DoorInspectionEnv(cfg)
    traces, contacts = [], []
    try:
        env.reset()
        door = env._door  # noqa: SLF001
        names = list(door.body_names)
        frame, panel = names.index("Frame"), names.index("Panel")
        actors = [cfg.door.prim_path + "/" + name for name in names]
        view = env.sim.physics_sim_view.create_rigid_contact_view(
            actors, max_contact_data_count=4096
        )
        initial_pos = tensor(door.data.body_link_pos_w)[0].cpu().numpy().copy()
        initial_quat = tensor(door.data.body_link_quat_w)[0].cpu().numpy().copy()
        require(
            np.linalg.norm(initial_pos[frame] - recipe["hinge_m"]) < 1e-4,
            "Frame/hinge authored placement mismatch",
        )
        zero = torch.zeros((1, 1), device=device)
        max_frame_drift = max_hinge_error = max_penetration = max_frame_rotation = 0.0
        max_passive = max_speed_reset = max_angle_reset = 0.0
        maximum = 0.0
        dt = cfg.sim.dt * cfg.decimation

        def sample(phase):
            nonlocal max_frame_drift, max_hinge_error, max_penetration, max_frame_rotation
            angle, speed = [float(v[0]) for v in env.hinge_state()]
            pos = tensor(door.data.body_link_pos_w)[0].cpu().numpy()
            quat = tensor(door.data.body_link_quat_w)[0].cpu().numpy()
            require(
                np.isfinite(np.r_[angle, speed, pos.ravel(), quat.ravel()]).all(),
                "Non-finite physics state",
                category="physics",
            )
            max_frame_drift = max(
                max_frame_drift, float(np.linalg.norm(pos[frame] - initial_pos[frame]))
            )
            max_hinge_error = max(max_hinge_error, float(np.linalg.norm(pos[panel] - pos[frame])))
            r = Rotation.from_quat(quat[frame, [1, 2, 3, 0]])
            r0 = Rotation.from_quat(initial_quat[frame, [1, 2, 3, 0]])
            max_frame_rotation = max(max_frame_rotation, float((r * r0.inv()).magnitude()))
            raw = [v.numpy() for v in view.get_raw_contact_data(cfg.sim.dt)]
            force, point, normal, separation, counts, starts, other = raw
            force, separation, counts, starts, other = [
                v.reshape(-1) for v in (force, separation, counts, starts, other)
            ]
            require(
                np.all(counts >= 0)
                and np.all(starts >= 0)
                and np.all(starts + counts <= len(force))
                and counts.sum() < len(force),
                "Contact buffer invalid/saturated",
                category="runtime",
                status="unresolved",
            )
            active = np.concatenate(
                [np.arange(s, s + c) for s, c in zip(starts, counts, strict=True)]
            )
            if len(active):
                require(
                    np.isfinite(
                        np.r_[
                            force[active],
                            separation[active],
                            point[active].ravel(),
                            normal[active].ravel(),
                        ]
                    ).all(),
                    "Invalid raw contact values",
                    category="physics",
                )
                max_penetration = max(max_penetration, float(max(0, -separation[active].min())))
                contacts.append(
                    dict(
                        phase=phase,
                        angle_deg=float(np.rad2deg(angle)),
                        minimum_separation_m=float(separation[active].min()),
                        maximum_normal_force_n=float(force[active].max()),
                        count=len(active),
                    )
                )
                # Bearing/rubbing contact is not itself a blockage. Progress to
                # the stop, penetration and stability decide functional readiness.
            traces.append([angle, speed, *pos[frame], *pos[panel]])
            return angle, speed

        for repeat in range(3):
            env.reset()
            a, v = sample("reset")
            max_angle_reset = max(max_angle_reset, abs(a))
            max_speed_reset = max(max_speed_reset, abs(v))
            for _ in range(120):
                env.step(zero)
                a, _ = sample("passive")
                max_passive = max(max_passive, abs(a))
            if repeat == 0:
                for tick in range(int((limit / 0.35 + 3) / dt)):
                    angle, speed = [float(v[0]) for v in env.hinge_state()]
                    target = min(limit + np.deg2rad(1), 0.35 * (tick + 1) * dt)
                    torque = np.clip(25 * (target - angle) - 4 * speed, -15, 15)
                    door.set_joint_effort_target_index(
                        target=torch.tensor([[torque]], device=device, dtype=torch.float32)
                    )
                    env.step(zero)
                    a, _ = sample("opening")
                    maximum = max(maximum, a)
        require(
            max_angle_reset <= np.deg2rad(0.1) and max_speed_reset <= 0.01,
            "Closed reset outside Phase 4 tolerances",
            category="physics",
        )
        require(
            max_passive <= np.deg2rad(0.25) and max_frame_drift <= 1e-4,
            "Passive/frame drift outside Phase 4 tolerances",
            category="physics",
        )
        require(
            max_frame_rotation <= np.deg2rad(0.1) and max_hinge_error <= 0.001,
            "Unstable fixed frame/hinge",
            category="physics",
        )
        require(
            max_penetration <= 0.002,
            "Penetration exceeds 2 mm contact-offset scale",
            category="physics",
        )
        require(
            abs(maximum - limit) <= np.deg2rad(1),
            "Opening does not reach geometric stop",
            category="geometry_or_recipe",
        )
        return dict(
            status="pass",
            scope="isolated_gpu_asset_physics_not_expert_qualification",
            device=str(env.device),
            gpu=torch.cuda.get_device_name(device),
            max_reset_deg=float(np.rad2deg(max_angle_reset)),
            max_reset_speed=max_speed_reset,
            passive_drift_deg=float(np.rad2deg(max_passive)),
            frame_drift_m=max_frame_drift,
            frame_rotation_deg=float(np.rad2deg(max_frame_rotation)),
            hinge_error_m=max_hinge_error,
            penetration_m=max_penetration,
            maximum_deg=float(np.rad2deg(maximum)),
            mechanical_limit_deg=checked["mechanical_limit_deg"],
            release_files=checked["release_files"],
            resets=3,
        )
    finally:
        np.savez_compressed(attempt / "physics_trace.npz", state=np.asarray(traces))
        from .preparation import write_json

        write_json(attempt / "contacts.json", contacts)
        env.close()
