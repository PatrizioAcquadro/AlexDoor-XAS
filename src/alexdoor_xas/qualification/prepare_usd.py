"""Installed Kit conversion, component inspection and canonical B1 authoring.

Import after AppLauncher. Visual geometry/materials pass through glTF; unsupported
conversion or missing material dependencies are unresolved, never silently accepted.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from alexdoor_xas.door_qualification import (
    DoorDimensions,
    connected_mesh_face_components,
    cuboid_inertia_kg_m2,
    geometry_fingerprint,
)

from .convex_geometry import clear_opening, mechanical_limit
from .preparation import (
    FORMATS,
    GROUPS,
    PreparationError,
    file_inventory,
    require,
    validate_recipe,
    write_json,
)


def convert(source, target):
    from isaacsim.core.experimental.utils.app import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter

    context = converter.AssetConverterContext()
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.export_preview_surface = True
    context.embed_textures = True
    # Referenced USD layers do not automatically rescale their metersPerUnit.
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = False
    context.baking_scales = True
    task = converter.get_instance().create_converter_task(str(source), str(target), None, context)
    require(
        asyncio.get_event_loop().run_until_complete(task.wait_until_finished()),
        f"Conversion failed: {task.get_error_message()}",
        status="unresolved",
        category="tool",
    )
    require(
        target.is_file() and target.stat().st_size > 0,
        "Converter produced no payload",
        status="unresolved",
        category="tool",
    )


def dependencies(source):
    """Find local sidecars before conversion; do not download dependencies."""
    source = Path(source).resolve()
    require(
        source.is_file() and source.suffix.lower() in FORMATS,
        "Missing/unsupported source",
        category="source",
        status="unresolved",
    )
    files = {source}
    if source.suffix.lower().startswith(".usd"):
        layers, assets, missing = UsdUtils.ComputeAllDependencies(str(source))
        require(
            not missing,
            f"Unresolved USD dependencies: {list(missing)}",
            status="unresolved",
            category="source",
        )
        for item in [*(layer.realPath for layer in layers), *assets]:
            # A package member is preserved by copying its enclosing USDZ.
            item = str(item).split("[", 1)[0]
            if item:
                files.add(Path(item).resolve())
    elif source.suffix.lower() == ".gltf":
        data = json.loads(source.read_text())
        for entry in data.get("buffers", []) + data.get("images", []):
            uri = entry.get("uri", "")
            if not uri or uri.startswith("data:"):
                continue
            require(
                "://" not in uri, "Remote glTF dependency", category="source", status="unresolved"
            )
            files.add((source.parent / unquote(uri)).resolve())
    elif source.suffix.lower() == ".obj":
        for line in source.read_text(errors="replace").splitlines():
            if line.strip().startswith("mtllib "):
                material = source.parent / line.strip().split(maxsplit=1)[1]
                files.add(material.resolve())
                require(
                    material.is_file(),
                    f"Missing {material}",
                    category="source",
                    status="unresolved",
                )
                for value in material.read_text().splitlines():
                    if value.strip().startswith(("map_", "bump ", "disp ", "decal ")):
                        files.add((material.parent / value.strip().split()[-1]).resolve())
    for path in files:
        require(
            path.is_file(), f"Missing dependency: {path}", status="unresolved", category="source"
        )
    return files


def load_source(source, output):
    """Inventory and snapshot inputs, then inspect individual connected mesh components."""
    source, output = Path(source).resolve(), Path(output)
    require(not output.exists() or not any(output.iterdir()), "Inspection output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    files = dependencies(source)
    inventory = file_inventory(files)
    # Preserve relative paths across layers, buffers and textures.
    import os

    common = Path(os.path.commonpath([str(p.parent) for p in files]))
    for original in files:
        target = output / "source" / original.relative_to(common)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    for item in inventory:
        item["snapshot"] = str(
            (output / "source" / Path(item["path"]).relative_to(common)).resolve()
        )
    local_source = output / "source" / source.relative_to(common)
    readable = local_source
    if source.suffix.lower() not in {".glb", ".gltf", ".obj"}:
        # FBX goes through the installed FBX SDK, then the common glTF inspection route.
        if source.suffix.lower() == ".fbx":
            converted = output / "converted.usd"
            convert(source, converted)
            dependencies(converted)
            local_source = converted
        # The installed USD->glTF exporter omits analytic primitives. Tessellate
        # cubes explicitly in a derived layer, retaining their original transforms.
        usd_stage = Usd.Stage.Open(str(local_source))
        flattened = output / "inspection.usda"
        usd_stage.Flatten().Export(str(flattened))
        usd_stage = Usd.Stage.Open(str(flattened))
        for prim in list(usd_stage.Traverse()):
            if prim.IsA(UsdGeom.Cube):
                size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                box = trimesh.creation.box(extents=[size] * 3)
                mesh_prim(usd_stage, str(prim.GetPath()), box.vertices, box.faces)
            elif prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
                require(
                    False,
                    f"Unsupported USD primitive {prim.GetTypeName()}",
                    status="unresolved",
                    category="tool",
                )
        usd_stage.GetRootLayer().Save()
        local_source = flattened
        readable = output / "inspection.glb"
        convert(local_source, readable)
    try:
        loaded = trimesh.load(readable, force="scene", process=False)
        components = []
        for mesh in loaded.dump(concatenate=False):
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                continue
            for faces in connected_mesh_face_components(mesh.vertices, mesh.faces):
                components.append(mesh.submesh([faces], append=True, repair=False))
    except Exception as exc:
        raise PreparationError(
            f"Cannot inspect converted geometry: {exc}", status="unresolved", category="tool"
        ) from exc
    require(bool(components), "No usable mesh geometry", category="asset")
    require(
        len(components) <= 512,
        "More than 512 components; bounded preparation limit",
        category="tool",
        status="unresolved",
    )
    triangles = sum(len(m.faces) for m in components)
    require(triangles <= 250_000, "More than 250,000 visual triangles", category="asset")
    textures = []
    summaries = []
    for index, mesh in enumerate(components):
        require(
            np.isfinite(mesh.vertices).all() and np.all(mesh.area_faces > 1e-14),
            f"Degenerate geometry in component {index}",
            category="asset",
        )
        material = getattr(mesh.visual, "material", None)
        for name in (
            "image",
            "baseColorTexture",
            "normalTexture",
            "emissiveTexture",
            "metallicRoughnessTexture",
            "occlusionTexture",
        ):
            image = getattr(material, name, None)
            if image is not None:
                require(max(image.size) <= 4096, "Texture exceeds 4K", category="asset")
                textures.append({"component": index, "channel": name, "size": list(image.size)})
        summaries.append(
            {
                "index": index,
                "triangles": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "material": getattr(material, "name", None),
            }
        )
    merged = trimesh.util.concatenate(components)
    summary = {
        "status": "pass",
        "scope": "local_inventory",
        "source": str(source),
        "source_sha256": next(item["sha256"] for item in inventory if item["path"] == str(source)),
        "files": inventory,
        "triangles": triangles,
        "textures": textures,
        "components": summaries,
        "coordinate_note": (
            "Recipe transforms inspected mesh coordinates; USD/FBX use glTF Y-up meters"
        ),
        "geometry_fingerprint": geometry_fingerprint(merged.vertices, merged.faces),
        "limits": [
            "glTF material conversion; review appearance before admission",
            "FBX external textures require local dependency review",
        ],
    }
    write_json(output / "inspect.json", summary)
    return components, summary


def mesh_prim(stage, path, vertices, faces):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def cooked_hulls(mesh, approximation="auto"):
    """Bake installed PhysX convex decomposition so clearance uses the same hulls."""
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    prim = mesh_prim(stage, "/Mesh", mesh.vertices, mesh.faces).GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    require(
        approximation in {"auto", "convexHull", "convexDecomposition"},
        "Unsupported collider approximation",
    )
    approximation = (
        ("convexHull" if mesh.is_convex else "convexDecomposition")
        if (approximation == "auto")
        else approximation
    )
    api.CreateApproximationAttr(approximation)
    if approximation == "convexHull":
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(64)
    else:
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(64)
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage)
    answer = []

    def ready(result, convexes):
        if result == PhysxCollisionRepresentationResult.RESULT_VALID:
            answer.extend(np.array([[v.x, v.y, v.z] for v in c.vertices]) for c in convexes)

    try:
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id.ToLongInt(),
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt("/Mesh"),
            run_asynchronously=False,
            on_result=ready,
        )
    finally:
        cache.Erase(stage_id)
    require(bool(answer), "PhysX convex cooking failed", category="tool", status="unresolved")
    return answer


def normalize(source, recipe, output):
    output = Path(output)
    components, inventory = load_source(source, output)
    rotation, scale, translation, hinge = validate_recipe(recipe, len(components))
    groups, collision = {}, {}
    for name in GROUPS:
        groups[name], collision[name] = [], []
        for index in recipe["components"][name]:
            mesh = components[index].copy()
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = rotation * scale, translation
            mesh.apply_transform(matrix)
            groups[name].append(mesh)
            collision[name].extend(
                cooked_hulls(mesh, recipe.get("colliders", {}).get(str(index), "auto"))
            )
    dimensions = DoorDimensions.from_mapping(recipe["dimensions_m"])
    panel_bounds = trimesh.util.concatenate(groups["Panel"]).bounds
    require(
        np.allclose(
            np.diff(panel_bounds, axis=0)[0],
            [dimensions.thickness_m, dimensions.width_m, dimensions.height_m],
            atol=0.001,
        ),
        "Measured panel dimensions differ from recipe by >1 mm",
    )
    sign = 1 if recipe["handedness"] == "left" else -1
    require(
        abs(hinge[1] - panel_bounds[1 if sign > 0 else 0, 1]) <= 0.01,
        "Hinge does not match original handedness/panel edge",
    )
    require(
        abs(panel_bounds[:, 1].mean()) <= 0.001
        and abs(min(m.bounds[0, 2] for m in groups["Frame"])) <= 0.001,
        "Canonical origin must center opening at floor level",
    )
    clear_opening(collision["Frame"], panel_bounds)
    limit = mechanical_limit(collision, hinge, recipe["handedness"])
    canonical = output / "door.usda"
    _author(canonical, groups, collision, hinge, recipe, limit)
    write_json(output / "recipe.json", recipe)
    write_json(
        output / "collision.json", {k: [p.tolist() for p in v] for k, v in collision.items()}
    )
    opening_to_hinge = np.eye(4)
    opening_to_hinge[:3, 3] = hinge
    opening_to_panel_center = np.eye(4)
    opening_to_panel_center[:3, 3] = panel_bounds.mean(0)
    result = {
        "status": "pass",
        "scope": "normalized_only",
        "usd": str(canonical.resolve()),
        "mechanical_limit_deg": limit,
        "dimensions_m": dimensions.to_dict(),
        "handedness": recipe["handedness"],
        "hinge_m": hinge.tolist(),
        "opening_center_m": [0, 0, 0],
        "panel_center_m": panel_bounds.mean(0).tolist(),
        "opening_to_hinge": opening_to_hinge.tolist(),
        "opening_to_panel_center_closed": opening_to_panel_center.tolist(),
        "geometry_fingerprint": inventory["geometry_fingerprint"],
        "release_files": file_inventory(
            [
                *dependencies(canonical),
                output / "recipe.json",
                output / "inspect.json",
                *(Path(item["snapshot"]) for item in inventory["files"]),
            ]
        ),
    }
    write_json(output / "normalize.json", result)
    return result


def _author(path, groups, collision, hinge_pos, recipe, limit):
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Door").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(True)
    root.SetCustomDataByKey(
        "b1",
        {
            "handedness": recipe["handedness"],
            "opening_center_m": Gf.Vec3d(0),
            "hinge_m": Gf.Vec3d(*hinge_pos),
        },
    )
    material = UsdShade.Material.Define(stage, "/Door/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.5)
    api.CreateDynamicFrictionAttr(0.5)
    api.CreateRestitutionAttr(0.0)
    for name in GROUPS:
        if not groups[name]:
            continue
        body = UsdGeom.Xform.Define(stage, f"/Door/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*hinge_pos))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateSolverPositionIterationCountAttr(
            16
        )
        PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateSolverVelocityIterationCountAttr(
            4
        )
        joined = trimesh.util.concatenate(groups[name])
        size, center = joined.extents, joined.bounds.mean(0) - hinge_pos
        mass = {"Frame": 50.0, "Panel": 25.0, "Handle": 0.5}[name]
        inertia = mass * (np.dot(size, size) - size**2) / 12
        if name == "Panel":
            inertia = cuboid_inertia_kg_m2(DoorDimensions.from_mapping(recipe["dimensions_m"]))
        m = UsdPhysics.MassAPI.Apply(body.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        scene = trimesh.Scene()
        for index, mesh in enumerate(groups[name]):
            mesh = mesh.copy()
            mesh.apply_translation(-hinge_pos)
            scene.add_geometry(mesh, node_name=f"component_{index}")
        glb, usd = path.parent / f"{name}.glb", path.parent / f"{name}.usd"
        glb.write_bytes(scene.export(file_type="glb"))
        convert(glb, usd)
        visual = UsdGeom.Xform.Define(stage, f"/Door/{name}/Visual").GetPrim()
        visual.GetReferences().AddReference(usd.name)
        for index, points in enumerate(collision[name]):
            points = points - hinge_pos
            shape = ConvexHull(points)
            mesh = mesh_prim(stage, f"/Door/{name}/Collision_{index}", points, shape.simplices)
            mesh.CreatePurposeAttr("guide")
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(0.002)
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Door/FixFrame")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Door/Frame")])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*hinge_pos))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Door/Hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/Door/Frame")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/Door/Panel")])
    hinge.CreateAxisAttr("Z")
    hinge.CreateCollisionEnabledAttr(True)
    q = Gf.Quatf(1, 0, 0, 0) if recipe["handedness"] == "left" else Gf.Quatf(0, 1, 0, 0)
    hinge.CreateLocalRot0Attr(q)
    hinge.CreateLocalRot1Attr(q)
    hinge.CreateLowerLimitAttr(0)
    hinge.CreateUpperLimitAttr(limit)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0)
    drive.CreateDampingAttr(4)
    if groups["Handle"]:
        joint = UsdPhysics.FixedJoint.Define(stage, "/Door/FixHandle")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Door/Panel")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Door/Handle")])
        joint.CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
