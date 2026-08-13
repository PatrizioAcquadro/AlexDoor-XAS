#!/usr/bin/env python
"""Inspect and normalize one admitted Phase 4.1 door into canonical USD.

The local source is decomposed into existing connected components.  The generated
recipe records only component selection, one uniform scale, rigid transforms,
material localization, pivot correction, and common physics authoring.  Meshes are
never deformed or decimated.  Extracted GLBs are converted with the installed
``omni.kit.asset_converter`` extension.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--slot", type=int, required=True)
parser.add_argument("--target-height-m", type=float, default=2.0)
parser.add_argument("--recipe", type=Path, help="Use an existing reviewed recipe.")
parser.add_argument(
    "--panel-component",
    type=int,
    help="Use this inspected connected component as the panel seed.",
)
parser.add_argument("--inspect-only", action="store_true")
parser.add_argument("--clean-shutdown", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Runtime imports after AppLauncher.
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from isaacsim.core.experimental.utils.app import enable_extension  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.door_qualification import (  # noqa: E402
    FRICTION,
    HINGE_DAMPING_NM_S_RAD,
    HINGE_LIMIT_DEG,
    MAX_TEXTURE_EDGE_PX,
    MAX_TRIANGLES,
    PANEL_MASS_KG,
    RESTITUTION,
    DoorDimensions,
    QualificationError,
    connected_mesh_face_components,
    cuboid_inertia_kg_m2,
    dump_json,
    geometry_fingerprint,
    handedness_sign,
    load_json,
    sha256_file,
)

WORKLIST = paths.PHASE4_1_EVIDENCE_DIR / "worklist.json"
ANCHOR_M = (-0.7007639, -0.4460541, 1.0096998)


def _progress(message: str) -> None:
    print(f"[phase4.1 normalize slot={args.slot:02d}] {message}", flush=True)


def _slot_entry(slot_id: int) -> tuple[dict, dict]:
    worklist = load_json(WORKLIST)
    for slot in worklist["slots"]:
        if int(slot["slot"]) == slot_id:
            if slot["state"] != "ingested":
                raise QualificationError(f"slot {slot_id} is not ingested: {slot['state']}")
            return worklist, slot
    raise QualificationError(f"slot {slot_id} is vacant")


def _source_path(slot: dict) -> Path:
    path = paths.REPO_ROOT / slot["source_path"]
    if not path.is_file() or sha256_file(path) != slot["source_sha256"]:
        raise QualificationError("source payload is missing or differs from its ingest checksum")
    return path


def _load_components(source: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    components: list[trimesh.Trimesh] = []
    for mesh in scene.dump(concatenate=False):
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        face_groups = connected_mesh_face_components(mesh.vertices, mesh.faces)
        for face_indices in face_groups:
            component = mesh.submesh([face_indices], append=True, repair=False)
            if isinstance(component, trimesh.Trimesh) and len(component.faces):
                components.append(component)
    if not components:
        raise QualificationError("source has no triangle meshes")
    return components


def _principal_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    covariance = np.cov((vertices - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    rotation = vectors[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    canonical = (vertices - center) @ rotation
    extents = np.ptp(canonical, axis=0)
    return rotation, extents


def _component_summary(component: trimesh.Trimesh, index: int) -> dict:
    _, extents = _principal_frame(component)
    return {
        "index": index,
        "vertices": int(len(component.vertices)),
        "triangles": int(len(component.faces)),
        "surface_area_source_units2": float(component.area),
        "principal_extents_source_units": [float(value) for value in extents],
    }


def _panel_index(
    components: list[trimesh.Trimesh], reviewed_index: int | None = None
) -> int:
    if reviewed_index is not None:
        if not 0 <= reviewed_index < len(components):
            raise QualificationError(
                f"reviewed panel component {reviewed_index} is outside the component list"
            )
        return reviewed_index
    candidates: list[tuple[float, int]] = []
    for index, component in enumerate(components):
        _, extents = _principal_frame(component)
        x, y, z = extents
        if y <= 0.0 or z <= 0.0:
            continue
        height_ratio = z / y
        thickness_ratio = x / y
        plausible = 1.4 <= height_ratio <= 4.5 and thickness_ratio <= 0.35
        score = float(component.area) * (4.0 if plausible else 1.0)
        candidates.append((score, index))
    if not candidates:
        raise QualificationError("no non-degenerate connected component can be a door panel")
    return max(candidates)[1]


def _texture_edge(component: trimesh.Trimesh) -> int:
    material = getattr(getattr(component, "visual", None), "material", None)
    image = getattr(material, "image", None)
    if image is None:
        return 0
    return int(max(image.size))


def _build_auto_recipe(slot: dict, components: list[trimesh.Trimesh]) -> dict:
    panel_index = _panel_index(components, args.panel_component)
    panel = components[panel_index]
    rotation, extents = _principal_frame(panel)
    scale = float(args.target_height_m / extents[2])
    dimensions = DoorDimensions(
        width_m=float(extents[1] * scale),
        height_m=float(extents[2] * scale),
        thickness_m=float(extents[0] * scale),
    )
    dimensions.validate()
    panel_center = np.asarray(panel.vertices, dtype=np.float64).mean(axis=0)
    panel_diag = float(np.linalg.norm(extents))
    panel_components = [panel_index]
    frame_components: list[int] = []
    handle_components: list[int] = []
    for index, component in enumerate(components):
        if index == panel_index:
            continue
        center = (np.asarray(component.vertices).mean(axis=0) - panel_center) @ rotation * scale
        _, component_extents = _principal_frame(component)
        component_size = np.sort(component_extents * scale)
        diagonal = float(np.linalg.norm(component_size))
        inside_panel = abs(center[1]) <= dimensions.width_m * 0.65 and abs(center[2]) <= (
            dimensions.height_m * 0.55
        )
        frame_like = (
            component_size[2] >= dimensions.height_m * 0.5
            and component_size[1] <= dimensions.width_m * 0.25
        )
        if frame_like:
            frame_components.append(index)
        elif inside_panel and diagonal <= panel_diag * scale * 0.25:
            handle_components.append(index)
        elif inside_panel:
            panel_components.append(index)
        else:
            frame_components.append(index)
    if not frame_components:
        raise QualificationError("connected-component separation found no distinct door frame")
    handedness = slot["candidate"]["handedness"]
    side = handedness_sign(handedness)
    rigid_rotation = rotation.copy()
    if side < 0.0:
        # A 180 degree rigid Z rotation places an existing, unmirrored panel on -Y.
        rigid_rotation = rigid_rotation @ np.diag((-1.0, -1.0, 1.0))
    return {
        "schema": "alexdoor.phase4_1_normalization_recipe.v1",
        "asset_id": f"door_{int(slot['slot']):02d}",
        "source_uid": slot["candidate"]["source_uid"],
        "source_sha256": slot["source_sha256"],
        "handedness": handedness,
        "component_selection": {
            "panel": panel_components,
            "frame": frame_components,
            "handle": handle_components,
        },
        "uniform_scale": scale,
        "rigid_source_to_canonical_rotation": rigid_rotation.tolist(),
        "source_panel_center": panel_center.tolist(),
        "dimensions_m": dimensions.to_dict(),
        "operations": [
            "select_existing_connected_components",
            "uniform_scale",
            "rigid_axis_alignment",
            "rigid_hinge_pivot_translation",
            "localize_materials_via_glb",
            "add_common_collision_proxies_and_articulation",
        ],
        "deformation": False,
        "decimation": False,
    }


def _transform_components(
    components: list[trimesh.Trimesh], recipe: dict
) -> dict[str, list[trimesh.Trimesh]]:
    rotation = np.asarray(recipe["rigid_source_to_canonical_rotation"], dtype=np.float64)
    center = np.asarray(recipe["source_panel_center"], dtype=np.float64)
    scale = float(recipe["uniform_scale"])
    dimensions = DoorDimensions.from_mapping(recipe["dimensions_m"])
    side = handedness_sign(recipe["handedness"])
    translation = np.array(
        [dimensions.thickness_m / 2.0, side * dimensions.width_m / 2.0, 0.0]
    )
    groups: dict[str, list[trimesh.Trimesh]] = {}
    used: list[int] = []
    for name, indices in recipe["component_selection"].items():
        groups[name] = []
        for index in indices:
            if int(index) in used or not 0 <= int(index) < len(components):
                raise QualificationError(f"invalid or repeated component index {index}")
            used.append(int(index))
            mesh = components[int(index)].copy()
            vertices = (np.asarray(mesh.vertices) - center) @ rotation * scale + translation
            mesh.vertices = vertices
            groups[name].append(mesh)
    if sorted(used) != list(range(len(components))):
        raise QualificationError("recipe must classify every connected component exactly once")
    if not groups["panel"] or not groups["frame"]:
        raise QualificationError("recipe requires non-empty panel and frame groups")
    return groups


def _export_group_glb(group: list[trimesh.Trimesh], target: Path) -> None:
    scene = trimesh.Scene()
    for index, mesh in enumerate(group):
        scene.add_geometry(mesh, node_name=f"component_{index:03d}")
    target.write_bytes(scene.export(file_type="glb"))


async def _convert_asset(source: Path, target: Path) -> None:
    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter  # noqa: PLC0415

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.export_preview_surface = True
    context.embed_textures = True
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    context.baking_scales = True
    context.use_double_precision_to_usd_transform_op = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(target), None, context
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"asset conversion failed: {task.get_error_message()}")


def _mesh_readable_source(source: Path, slot_id: int) -> Path:
    if source.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        _progress(f"using mesh-readable source {source.name}")
        return source
    target = paths.PHASE4_1_EVIDENCE_DIR / f"door_{slot_id:02d}" / "source_inspection.glb"
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size > 0
        and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        _progress(f"reusing source inspection GLB ({target.stat().st_size} bytes)")
        return target
    _progress(f"converting {source.name} to source inspection GLB")
    asyncio.get_event_loop().run_until_complete(_convert_asset(source, target))
    if not target.is_file() or target.stat().st_size <= 0:
        raise QualificationError(f"USD-family source conversion produced no GLB: {target}")
    _progress(f"source inspection GLB ready ({target.stat().st_size} bytes)")
    return target


def _default_prim_path(path: Path) -> Sdf.Path:
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise QualificationError(f"converted visual has no default prim: {path}")
    return stage.GetDefaultPrim().GetPath()


def _add_visual_reference(stage, parent_path: str, visual_path: Path, normalized_dir: Path) -> None:
    prim = UsdGeom.Xform.Define(stage, f"{parent_path}/Visual").GetPrim()
    relative = os.path.relpath(visual_path, normalized_dir)
    prim.GetReferences().AddReference(relative, _default_prim_path(visual_path))


def _collision_cube(stage, path: str, translation, scale, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        material, materialPurpose="physics"
    )


def _apply_mass(prim, mass: float, inertia: tuple[float, float, float]) -> None:
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _author_canonical_usd(
    target: Path,
    visuals: dict[str, Path],
    recipe: dict,
    slot: dict,
    geometry_digest: str,
    triangles: int,
    texture_edge_px: int,
) -> None:
    dimensions = DoorDimensions.from_mapping(recipe["dimensions_m"])
    side = handedness_sign(recipe["handedness"])
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor_xf = UsdGeom.Xformable(floor.GetPrim())
    floor_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    floor_xf.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 0.05))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(400.0)

    door_root = UsdGeom.Xform.Define(stage, "/World/Door").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(door_root)
    door_root.SetCustomData(
        {
            "alexdoor": {
                "schema": "alexdoor.phase4_1_normalized_door.v1",
                "asset_id": recipe["asset_id"],
                "source_uid": recipe["source_uid"],
                "source_sha256": recipe["source_sha256"],
                "geometry_fingerprint": geometry_digest,
                "handedness": recipe["handedness"],
                "width_m": dimensions.width_m,
                "height_m": dimensions.height_m,
                "thickness_m": dimensions.thickness_m,
                "triangles": triangles,
                "texture_max_px": texture_edge_px,
            }
        }
    )
    material = UsdShade.Material.Define(stage, "/World/Door/PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(FRICTION)
    physics_material.CreateDynamicFrictionAttr(FRICTION)
    physics_material.CreateRestitutionAttr(RESTITUTION)

    frame = UsdGeom.Xform.Define(stage, "/World/Door/Doorframe")
    door = UsdGeom.Xform.Define(stage, "/World/Door/Door")
    handle = UsdGeom.Xform.Define(stage, "/World/Door/Handle")
    body_anchor = (
        ANCHOR_M[0],
        ANCHOR_M[1] + (0.0 if side > 0.0 else dimensions.width_m),
        ANCHOR_M[2],
    )
    for body in (frame, door, handle):
        body_xform = UsdGeom.Xformable(body.GetPrim())
        body_xform.AddTranslateOp().Set(Gf.Vec3d(*body_anchor))
        body_xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    _add_visual_reference(stage, "/World/Door/Doorframe", visuals["frame"], target.parent)
    _add_visual_reference(stage, "/World/Door/Door", visuals["panel"], target.parent)
    if "handle" in visuals:
        _add_visual_reference(stage, "/World/Door/Handle", visuals["handle"], target.parent)

    gap = 0.003
    frame_depth = max(0.12, dimensions.thickness_m * 2.0)
    jamb = 0.08
    _collision_cube(
        stage,
        "/World/Door/Door/PanelCollider",
        (dimensions.thickness_m / 2.0, side * dimensions.width_m / 2.0, 0.0),
        (dimensions.thickness_m, dimensions.width_m, dimensions.height_m),
        material,
    )
    _collision_cube(
        stage,
        "/World/Door/Doorframe/HingeJambCollider",
        (0.0, -side * (jamb / 2.0 + gap), 0.0),
        (frame_depth, jamb, dimensions.height_m + 0.16),
        material,
    )
    _collision_cube(
        stage,
        "/World/Door/Doorframe/FarJambCollider",
        (0.0, side * (dimensions.width_m + jamb / 2.0 + gap), 0.0),
        (frame_depth, jamb, dimensions.height_m + 0.16),
        material,
    )
    _collision_cube(
        stage,
        "/World/Door/Doorframe/HeaderCollider",
        (0.0, side * dimensions.width_m / 2.0, dimensions.height_m / 2.0 + jamb / 2.0),
        (frame_depth, dimensions.width_m + 2.0 * jamb, jamb),
        material,
    )

    frame_inertia = (10.0, 10.0, 10.0)
    _apply_mass(frame.GetPrim(), 50.0, frame_inertia)
    _apply_mass(door.GetPrim(), PANEL_MASS_KG, cuboid_inertia_kg_m2(dimensions))
    _apply_mass(handle.GetPrim(), 0.5, (0.002, 0.002, 0.002))

    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Door/FixDoorframe")
    fixed.CreateBody1Rel().SetTargets([frame.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*body_anchor))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Door/Doorframe/Hinge")
    hinge.CreateBody0Rel().SetTargets([frame.GetPath()])
    hinge.CreateBody1Rel().SetTargets([door.GetPath()])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLowerLimitAttr(HINGE_LIMIT_DEG[0])
    hinge.CreateUpperLimitAttr(HINGE_LIMIT_DEG[1])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    if side > 0.0:
        joint_frame = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    else:
        joint_frame = Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    hinge.CreateLocalRot0Attr(joint_frame)
    hinge.CreateLocalRot1Attr(joint_frame)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(HINGE_DAMPING_NM_S_RAD)

    handle_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Door/FixHandle")
    handle_joint.CreateBody0Rel().SetTargets([door.GetPath()])
    handle_joint.CreateBody1Rel().SetTargets([handle.GetPath()])
    handle_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    handle_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()


def _normalize(worklist: dict, slot: dict, source: Path, components) -> None:
    recipe_path = args.recipe or paths.PHASE4_1_RECIPES_DIR / f"door_{args.slot:02d}.json"
    if not recipe_path.is_absolute():
        recipe_path = paths.REPO_ROOT / recipe_path
    recipe_path = recipe_path.resolve()
    if args.recipe:
        _progress(f"loading reviewed recipe {recipe_path.name}")
        recipe = load_json(recipe_path)
    else:
        _progress("building automatic normalization recipe")
        recipe = _build_auto_recipe(slot, components)
        dump_json(recipe_path, recipe)
    if recipe["source_sha256"] != slot["source_sha256"]:
        raise QualificationError("recipe source checksum differs from ingested payload")
    groups = _transform_components(components, recipe)
    _progress(
        "classified components as "
        + ", ".join(f"{name}={len(group)}" for name, group in groups.items())
    )
    triangles = sum(len(component.faces) for component in components)
    texture_edge = max((_texture_edge(component) for component in components), default=0)
    if triangles > MAX_TRIANGLES:
        raise QualificationError(f"local triangle count exceeds {MAX_TRIANGLES}: {triangles}")
    if texture_edge > MAX_TEXTURE_EDGE_PX:
        raise QualificationError(
            f"local texture edge exceeds {MAX_TEXTURE_EDGE_PX}px: {texture_edge}px"
        )
    merged = trimesh.util.concatenate(components)
    fingerprint = geometry_fingerprint(merged.vertices, merged.faces)
    accepted_fingerprints = {
        other.get("geometry_fingerprint")
        for other in worklist["slots"]
        if other is not slot and other.get("geometry_fingerprint")
    }
    if fingerprint in accepted_fingerprints:
        raise QualificationError("geometry fingerprint duplicates another local candidate")

    target_dir = paths.PHASE4_1_NORMALIZED_DIR / f"door_{args.slot:02d}"
    if target_dir.exists() and any(target_dir.iterdir()):
        raise QualificationError(f"normalized target is not empty: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    visuals: dict[str, Path] = {}
    for name, group in groups.items():
        if not group:
            continue
        _progress(f"exporting and converting {name} visual ({len(group)} components)")
        glb = target_dir / f"{name}.glb"
        usd = target_dir / f"{name}.usd"
        _export_group_glb(group, glb)
        asyncio.get_event_loop().run_until_complete(_convert_asset(glb, usd))
        visuals[name] = usd
    normalized_usd = target_dir / "door.usda"
    _progress("authoring canonical door USD")
    _author_canonical_usd(
        normalized_usd,
        visuals,
        recipe,
        slot,
        fingerprint,
        triangles,
        texture_edge,
    )
    slot.update(
        {
            "state": "normalized",
            "recipe_path": str(recipe_path.relative_to(paths.REPO_ROOT)),
            "normalized_path": str(normalized_usd.relative_to(paths.REPO_ROOT)),
            "normalized_sha256": sha256_file(normalized_usd),
            "geometry_fingerprint": fingerprint,
            "dimensions_m": recipe["dimensions_m"],
            "triangles": triangles,
            "texture_max_px": texture_edge,
        }
    )
    dump_json(WORKLIST, worklist)
    print(
        f"PASS normalize: slot={args.slot} triangles={triangles} texture={texture_edge}px "
        f"fingerprint={fingerprint} usd={normalized_usd}"
    )


def main() -> int:
    rc = 0
    try:
        if not 1 <= args.slot <= 24:
            raise QualificationError("slot must be in 1..24")
        worklist, slot = _slot_entry(args.slot)
        source = _source_path(slot)
        _progress(f"validated source checksum for {source.name}")
        readable_source = _mesh_readable_source(source, args.slot)
        _progress(f"loading and splitting connected components from {readable_source.name}")
        components = _load_components(readable_source)
        _progress(f"loaded {len(components)} connected components")
        summary = [
            _component_summary(component, index)
            for index, component in enumerate(components)
        ]
        inspect_path = paths.PHASE4_1_EVIDENCE_DIR / f"door_{args.slot:02d}" / "components.json"
        dump_json(inspect_path, {"source": str(source), "components": summary})
        print(json.dumps({"slot": args.slot, "components": summary}, indent=2), flush=True)
        if not args.inspect_only:
            _normalize(worklist, slot, source, components)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    finally:
        # Kit shutdown may terminate the interpreter with status 0.  On failure,
        # preserve the qualification error as the process status instead.
        if args.clean_shutdown and rc == 0:
            simulation_app.close()
    return rc


if __name__ == "__main__":
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
