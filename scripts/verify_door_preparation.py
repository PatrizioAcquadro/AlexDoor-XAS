#!/usr/bin/env python
"""Exercise B1 preparation on Phase 4 synthetics; no real candidate is admitted."""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case", type=int, choices=range(4))
parser.add_argument("--physics", action="store_true")
parser.add_argument("--formats", action="store_true")
parser.add_argument("--invalid", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402

from alexdoor_xas.assets.synthetic_door import CASES, author_synthetic_door  # noqa: E402
from alexdoor_xas.qualification.preparation import (  # noqa: E402
    PreparationError,
    new_attempt,
    require,
    write_json,
)
from alexdoor_xas.qualification.prepare_usd import (  # noqa: E402
    convert,
    cooked_hulls,
    load_source,
    mesh_prim,
    normalize,
)


def synthetic_source(door, output):
    source = output / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(author_synthetic_door(stage, "/Door", door))
    stage.GetRootLayer().Save()
    components, _ = load_source(source, output / "inventory")
    recipe = synthetic_recipe(components, door)
    write_json(output / "recipe.json", recipe)
    return source, recipe


def synthetic_recipe(components, door):
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    groups = {"Frame": [], "Panel": [], "Handle": []}
    for i, mesh in enumerate(components):
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
        if np.allclose(mesh.extents, [door.thickness, door.width, door.height], atol=1e-5):
            groups["Panel"].append(i)
        elif np.allclose(mesh.extents, [0.08, 0.12, 0.035], atol=1e-5):
            groups["Handle"].append(i)
        else:
            groups["Frame"].append(i)
    recipe = dict(
        handedness=door.handedness,
        scale=1,
        rotation=rotation.tolist(),
        translation_m=[0, 0, 0],
        opening_center_source=[0, 0, 0],
        hinge_m=door.hinge.tolist(),
        components=groups,
        leaf_components=groups["Panel"],
        dimensions_m=dict(width_m=door.width, height_m=door.height, thickness_m=door.thickness),
        modifications=["Explicit synthetic component selection; nominal B1 physics"],
        source_dependencies=[],
    )
    return recipe


def format_checks(reference, output):
    import trimesh
    from PIL import Image

    from alexdoor_xas.qualification.verify_prepared import static_check

    output.mkdir()
    scene = trimesh.load(reference / "inspection.glb", force="scene", process=False)
    # A conspicuous texture proves external-image preservation as well as geometry.
    pattern = np.zeros((64, 64, 3), dtype=np.uint8)
    pattern[:, :32] = [220, 50, 20]
    pattern[:, 32:] = [20, 150, 220]
    image = Image.fromarray(pattern)
    for mesh in scene.geometry.values():
        uv = mesh.vertices[:, [1, 2]].copy()
        uv -= uv.min(0)
        uv /= np.maximum(np.ptp(uv, axis=0), 1e-8)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=image)
        )
    glb = output / "textured.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    exported = scene.export(file_type="gltf")
    for name, content in exported.items():
        (output / name).write_bytes(content)
    gltf = output / "model.gltf"
    data = json.loads(gltf.read_text())
    for index, item in enumerate(data.get("images", [])):
        view = data["bufferViews"][item.pop("bufferView")]
        buffer = (output / data["buffers"][view["buffer"]]["uri"]).read_bytes()
        image_path = f"external-{index}.png"
        offset = view.get("byteOffset", 0)
        (output / image_path).write_bytes(buffer[offset : offset + view["byteLength"]])
        item["uri"] = image_path
    write_json(gltf, data)
    obj = output / "textured.obj"
    obj_text, textures = trimesh.exchange.obj.export_obj(scene, return_texture=True)
    obj.write_text(obj_text)
    for name, content in textures.items():
        (output / name).write_bytes(content)
    usd = output / "textured.usd"
    convert(glb, usd)
    stage = Usd.Stage.Open(str(usd))
    for ext in ("usda", "usdc"):
        stage.Export(str(output / f"textured.{ext}"))
    usdz = output / "textured.usdz"
    require(
        UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usd)), str(usdz)), "USDZ packaging failed"
    )
    fbx = output / "textured.fbx"
    convert(usd, fbx)
    results = []
    for source in (
        glb,
        gltf,
        obj,
        usd,
        usdz,
        output / "textured.usda",
        output / "textured.usdc",
        fbx,
    ):
        directory = output / source.suffix[1:]
        directory.mkdir()
        components, inventory = load_source(source, directory / "inventory")
        require(bool(inventory["textures"]), f"Texture disappeared from {source.name}")
        recipe = synthetic_recipe(components, CASES[0])
        target = directory / "prepared"
        target.mkdir()
        normalized = normalize(source, recipe, target)
        checked = static_check(target)
        write_json(target / "static.json", checked)
        result = {
            "format": source.suffix,
            "status": "pass",
            "usd": normalized["usd"],
            "texture_channels": len(inventory["textures"]),
        }
        results.append(result)
        print(f"FORMAT {json.dumps(result)}", flush=True)
    write_json(output / "formats.json", results)
    return results


def invalid_checks(reference, output):
    import trimesh

    from alexdoor_xas.qualification.convex_geometry import Convex, overlap
    from alexdoor_xas.qualification.verify_prepared import static_check

    output.mkdir()
    results = []
    for case in (
        "missing_dependency",
        "zero_mass",
        "negative_inertia",
        "missing_handle_collision",
        "blocked_frame",
        "wrong_limit",
        "wrong_visual_scale",
        "initial_interpenetration",
    ):
        target = output / case
        shutil.copytree(reference, target)
        path = target / "door.usda"
        stage = Usd.Stage.Open(str(path))
        if case == "missing_dependency":
            visual = stage.GetPrimAtPath("/Door/Panel/Visual")
            visual.GetReferences().ClearReferences()
            visual.GetReferences().AddReference("missing.usd")
        elif case == "zero_mass":
            stage.GetPrimAtPath("/Door/Panel").GetAttribute("physics:mass").Set(0)
        elif case == "negative_inertia":
            stage.GetPrimAtPath("/Door/Panel").GetAttribute("physics:diagonalInertia").Set(
                (-1, 1, 1)
            )
        elif case == "missing_handle_collision":
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/Door/Handle")):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
        elif case == "wrong_limit":
            stage.GetPrimAtPath("/Door/Hinge").GetAttribute("physics:upperLimit").Set(90)
        elif case == "wrong_visual_scale":
            stage.GetPrimAtPath("/Door/Panel/Visual").GetAttribute("xformOp:scale").Set(
                (100, 100, 100)
            )
        elif case == "initial_interpenetration":
            stage.GetPrimAtPath("/Door/Handle").GetAttribute("xformOp:translate").Set(
                (0.06, 0.9225, 0)
            )
        else:
            # A frame hull enclosing its opening preserves its outer bounds but is invalid.
            points = []
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/Door/Frame")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    points.extend(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            box = trimesh.Trimesh(vertices=points, process=False).convex_hull
            prim = mesh_prim(stage, "/Door/Frame/Blocker", box.vertices, box.faces).GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(0.002)
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateRestOffsetAttr(0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material(stage.GetPrimAtPath("/Door/PhysicsMaterial")),
                materialPurpose="physics",
            )
        stage.GetRootLayer().Save()
        try:
            static_check(target)
        except PreparationError as exc:
            expected = {
                "missing_dependency": "dependencies",
                "zero_mass": "mass/inertia",
                "negative_inertia": "mass/inertia",
                "missing_handle_collision": "Missing collision",
                "blocked_frame": "obstructs",
                "wrong_limit": "joint limit",
                "wrong_visual_scale": "visual/collision",
                "initial_interpenetration": "intersects",
            }
            require(expected[case] in str(exc), f"Unexpected rejection for {case}: {exc}")
            results.append({"case": case, "status": "detected", "reason": str(exc)})
        else:
            raise AssertionError(f"Invalid fixture passed: {case}")
    # Exercise actual PhysX decomposition on a concave connected L extrusion.
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=float
    )
    vertices = np.vstack([vertices, vertices + [0, 0, 0.1]])
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 5, 3],
        [3, 5, 4],
        [6, 7, 8],
        [6, 8, 9],
        [6, 9, 11],
        [9, 10, 11],
    ]
    for a in range(6):
        b = (a + 1) % 6
        faces.extend([[a, b, b + 6], [a, b + 6, a + 6]])
    shape = trimesh.Trimesh(vertices, faces, process=True)
    hulls = cooked_hulls(shape, "convexDecomposition")
    probe = trimesh.creation.box(extents=[0.2, 0.2, 0.05])
    probe.apply_translation([1.65, 1.65, 0.05])
    require(
        not any(overlap(Convex(p), Convex(probe.vertices)) for p in hulls),
        "Decomposition filled the concavity",
    )
    results.append({"case": "concave_decomposition", "status": "pass", "hulls": len(hulls)})
    degenerate = output / "degenerate.glb"
    degenerate.write_bytes(
        trimesh.Scene(
            trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces=[[0, 1, 2]], process=False
            )
        ).export(file_type="glb")
    )
    try:
        load_source(degenerate, output / "degenerate")
    except PreparationError as exc:
        require("Degenerate" in str(exc), f"Unexpected degenerate-mesh rejection: {exc}")
        results.append({"case": "degenerate_geometry", "status": "detected", "reason": str(exc)})
    else:
        raise AssertionError("Degenerate mesh passed")
    write_json(output / "invalid.json", results)
    return results


def runtime_obstruction(reference, output):
    from unittest.mock import patch

    from alexdoor_xas.envs.door_task.door_inspection import DoorInspectionEnv
    from alexdoor_xas.qualification.verify_prepared import physics_check

    shutil.copytree(reference, output)
    original = DoorInspectionEnv._setup_scene  # noqa: SLF001

    def obstructed_setup(env):
        original(env)
        door = CASES[0]
        center = door.hinge + door.rotation(np.deg2rad(40)) @ np.array([0, -0.6 * door.width, 1.0])
        cube = UsdGeom.Cube.Define(env.sim.stage, "/World/InjectedObstruction")
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 0.2))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    with patch.object(DoorInspectionEnv, "_setup_scene", obstructed_setup):
        try:
            physics_check(output, args.device)
        except PreparationError as exc:
            require(
                "Contact obstructs free opening" in str(exc), f"Unexpected physics rejection: {exc}"
            )
            result = dict(status="detected", case="gpu_obstruction", reason=str(exc))
            write_json(output / "physics.json", result)
            return result
        raise AssertionError("Injected physical obstacle passed")


def main():
    results = []
    for index, door in enumerate(CASES):
        if args.case is not None and index != args.case:
            continue
        output = new_attempt(args.output, door.name.replace(".", "-"))
        print(f"PREPARATION {door.name}: {output}", flush=True)
        source, recipe = synthetic_source(door, output)
        target = output / "prepared"
        target.mkdir()
        result = normalize(source, recipe, target)
        hinge_frame = np.array(result["opening_to_hinge"])
        require(
            np.allclose(hinge_frame[:3, 2], [0, 0, door.sign])
            and np.isclose(np.linalg.det(hinge_frame[:3, :3]), 1),
            "Hinge transform must expose the actual positive joint axis without reflection",
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.formats:
        format_checks(Path(results[0]["usd"]).parent, args.output / "formats")
    if args.invalid:
        invalid_checks(Path(results[0]["usd"]).parent, args.output / "invalid")
    if args.physics:
        from alexdoor_xas.qualification.verify_prepared import physics_check, static_check

        for result in results:
            target = Path(result["usd"]).parent
            write_json(target / "static.json", static_check(target))
            write_json(target / "physics.json", physics_check(target, args.device))
        if args.invalid:
            runtime_obstruction(Path(results[0]["usd"]).parent, args.output / "gpu-obstruction")
    write_json(args.output / "summary.json", {"status": "pass", "results": results})


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
