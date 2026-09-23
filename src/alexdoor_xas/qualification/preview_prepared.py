"""Rendered views of a normalized door for manual local appearance review."""

from pathlib import Path

import numpy as np

from .preparation import require


def preview(attempt):
    import omni.usd
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux

    attempt = Path(attempt).resolve()
    context = omni.usd.get_context()
    context.new_stage()
    sim = SimulationContext(SimulationCfg(device="cuda:0", dt=1 / 120))
    stage = sim.stage
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/World/Door").GetPrim()
    root.GetReferences().AddReference(str(attempt / "door.usda"))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(400)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((-50, 20, 0))
    images, cameras = [], []
    for name, position in (("front", (-3, -2, 2.2)), ("rear", (3, 2, 2.2))):
        camera = UsdGeom.Camera.Define(stage, f"/World/Camera_{name}")
        camera.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(0, 0, 1.05), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        camera.CreateFocalLengthAttr(32)
        cameras.append(
            (
                name,
                Camera(
                    CameraCfg(
                        prim_path=str(camera.GetPath()),
                        width=640,
                        height=640,
                        data_types=["rgb"],
                        spawn=None,
                    )
                ),
            )
        )
    sim.reset()
    for name, sensor in cameras:
        print(f"Rendering {name}", flush=True)
        for _ in range(16):
            sim.render()
            sensor.update(0, force_recompute=True)
        pixels = sensor.data.output["rgb"]
        pixels = pixels.torch if hasattr(pixels, "torch") else pixels
        pixels = np.asarray(pixels[0].cpu())
        require(
            pixels.shape[:2] == (640, 640) and pixels[..., :3].std() > 5,
            "Empty/blank preview",
            status="unresolved",
            category="render",
        )
        target = attempt / f"preview-{name}.png"
        Image.fromarray(pixels).save(target)
        images.append(str(target))
    return {"status": "pass", "scope": "render_capture_requires_visual_review", "images": images}
