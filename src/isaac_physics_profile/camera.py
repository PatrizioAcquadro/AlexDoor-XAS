"""Keep Isaac RTX and measured poses aligned for a Steel-mounted camera."""

import numpy as np


class SteelMountedCamera:
    def __init__(self, sim, robot, camera, body_path):
        import torch
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation

        self.sim, self.robot, self.camera = sim, robot, camera
        self.body_name = body_path.rsplit("/", 1)[-1]
        stage = sim.stage
        xf = UsdGeom.XformCache()
        parent = np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))).T
        optic = np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(camera.cfg.prim_path))).T
        local = np.linalg.solve(parent, optic)
        rotation = local[:3, :3] / np.linalg.norm(local[:3, :3], axis=0)
        self.offset = torch.tensor(local[:3, 3], device=sim.device, dtype=torch.float32)[None]
        self.rotation = torch.tensor(
            Rotation.from_matrix(rotation).as_quat(),
            device=sim.device,
            dtype=torch.float32,
        )[None]

    def bind(self):
        from isaaclab_physx.sim.views.fabric_frame_view import FabricFrameView

        self.body = self.robot.data.body_names.index(self.body_name)
        view = FabricFrameView(
            self.camera.cfg.prim_path, device=self.camera.device, stage=self.camera.stage
        )
        if view.count != 1:
            raise ValueError("The mounted-camera bridge requires one camera")
        view._use_fabric = True
        self.camera._view = view

    def sync(self):
        from isaaclab.utils.math import combine_frame_transforms

        self.sim.forward()
        manager = self.sim.physics_manager
        manager._mark_transforms_dirty()
        manager.sync_transforms_to_usd()
        data = self.robot.data
        p, q = combine_frame_transforms(
            data.body_link_pos_w.torch[:, self.body],
            data.body_link_quat_w.torch[:, self.body],
            self.offset,
            self.rotation,
        )
        self.camera.set_world_poses(p, q, convention="opengl")
