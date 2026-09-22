"""Conservative robot/door clearance bounds; raw contacts remain the validity gate."""

import itertools

import numpy as np


class DoorClearance:
    def __init__(self, env):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.env = env
        stage = env.sim.stage
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.guide,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.render,
            ],
            False,
            True,
        )
        self.local = {}
        for actor in env.contacts.actors:
            prim = stage.GetPrimAtPath(actor)
            world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            corners = []
            for collider in Usd.PrimRange(prim):
                if not collider.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                # Stop at a separately simulated descendant actor.
                owner = collider
                while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                if str(owner.GetPath()) != actor:
                    continue
                bounds = cache.ComputeWorldBound(collider).ComputeAlignedRange()
                if bounds.IsEmpty():
                    continue
                lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
                points = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
                corners.extend((points - world[:3, 3]) @ world[:3, :3])
            if corners:
                self.local[actor] = np.array(corners)
        if not self.local:
            raise RuntimeError("No imported collision bounds for clearance")

    def measure(self, door, angle):
        from scipy.spatial.transform import Rotation

        poses = self.env.contacts.bodies.get_transforms().numpy()
        panel = door.panel_rectangle(angle)
        panel_bounds = (np.r_[panel.min(0), 0.01], np.r_[panel.max(0), door.height + 0.01])
        handle_c = door.hinge + door.rotation(angle) @ np.array(
            [-0.06, -door.sign * 0.85 * door.width, 1.05]
        )
        half = np.abs(door.rotation(angle)) @ np.array([0.04, 0.06, 0.0175])
        targets = [("panel", panel_bounds), ("handle", (handle_c - half, handle_c + half))]
        for side in (-1, 1):
            c = np.array([0, side * (door.width / 2 + 0.045), door.height / 2])
            half = np.array([0.06, 0.04, door.height / 2])
            targets.append(("frame", (c - half, c + half)))
        c = np.array([0, 0, door.height + 0.06])
        half = np.array([0.06, (door.width + 0.17) / 2, 0.04])
        targets.append(("frame", (c - half, c + half)))
        distal_actors = {surface.actor for surface in self.env.contacts.surfaces}
        minimum = float("inf")
        for actor, pose in zip(self.env.contacts.actors, poses, strict=True):
            if actor not in self.local:
                continue
            points = self.local[actor] @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
            lo, hi = points.min(0), points.max(0)
            for name, (other_lo, other_hi) in targets:
                if name == "panel" and actor in distal_actors:
                    continue
                gap = np.maximum(np.maximum(lo - other_hi, other_lo - hi), 0)
                minimum = min(minimum, float(np.linalg.norm(gap)))
        return minimum
