"""Prepared-door geometry in the canonical opening frame; no asset rewriting."""

from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from alexdoor_xas.assets.synthetic_door import SyntheticDoor


class PreparedAssetError(ValueError):
    """A loaded prepared asset contradicts its published structural contract."""


class PreparedDoor:
    def __init__(self, folder, record, recipe):
        self.folder = Path(folder).resolve()
        self._contacts = {}
        self.record = record
        self.recipe = recipe
        self.name = self.folder.name
        self.handedness = record["handedness"]
        if self.handedness not in ("left", "right"):
            raise ValueError("Unknown door handedness")
        self.sign = 1 if self.handedness == "left" else -1
        dimensions = record["dimensions_m"]
        self.width = float(dimensions["width_m"])
        self.height = float(dimensions["height_m"])
        self.thickness = float(dimensions["thickness_m"])
        self.hinge = np.asarray(record["hinge_m"], dtype=float)
        self.center = np.asarray(record["panel_center_m"], dtype=float)
        self.mechanical_stop = float(np.deg2rad(record["mechanical_limit_deg"]))
        self.usd = (self.folder / record["usd"]).resolve()
        if not self.usd.is_relative_to(self.folder / "prepared") or not self.usd.is_file():
            raise ValueError("Prepared USD is missing or outside its published folder")
        values = [
            self.width,
            self.height,
            self.thickness,
            self.mechanical_stop,
            *self.hinge,
            *self.center,
        ]
        if not np.isfinite(values).all() or min(values[:4]) <= 0:
            raise ValueError("Invalid prepared geometry")

    rotation = SyntheticDoor.rotation

    def load_stage(self, stage, root):
        """Read composed initial geometry before simulation starts."""
        from pxr import Usd, UsdGeom, UsdPhysics

        self.shapes = {name: [] for name in ("Panel", "Frame", "Handle")}
        self.leaf_shapes = []
        leaf = set(self.recipe.get("leaf_components", self.recipe["components"]["Panel"]))
        for name in self.shapes:
            body = stage.GetPrimAtPath(root + "/" + name)
            if not body:
                if name == "Handle":
                    continue
                raise PreparedAssetError(f"Missing {name} body")
            for prim in Usd.PrimRange(body):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    raise PreparedAssetError("Prepared colliders must be baked convex meshes")
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
                points = points @ matrix[:3, :3].T + matrix[:3, 3]
                self.shapes[name].append(points)
                components = prim.GetAttribute("b1:sourceComponents").Get() or []
                if name == "Panel" and leaf.intersection(components):
                    self.leaf_shapes.append(points)
        if not self.leaf_shapes or not self.shapes["Frame"]:
            raise PreparedAssetError("Missing leaf or frame collision geometry")
        self._leaf_planes = [ConvexHull(p).equations for p in self.leaf_shapes]
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root + "/Hinge"))
        if not joint or not np.isclose(
            joint.GetUpperLimitAttr().Get(), np.rad2deg(self.mechanical_stop), atol=1e-4
        ):
            raise PreparedAssetError("Prepared record and physical hinge stop disagree")
        self.damping = float(UsdPhysics.DriveAPI(joint.GetPrim(), "angular").GetDampingAttr().Get())

    def front_x(self, y, z):
        """First intersection along +X with the actual leaf convex union."""
        hits = []
        for planes in self._leaf_planes:
            a = planes[:, 0]
            b = planes[:, 1] * y + planes[:, 2] * z + planes[:, 3]
            flat = np.abs(a) < 1e-10
            if np.any(b[flat] > 1e-7):
                continue
            lower = np.max(-b[a < -1e-10] / a[a < -1e-10], initial=-np.inf)
            upper = np.min(-b[a > 1e-10] / a[a > 1e-10], initial=np.inf)
            if np.isfinite(lower) and lower <= upper + 1e-7:
                hits.append(lower)
        if not hits:
            raise ValueError("Prescribed point has no collidable leaf surface")
        return min(hits)

    def closed_contact(self, fraction, height):
        if not 0 < fraction < 1:
            raise ValueError("Contact fraction outside panel")
        key = (fraction, height)
        if key in self._contacts:
            return self._contacts[key]
        y = self.hinge[1] - self.sign * fraction * self.width
        if not self.center[2] - self.height / 2 <= height <= self.center[2] + self.height / 2:
            raise ValueError("Contact height outside panel")
        point = np.array([self.front_x(float(y), float(height)), y, height])
        self._contacts[key] = point
        return point

    def contact_pose(self, angle, fraction, height, normal_offset=0.0):
        point = self.closed_contact(fraction, height).copy()
        point[0] += normal_offset
        rotation = self.rotation(angle)
        return self.hinge + rotation @ (point - self.hinge), rotation

    def footprint_inside(self, points, angle):
        closed = (points - self.hinge) @ self.rotation(angle) + self.hinge
        lo = self.center - np.array([self.thickness, self.width, self.height]) / 2
        hi = self.center + np.array([self.thickness, self.width, self.height]) / 2
        if not np.all((closed[:, 1:] >= lo[1:]) & (closed[:, 1:] <= hi[1:])):
            return False
        try:
            for point in closed:
                self.front_x(float(point[1]), float(point[2]))
        except ValueError:
            return False
        return True

    def collision_bounds(self, angle):
        rotation = self.rotation(angle)
        targets = []
        for name, shapes in self.shapes.items():
            for points in shapes:
                if name != "Frame":
                    points = (points - self.hinge) @ rotation.T + self.hinge
                targets.append((name.lower(), (points.min(0), points.max(0))))
        return targets

    @cached_property
    def frame_points(self):
        # Actual front-facing hull facet centroids, sampled deterministically.
        samples = []
        for points in self.shapes["Frame"]:
            hull = ConvexHull(points)
            faces = hull.simplices[hull.equations[:, 0] < -0.5]
            samples.extend(points[faces].mean(1))
        points = np.asarray(samples)
        return points[np.linspace(0, len(points) - 1, min(len(points), 40), dtype=int)]

    def pedestal_intersections(self, stage, pedestal_root):
        """Positive-volume initial overlap with the fixed pedestal, using convex SAT."""
        from itertools import product

        from pxr import Usd, UsdGeom, UsdPhysics

        from .convex_geometry import Convex, overlap

        witnesses = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(pedestal_root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
                points = np.array(list(product([-half, half], repeat=3)))
            else:
                raise ValueError("Unsupported pedestal collision shape; cannot establish clearance")
            matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            points = points @ matrix[:3, :3].T + matrix[:3, 3]
            pedestal = Convex(points)
            for name, shapes in self.shapes.items():
                for index, shape in enumerate(shapes):
                    if overlap(pedestal, Convex(shape), tolerance=0.002):
                        witnesses.append(
                            dict(
                                pedestal_collider=str(prim.GetPath()),
                                door_body=name,
                                door_collider_index=index,
                                tolerance_m=0.002,
                                pedestal_vertices=pedestal.points.tolist(),
                                door_vertices=shape.tolist(),
                            )
                        )
        return witnesses
