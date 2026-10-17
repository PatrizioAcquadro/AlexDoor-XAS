"""Calibrated robot geometry for observed occlusion and conservative sweep covers."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.b1 import checked_pose
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.assets.purdue import origin_matrix, read_stl_vertices
from alexdoor_xas.perception.geometry import deproject


def mesh_points(path):
    path = Path(path)
    if path.suffix.lower() == ".stl":
        points = read_stl_vertices(path)
    elif path.suffix.lower() == ".obj":
        points = np.array(
            [
                [float(x) for x in line.split()[1:4]]
                for line in path.read_text().splitlines()
                if line.startswith("v ")
            ]
        )
    else:
        raise ValueError("Unsupported calibrated robot mesh")
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Invalid calibrated robot mesh")
    return points


@dataclass(frozen=True)
class CollisionCell:
    shape: int
    lower: np.ndarray
    upper: np.ndarray
    path: tuple = ()


class RobotGeometry:
    """Robot CAD and measured joints only; no environment assets or task targets.

    Convex collision cells cover every shape volume, not just sampled vertices.
    World balls add supplied joint-travel bounds over the entire arm chain.
    Occlusion uses conservative link-local bounds of both visual/collision meshes.
    """

    def __init__(self, urdf, moving_joints):
        self.urdf = Path(urdf)
        xml = ET.parse(self.urdf).getroot()
        self.parents = {j.find("child").get("link"): j for j in xml.findall("joint")}
        self.links = tuple(e.get("name") for e in xml.findall("link"))
        self.moving_joints = tuple(moving_joints)
        self.occluders, self.shapes = [], []
        for link in xml.findall("link"):
            name = link.get("name")
            chain, ancestor = [], name
            while ancestor in self.parents:
                joint = self.parents[ancestor]
                chain.append(joint)
                ancestor = joint.find("parent").get("link")
            chain.reverse()
            for kind in ("visual", "collision"):
                for index, element in enumerate(link.findall(kind)):
                    local = origin_matrix(element.find("origin"))
                    mesh = element.find("geometry/mesh")
                    if mesh is not None:
                        points = mesh_points(self.urdf.parent / mesh.get("filename"))
                        points = points * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                        points = points @ local[:3, :3].T + local[:3, 3]
                        lo, hi = points.min(0), points.max(0)
                    else:
                        sphere = element.find("geometry/sphere")
                        if sphere is None:
                            raise ValueError("Unsupported calibrated robot primitive")
                        radius = float(sphere.get("radius"))
                        if not np.isfinite(radius) or radius <= 0:
                            raise ValueError("Invalid calibrated robot sphere")
                        lo, hi = local[:3, 3] - radius, local[:3, 3] + radius
                        points = None
                    self.occluders.append((name, lo, hi))
                    if kind != "collision" or not any(
                        j.get("name") in self.moving_joints for j in chain
                    ):
                        continue
                    if points is None:
                        raise ValueError("Moving robot collision shape must be a convex mesh")
                    levers = {}
                    for k, joint in enumerate(chain):
                        if joint.get("name") in self.moving_joints:
                            levers[joint.get("name")] = (
                                1.0
                                if joint.get("type") == "prismatic"
                                else np.linalg.norm(points, axis=1).max()
                                + sum(
                                    np.linalg.norm(origin_matrix(j.find("origin"))[:3, 3])
                                    for j in chain[k + 1 :]
                                )
                            )
                    self.shapes.append(
                        dict(
                            link=name,
                            index=index,
                            lower=lo,
                            upper=hi,
                            planes=ConvexHull(points).equations,
                            levers=levers,
                        )
                    )
        merged = {}
        for name, lo, hi in self.occluders:
            previous = merged.get(name, (lo, hi))
            merged[name] = (np.minimum(previous[0], lo), np.maximum(previous[1], hi))
        self.occluders = [(name, lo, hi) for name, (lo, hi) in merged.items()]
        if not self.shapes:
            raise ValueError("Missing moving robot collision geometry")

    def transforms(self, positions, root_world):
        root = np.asarray(root_world, dtype=float)
        if root.shape != (4, 4) or not np.allclose(root[3], [0, 0, 0, 1]):
            raise ValueError("Invalid calibrated robot root")
        checked_pose(ObjectFrame(root[:3, 3], root[:3, :3]))
        matrices = {}

        def forward(link):
            if link in matrices:
                return matrices[link]
            if link not in self.parents:
                matrices[link] = root
                return root
            joint = self.parents[link]
            base = forward(joint.find("parent").get("link")) @ origin_matrix(joint.find("origin"))
            motion, kind = np.eye(4), joint.get("type")
            if kind != "fixed":
                value = float(positions.get(joint.get("name"), np.nan))
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                if (
                    not np.isfinite(value)
                    or axis.shape != (3,)
                    or not np.isfinite(axis).all()
                    or not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-8)
                ):
                    raise ValueError("Missing calibrated robot joint state")
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * value
                else:
                    raise ValueError("Unsupported calibrated robot joint")
            matrices[link] = base @ motion
            return matrices[link]

        for link in self.links:
            forward(link)
        return matrices

    def occluded_pixels(
        self, sensor, positions, root_world, calibration_bound, *, pixel_domain=None
    ):
        """Remove known robot returns; excluded material remains unobserved."""
        depth = np.asarray(sensor["depth_m"])[..., 0]
        valid = np.asarray(sensor["valid_depth"])[..., 0]
        if pixel_domain is not None:
            domain = np.asarray(pixel_domain)
            if domain.dtype != bool or domain.shape != valid.shape:
                raise ValueError("Invalid robot occlusion pixel domain")
            valid = valid & domain
        v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
        points = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
        bounds = np.asarray(calibration_bound, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or (bounds < 0).any():
            raise ValueError("Unmeasured robot occlusion error")
        error = bounds[0] + 2 * np.linalg.norm(
            points - sensor["camera_world"][:3, 3], axis=1
        ) * np.sin(min(bounds[1], np.pi) / 2)
        matrices = self.transforms(positions, root_world)
        inside = np.zeros(len(points), bool)
        if not len(points):
            return np.zeros(depth.shape, bool)
        camera = sensor["camera_world"]
        maximum_error = float(error.max())
        signs = np.array(list(product((0, 1), repeat=3)))
        for name, lo, hi in self.occluders:
            transform = matrices[name]
            corners = np.where(signs, hi + maximum_error, lo - maximum_error)
            world = corners @ transform[:3, :3].T + transform[:3, 3]
            optical = (world - camera[:3, 3]) @ camera[:3, :3]
            minimum_z, maximum_z = optical[:, 2].min(), optical[:, 2].max()
            possible = (~inside) & (depth[v, u] >= max(0.0, minimum_z)) & (depth[v, u] <= maximum_z)
            if minimum_z > 1e-9:
                image = optical @ sensor["intrinsics"].T
                uv = image[:, :2] / image[:, 2, None]
                lower, upper = uv.min(0) - 1, uv.max(0) + 1
                possible &= (u >= lower[0]) & (u <= upper[0]) & (v >= lower[1]) & (v <= upper[1])
            ids = np.flatnonzero(possible)
            if not len(ids):
                continue
            local = (points[ids] - transform[:3, 3]) @ transform[:3, :3]
            inside[ids] = ((local >= lo - error[ids, None]) & (local <= hi + error[ids, None])).all(
                1
            )
        mask = np.zeros(depth.shape, bool)
        mask[v[inside], u[inside]] = True
        return mask

    def cells(self):
        return tuple(CollisionCell(i, s["lower"], s["upper"]) for i, s in enumerate(self.shapes))

    def subdivide(self, cell):
        midpoint = (cell.lower + cell.upper) / 2
        planes = self.shapes[cell.shape]["planes"]
        cells = []
        for index, signs in enumerate(product((0, 1), repeat=3)):
            lo, hi = np.where(signs, midpoint, cell.lower), np.where(signs, cell.upper, midpoint)
            center, half = (lo + hi) / 2, (hi - lo) / 2
            if not np.any(
                planes[:, :3] @ center + planes[:, 3]
                > np.abs(planes[:, :3]) @ half
                + 32 * np.finfo(float).eps * max(1.0, np.linalg.norm(center))
            ):
                cells.append(CollisionCell(cell.shape, lo, hi, cell.path + (index,)))
        return tuple(cells)

    def velocity_sweep(self, positions, root_world, velocity, acceleration, duration_s):
        """Prepare whole-volume balls for coupled velocity and bounded acceleration.

        The nominal constant-velocity center curve lies within its endpoint chord
        plus a conservative second-derivative reserve. A joint acceleration box
        covers every departure over the entire interval. Kinematics are computed
        once per proposed interval, then reused for every adaptive collision cell.
        Stopping, scene/model error and bound-domain checks belong to the caller.
        """
        velocity = np.asarray(velocity, float)
        acceleration = np.asarray(acceleration, float)
        n = len(self.moving_joints)
        if (
            velocity.shape != (n,)
            or acceleration.shape != (n,)
            or not np.isfinite(np.r_[velocity, acceleration, duration_s]).all()
            or (acceleration < 0).any()
            or duration_s <= 0
        ):
            raise ValueError("Unmeasured robot velocity/acceleration envelope")
        kinds = {j.get("name"): j.get("type") for j in self.parents.values()}
        if any(kinds[name] not in ("revolute", "continuous") for name in self.moving_joints):
            raise ValueError("Velocity sweep requires a revolute moving chain")
        positions = dict(positions)
        delta = velocity * duration_s
        end = dict(positions)
        end.update(
            (name, positions[name] + value)
            for name, value in zip(self.moving_joints, delta, strict=True)
        )
        start_m, end_m = self.transforms(positions, root_world), self.transforms(end, root_world)
        travel = np.abs(delta)
        error = 0.5 * acceleration * duration_s**2
        weights = travel.sum() * travel / 8 + 2 * np.sin(np.minimum(error, np.pi) / 2)
        levers = []
        for shape in self.shapes:
            offsets, ancestors, suffix = np.zeros(n), np.zeros(n), 0.0
            link = shape["link"]
            while link in self.parents:
                joint = self.parents[link]
                name = joint.get("name")
                if name in self.moving_joints:
                    index = self.moving_joints.index(name)
                    offsets[index], ancestors[index] = suffix, 1
                suffix += np.linalg.norm(origin_matrix(joint.find("origin"))[:3, 3])
                if joint.get("type") == "prismatic":
                    suffix += abs(positions[name])
                link = joint.find("parent").get("link")
            levers.append((offsets, ancestors))

        def ball(cell):
            local = (cell.lower + cell.upper) / 2
            link = self.shapes[cell.shape]["link"]
            c0 = local @ start_m[link][:3, :3].T + start_m[link][:3, 3]
            c1 = local @ end_m[link][:3, :3].T + end_m[link][:3, 3]
            offsets, ancestors = levers[cell.shape]
            # Downstream segment lengths bound the center derivatives even when
            # a covering cell's center lies outside the actual convex hull.
            reserve = (offsets + ancestors * np.linalg.norm(local)) @ weights
            radius = (
                np.linalg.norm(cell.upper - cell.lower) / 2 + np.linalg.norm(c1 - c0) / 2 + reserve
            )
            return (c0 + c1) / 2, float(radius)

        return ball

    def world_ball(self, cell, matrices, joint_travel):
        values = np.array([joint_travel.get(name, np.nan) for name in self.moving_joints])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Unmeasured robot sweep travel")
        shape = self.shapes[cell.shape]
        transform = matrices[shape["link"]]
        center = ((cell.lower + cell.upper) / 2) @ transform[:3, :3].T + transform[:3, 3]
        radius = float(np.linalg.norm(cell.upper - cell.lower) / 2)
        extra = 0.0
        # Telescope proximal-to-distal joint changes. Before each change its
        # untouched suffix and axis share a rigid prefix transform, so their
        # initial perpendicular distance still bounds the rotation lever.
        # This covers every point in the cell, including simultaneous motion;
        # a whole-chain length is unnecessary for each small collision cell.
        link = shape["link"]
        while link in self.parents:
            joint = self.parents[link]
            parent = joint.find("parent").get("link")
            name = joint.get("name")
            if name in shape["levers"]:
                travel = joint_travel[name]
                if joint.get("type") == "prismatic":
                    extra += travel
                else:
                    frame = matrices[parent] @ origin_matrix(joint.find("origin"))
                    axis = frame[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    offset = center - frame[:3, 3]
                    lever = np.linalg.norm(offset - axis * (offset @ axis)) + radius
                    extra += 2 * lever * np.sin(min(travel, np.pi) / 2)
            link = parent
        return center, radius + float(extra)
