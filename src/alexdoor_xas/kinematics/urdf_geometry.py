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


def stop_box_cover(lower, upper, radial_bound_m, maximum_radius_m, *, certify=None):
    """Cover the complete calibrated box/sphere intersection with finite balls."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(np.r_[lower, upper, radial_bound_m, maximum_radius_m]).all()
        or (lower > 0).any()
        or (upper < 0).any()
        or min(radial_bound_m, maximum_radius_m) <= 0
    ):
        raise ValueError("invalid_continuous_stop_box")
    pending, result = [(lower.copy(), upper.copy(), ())], []
    while pending:
        lo, hi, path = pending.pop()
        nearest = np.maximum(lo, np.minimum(np.zeros(3), hi))
        if np.linalg.norm(nearest) > radial_bound_m:
            continue
        radius = np.linalg.norm(hi - lo) / 2
        center = (lo + hi) / 2
        if certify is not None:
            if certify(center, float(radius), path):
                result.append((center, float(radius)))
                continue
            if radius <= maximum_radius_m:
                return None
        elif radius <= maximum_radius_m:
            result.append((center, float(radius)))
            continue
        if len(pending) + len(result) >= 4096:
            raise ValueError("continuous_stop_cover_budget_exhausted")
        axis = int(np.argmax(hi - lo))
        middle = (lo[axis] + hi[axis]) / 2
        left, right = hi.copy(), lo.copy()
        left[axis] = right[axis] = middle
        pending.extend(((lo, left, path + (0,)), (right, hi, path + (1,))))
    return tuple(result)


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
        self.joint_origins = {
            link: origin_matrix(joint.find("origin")) for link, joint in self.parents.items()
        }
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
            base = forward(joint.find("parent").get("link")) @ self.joint_origins[link]
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
        if getattr(sensor["depth_m"], "is_cuda", False):
            return self._occluded_pixels_cuda(
                sensor, positions, root_world, calibration_bound, pixel_domain=pixel_domain
            )
        depth = np.asarray(sensor["depth_m"])[..., 0]
        valid = np.asarray(sensor["valid_depth"])[..., 0]
        if pixel_domain is not None:
            domain = np.asarray(pixel_domain)
            if domain.dtype != bool or domain.shape != valid.shape:
                raise ValueError("Invalid robot occlusion pixel domain")
            valid = valid & domain
        v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
        observed_depth = depth[v, u]
        points = deproject(
            observed_depth, np.c_[u, v], sensor["intrinsics"], sensor["camera_world"]
        )
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
            possible = (
                (~inside) & (observed_depth >= max(0.0, minimum_z)) & (observed_depth <= maximum_z)
            )
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

    def _occluded_pixels_cuda(
        self, sensor, positions, root_world, calibration_bound, *, pixel_domain=None
    ):
        """Same complete local volume membership; four-shape batches bound VRAM."""
        import torch

        depth, valid = sensor["depth_m"][..., 0], sensor["valid_depth"][..., 0]
        if pixel_domain is not None:
            domain = torch.as_tensor(pixel_domain, device=depth.device)
            if domain.dtype != torch.bool or domain.shape != valid.shape:
                raise ValueError("Invalid robot occlusion pixel domain")
            valid = valid & domain
        bounds = np.asarray(calibration_bound, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or (bounds < 0).any():
            raise ValueError("Unmeasured robot occlusion error")
        matrices = self.transforms(positions, root_world)
        with torch.inference_mode():
            v, u = torch.nonzero(valid & torch.isfinite(depth) & (depth > 0), as_tuple=True)
            result = torch.zeros(depth.shape, dtype=torch.bool, device=depth.device)
            if not len(v):
                return result.cpu().numpy()
            k = torch.as_tensor(sensor["intrinsics"], dtype=torch.float64, device=depth.device)
            camera = torch.as_tensor(
                sensor["camera_world"], dtype=torch.float64, device=depth.device
            )
            pixels = torch.stack((u, v, torch.ones_like(u)), dim=1).to(torch.float64)
            rays = pixels @ torch.linalg.inv(k).T
            points = (rays * depth[v, u, None]) @ camera[:3, :3].T + camera[:3, 3]
            error = bounds[0] + 2 * torch.linalg.vector_norm(
                points - camera[:3, 3], dim=1
            ) * np.sin(min(bounds[1], np.pi) / 2)
            inside = torch.zeros(len(points), dtype=torch.bool, device=depth.device)
            for start in range(0, len(self.occluders), 4):
                shapes = self.occluders[start : start + 4]
                transforms = torch.as_tensor(
                    np.stack([matrices[name] for name, _, _ in shapes]),
                    dtype=torch.float64,
                    device=depth.device,
                )
                lower = torch.as_tensor(
                    np.stack([lo for _, lo, _ in shapes]), dtype=torch.float64, device=depth.device
                )
                upper = torch.as_tensor(
                    np.stack([hi for _, _, hi in shapes]), dtype=torch.float64, device=depth.device
                )
                local = torch.bmm(points[None] - transforms[:, None, :3, 3], transforms[:, :3, :3])
                contained = (local >= lower[:, None] - error[None, :, None]) & (
                    local <= upper[:, None] + error[None, :, None]
                )
                inside |= contained.all(dim=2).any(dim=0)
            result[v[inside], u[inside]] = True
            return result.cpu().numpy()

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

    def refine_normal_cover(self, cell, matrices, normal, margin_m, minimum_cell_m):
        """Tighten a loose half-space cover; children require fresh full proofs.

        A cell's enclosing ball can extend farther toward a plane than its box.
        Spend extra spatial work when that geometric overreach exceeds the proved
        margin. This is a refinement rule, never a new separation criterion.
        """
        if np.max(cell.upper - cell.lower) <= minimum_cell_m:
            return ()
        half = (cell.upper - cell.lower) / 2
        rotation = matrices[self.shapes[cell.shape]["link"]][:3, :3]
        projection = np.abs(rotation.T @ normal) @ half
        overreach = max(0.0, float(np.linalg.norm(half) - projection))
        return self.subdivide(cell) if margin_m <= overreach else ()

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
                suffix += np.linalg.norm(self.joint_origins[link][:3, 3])
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

    def convex_velocity_sweep(self, positions, root_world, velocity, acceleration, duration_s):
        """Enclose complete cells by a linear-motion hull and Taylor radius.

        All local vertices, time endpoints and acceleration-box corners are kept.
        For a revolute chain every mixed point derivative is bounded by a full
        downstream lever L; the norm remainder is L * sum(axis_angle_reach)**2 / 2.
        This motion enclosure excludes stopping and model/scene error; callers
        must add those bounds and independently prove the resulting full volume.
        """
        velocity, acceleration = np.asarray(velocity, float), np.asarray(acceleration, float)
        n = len(self.moving_joints)
        if (
            velocity.shape != (n,)
            or acceleration.shape != (n,)
            or not np.isfinite(np.r_[velocity, acceleration, duration_s]).all()
            or (acceleration < 0).any()
            or duration_s <= 0
        ):
            raise ValueError("Unmeasured robot velocity/acceleration envelope")
        joints = {j.get("name"): (link, j) for link, j in self.parents.items()}
        if any(
            joints[name][1].get("type") not in ("revolute", "continuous")
            for name in self.moving_joints
        ):
            raise ValueError("Velocity sweep requires a revolute moving chain")
        matrices = self.transforms(positions, root_world)
        axes, origins = [], []
        for name in self.moving_joints:
            link, joint = joints[name]
            frame = matrices[joint.find("parent").get("link")] @ self.joint_origins[link]
            axes.append(frame[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" "))
            origins.append(frame[:3, 3])
        axes, origins = np.asarray(axes), np.asarray(origins)
        axis_lengths = np.linalg.norm(axes, axis=1)
        error = 0.5 * acceleration * duration_s**2
        reach = np.abs(velocity) * duration_s + error
        acceleration_corners = np.array(list(product((-1, 1), repeat=n))) * error
        corners = np.array(list(product((False, True), repeat=3)))
        levers = []
        for shape in self.shapes:
            offsets, ancestors, suffix = np.zeros(n), np.zeros(n, bool), 0.0
            link = shape["link"]
            while link in self.parents:
                joint = self.parents[link]
                name = joint.get("name")
                if name in self.moving_joints:
                    index = self.moving_joints.index(name)
                    offsets[index], ancestors[index] = suffix, True
                suffix += np.linalg.norm(self.joint_origins[link][:3, 3])
                if joint.get("type") == "prismatic":
                    suffix += abs(positions[name])
                link = joint.find("parent").get("link")
            levers.append((offsets, ancestors))

        def convex(cell):
            local = np.where(corners, cell.upper, cell.lower)
            transform = matrices[self.shapes[cell.shape]["link"]]
            world = local @ transform[:3, :3].T + transform[:3, 3]
            offsets, ancestors = levers[cell.shape]
            jacobian = np.cross(axes[None], world[:, None] - origins[None])
            jacobian[:, ~ancestors] = 0
            nominal = np.einsum("vjk,j->vk", jacobian, velocity) * duration_s
            uncertain = np.einsum("sj,vjk->vsk", acceleration_corners, jacobian)
            vertices = (
                world[:, None, None, :]
                + np.array([0.0, 1.0])[None, :, None, None] * nominal[:, None, None, :]
                + uncertain[:, None, :, :]
            ).reshape(-1, 3)
            lever = np.max((offsets + np.linalg.norm(local, axis=1).max()) * ancestors)
            remainder = 0.5 * lever * (reach[ancestors] * axis_lengths[ancestors]).sum() ** 2
            return vertices, float(remainder)

        return convex

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
                    frame = matrices[parent] @ self.joint_origins[link]
                    axis = frame[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    offset = center - frame[:3, 3]
                    lever = np.linalg.norm(offset - axis * (offset @ axis)) + radius
                    extra += 2 * lever * np.sin(min(travel, np.pi) / 2)
            link = parent
        return center, radius + float(extra)

    def point_advance_bound(
        self, link, matrices, points, velocity_step, acceleration_step, normal, normal_cone
    ):
        """Bound each point's inward advance under the calibrated joint interval."""
        nominal = np.zeros_like(points)
        acceleration = np.zeros(len(points))
        travel = np.zeros(len(points))
        angle = 0.0
        lever = np.linalg.norm(points - matrices[link][:3, 3], axis=1)
        max_lever = lever.copy()
        while link in self.parents:
            joint = self.parents[link]
            parent = joint.find("parent").get("link")
            name = joint.get("name")
            lever += np.linalg.norm(self.joint_origins[link][:3, 3])
            if name in self.moving_joints:
                if joint.get("type") not in ("revolute", "continuous"):
                    raise ValueError("Point advance requires a revolute moving chain")
                frame = matrices[parent] @ self.joint_origins[link]
                axis = frame[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                jacobian = np.cross(axis, points - frame[:3, 3])
                reach = abs(velocity_step[name]) + acceleration_step[name]
                nominal += jacobian * velocity_step[name]
                acceleration += abs(jacobian @ normal) * acceleration_step[name]
                travel += np.linalg.norm(jacobian, axis=1) * reach
                angle += reach
                max_lever = lever.copy()
            link = parent
        # A full downstream lever bounds every mixed point derivative, as for
        # convex_velocity_sweep. The remainder covers every time in the interval.
        remainder = 0.5 * max_lever * angle**2
        direction_error = 2 * np.sin(min(normal_cone, np.pi) / 2) * (travel + remainder)
        return np.maximum(0.0, nominal @ normal) + acceleration + remainder + direction_error
