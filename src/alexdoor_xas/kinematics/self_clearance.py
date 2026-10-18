"""Complete robot CAD self-clearance with the importer's rigid-body pair semantics."""

import xml.etree.ElementTree as ET
from itertools import combinations, product

import numpy as np

from alexdoor_xas.assets.purdue import origin_matrix
from alexdoor_xas.kinematics.urdf_geometry import mesh_points
from alexdoor_xas.qualification.convex_geometry import Convex


class RobotSelfClearance:
    """Separate shared rigid motion from relative swept-shape uncertainty.

    Pair filters are supplied from the unchanged robot importer. Fixed collision
    descendants inherit their owning rigid body's filters, matching authored USD.
    The query returns a geometric lower bound, never a qualification flag.
    """

    def __init__(self, model, filter_pairs):
        self.model = model
        self.filters = {frozenset(pair) for pair in filter_pairs}

        def group(link):
            while link in model.parents and model.parents[link].get("type") == "fixed":
                link = model.parents[link].find("parent").get("link")
            return link

        adjacent = {
            frozenset((group(j.find("parent").get("link")), group(j.find("child").get("link"))))
            for j in model.parents.values()
            if j.get("type") != "fixed"
        }
        moving = {(s["link"], s["index"]): i for i, s in enumerate(model.shapes)}
        self.shapes = []
        for link in ET.parse(model.urdf).getroot().findall("link"):
            name = link.get("name")
            for index, element in enumerate(link.findall("collision")):
                local = origin_matrix(element.find("origin"))
                mesh = element.find("geometry/mesh")
                sphere = None
                if mesh is not None:
                    points = mesh_points(model.urdf.parent / mesh.get("filename"))
                    points *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                else:
                    radius = float(element.find("geometry/sphere").get("radius"))
                    points = np.array(list(product((-radius, radius), repeat=3)))
                    sphere = (local[:3, 3], radius)
                convex = Convex(points @ local[:3, :3].T + local[:3, 3])
                ancestors = set()
                ancestor = name
                while ancestor in model.parents:
                    joint = model.parents[ancestor]
                    if joint.get("type") != "fixed":
                        ancestors.add(joint.get("name"))
                    ancestor = joint.find("parent").get("link")
                self.shapes.append(
                    dict(
                        link=name,
                        group=group(name),
                        convex=convex,
                        moving=moving.get((name, index)),
                        sphere=sphere,
                        joints=ancestors,
                        center=(convex.points.min(0) + convex.points.max(0)) / 2,
                        radius=(
                            sphere[1]
                            if sphere is not None
                            else np.linalg.norm(convex.points.max(0) - convex.points.min(0)) / 2
                        ),
                    )
                )
        self.pairs = []
        for i, j in combinations(range(len(self.shapes)), 2):
            a, b = self.shapes[i], self.shapes[j]
            groups = frozenset((a["group"], b["group"]))
            if (
                a["moving"] is None
                and b["moving"] is None
                or len(groups) == 1
                or groups in adjacent
                or groups in self.filters
            ):
                continue
            shared = a["joints"] & b["joints"]
            self.pairs.append((i, j, shared))

    def query(self, matrices, joint_travel, *, relative_model_error_m):
        if not np.isfinite(relative_model_error_m) or relative_model_error_m < 0:
            raise ValueError("Unmeasured relative robot-model error")
        required = set().union(
            *(self.shapes[i]["joints"] | self.shapes[j]["joints"] for i, j, _ in self.pairs)
        )
        if any(
            name not in joint_travel
            or joint_travel[name] is None
            or not np.isfinite(joint_travel[name])
            or joint_travel[name] < 0
            for name in required
        ):
            raise ValueError("Unmeasured full-robot joint travel")
        world, bounds = [], []
        for shape in self.shapes:
            t, c = matrices[shape["link"]], shape["convex"]
            sphere = shape["sphere"]
            points = c.points @ t[:3, :3].T + t[:3, 3]
            bounds.append((points.min(0), points.max(0)))
            world.append(
                (
                    points,
                    c.normals @ t[:3, :3].T,
                    c.edges @ t[:3, :3].T,
                    None if sphere is None else (sphere[0] @ t[:3, :3].T + t[:3, 3], sphere[1]),
                )
            )
        extra_cache, joint_frames = {}, {}
        minimum, pair = np.inf, None
        for i, j, shared in self.pairs:
            extra = []
            for index in (i, j):
                key = (index, tuple(sorted(shared)))
                if key not in extra_cache:
                    shape = self.shapes[index]
                    t = matrices[shape["link"]]
                    center = shape["center"] @ t[:3, :3].T + t[:3, 3]
                    allowance = 0.0
                    link = shape["link"]
                    while link in self.model.parents:
                        joint = self.model.parents[link]
                        name = joint.get("name")
                        parent = joint.find("parent").get("link")
                        if name in shape["joints"] and name not in shared:
                            travel = joint_travel[name]
                            if joint.get("type") == "prismatic":
                                allowance += travel
                            else:
                                if name not in joint_frames:
                                    frame = matrices[parent] @ origin_matrix(joint.find("origin"))
                                    axis = frame[:3, :3] @ np.fromstring(
                                        joint.find("axis").get("xyz"), sep=" "
                                    )
                                    joint_frames[name] = frame[:3, 3], axis
                                joint_origin, axis = joint_frames[name]
                                offset = center - joint_origin
                                lever = (
                                    np.linalg.norm(offset - axis * (offset @ axis))
                                    + shape["radius"]
                                )
                                allowance += 2 * lever * np.sin(min(travel, np.pi) / 2)
                        link = parent
                    extra_cache[key] = allowance
                extra.append(extra_cache[key])
            a, na, ea, sa = world[i]
            b, nb, eb, sb = world[j]
            reserve = sum(extra) + relative_model_error_m
            alo, ahi = bounds[i]
            blo, bhi = bounds[j]
            gap = float(np.max(np.maximum(blo - ahi, alo - bhi)))
            if gap <= reserve:
                gap = max(gap, projection_gap(a, sa, b, sb, np.concatenate((na, nb))))
                if gap <= reserve:
                    cross = np.cross(ea[:, None], eb).reshape(-1, 3)
                    length = np.linalg.norm(cross, axis=1)
                    if (length > 1e-8).any():
                        gap = max(gap, projection_gap(a, sa, b, sb, cross[length > 1e-8]))
            margin = gap - reserve
            if margin < minimum:
                minimum, pair = margin, (self.shapes[i]["link"], self.shapes[j]["link"])
            if margin <= 0:
                break
        return float(minimum), pair


def projection_gap(a, sa, b, sb, axes):
    """Any normalized separating axis supplies a valid distance lower bound."""
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]

    def projection(points, sphere):
        if sphere is not None:
            center, radius = sphere
            return center @ axes.T - radius, center @ axes.T + radius
        values = points @ axes.T
        return values.min(0), values.max(0)

    alo, ahi = projection(a, sa)
    blo, bhi = projection(b, sb)
    return float(np.max(np.maximum(blo - ahi, alo - bhi)))
