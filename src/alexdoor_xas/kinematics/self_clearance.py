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
                self.shapes.append(
                    dict(
                        link=name,
                        group=group(name),
                        convex=convex,
                        moving=moving.get((name, index)),
                        sphere=sphere,
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
            shared = (
                set(model.shapes[a["moving"]]["levers"]) & set(model.shapes[b["moving"]]["levers"])
                if a["moving"] is not None and b["moving"] is not None
                else set()
            )
            self.pairs.append((i, j, shared))

    def query(self, matrices, joint_travel, *, relative_model_error_m):
        if not np.isfinite(relative_model_error_m) or relative_model_error_m < 0:
            raise ValueError("Unmeasured relative robot-model error")
        world = []
        for shape in self.shapes:
            t, c = matrices[shape["link"]], shape["convex"]
            sphere = shape["sphere"]
            world.append(
                (
                    c.points @ t[:3, :3].T + t[:3, 3],
                    c.normals @ t[:3, :3].T,
                    c.edges @ t[:3, :3].T,
                    None if sphere is None else (sphere[0] @ t[:3, :3].T + t[:3, 3], sphere[1]),
                )
            )
        extra_cache = {}
        cells = self.model.cells()
        minimum, pair = np.inf, None
        for i, j, shared in self.pairs:
            extra = []
            for index in (i, j):
                key = (index, tuple(sorted(shared)))
                if key not in extra_cache:
                    moved = self.shapes[index]["moving"]
                    allowance = 0.0
                    if moved is not None:
                        cell = cells[moved]
                        _, radius = self.model.world_ball(
                            cell,
                            matrices,
                            {
                                name: 0.0 if name in shared else value
                                for name, value in joint_travel.items()
                            },
                        )
                        allowance = radius - np.linalg.norm(cell.upper - cell.lower) / 2
                    extra_cache[key] = allowance
                extra.append(extra_cache[key])
            a, na, ea, sa = world[i]
            b, nb, eb, sb = world[j]
            reserve = sum(extra) + relative_model_error_m
            gap = float(np.max(np.maximum(b.min(0) - a.max(0), a.min(0) - b.max(0))))
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
