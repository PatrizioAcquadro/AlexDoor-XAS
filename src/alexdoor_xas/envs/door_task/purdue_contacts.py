"""Contact ownership and distal-face classification on the imported rigid bodies."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.frames import quat_to_rot_matrix


@dataclass(frozen=True)
class DistalSurface:
    actor: str
    finger: str
    vertices: np.ndarray
    forward: np.ndarray

    def contains(self, point, normal, tolerance=0.003):
        """Convex hull inclusion, forward extremum, and opposing surface normal."""
        points = self.vertices
        if abs(np.dot(point, self.forward) - np.max(points @ self.forward)) > tolerance:
            return False
        if abs(np.dot(normal, self.forward)) < 0.7:
            return False
        triangles = points.reshape(-1, 3, 3)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-12
        normals = normals[keep] / lengths[keep, None]
        anchors = triangles[keep, 0]
        sign = np.sum(normals * (points.mean(axis=0) - anchors), axis=1) > 0
        normals[sign] *= -1
        return bool(np.all(np.sum(normals * (point - anchors), axis=1) <= tolerance))


def imported_surfaces(stage, robot_path, wrist_from_base):
    """Resolve collision meshes to their actual rigid owners after fixed-link merging."""
    from pxr import Usd, UsdGeom, UsdPhysics

    surfaces = []
    root = stage.GetPrimAtPath(robot_path)
    base = next(p for p in Usd.PrimRange(root) if p.GetName() == "RIGHT_GRIPPER_Y_LINK")
    base_world = (
        np.array(UsdGeom.Xformable(base).ComputeLocalToWorldTransform(0)).T @ wrist_from_base
    )
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if "right_" not in path or not prim.GetName().endswith("UMI_V1_CONTACT_CONVEX"):
            continue
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        actor = prim
        while actor and not actor.HasAPI(UsdPhysics.RigidBodyAPI):
            actor = actor.GetParent()
        if not actor:
            raise RuntimeError(f"No rigid owner for {path}")
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = mesh.GetFaceVertexCountsAttr().Get()
        triangles, offset = [], 0
        for count in counts:
            face = indices[offset : offset + count]
            triangles.extend(vertices[[face[0], face[i], face[i + 1]]] for i in range(1, count - 1))
            offset += count
        world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        actor_world = np.array(UsdGeom.Xformable(actor).ComputeLocalToWorldTransform(0)).T
        local = np.linalg.inv(actor_world) @ world
        points = np.array(triangles).reshape(-1, 3) @ local[:3, :3].T + local[:3, 3]
        forward = actor_world[:3, :3].T @ base_world[:3, 0]
        finger = "positive" if "right_WSG32_POSITIVE_JAW_LINK" in path else "negative"
        surfaces.append(DistalSurface(str(actor.GetPath()), finger, points, forward))
    if len(surfaces) != 2:
        raise RuntimeError(f"Expected two right distal surfaces, got {len(surfaces)}")
    return surfaces


class ContactObserver:
    """All robot/pedestal contacts; normal forces only, with exact actor ownership."""

    def __init__(
        self, sim, stage, robot_path, panel_path, pedestal_path, ground_path, wrist_from_base
    ):
        from pxr import UsdPhysics

        self.surfaces = imported_surfaces(stage, robot_path, wrist_from_base)
        self.panel_path = panel_path
        roots = (robot_path, pedestal_path, robot_path.rsplit("/", 1)[0] + "/Zed")
        self.actors = sorted(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and any((str(p.GetPath()) == r or str(p.GetPath()).startswith(r + "/")) for r in roots)
        )
        if not self.actors:
            raise RuntimeError("No contact sensing bodies")
        self.view = sim.physics_sim_view.create_rigid_contact_view(
            self.actors, max_contact_data_count=128 * len(self.actors)
        )
        self.bodies = sim.physics_sim_view.create_rigid_body_view(self.actors)
        self.actors = list(self.bodies.prim_paths)
        # View orders must match; both are constructed from exact paths.
        if list(self.view.sensor_paths) != self.actors:
            raise RuntimeError("Contact and pose sensor ownership orders differ")
        self.structural_pairs = set()
        pedestal_actors = [a for a in self.actors if a.startswith(pedestal_path + "/")]
        root_actors = [a for a in self.actors if a.endswith("/PEDESTAL_LINK")]
        for pedestal in pedestal_actors:
            self.structural_pairs.add(frozenset((pedestal, ground_path)))
            for root in root_actors:
                self.structural_pairs.add(frozenset((pedestal, root)))
        self.dt = sim.get_physics_dt()
        self.last = None

    def read(self):
        import warp as wp

        raw = self.view.get_raw_contact_data(self.dt)
        force, point, normal, separation, counts, starts, other = [v.numpy() for v in raw]
        counts, starts, other, force, separation = [
            v.reshape(-1) for v in (counts, starts, other, force, separation)
        ]
        if len(counts) != len(self.actors) or np.any(counts < 0) or np.any(starts < 0):
            raise RuntimeError("Invalid contact sensor ranges")
        if np.any(starts + counts > len(force)) or counts.sum() >= len(force):
            raise RuntimeError("Contact buffer incomplete or saturated")
        active = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts, strict=True)])
        ids = np.unique(other[active])
        paths = self.view.get_other_actor_paths_from_ids(
            wp.array(ids, dtype=wp.uint64, device="cpu")
        )
        mapping = dict(zip(ids.tolist(), map(str, paths), strict=True))
        poses = self.bodies.get_transforms().numpy()
        forces = {name: np.zeros(3) for name in ("negative", "positive")}
        contacts = []
        for index, actor in enumerate(self.actors):
            rotation = quat_to_rot_matrix(poses[index, 3:])
            for slot in range(starts[index], starts[index] + counts[index]):
                partner = mapping.get(int(other[slot]), "")
                if (
                    not partner
                    or not np.isfinite(
                        np.r_[force[slot], point[slot], normal[slot], separation[slot]]
                    ).all()
                ):
                    raise RuntimeError("Contact ownership or data unavailable")
                # Internal pairs appear from both bodies; retain one physical record.
                if partner in self.actors and actor > partner:
                    continue
                local_point = rotation.T @ (point[slot] - poses[index, :3])
                local_normal = rotation.T @ normal[slot]
                matches = [
                    s
                    for s in self.surfaces
                    if s.actor == actor and s.contains(local_point, local_normal)
                ]
                category = "forbidden"
                if frozenset((actor, partner)) in self.structural_pairs:
                    category = "support"
                elif partner == self.panel_path and len(matches) == 1:
                    category = matches[0].finger
                    forces[category] += force[slot] * normal[slot]
                contacts.append(
                    dict(
                        actor=actor,
                        partner=partner,
                        category=category,
                        point=point[slot].tolist(),
                        normal=normal[slot].tolist(),
                        force_n=float(force[slot]),
                        separation_m=float(separation[slot]),
                    )
                )
        self.last = dict(
            components="normal_only",
            contacts=contacts,
            normal_force_per_finger_w={k: v.tolist() for k, v in forces.items()},
            normal_force_w=sum(forces.values()).tolist(),
            forbidden=any(c["category"] == "forbidden" for c in contacts),
        )
        return self.last
