"""Truth-assisted visibility evidence, isolated from observation/policy interfaces."""

import numpy as np


def read_visual_triangles(stage, root):
    """Read rendered mesh surfaces, excluding invisible and guide collision meshes."""
    from pxr import Usd, UsdGeom

    triangles = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == "invisible" or mesh.ComputePurpose() not in (
            "default",
            "render",
        ):
            continue
        counts = mesh.GetFaceVertexCountsAttr().Get()
        if counts is None or len(counts) == 0:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.array(mesh.ComputeLocalToWorldTransform(0)).T
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        start = 0
        for count in counts:
            face = indices[start : start + count]
            triangles.extend(points[[face[0], face[i], face[i + 1]]] for i in range(1, count - 1))
            start += count
    return np.asarray(triangles, dtype=float).reshape(-1, 3, 3)


def front_mesh_points(triangles, yz):
    """First +X intersection with actual triangles; retain visual recesses and holes."""
    a = triangles[:, 0]
    e, f = triangles[:, 1] - a, triangles[:, 2] - a
    det = e[:, 1] * f[:, 2] - e[:, 2] * f[:, 1]
    nonzero = np.abs(det) > 1e-12
    a, e, f, det = (x[nonzero] for x in (a, e, f, det))
    yz = np.asarray(yz, dtype=float).reshape(-1, 2)
    # Bound temporary arrays by ray/triangle pairs, not the number of rays alone.
    # Prepared meshes have at most 250,000 triangles; keep every ray and triangle.
    batch_size = max(1, 1_000_000 // max(len(a), 1))
    x = np.empty(len(yz))
    for start in range(0, len(yz), batch_size):
        batch = slice(start, start + batch_size)
        d = yz[batch, None] - a[None, :, 1:]
        u = (d[:, :, 0] * f[:, 2] - d[:, :, 1] * f[:, 1]) / det
        v = (e[:, 1] * d[:, :, 1] - e[:, 2] * d[:, :, 0]) / det
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        xs = a[:, 0] + u * e[:, 0] + v * f[:, 0]
        x[batch] = np.min(np.where(inside, xs, np.inf), axis=1, initial=np.inf)
    return np.column_stack([np.where(np.isfinite(x), x, np.nan), yz])


def prepared_samples(door, fraction, height):
    """Cache closed visual samples; physics still uses the original collision surfaces."""
    key = (fraction, height)
    if key not in door._visibility_samples:
        groups = {
            name: front_mesh_points(
                door.visual_triangles["Panel"],
                [(door.hinge[1] - door.sign * f * door.width, z) for f, z in locations],
            )
            for name, locations in panel_locations(door.width, fraction, height).items()
        }
        triangles = door.visual_triangles["Frame"]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        front = np.abs(normal[:, 0]) > 0.5 * np.linalg.norm(normal, axis=1)
        ys = np.unique(np.round(triangles[front].mean(axis=1)[:, 1], 6))
        points = front_mesh_points(
            triangles, [(y, z) for y in ys for z in np.linspace(height - 0.3, height + 0.3, 7)]
        )
        points = points[np.isfinite(points).all(axis=1)]
        groups["frame"] = points[np.linspace(0, len(points) - 1, min(len(points), 40), dtype=int)]
        door._visibility_samples[key] = groups
    return door._visibility_samples[key]


def visible_points(points, camera_position, camera_rotation, intrinsics, depth, tolerance=0.02):
    """Project world points and test actual optical-axis depth, including occlusion."""
    if not len(points):
        return np.zeros(0, dtype=bool)
    optical = (np.asarray(points) - camera_position) @ camera_rotation
    image = optical @ intrinsics.T
    h, w = depth.shape[:2]
    finite = np.isfinite(optical).all(-1) & (optical[:, 2] > 0)
    uv = np.zeros((len(points), 2), dtype=int)
    uv[finite] = np.rint(image[finite, :2] / image[finite, 2:]).astype(int)
    inside = finite & (uv[:, 0] >= 1) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 1)
    visible = np.zeros(len(points), dtype=bool)
    for i in np.flatnonzero(inside):
        x, y = uv[i]
        measured = depth[y - 1 : y + 2, x - 1 : x + 2].reshape(-1)
        visible[i] = np.any(
            np.isfinite(measured) & (measured > 0) & (np.abs(measured - optical[i, 2]) <= tolerance)
        )
    return visible


def panel_locations(width, fraction, height):
    return {
        "panel": [
            (f, z)
            for f in np.linspace(0.15, 0.85, 5)
            for z in np.linspace(height - 0.2, height + 0.2, 5)
        ],
        "contact_surround": [
            (fraction + dy / width, height + dz)
            for dy, dz in (
                (-0.06, -0.06),
                (-0.06, 0.06),
                (0.06, -0.06),
                (0.06, 0.06),
                (0, -0.09),
                (0, 0.09),
                (-0.09, 0),
                (0.09, 0),
            )
        ],
    }


def surface_groups(door, angle, fraction, height):
    def surface(f, z):
        try:
            return door.contact_pose(angle, f, z)[0]
        except ValueError:
            return np.full(3, np.nan)

    return {
        **{
            name: [surface(f, z) for f, z in locations]
            for name, locations in panel_locations(door.width, fraction, height).items()
        },
        "frame": [
            np.array([-0.06, side * (door.width / 2 + 0.045), z])
            for side in (-1, 1)
            for z in np.linspace(height - 0.3, height + 0.3, 7)
        ],
    }


def visibility_groups(door, angle, fraction, height):
    if not hasattr(door, "visual_triangles"):
        return surface_groups(door, angle, fraction, height)
    return {
        name: points
        if name == "frame"
        else (points - door.hinge) @ door.rotation(angle).T + door.hinge
        for name, points in prepared_samples(door, fraction, height).items()
    }


def measure_visibility(env, door, angle, fraction, height):
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.envs.door_task.door_push_purdue_env import tensor

    sample = env.capture.sample
    camera = env.camera.data
    position = tensor(camera.pos_w)[0].cpu().numpy()
    rotation = Rotation.from_quat(tensor(camera.quat_w_ros)[0].cpu().numpy()).as_matrix()
    intrinsics = tensor(camera.intrinsic_matrices)[0].cpu().numpy()
    depth = np.where(
        sample.valid_depth[0, ..., 0].cpu().numpy(),
        sample.depth_m[0, ..., 0].cpu().numpy(),
        np.nan,
    )
    groups = visibility_groups(door, angle, fraction, height)
    counts = {
        name: int(visible_points(points, position, rotation, intrinsics, depth).sum())
        for name, points in groups.items()
    }
    counts["passed"] = (
        counts["panel"] >= 6 and counts["contact_surround"] >= 2 and counts["frame"] >= 2
    )
    return counts
