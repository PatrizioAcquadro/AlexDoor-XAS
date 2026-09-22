"""Truth-assisted visibility evidence, isolated from observation/policy interfaces."""

import numpy as np


def visible_points(points, camera_position, camera_rotation, intrinsics, depth, tolerance=0.02):
    """Project world points and test actual optical-axis depth, including occlusion."""
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


def measure_visibility(env, door, angle, fraction, height):
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.envs.door_task.door_push_purdue_env import tensor

    sample = env.capture.sample
    camera = env.camera.data
    position = tensor(camera.pos_w)[0].cpu().numpy()
    rotation = Rotation.from_quat(tensor(camera.quat_w_ros)[0].cpu().numpy()).as_matrix()
    intrinsics = tensor(camera.intrinsic_matrices)[0].cpu().numpy()
    depth = sample.depth_m[0, ..., 0].cpu().numpy()
    groups = {
        "panel": [
            door.contact_pose(angle, f, z)[0]
            for f in np.linspace(0.15, 0.85, 5)
            for z in np.linspace(height - 0.2, height + 0.2, 5)
        ],
        "contact_surround": [
            door.contact_pose(angle, fraction + dy / door.width, height + dz)[0]
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
        "frame": [
            np.array([-0.06, side * (door.width / 2 + 0.045), z])
            for side in (-1, 1)
            for z in np.linspace(height - 0.3, height + 0.3, 7)
        ],
    }
    counts = {
        name: int(visible_points(points, position, rotation, intrinsics, depth).sum())
        for name, points in groups.items()
    }
    counts["passed"] = (
        counts["panel"] >= 6 and counts["contact_surround"] >= 2 and counts["frame"] >= 2
    )
    return counts
