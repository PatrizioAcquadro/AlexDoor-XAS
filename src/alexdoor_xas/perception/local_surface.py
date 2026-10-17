"""Local rigid pose constrained by measured plane/edge geometry and native pairs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfacePose:
    camera_from_map: np.ndarray
    registration_origin_map: np.ndarray
    position_bound_m: float
    rotation_bound_rad: float
    normal_rotation_bound_rad: float
    pairs: int


def _angle(a, b):
    return float(np.arccos(np.clip(a @ b, -1, 1)))


def observed_basis(surface, point_error_m):
    """Bound a plane/edge basis using finite baselines, not fit residual alone.

    The caller must justify the metric point-error bound. A rigid planar surface
    and a straight observed silhouette are explicit model assumptions. Missing or
    poorly conditioned evidence remains unavailable; gravity only fixes edge sign.
    """
    if not np.isfinite(point_error_m) or point_error_m <= 0:
        raise ValueError("missing_metric_point_error")
    points = np.asarray(surface.points, float)
    normal = np.asarray(surface.normal, float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 5
        or not np.isfinite(points).all()
        or normal.shape != (3,)
        or not np.isfinite(normal).all()
        or abs(np.linalg.norm(normal) - 1) > 1e-6
    ):
        raise ValueError("invalid_observed_plane")
    # Select a wide triangle using observed geometry only. Every selected vertex
    # retains the complete point/model error; a tiny fitting residual is not it.
    a = points[np.linalg.norm(points - points[0], axis=1).argmax()]
    b = points[np.linalg.norm(points - a, axis=1).argmax()]
    baseline = b - a
    c = points[np.linalg.norm(np.cross(points - a, baseline), axis=1).argmax()]
    v, w = b - a, c - a
    cross = np.cross(v, w)
    area = np.linalg.norm(cross)
    error = point_error_m + float(np.max(np.abs(points @ normal - surface.offset)))
    perturbation = 2 * error * (np.linalg.norm(v) + np.linalg.norm(w)) + 4 * error**2
    if area <= perturbation:
        raise ValueError("unconditioned_observed_plane")
    triangle_normal = cross / area
    if triangle_normal @ normal < 0:
        triangle_normal = -triangle_normal
    normal_bound = _angle(normal, triangle_normal) + np.arcsin(perturbation / area)
    alternatives = []
    for key, values in surface.edge_points.items():
        if not key.startswith("width_"):
            continue
        edge = np.asarray(values, float)
        if edge.ndim != 2 or edge.shape[1:] != (3,) or len(edge) < 5:
            continue
        if not np.isfinite(edge).all():
            continue
        center = edge.mean(0)
        _, _, vt = np.linalg.svd(edge - center, full_matrices=False)
        direction = vt[0]
        direction *= 1 if direction[2] >= 0 else -1
        coordinate = edge @ direction
        chord = edge[coordinate.argmax()] - edge[coordinate.argmin()]
        length = np.linalg.norm(chord)
        line_residual = np.linalg.norm(np.cross(edge - center, direction), axis=1).max()
        edge_error = point_error_m + line_residual
        if length <= 2 * edge_error or direction[2] < 0.7:
            continue
        chord /= length
        edge_bound = _angle(direction, chord) + np.arcsin(2 * edge_error / length)
        # Orthogonalization sensitivity grows near a parallel normal/edge.
        cosine = abs(direction @ normal)
        combined = normal_bound + edge_bound
        if cosine + np.sin(min(combined, np.pi / 2)) >= 1:
            continue
        bound = 2 * combined / np.sqrt(1 - cosine**2)
        z = direction - normal * (normal @ direction)
        z /= np.linalg.norm(z)
        alternatives.append((bound, np.column_stack((normal, np.cross(z, normal), z))))
    if not alternatives:
        raise ValueError("missing_conditioned_observed_edge")
    bound, basis = min(alternatives, key=lambda item: item[0])
    if bound >= np.pi / 2:
        raise ValueError("unconditioned_observed_basis")
    return basis, float(bound), float(normal_bound)


def constrained_surface_pose(
    initial_surface,
    current_surface,
    initial_camera,
    current_camera,
    source_points,
    current_points,
    point_error_m,
):
    """Use all supplied measured native inliers; recompute their residual envelope.

    No native pose, plane residual, semantic label or favorable subset supplies a
    complete bound. Calibration, initial geometry, temporal growth and collision
    model disagreement remain separate contributions at the consumer.
    """
    initial_basis, initial_error, _ = observed_basis(initial_surface, point_error_m)
    current_basis, current_error, normal_error = observed_basis(current_surface, point_error_m)
    source, current = np.asarray(source_points, float), np.asarray(current_points, float)
    cameras = (np.asarray(initial_camera, float), np.asarray(current_camera, float))
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or source.shape != current.shape
        or len(source) < 5
        or not np.isfinite(source).all()
        or not np.isfinite(current).all()
        or any(c.shape != (4, 4) or not np.isfinite(c).all() for c in cameras)
    ):
        raise ValueError("missing_measured_surface_correspondences")
    rotation = cameras[1][:3, :3].T @ current_basis @ initial_basis.T @ cameras[0][:3, :3]
    translation = np.mean(current - source @ rotation.T, axis=0)
    residual = np.linalg.norm(source @ rotation.T + translation - current, axis=1)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return SurfacePose(
        pose,
        source.mean(0),
        float(residual.max() + point_error_m),
        min(np.pi, initial_error + current_error),
        normal_error,
        len(source),
    )
