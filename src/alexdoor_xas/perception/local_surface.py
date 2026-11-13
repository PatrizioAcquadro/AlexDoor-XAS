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
    normal_position_bound_m: float
    normal_anchor_lever_m: float
    pairs: int
    normal_origin_world: np.ndarray
    normal_point_bound_m: float


def _angle(a, b):
    return float(np.arccos(np.clip(a @ b, -1, 1)))


def _orthogonal_rotation_bound(first, second):
    """Bound one rotation from angle bounds on two orthogonal unit axes.

    For rotation axis u and angle t, sin(a_i/2)^2 = sin(t/2)^2(1-u_i^2).
    Two orthogonal axes therefore bound sin(t/2)^2 by their sum, without
    assuming their measurement errors are independent.
    """
    sine = np.hypot(np.sin(first / 2), np.sin(second / 2))
    return float(2 * np.arcsin(min(1.0, sine)))


def _normal_span_bound(points, normal, first, error):
    """Bound normal tilt from two measured chords spanning the planar core.

    For the true unit normal m and either chord d, |m.d| <= 2*error.
    Writing m = tangent*p + c*normal gives |B*p| <= 2*error + |d.normal|.
    The largest norm over that two-dimensional box bounds sin(tilt), without
    independent errors. The caller must establish the original hemisphere.
    """
    first = first - normal * (first @ normal)
    length = np.linalg.norm(first)
    if length == 0:
        return np.pi / 2
    first /= length
    tangent = np.column_stack((first, np.cross(normal, first)))
    chords = []
    for direction in tangent.T:
        coordinate = points @ direction
        chords.append(points[coordinate.argmax()] - points[coordinate.argmin()])
    chords = np.asarray(chords)
    matrix = chords @ tangent
    if np.linalg.det(matrix) == 0:
        return np.pi / 2
    allowance = 2 * error + np.abs(chords @ normal)
    corners = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * allowance
    sine = np.linalg.norm(np.linalg.solve(matrix, corners.T).T, axis=1).max()
    return float(np.arcsin(min(1.0, sine)))


def observed_basis(surface, point_error_m, *, normal_points=None):
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
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm
    offset = surface.offset / normal_norm
    # Additional normal baselines must be current depth measurements of this
    # same tracked material. They never extend footprint or edge coverage.
    if normal_points is not None:
        points = np.asarray(normal_points, float)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < 5
            or not np.isfinite(points).all()
        ):
            raise ValueError("invalid_observed_normal_support")
    # Orientation uses a finite measured planar core at the declared point
    # resolution. The wider surface/relief coverage and its original tolerance
    # remain unchanged at material consumers. Never interpret all shallow relief
    # as either orientation noise or a smaller complete geometry bound.
    points = points[abs(points @ normal - offset) <= point_error_m]
    if len(points) < 5:
        raise ValueError("missing_measured_planar_core")
    # Each triangle independently bounds the same plane using the COMPLETE
    # core's point/model error. Choose the strongest certificate, never a
    # favorable residual subset. Axis extrema add bounded geometric alternatives
    # to the original seed; no metric error or acceptance threshold is reduced.
    error = point_error_m + float(np.max(np.abs(points @ normal - offset)))
    seeds = {0}
    for axis in range(3):
        seeds.update((int(points[:, axis].argmin()), int(points[:, axis].argmax())))
    normal_bounds = []
    for seed in sorted(seeds):
        a = points[np.linalg.norm(points - points[seed], axis=1).argmax()]
        b = points[np.linalg.norm(points - a, axis=1).argmax()]
        baseline = b - a
        c = points[np.linalg.norm(np.cross(points - a, baseline), axis=1).argmax()]
        v, w = b - a, c - a
        cross = np.cross(v, w)
        area = np.linalg.norm(cross)
        perturbation = 2 * error * (np.linalg.norm(v) + np.linalg.norm(w)) + 4 * error**2
        if area <= perturbation:
            continue
        triangle_normal = cross / area
        if triangle_normal @ normal < 0:
            triangle_normal = -triangle_normal
        bound = _angle(normal, triangle_normal) + np.arcsin(perturbation / area)
        if bound < np.pi / 2:
            # Retain the complete core's error and original triangle conditioning.
            # Independent measured spans can tighten its conservative cross bound.
            bound = min(bound, _normal_span_bound(points, normal, v, error))
        normal_bounds.append(bound)
    if not normal_bounds:
        raise ValueError("unconditioned_observed_plane")
    normal_bound = min(normal_bounds)
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
        old_bound = 2 * combined / np.sqrt(1 - cosine**2)
        # The fitted edge differs from its projected unit z axis by asin(cosine).
        # Keep both original axis errors; orthogonality is the existing rigid
        # plane/straight-edge model, not an additional precision assumption.
        edge_from_z = edge_bound + np.arcsin(min(cosine, 1.0))
        bound = min(old_bound, _orthogonal_rotation_bound(normal_bound, edge_from_z))
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
    *,
    map_anchor,
    current_normal_points=None,
):
    """Use all supplied measured native inliers; recompute their residual envelope.

    No native pose, plane residual, semantic label or favorable subset supplies a
    complete bound. Calibration, initial geometry, temporal growth and collision
    model disagreement remain separate contributions at the consumer.
    """
    initial_basis, initial_error, _ = observed_basis(initial_surface, point_error_m)
    current_basis, current_error, normal_error = observed_basis(
        current_surface, point_error_m, normal_points=current_normal_points
    )
    source, current = np.asarray(source_points, float), np.asarray(current_points, float)
    anchor = np.asarray(map_anchor, float)
    cameras = (np.asarray(initial_camera, float), np.asarray(current_camera, float))
    if (
        anchor.shape != (3,)
        or not np.isfinite(anchor).all()
        or source.ndim != 2
        or source.shape[1:] != (3,)
        or source.shape != current.shape
        or len(source) < 5
        or not np.isfinite(source).all()
        or not np.isfinite(current).all()
        or any(c.shape != (4, 4) or not np.isfinite(c).all() for c in cameras)
    ):
        raise ValueError("missing_measured_surface_correspondences")
    initial_anchor = cameras[0][:3, :3] @ anchor + cameras[0][:3, 3]
    initial_residual = np.max(
        np.abs(initial_surface.points @ initial_surface.normal - initial_surface.offset)
    )
    if (
        abs(initial_surface.normal @ initial_anchor - initial_surface.offset)
        > point_error_m + initial_residual
    ):
        raise ValueError("material_anchor_outside_observed_plane")
    rotation = cameras[1][:3, :3].T @ current_basis @ initial_basis.T @ cameras[0][:3, :3]
    translation = np.mean(current - source @ rotation.T, axis=0)
    # The initial material anchor lies on the observed reference plane. Its
    # normal coordinate is measured again by the current plane, independently
    # of native point translation. Tangential coordinates still use all pairs.
    world_anchor = cameras[1][:3, :3] @ (rotation @ anchor + translation) + cameras[1][:3, 3]
    correction = current_surface.offset - current_surface.normal @ world_anchor
    translation += cameras[1][:3, :3].T @ current_surface.normal * correction
    world_anchor += current_surface.normal * correction
    normal_origin = current_surface.points.mean(0)
    normal_lever = float(np.linalg.norm(world_anchor - normal_origin))
    normal_point_bound = float(
        point_error_m
        + np.max(np.abs(current_surface.points @ current_surface.normal - current_surface.offset))
    )
    normal_position = normal_point_bound + 2 * normal_lever * np.sin(min(normal_error, np.pi) / 2)
    residual = np.linalg.norm(source @ rotation.T + translation - current, axis=1)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return SurfacePose(
        pose,
        source.mean(0),
        float(residual.max() + point_error_m),
        min(np.pi, initial_error + current_error),
        normal_error,
        float(normal_position),
        normal_lever,
        len(source),
        normal_origin,
        normal_point_bound,
    )
