"""Local rigid pose constrained by measured plane/edge geometry and native pairs."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_opening, label

from alexdoor_xas.perception.geometry import Surface, deproject, extent_edges, plane_fit


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
    rotation_basis_world: np.ndarray | None = None
    rotation_axis_chord_bounds: np.ndarray | None = None


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


def _planar_core_bound(points, normal, offset, point_error_m):
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
    return min(normal_bounds)


def _observed_basis_axes(surface, point_error_m, *, normal_points=None):
    """Bound a plane/edge basis using finite baselines and a metric error bound."""
    if not np.isfinite(point_error_m) or point_error_m <= 0:
        raise ValueError("missing_metric_point_error")
    if normal_points is not None:
        normal_points = np.asarray(normal_points, float)
        if (
            normal_points.ndim != 2
            or normal_points.shape[1:] != (3,)
            or len(normal_points) < 5
            or not np.isfinite(normal_points).all()
        ):
            raise ValueError("invalid_observed_normal_support")
    points = np.asarray(surface.points if normal_points is None else normal_points, float)
    normal = np.asarray(surface.normal, float)
    normal_norm = np.linalg.norm(normal)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 5
        or not np.isfinite(points).all()
        or normal.shape != (3,)
        or not np.isfinite(normal).all()
        or abs(normal_norm - 1) > 1e-6
    ):
        raise ValueError("invalid_observed_plane")
    normal = normal / normal_norm
    normal_bound = _planar_core_bound(points, normal, surface.offset / normal_norm, point_error_m)
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
        # Keep the original normal/edge errors separately. The cross-product
        # column is bounded by their chord sum; the scalar SO(3) bound remains
        # an independent ceiling for every column. No independence is assumed.
        normal_chord = 2 * np.sin(min(normal_bound, np.pi) / 2)
        edge_chord = 2 * np.sin(min(edge_from_z, np.pi) / 2)
        axes = np.minimum(
            2 * np.sin(bound / 2),
            [normal_chord, normal_chord + edge_chord, edge_chord],
        )
        alternatives.append((bound, np.column_stack((normal, np.cross(z, normal), z)), axes))
    if not alternatives:
        raise ValueError("missing_conditioned_observed_edge")
    bound, basis, axes = min(alternatives, key=lambda item: item[0])
    if bound >= np.pi / 2:
        raise ValueError("unconditioned_observed_basis")
    return basis, float(bound), float(normal_bound), axes


def observed_basis(surface, point_error_m, *, normal_points=None):
    """Preserve the public scalar basis contract and its original bounds."""
    return _observed_basis_axes(surface, point_error_m, normal_points=normal_points)[:3]


def measured_material_surface(sensor, mask, pixels, predicted, reference, config, diagnostics):
    """Associate measured components to the transported seed before measuring edges."""
    depth = sensor["depth_m"][..., 0]
    valid = sensor["valid_depth"][..., 0] & np.isfinite(depth) & (depth > 0)
    intrinsics, camera = sensor["intrinsics"], sensor["camera_world"]
    minimum = config["min_points"]
    point_error = config["local_surface_constraint"]["point_error_m"]
    u, v = pixels.T
    inside = mask[v, u] & valid[v, u]
    u, v = u[inside], v[inside]
    points = deproject(depth[v, u], np.c_[u, v], intrinsics, camera)
    diagnostics["alignment_input_points"] = len(points)
    fit = plane_fit(points, point_error, minimum, np.random.default_rng(int(sensor["frame"])))
    if fit is None:
        raise ValueError("missing_measured_planar_core")
    normal, offset, inliers, residual = fit
    if inliers.sum() <= len(points) / 2:
        raise ValueError("ambiguous_measured_material_plane")
    if normal @ reference < 0:
        normal, offset = -normal, -offset
    if abs(normal[2]) >= 0.3:
        raise ValueError("invalid_observed_plane")

    vv, uu = np.nonzero(mask & valid)
    dense = deproject(depth[vv, uu], np.c_[uu, vv], intrinsics, camera)
    on_plane = abs(dense @ normal - offset) <= point_error
    raster = np.zeros_like(mask)
    raster[vv[on_plane], uu[on_plane]] = True
    components, _ = label(binary_opening(raster))
    root_labels = components[v[inliers], u[inliers]]
    counts = np.bincount(root_labels)
    associated = np.flatnonzero(counts[1:] >= minimum) + 1
    diagnostics["component_root_points"] = counts[associated].tolist()
    if counts[associated].sum() <= len(points) / 2:
        raise ValueError("insufficient_connected_material_support")
    core = points[inliers][np.isin(root_labels, associated)]
    # An occlusion or shallow relief may split one leaf. Each part needs its
    # own projected seed support. Never bridge a gap to manufacture an edge.
    edges = {}
    dense_labels = components[vv, uu]
    for component in associated:
        part = components == component
        _, lines = extent_edges(
            dense[dense_labels == component],
            normal,
            part,
            sensor,
            config["plane_tolerance_m"],
            return_points=True,
            foreground_reference=predicted,
        )
        edges.update((f"{key}_{component}", line) for key, line in lines.items())
    diagnostics["measured_edges"] = list(edges)
    return Surface(core, normal, offset, residual, len(core) / len(points), edge_points=edges)


def intersect_normal_cones(cones):
    """Enclose the common unit normals by projecting onto their cap halfspaces."""
    normals = np.asarray([normal for normal, _ in cones], dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    radii = np.asarray([radius for _, radius in cones])
    if not np.isfinite(radii).all() or (radii < 0).any() or (radii >= np.pi / 2).any():
        raise ValueError("unconditioned_local_normal_cone")
    cosines = np.cos(radii)
    # In 3D the projection uses at most three linearly independent active planes.
    # Its direction gives a bound via the same nonnegative dual weights.
    roundoff = 64 * np.finfo(float).eps
    best = None
    for size in range(1, min(3, len(cones)) + 1):
        for active in combinations(range(len(cones)), size):
            selected, limits = normals[list(active)], cosines[list(active)]
            try:
                weights = np.linalg.solve(selected @ selected.T, limits)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(weights).all() or min(weights) < 0:
                continue
            point = weights @ selected
            length = np.linalg.norm(point)
            if (
                length > 0
                and (normals @ point >= cosines - roundoff).all()
                and (best is None or length < best[0])
            ):
                best = length, point, float(weights @ limits / length)
    if best is None or best[0] > 1 + roundoff:
        raise ValueError("inconsistent_local_normal_observations")
    length, point, lower_dot = best
    return tuple(point / length), float(np.arccos(np.clip(lower_dot - roundoff, -1, 1)))


def constrained_surface_pose(
    initial_surface,
    current_surface,
    initial_camera,
    current_camera,
    source_points,
    current_points,
    point_error_m,
    *,
    current_normal_points=None,
    map_anchor,
):
    """Use all supplied measured native inliers; recompute their residual envelope.

    No native pose, plane residual, semantic label or favorable subset supplies a
    complete bound. Calibration, initial geometry, temporal growth and collision
    model disagreement remain separate contributions at the consumer.
    """
    initial_basis, initial_error, _, initial_axes = _observed_basis_axes(
        initial_surface, point_error_m
    )
    current_basis, current_error, normal_error, current_axes = _observed_basis_axes(
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
    # The same maximum residual plus point error bounds translation at every
    # measured pair, and by convexity at their centroid. Choose among those
    # anchors by the COMPLETE observed root's largest lever, never by an action
    # margin. Pose, residual envelope and rotation bounds remain unchanged.
    reference = (initial_surface.points - cameras[0][:3, 3]) @ cameras[0][:3, :3]
    anchors = np.vstack((source.mean(0), source))
    origin = min(anchors, key=lambda point: np.linalg.norm(reference - point, axis=1).max())
    return SurfacePose(
        pose,
        origin.copy(),
        float(residual.max() + point_error_m),
        min(np.pi, initial_error + current_error),
        normal_error,
        float(normal_position),
        normal_lever,
        len(source),
        normal_origin,
        normal_point_bound,
        current_basis.copy(),
        np.minimum(
            2 * np.sin(min(np.pi, initial_error + current_error) / 2),
            initial_axes + current_axes,
        ),
    )
