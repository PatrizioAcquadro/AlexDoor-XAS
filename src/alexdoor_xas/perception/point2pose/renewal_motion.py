"""Experimental pending-promotion evidence; no map, mask or registration mutations."""

import numpy as np


def transform(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def rigid_fit(source, current):
    a, b = source.mean(0), current.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (current - b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, b - rotation @ a
    return pose


def spatial_support(points):
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 3:
        return dict(count=len(points), second_rms_m=0.0)
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False) / np.sqrt(len(points))
    return dict(count=len(points), second_rms_m=float(singular[1]))


def observed_rigid_motion(source, current, *, minimum=5, tolerance_m=0.004):
    """Independent consensus of measured same-ID points, never a supplied object pose."""
    source, current = np.asarray(source, float), np.asarray(current, float)
    finite = np.isfinite(source).all(1) & np.isfinite(current).all(1)
    source, current = source[finite], current[finite]
    if len(source) < minimum:
        return None, dict(reason="insufficient_pairs", count=len(source))
    # Local RNG does not consume or reset the native model/registration seed.
    rng = np.random.default_rng(0)
    best = np.zeros(len(source), bool)
    for _ in range(64):
        sample = rng.choice(len(source), 3, replace=False)
        if spatial_support(source[sample])["second_rms_m"] <= tolerance_m:
            continue
        pose = rigid_fit(source[sample], current[sample])
        keep = np.linalg.norm(transform(pose, source) - current, axis=1) <= tolerance_m
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < minimum or best.mean() < 0.6:
        return None, dict(reason="weak_consensus", count=int(best.sum()), total=len(source))
    pose = rigid_fit(source[best], current[best])
    residual = np.linalg.norm(transform(pose, source) - current, axis=1)
    best &= residual <= tolerance_m
    support = spatial_support(source[best])
    support.update(current_second_rms_m=spatial_support(current[best])["second_rms_m"])
    if (
        support["count"] < minimum
        or min(support["second_rms_m"], support["current_second_rms_m"]) <= 0.008
    ):
        return None, dict(reason="degenerate_support", **support)
    support.update(reason="supported", rms_m=float(np.sqrt(np.mean(residual[best] ** 2))))
    return pose, support


def lift(depth, pixels, intrinsics):
    """Integer RGB-D lifts, matching the maintained native convention."""
    pixels = np.asarray(pixels, float)
    finite = np.isfinite(pixels).all(1)
    xy = np.where(finite[:, None], pixels, -1).astype(int)
    h, w = depth.shape
    inside = finite & (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
    points = np.full((len(xy), 3), np.nan)
    rows = np.flatnonzero(inside)
    z = depth[xy[rows, 1], xy[rows, 0]]
    points[rows] = np.column_stack(
        (
            (xy[rows, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (xy[rows, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    points[~np.isfinite(points).all(1) | (points[:, 2] <= 0)] = np.nan
    return points


def background_camera_motion(anchor, current):
    """RGB-D background matches compensate camera motion without extrinsics or truth.

    Classical image matching is confined to this short camera-motion measurement;
    it does not add P2P queries, replace TAPIR or change SuperPoint/SAM3.
    """
    import cv2

    old_rgb, old_depth, old_mask, old_k = anchor
    rgb, depth, mask, k = current
    old_gray, gray = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (old_rgb, rgb))
    allowed = cv2.dilate(old_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    allowed &= np.isfinite(old_depth) & (old_depth > 0)
    uv = cv2.goodFeaturesToTrack(
        old_gray,
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=12,
        mask=allowed.astype(np.uint8),
    )
    if uv is None or len(uv) < 20:
        return None, dict(reason="insufficient_background_features")
    new, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, uv, None)
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, new, None)
    xy = np.rint(new[:, 0]).astype(int)
    h, w = depth.shape
    keep = (ok[:, 0] > 0) & (back_ok[:, 0] > 0)
    keep &= np.linalg.norm(back[:, 0] - uv[:, 0], axis=1) <= 1.0
    keep &= (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
    safe = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    rows = np.flatnonzero(keep)
    keep[rows] &= safe[xy[rows, 1], xy[rows, 0]]
    a, b = lift(old_depth, uv[keep, 0], old_k), lift(depth, new[keep, 0], k)
    pose, stats = observed_rigid_motion(a, b, minimum=20)
    if pose is not None and stats["second_rms_m"] <= 0.05:
        return None, dict(stats, reason="narrow_background")
    return pose, stats


def motion_evidence(
    anchor_point,
    current_point,
    body_motion,
    camera_motion,
    *,
    elapsed_s,
    deadline_s,
    observation_tolerance_m=0.004,
    pixel_size_m=0.0,
    body_rms_m=0.0,
    camera_rms_m=0.0,
):
    """Three-way causal decision, to be stored inside the existing pending metadata.

    Independent endpoint observation errors combine in quadrature. A motion must
    separate the two prediction balls; an ambiguous/unsupported point never passes.
    The caller retains native good/TTL/budget rules and drops expired pending points.
    """
    record = dict(
        decision="insufficient", elapsed_s=float(elapsed_s), expired=bool(elapsed_s > deadline_s)
    )
    if not np.isfinite(elapsed_s) or elapsed_s <= 0 or record["expired"]:
        return dict(record, reason="time_budget" if record["expired"] else "timestamp")
    if body_motion is None or camera_motion is None:
        return dict(record, reason="unreliable_motion")
    if not np.isfinite([anchor_point, current_point]).all():
        return dict(record, reason="invalid_observation")
    tolerance = float(
        np.sqrt(2 * observation_tolerance_m**2 + pixel_size_m**2 + body_rms_m**2 + camera_rms_m**2)
    )
    moving = transform(body_motion, anchor_point)
    static = transform(camera_motion, anchor_point)
    separation = float(np.linalg.norm(moving - static))
    body_error = float(np.linalg.norm(current_point - moving))
    static_error = float(np.linalg.norm(current_point - static))
    record.update(
        tolerance_m=tolerance,
        separation_m=separation,
        body_error_m=body_error,
        static_error_m=static_error,
    )
    if separation <= 2 * tolerance:
        return dict(record, reason="indistinguishable_motion")
    if body_error <= tolerance and static_error - body_error > tolerance:
        return dict(record, decision="body", reason="body_supported")
    if static_error <= tolerance and body_error - static_error > tolerance:
        return dict(record, decision="static", reason="static_supported")
    return dict(record, reason="neither_or_ambiguous")
