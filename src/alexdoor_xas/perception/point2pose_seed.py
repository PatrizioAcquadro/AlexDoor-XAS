"""SAM2 preservation checks from the same measured RGB-D component."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def candidate_references(mask, depth, valid, intrinsics, error_m):
    depth = np.asarray(depth).squeeze(-1)
    valid = np.asarray(valid).squeeze(-1).astype(bool)
    mask = np.asarray(mask, bool) & valid & np.isfinite(depth) & (depth > 0)
    if not mask.any():
        raise ValueError("missing_candidate_measured_support")
    # Use the metric pixel uncertainty to avoid prompting on uncertain borders.
    margin = max(
        1, int(np.ceil(error_m * max(intrinsics[0, 0], intrinsics[1, 1]) / depth[mask].min()))
    )
    inside = distance_transform_edt(mask) > margin
    pixels = np.argwhere(inside)
    if len(pixels) < 5:
        raise ValueError("insufficient_candidate_core")
    # Spread positive references over the observed face, rather than row-end edges.
    points = [pixels[len(pixels) // 2]]
    distance = np.full(len(pixels), np.inf)
    for _ in range(4):
        distance = np.minimum(distance, np.linalg.norm(pixels - points[-1], axis=1))
        points.append(pixels[distance.argmax()])
    return np.asarray(points)


def verify_candidate(expected, actual, positives, competitors=()):
    if actual.shape != expected.shape or not actual[positives[:, 0], positives[:, 1]].all():
        raise ValueError("sam2_initialization_candidate_mismatch")
    for other in competitors:
        core = distance_transform_edt(other) > 1
        if core.any() and (actual & core).any():
            raise ValueError("sam2_initialization_competing_candidate")


def initialization_checks(masks, regenerated, prompts, *, diagnostic_only=False):
    """Report each guard in offline diagnosis; operational initialization stays strict."""
    checks = []
    for index, (expected, actual, positives) in enumerate(
        zip(masks, regenerated, prompts, strict=True)
    ):
        reason = None
        try:
            verify_candidate(expected, actual, positives)
        except ValueError as error:
            if not diagnostic_only:
                raise
            reason = str(error)
        checks.append(dict(object_id=index, accepted=reason is None, reason=reason))
    return checks
