"""Subpixel RGB correspondences checked against metric depth and rigid motion."""

import numpy as np

from alexdoor_xas.perception.geometry import Surface, deproject, rigid_fit


def sample_points(sensor, pixels):
    """Bilinear optical depth, rejecting invalid pixels and depth discontinuities."""
    depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
    h, w = depth.shape
    xy = np.floor(pixels).astype(int)
    good = (xy >= 0).all(1) & (xy[:, 0] < w - 1) & (xy[:, 1] < h - 1)
    indices = np.flatnonzero(good)
    x, y = xy[good].T
    values = np.c_[depth[y, x], depth[y, x + 1], depth[y + 1, x], depth[y + 1, x + 1]]
    usable = valid[y, x] & valid[y, x + 1] & valid[y + 1, x] & valid[y + 1, x + 1]
    usable &= np.isfinite(values).all(1) & (values > 0).all(1) & (np.ptp(values, axis=1) < 0.01)
    indices, values = indices[usable], values[usable]
    dx, dy = (pixels[indices] - xy[indices]).T
    z = np.sum(values * np.c_[(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy], axis=1)
    return indices, deproject(z, pixels[indices], sensor["intrinsics"], sensor["camera_world"])


class PixelMotionTracker:
    """Follow fixed observed points; never infer physical matches from patch centers."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = self.pixels = self.source = None
        self.diagnostics = dict(pixel_motion="acquiring", tracked_pixels=0)

    def update(self, sensor, reference):
        import cv2

        gray = cv2.cvtColor(sensor["rgb"], cv2.COLOR_RGB2GRAY)
        if self.gray is None:
            depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
            v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
            points = deproject(
                depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"]
            )
            local = points @ reference.basis
            bounds = reference.bounds
            keep = abs(points @ reference.normal - reference.offset) < 0.008
            keep &= (
                (local[:, 1:] > bounds[0, 1:] + 0.008) & (local[:, 1:] < bounds[1, 1:] - 0.008)
            ).all(1)
            mask = np.zeros(depth.shape, np.uint8)
            mask[v[keep], u[keep]] = 255
            pixels = cv2.goodFeaturesToTrack(gray, 300, 0.005, 8, mask=mask)
            if pixels is None:
                self.diagnostics.update(pixel_motion="unobserved_texture", tracked_pixels=0)
                return None
            pixels = pixels.reshape(-1, 2)
            indices, source = sample_points(sensor, pixels)
            if len(source) < 12:
                self.diagnostics.update(
                    pixel_motion="insufficient_metric_features", tracked_pixels=len(source)
                )
                return None
            self.gray, self.pixels, self.source = gray, pixels[indices], source
            self.diagnostics.update(pixel_motion="acquiring", tracked_pixels=len(source))
            return None
        following, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.pixels.astype(np.float32), None, winSize=(21, 21), maxLevel=3
        )
        if following is None:
            self.reset()
            self.diagnostics["pixel_motion"] = "lost_visual_matches"
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, following, None, winSize=(21, 21), maxLevel=3
        )
        good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        good &= np.linalg.norm(backward - self.pixels, axis=1) < 0.7
        source, pixels = self.source[good], following[good]
        indices, target = sample_points(sensor, pixels)
        source, pixels = source[indices], pixels[indices]
        if len(source) < 12:
            self.reset()
            self.diagnostics["pixel_motion"] = "lost_metric_matches"
            return None
        best = None
        rng = np.random.default_rng(int(sensor["frame"]))
        for _ in range(32):
            take = rng.choice(len(source), 4, replace=False)
            fitted = rigid_fit(source[take], target[take])
            if fitted is None:
                continue
            r, t, _ = fitted
            inliers = np.linalg.norm(source @ r.T + t - target, axis=1) < 0.004
            if inliers.sum() >= 12 and (best is None or inliers.sum() > best.sum()):
                best = inliers
        if best is None or best.mean() < 0.7:
            self.reset()
            self.diagnostics["pixel_motion"] = "ambiguous_rigid_motion"
            return None
        fitted = rigid_fit(source[best], target[best])
        if fitted is None or fitted[2] > 0.004:
            self.reset()
            self.diagnostics["pixel_motion"] = "unreliable_rigid_motion"
            return None
        self.gray, self.pixels, self.source = gray, pixels[best], source[best]
        self.diagnostics.update(
            pixel_motion="tracked",
            tracked_pixels=int(best.sum()),
            pixel_motion_residual_m=fitted[2],
        )
        return fitted


def moved_surface(reference, motion):
    r, t, residual = motion
    normal = r @ reference.normal
    return Surface(
        reference.points @ r.T + t,
        normal,
        reference.offset + float(normal @ t),
        reference.descriptor,
        reference.anchors @ r.T + t,
        reference.features,
        residual,
        1.0,
        reference.views.copy(),
    )
