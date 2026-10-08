"""Observed-only extent adaptation around the official CUDA TSDF implementation."""

import math

import numpy as np


def observed_bounds(points, padding):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("missing_observed_tsdf_extent")
    if not np.isfinite(points).all() or not np.isfinite(padding) or padding <= 0:
        raise ValueError("invalid_observed_tsdf_extent")
    return np.stack((points.min(0) - padding, points.max(0) + padding), axis=1)


def voxel_count(bounds, voxel_size):
    bounds = np.asarray(bounds)
    if (
        bounds.shape != (3, 2)
        or not np.isfinite(bounds).all()
        or not np.isfinite(voxel_size)
        or voxel_size <= 0
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("invalid_tsdf_grid")
    return math.prod(math.ceil(float(d) / voxel_size) for d in bounds[:, 1] - bounds[:, 0])


def panel_sdf_builder(base, depth_error_m):
    """Keep official fusion/filtering; expand observed volumes and replay their keyframes."""

    class PanelSDFBuilder(base):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.padding = max(self.padding, depth_error_m + 5 * self.voxel_size)
            self.envelopes, self.history = {}, {}
            self.rebuilds = 0

        def _init_object_volume(self, obj, pts_obj):
            import psutil
            import pycuda.driver as cuda
            from point2pose.modules.reconstruction.sdf_builder import TSDFVolume

            bounds = self.envelopes[obj.id]
            count = voxel_count(bounds, self.voxel_size)
            free, total = cuda.mem_get_info()
            # Reserve 20% of device memory for concurrent rendering/model workspaces.
            required = count * 3 * np.dtype(np.float32).itemsize
            if required > free - total // 5:
                raise MemoryError(f"tsdf_memory_budget_exceeded: {required} bytes, {free} free")
            if required * 3 > psutil.virtual_memory().available:
                raise MemoryError("tsdf_host_memory_budget_exceeded")
            obj.sdf_volume = TSDFVolume(bounds, self.voxel_size, use_gpu=True)
            if not obj.sdf_volume.gpu_mode:
                raise RuntimeError("CUDA TSDF required; CPU fallback refused")
            obj.sdf_num_integrated = 0

        def integrate_keyframe(self, obj, keyframe):
            if keyframe is None or keyframe.dense_pts is None or not len(keyframe.dense_pts):
                return False
            transform = np.linalg.inv(keyframe.pose)
            points = np.asarray(keyframe.dense_pts) @ transform[:3, :3].T + transform[:3, 3]
            points = points[np.isfinite(points).all(1)]
            if len(points) < self.min_points:
                return False
            # Bound the radial filter by observed extent, never a small-object constant.
            self.max_radius = float(np.linalg.norm(np.ptp(points, axis=0)) + self.padding)
            # Use the authors' accepted cloud, not outliers they already reject.
            points = self._filter_points(points)
            if len(points) < self.min_points:
                return False
            bounds = observed_bounds(points, self.padding)
            previous = self.envelopes.get(obj.id)
            volume = getattr(obj, "sdf_volume", None)
            grows = volume is not None and np.any(
                (points < volume._vol_bnds[:, 0]) | (points > volume._vol_bnds[:, 1])
            )
            if previous is not None:
                bounds[:, 0] = np.minimum(previous[:, 0], bounds[:, 0])
                bounds[:, 1] = np.maximum(previous[:, 1], bounds[:, 1])
            self.envelopes[obj.id] = bounds
            history = self.history.setdefault(obj.id, [])
            if grows:
                obj.sdf_volume = obj.sdf = None
                del volume
                for old in history:
                    self.max_radius = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
                    super().integrate_keyframe(obj, old)
                self.rebuilds += 1
            integrated = super().integrate_keyframe(obj, keyframe)
            if integrated:
                history.append(keyframe)
            return integrated

    return PanelSDFBuilder
