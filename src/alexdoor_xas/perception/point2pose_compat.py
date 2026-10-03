"""Measured runtime adaptations; preserve official lifting and registration results."""

import functools
import inspect

import numpy as np


def measured_depth_lifting(native, interval):
    signature = inspect.signature(native)

    @functools.wraps(native)
    def lift(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        values = call.arguments
        values["min_depth"], values["max_depth"] = interval
        if values["fill_missing_depth"] and not values["compute_depth_uncertainty"]:
            if values["window_size"] < 1 or values["window_size"] % 2 != 1:
                raise ValueError("window_size must be an odd integer >= 1")
            pixels = np.atleast_2d(values["pixel"])
            depth = values["depth_image"]
            x, y = pixels[:, 0].astype(int), pixels[:, 1].astype(int)
            inside = (x >= 0) & (y >= 0) & (x < depth.shape[1]) & (y < depth.shape[0])
            # Official fill changes only nonfinite in-bounds samples. On measured
            # samples its NxWxW gather is unused (0.126s on the 600x960 pilot).
            if np.isfinite(depth[y[inside], x[inside]]).all():
                values["fill_missing_depth"] = False
        return native(**values)

    return lift


def install_lifting(interval):
    from point2pose.utils import camera

    camera.convert_pixel_to_world = measured_depth_lifting(camera.convert_pixel_to_world, interval)
    native_crop = camera.extract_cropped_point_cloud
    signature = inspect.signature(native_crop)
    cache = {}

    @functools.wraps(native_crop)
    def crop(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        call.arguments["min_depth"], call.arguments["max_depth"] = interval
        if np.isfinite(call.arguments["frame"].depth).all():
            call.arguments.update(fill_missing_depth=False, window_size=3, min_neighbors=1)
        key = (id(call.arguments["frame"]), *(v for k, v in call.arguments.items() if k != "frame"))
        if key not in cache:
            cache[key] = native_crop(**call.arguments)
        return cache[key]

    camera.extract_cropped_point_cloud = crop
    return cache
