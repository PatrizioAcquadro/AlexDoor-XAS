"""Official CAD-free Point2Pose in an isolated CUDA process; no door truth inputs."""

import ctypes
import json
import os
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from alexdoor_xas.perception.visual_worker import receive, send


def pack_array(value):
    import numpy as np

    value = np.asarray(value)
    return dict(
        data=np.ascontiguousarray(value).tobytes(), shape=value.shape, dtype=value.dtype.str
    )


@contextmanager
def native_logs():
    """GTSAM writes through C stdout; keep it out of the framed IPC stream."""
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


def unpack_array(value):
    import numpy as np

    return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()


class OfficialPipeline:
    def __init__(self, root, request):
        import numpy as np
        import torch
        from omegaconf import OmegaConf

        from alexdoor_xas.perception.point2pose_compat import install_lifting
        from alexdoor_xas.perception.point2pose_scale import panel_sdf_builder

        if not torch.cuda.is_available():
            raise RuntimeError("Point2Pose requires CUDA; no CPU fallback")
        torch.set_num_threads(4)
        config = OmegaConf.load(root / "upstream/configs/ycbinisaac/eccv_final.yaml")
        params = config.pipeline.params
        near, far = request["depth_interval_m"]
        self.crop_cache = install_lifting((near, far))
        from point2pose.pipeline.modular_pipeline import ModularPipeline

        params.update(
            min_depth=near,
            max_depth=far,
            use_segmenter=True,
            estimate_init_pose=False,
            use_key_frame_graph=True,
            save_pose=False,
            save_meta_data=False,
            debug_dir=request["log_dir"],
            max_num_obj=request["num_objects"],
        )
        config.register.params.update(select_3d_dist_min_depth=near, select_3d_dist_max_depth=far)
        config.segmenter.params.checkpoint = str(root / "checkpoints/sam2.1_hiera_large.pt")
        config.tracker.params.checkpoint_path = str(
            root / "checkpoints/causal_bootstapir_checkpoint.pt"
        )
        config.visualization.params.save_images = False
        config.sampler.params.debug_dir = request["log_dir"]
        self.pipeline = ModularPipeline(config)
        builder = panel_sdf_builder(type(self.pipeline.sdf_builder), request["depth_error_m"])
        self.pipeline.sdf_builder = builder(config.reconstructor.params)
        self.config = OmegaConf.to_container(config, resolve=True)
        self.torch, self.np = torch, np
        self.last_frontend = None
        native_step = self.pipeline.frontend.step

        def capture(*args):
            self.last_frontend = native_step(*args)
            return self.last_frontend

        self.pipeline.frontend.step = capture
        self.initialized = False
        self.diagnostic_only = bool(request.get("diagnostic_only", False))
        self.initialization_checks = []
        self.models = (
            self.pipeline.frontend.tracker._model,
            self.pipeline.frontend.segmenter.predictor,
            self.pipeline.kf_manager.sampler.super_point_extractor,
        )
        self.index = 0
        self.log_dir = Path(request["log_dir"])
        self.depth_error_m = request["depth_error_m"]
        self.runtime = dict(
            device="cuda:0",
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            numpy=np.__version__,
            python=sys.version.split()[0],
            sources=json.loads((root / "sources.json").read_text()),
            config=self.config,
            training_started=False,
            qualified=False,
            diagnostic_only=self.diagnostic_only,
        )

    def infer(self, request):
        from point2pose.data_types.frame import Frame

        np, torch = self.np, self.torch
        sensor = {key: unpack_array(value) for key, value in request["sensor"].items()}
        started = time.perf_counter()
        if any(next(model.parameters()).device.type != "cuda" for model in self.models):
            raise RuntimeError("CUDA required for every Point2Pose model")
        self.crop_cache.clear()
        torch.cuda.reset_peak_memory_stats()
        depth = sensor["depth_m"].squeeze(-1).astype(np.float32)
        depth[~sensor["valid_depth"].squeeze(-1).astype(bool)] = 0
        masks = request.get("masks")
        if masks is not None:
            if self.initialized or (
                request["mask_frame"] != int(sensor["frame"])
                or request["mask_time_s"] != float(sensor["time_s"])
            ):
                raise ValueError("unsynchronized_point2pose_initialization")
            masks = np.stack([unpack_array(mask) for mask in masks]).astype(bool)
        elif not self.initialized:
            raise ValueError("missing_automatic_candidate_masks")
        frame = Frame(
            id=self.index,
            rgb=sensor["rgb"],
            depth=depth,
            intrinsics=sensor["intrinsics"],
            depth_factor=1.0,
            timestamp=float(sensor["time_s"]),
            mask=None if masks is None else torch.as_tensor(masks[:, None], device="cuda"),
        )
        if not self.initialized:
            from alexdoor_xas.perception.point2pose_seed import (
                candidate_references,
                initialization_checks,
            )

            prompts = [
                candidate_references(
                    mask,
                    sensor["depth_m"],
                    sensor["valid_depth"],
                    sensor["intrinsics"],
                    self.depth_error_m,
                )
                for mask in masks
            ]
            self.pipeline.initialize_first_frame(frame)
            regenerated = frame.mask.detach().float().cpu().numpy()[:, 0] > 0
            np.save(self.log_dir / "sam2-initial.npy", regenerated)
            self.initialization_checks = initialization_checks(
                masks, regenerated, prompts, diagnostic_only=self.diagnostic_only
            )
            self.initialized = True
        else:
            self.pipeline.step(frame)
        self.index += 1
        torch.cuda.synchronize()
        objects = []
        for obj in self.pipeline.objects:
            indices = getattr(obj, "curr_frame_indices", None)
            source, current = np.empty((0, 3)), np.empty((0, 3))
            measured = np.empty(0, bool)
            if indices is not None and len(indices):
                rows = obj.track_idx_2_obj_idx[indices]
                valid = rows >= 0
                source = obj.key_points[rows[valid]]
                current = obj.curr_frame_points_3d[valid]
                tracks = self.last_frontend.tracks[indices[valid]]
                uv = np.rint(tracks).astype(int)
                h, w = depth.shape
                measured = (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
                inside = np.flatnonzero(measured)
                measured[inside] &= sensor["valid_depth"][uv[inside, 1], uv[inside, 0], 0]
            volume = obj.sdf_volume
            objects.append(
                dict(
                    object_id=obj.id,
                    camera_from_map=pack_array(obj.pose),
                    lost=bool(obj.lost),
                    source_points=pack_array(source),
                    current_points=pack_array(current),
                    measured_correspondences=pack_array(measured),
                    inliers=pack_array(
                        np.asarray(
                            getattr(obj, "inliers", None)
                            if getattr(obj, "inliers", None) is not None
                            else []
                        )
                    ),
                    residuals=pack_array(
                        np.asarray(
                            getattr(obj, "residuals", None)
                            if getattr(obj, "residuals", None) is not None
                            else []
                        )
                    ),
                    key_points=pack_array(obj.key_points),
                    tsdf_gpu=bool(volume is not None and volume.gpu_mode),
                    tsdf_voxels=0 if volume is None else int(np.prod(volume._vol_dim)),
                    tsdf_bounds=None if volume is None else pack_array(volume._vol_bnds),
                )
            )
        return dict(
            objects=objects,
            initialization_checks=self.initialization_checks,
            masks=pack_array(frame.mask.detach().float().cpu().numpy()[:, 0] > 0),
            capture_s=float(sensor["time_s"]),
            frame=int(sensor["frame"]),
            latency_s=time.perf_counter() - started,
            torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_peak_bytes=torch.cuda.max_memory_allocated(),
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            tsdf_rebuilds=self.pipeline.sdf_builder.rebuilds,
        )


def main():
    root = Path(sys.argv[1]).resolve()
    for name in ("upstream", "tapnet"):
        sys.path.insert(0, str(root / name))
    os.environ["TORCH_HOME"] = str(root / "torch")
    request = receive(sys.stdin.buffer)
    try:
        with native_logs():
            pipeline = OfficialPipeline(root, request)
        send(sys.stdout.buffer, dict(ready=pipeline.runtime))
        while (request := receive(sys.stdin.buffer)) is not None:
            try:
                with native_logs():
                    if request.get("profile"):
                        import cProfile

                        profiler = cProfile.Profile()
                        result = profiler.runcall(pipeline.infer, request)
                        profiler.dump_stats(
                            str(pipeline.log_dir / f"profile-{pipeline.index}.pstats")
                        )
                    else:
                        result = pipeline.infer(request)
                send(sys.stdout.buffer, result)
            except Exception as error:
                send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
    except Exception as error:
        send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
