"""Bounded observed-only IPC for the official Point2Pose worker."""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from alexdoor_xas.perception.point2pose_worker import pack_array
from alexdoor_xas.perception.provider import CueEngine
from alexdoor_xas.perception.visual_worker import receive, send
from alexdoor_xas.recording.b1 import OBS_KEYS


class Point2PoseWorker:
    def __init__(
        self,
        root,
        depth_interval_m,
        depth_error_m,
        num_objects,
        log_dir,
        *,
        diagnostic_only=False,
        registration_diagnostics=False,
        use_key_frame_graph=True,
    ):
        boot_started = time.perf_counter()
        root, log_dir = Path(root).resolve(), Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        env.pop("PYTHONHOME", None)
        # CUDA wheels belong to this venv, not the launcher's shared Isaac ABI.
        env.pop("LD_LIBRARY_PATH", None)
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        env["PATH"] = f"{cuda}/bin:{env.get('PATH', '')}"
        compiler = os.environ.get("CXX") or shutil.which("g++-11")
        if compiler:
            env["NVCC_PREPEND_FLAGS"] = f"-ccbin {compiler}"
        self.log = (log_dir / "worker.log").open("a")
        self.process = subprocess.Popen(
            [
                str(root / "runtime/bin/python"),
                "-m",
                "alexdoor_xas.perception.point2pose_worker",
                str(root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=env,
        )
        self.gpu_resident_bytes = self.gpu_sampled_peak_bytes = None
        self.monitor_stop = threading.Event()

        def monitor():
            while not self.monitor_stop.is_set():
                try:
                    sample = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    for line in sample.stdout.splitlines():
                        pid, memory = line.split(",")
                        if int(pid) == self.process.pid:
                            value = int(memory) * 1024**2
                            self.gpu_resident_bytes = value
                            self.gpu_sampled_peak_bytes = max(
                                value, self.gpu_sampled_peak_bytes or 0
                            )
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                self.monitor_stop.wait(0.1)

        self.monitor = threading.Thread(target=monitor, daemon=True)
        self.monitor.start()
        try:
            send(
                self.process.stdin,
                dict(
                    depth_interval_m=depth_interval_m,
                    depth_error_m=depth_error_m,
                    num_objects=num_objects,
                    log_dir=str(log_dir),
                    diagnostic_only=diagnostic_only,
                    registration_diagnostics=registration_diagnostics,
                    use_key_frame_graph=use_key_frame_graph,
                ),
            )
            response = receive(self.process.stdout)
            if response is None or "error" in response:
                raise RuntimeError(f"Point2Pose initialization failed: {response}")
        except BaseException:
            # The caller cannot close a constructor that did not return.
            self.close()
            raise
        self.runtime = response["ready"]
        self.boot_latency_s = time.perf_counter() - boot_started
        self.runtime["boot_latency_s"] = self.boot_latency_s
        (log_dir / "runtime.json").write_text(json.dumps(self.runtime, indent=2) + "\n")
        self.first_request = True

    def infer(self, sensor, masks=None, *, profile=False):
        started = time.perf_counter()
        # Explicit whitelist: commands, prepared assets and annotations cannot cross IPC.
        request = dict(sensor={key: pack_array(sensor[key]) for key in OBS_KEYS})
        request["profile"] = profile
        if masks is not None:
            request.update(
                masks=[pack_array(mask) for mask in masks],
                mask_frame=int(sensor["frame"]),
                mask_time_s=float(sensor["time_s"]),
            )
        send(self.process.stdin, request)
        result = receive(self.process.stdout)
        if result is None or "error" in result:
            raise RuntimeError(f"Point2Pose inference failed: {result}")
        result["model_latency_s"] = result["latency_s"]
        result["latency_s"] = time.perf_counter() - started
        if self.first_request:
            result["latency_s"] += self.boot_latency_s
            self.first_request = False
        result.update(
            gpu_resident_bytes=self.gpu_resident_bytes,
            gpu_sampled_peak_bytes=self.gpu_sampled_peak_bytes,
            gpu_sampling_period_s=0.1,
        )
        return result

    def close(self):
        # Episode reset kills all model/graph/TSDF state, including a running request.
        self.monitor_stop.set()
        self.monitor.join(timeout=3)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


class Point2PoseEngine(CueEngine):
    """One in-flight acquisition; latest waiting acquisition replaces older ones.

    Prepared processes are recreated before the next episode's acquisitions;
    SAM2 memory and Point2Pose's causal state never cross an episode boundary.
    """

    def __init__(self, factory, *, replay):
        super().__init__(None, replay=replay)
        self.factory, self.latest = factory, None
        self.failed = False
        self.prepared = False

    def prepare(self):
        """Load before acquisition; native initialization sets the actual object count."""
        if self.worker is None:
            self.worker = self.factory(1)
            self.worker.first_request = False  # Startup predates every supplied capture.
        self.prepared = True

    def reset(self, *, preload=True):
        was_prepared, self.prepared = self.prepared, False
        self.generation += 1
        if self.worker is not None:
            self.worker.close()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            from concurrent.futures import ThreadPoolExecutor

            self.executor = ThreadPoolExecutor(max_workers=1)
        self.worker = self.pending = self.latest = None
        self.failed = False
        if preload and was_prepared:
            self.prepare()

    def submit(self, sensor, masks=None, *, start_s=None):
        if self.failed:
            return False
        import numpy as np

        copy_started = time.perf_counter()
        snapshot = {k: np.array(sensor[k], copy=True) for k in OBS_KEYS}
        for value in snapshot.values():
            value.setflags(write=False)
        masks = None if masks is None else tuple(np.array(m, copy=True) for m in masks)
        copy_s = time.perf_counter() - copy_started
        if self.pending is not None:
            self.latest = snapshot, masks, copy_s
            return False
        return self._dispatch(snapshot, masks, copy_s, start_s)

    def _dispatch(self, snapshot, masks, copy_s, start_s):
        if self.worker is None:
            if masks is None:
                return False
            self.worker = self.factory(len(masks))
        worker = self.worker

        def infer():
            result = worker.infer(snapshot, masks)
            result["latency_s"] += copy_s
            result["snapshot_copy_s"] = copy_s
            return result

        result = infer() if self.replay else self.executor.submit(infer)
        started = float(snapshot["time_s"]) if start_s is None else start_s
        if self.replay:
            result["available_s"] = started + result["latency_s"]
        self.pending = (self.generation, snapshot, started), result
        return True

    def poll(self, now):
        if self.pending is not None and not self.replay:
            (_, sensor, started), future = self.pending
            if future.done() and future.exception() is None:
                result = future.result()
                result.setdefault("available_s", max(now, started + result["latency_s"]))
        try:
            event = super().poll(now)
        except (RuntimeError, EOFError, BrokenPipeError, ValueError):
            self.pending = self.latest = None
            self.failed = True
            raise
        if event is not None and self.latest is not None:
            sensor, masks, copy_s = self.latest
            self.latest = None
            self._dispatch(sensor, masks, copy_s, start_s=now)
        return event

    def close(self):
        self.reset(preload=False)
        super().close()


class SeedCueEngine(CueEngine):
    """Release the stateless 6.0B CUDA worker once the immutable seed is ready."""

    def __init__(self, factory, *, replay):
        self.factory = factory
        self.first_reset = True
        super().__init__(factory(), replay=replay)

    def release(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.pending = None

    def restart(self):
        super().reset()
        if self.first_reset:
            self.first_reset = False
            return
        self.release()
        self.worker = self.factory()
        if not self.replay:
            from concurrent.futures import ThreadPoolExecutor

            self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.release()


def tracking_provider(
    recipe,
    models,
    calibration,
    output,
    *,
    replay,
    tool_fk=None,
    distal_faces=None,
    calibration_bound=None,
    relative_speed_bound=None,
):
    """Identical live/replay provider; only acquisition/completion scheduling differs."""
    from alexdoor_xas.perception.panel_tracking import PanelTracking
    from alexdoor_xas.perception.provider import GeometryProvider, ModelWorker

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "visual.log").open("a")
    started = time.perf_counter()
    cues = SeedCueEngine(lambda: ModelWorker(models, recipe.config, log=log), replay=replay)
    visual_startup_s = time.perf_counter() - started
    processes = 0

    def factory(count):
        nonlocal processes
        processes += 1
        return Point2PoseWorker(
            models / "point2pose",
            calibration["depth_interval_m"],
            recipe.config["plane_tolerance_m"],
            count,
            output / f"process-{processes}",
        )

    engine = Point2PoseEngine(factory, replay=replay)
    tracking = PanelTracking(
        engine,
        recipe.config,
        calibration=calibration,
        tool_fk=tool_fk,
        distal_faces=distal_faces,
        calibration_bound=calibration_bound,
        relative_speed_bound=relative_speed_bound,
    )
    tracking.before_start = cues.release
    provider = GeometryProvider(recipe, cues, tracking=tracking)
    provider.owned_log = log
    try:
        engine.prepare()
    except Exception:
        provider.close()
        raise
    (output / "startup.json").write_text(
        json.dumps(
            dict(
                visual_startup_s=visual_startup_s,
                point2pose_startup_s=engine.worker.boot_latency_s,
                prepared_before_acquisition=True,
            ),
            indent=2,
        )
        + "\n"
    )
    return provider
