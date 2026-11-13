"""Bounded observed-only IPC for the official Point2Pose worker."""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from alexdoor_xas.perception.ipc import pack_array, receive, send
from alexdoor_xas.perception.provider import CueEngine
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
        allow_partial_reference_batch=False,
        refit_seed_rollback=False,
        selected_registration_only=False,
        performance_controls=None,
        external_masks=True,
        reference_depth_edge_filter=False,
    ):
        if refit_seed_rollback or selected_registration_only:
            raise ValueError("retired_point2pose_variant")
        boot_started = time.perf_counter()
        root, log_dir = Path(root).resolve(), Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[3]),
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
                "alexdoor_xas.perception.point2pose.worker",
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
                    allow_partial_reference_batch=allow_partial_reference_batch,
                    refit_seed_rollback=refit_seed_rollback,
                    selected_registration_only=selected_registration_only,
                    performance_controls=performance_controls,
                    external_masks=external_masks,
                    reference_depth_edge_filter=reference_depth_edge_filter,
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
    SAM3 and Point2Pose causal state never cross an episode boundary.
    """

    def __init__(self, factory, *, replay, capture_session=False, clock=None):
        super().__init__(None, replay=replay)
        self.factory, self.latest = factory, None
        self.capture_session = capture_session
        self.clock = clock
        self.queue_stats = dict(submitted=0, replaced=0, published=0, skipped_capture_ids=0)
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
        self.queue_stats = dict(submitted=0, replaced=0, published=0, skipped_capture_ids=0)
        self.last_dispatch_frame = None
        if preload and was_prepared:
            self.prepare()

    def submit(self, sensor, masks=None, *, start_s=None):
        if self.failed:
            return False
        import numpy as np

        enqueued_s = (
            self.clock()
            if self.clock
            else (float(sensor["time_s"]) if start_s is None else start_s)
        )
        copy_started = time.perf_counter()
        snapshot = {k: np.array(sensor[k], copy=True) for k in OBS_KEYS}
        for value in snapshot.values():
            value.setflags(write=False)
        masks = None if masks is None else tuple(np.array(m, copy=True) for m in masks)
        copy_s = time.perf_counter() - copy_started
        if self.pending is not None:
            if self.latest is not None:
                self.queue_stats["replaced"] += 1
            self.latest = snapshot, masks, copy_s, enqueued_s
            return False
        return self._dispatch(snapshot, masks, copy_s, start_s, enqueued_s)

    def _dispatch(self, snapshot, masks, copy_s, start_s, enqueued_s):
        dispatched_wall = time.perf_counter()
        started = (
            self.clock()
            if self.clock
            else (float(snapshot["time_s"]) if start_s is None else start_s)
        )
        boot_s = 0.0
        if self.worker is None:
            boot_started = time.perf_counter()
            if masks is None and not self.capture_session:
                return False
            self.worker = self.factory(1 if masks is None else len(masks))
            if self.capture_session:
                self.prepared = True
                boot_s = time.perf_counter() - boot_started
        worker = self.worker
        frame = int(snapshot["frame"])
        previous = getattr(self, "last_dispatch_frame", None)
        if previous is not None:
            self.queue_stats["skipped_capture_ids"] += max(0, frame - previous - 1)
        self.last_dispatch_frame = frame
        self.queue_stats["submitted"] += 1

        def infer():
            execution_wait_s = time.perf_counter() - dispatched_wall
            result = worker.infer(snapshot, masks)
            result["latency_s"] += copy_s + boot_s
            result["restart_startup_s"] = boot_s
            result["snapshot_copy_s"] = copy_s
            result["queue_wait_s"] = max(
                0.0, started - enqueued_s - (copy_s if self.clock else 0.0)
            )
            result["ingress_delay_s"] = max(0.0, enqueued_s - float(snapshot["time_s"]))
            result["executor_wait_s"] = execution_wait_s
            result["dispatch_s"] = started
            result["compute_completed_s"] = (
                self.clock() if self.clock else started + time.perf_counter() - dispatched_wall
            )
            if not self.replay:
                result["available_s"] = result["compute_completed_s"]
            return result

        result = infer() if self.replay else self.executor.submit(infer)
        if self.replay:
            result["available_s"] = started + result["latency_s"]
        self.pending = (self.generation, snapshot, started), result
        return True

    def wait(self):
        """Wait for the captured frame's result without advancing simulation."""
        if self.pending is None or self.replay:
            return
        future = self.pending[1]
        check = getattr(self, "interruption_check", None)
        if check is None:
            future.result()
            return
        from concurrent.futures import TimeoutError

        while True:
            check()
            try:
                future.result(timeout=0.1)
                return
            except TimeoutError:
                if future.done():
                    raise

    def poll(self, now):
        if self.pending is not None and not self.replay:
            (_, sensor, started), future = self.pending
            if future.done() and future.exception() is None:
                result = future.result()
                result.setdefault("available_s", started + result["latency_s"])
        try:
            event = super().poll(now)
        except (RuntimeError, EOFError, BrokenPipeError, ValueError):
            self.pending = self.latest = None
            self.failed = True
            raise
        if event is not None:
            self.queue_stats["published"] += 1
        if event is not None and self.latest is not None:
            sensor, masks, copy_s, enqueued_s = self.latest
            self.latest = None
            self._dispatch(sensor, masks, copy_s, start_s=now, enqueued_s=enqueued_s)
        return event

    def close(self):
        self.reset(preload=False)
        super().close()


def tracking_provider(recipe, models, calibration, output, *, replay, **options):
    """Shared SAM3 inspection/tracking provider with operational scheduling."""
    from alexdoor_xas.perception.session import session_provider

    return session_provider(recipe, models, calibration, output, replay=replay, **options)
