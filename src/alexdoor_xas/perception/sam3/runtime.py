"""Observed-only IPC for one causal SAM3 text/video session."""

import os
import subprocess
import sys
import time

import numpy as np

from alexdoor_xas.perception.ipc import receive, send


def masks_for_seed(masks, current_ids, seed_ids, shape):
    """Missing original IDs receive empty masks, including during native recovery."""
    selected = np.zeros((len(seed_ids), *shape), bool)
    for index, identity in enumerate(seed_ids):
        if identity in current_ids:
            selected[index] = masks[current_ids.index(identity)]
    return selected


class Sam3Worker:
    def __init__(self, root, config, log):
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "alexdoor_xas.perception.sam3.worker", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=dict(os.environ, OMP_NUM_THREADS="4"),
        )
        try:
            send(self.process.stdin, config)
            response = receive(self.process.stdout)
            if response is None or "error" in response:
                raise RuntimeError(f"SAM3 startup failed: {response}")
            self.runtime = response["ready"]
            self.runtime["startup_s"] = time.perf_counter() - started
        except BaseException:
            self.close()
            raise
        self.latest_masks = self.latest = None

    def infer(self, rgb, frame, time_s):
        started = time.perf_counter()
        send(
            self.process.stdin,
            dict(rgb=rgb.tobytes(), shape=rgb.shape, frame=int(frame), time_s=float(time_s)),
        )
        result = receive(self.process.stdout)
        if result is None or "error" in result:
            raise RuntimeError(f"SAM3 inference failed: {result}")
        if result["frame"] != int(frame) or result["time_s"] != float(time_s):
            raise ValueError("unsynchronized_sam3_mask")
        masks = (
            np.unpackbits(
                np.frombuffer(result.pop("masks"), np.uint8), count=np.prod(result["shape"])
            )
            .reshape(result.pop("shape"))
            .astype(bool)
        )
        seed = result["seed_ids"]
        if len(seed) != 1:
            raise ValueError("sam3_requires_one_unambiguous_leaf_seed")
        selected = masks_for_seed(masks, result["ids"], seed, rgb.shape[:2])
        result["model_latency_s"] = result["latency_s"]
        result["latency_s"] = time.perf_counter() - started
        self.latest, self.latest_masks = result, selected
        return dict(
            masks=[np.packbits(selected[0].reshape(-1)).tobytes()],
            shape=rgb.shape[:2],
            latency_s=result["latency_s"],
            sam3=result,
        )

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
