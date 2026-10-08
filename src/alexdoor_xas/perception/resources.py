"""Sample simultaneous worker/coordinator residency; never affects inference decisions."""

import json
import os
import subprocess
import threading
from pathlib import Path


class SessionResources:
    def __init__(self, workers, output):
        self.pids = [os.getpid(), *(w.process.pid for w in workers)]
        self.worker_pids = self.pids[1:]
        self.capture_s = None
        self.latest = None
        self.peaks = dict(gpu_bytes=0, rss_bytes=0)
        self.stop = threading.Event()
        self.stream = Path(output).open("a", buffering=1)
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.is_set():
            try:
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                residency = {
                    int(pid): int(memory) * 1024**2
                    for pid, memory in (line.split(",") for line in gpu.stdout.splitlines())
                }
                rss = {}
                for pid in self.pids:
                    status = Path(f"/proc/{pid}/status").read_text()
                    rss[pid] = (
                        int(
                            next(
                                line.split()[1]
                                for line in status.splitlines()
                                if line.startswith("VmRSS:")
                            )
                        )
                        * 1024
                    )
                current = dict(
                    capture_s=self.capture_s,
                    gpu_bytes=sum(residency.get(pid, 0) for pid in self.worker_pids),
                    rss_bytes=sum(rss.values()),
                    per_process_rss=rss,
                    per_worker_gpu={pid: residency.get(pid, 0) for pid in self.worker_pids},
                )
                for key in self.peaks:
                    self.peaks[key] = max(self.peaks[key], current[key])
                self.latest = current
                self.stream.write(json.dumps(current) + "\n")
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
                self.stream.write(
                    json.dumps(dict(capture_s=self.capture_s, error=str(error))) + "\n"
                )
            self.stop.wait(0.25)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.stream.close()
