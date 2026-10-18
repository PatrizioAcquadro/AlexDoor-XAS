"""Linux host-RAM accounting shared by bounded pilot workloads."""

from pathlib import Path
from time import time

GIB = 1024**3


def host_memory_snapshot():
    def fields(path, names):
        return {
            key.rstrip(":"): int(value.split()[0]) * 1024
            for line in Path(path).read_text().splitlines()
            for key, value in [line.split(":", 1)]
            if key.rstrip(":") in names
        }

    memory = fields("/proc/meminfo", {"MemTotal", "MemAvailable"})
    process = fields("/proc/self/status", {"VmRSS", "VmHWM"})
    state = dict(
        time_s=time(),
        host_total_bytes=memory["MemTotal"],
        host_available_bytes=memory["MemAvailable"],
        process_rss_bytes=process["VmRSS"],
        process_peak_rss_bytes=process["VmHWM"],
        cgroup_memory_bytes=None,
        cgroup_peak_bytes=None,
    )
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    for group in groups:
        if group.startswith("0::"):
            root = Path("/sys/fs/cgroup") / group[3:].lstrip("/")
            for filename, key in (
                ("memory.current", "cgroup_memory_bytes"),
                ("memory.peak", "cgroup_peak_bytes"),
            ):
                path = root / filename
                if path.exists():
                    state[key] = int(path.read_text())
    return state


def require_host_memory(state, *, additional_bytes=0, reserve_bytes=12 * GIB):
    """Refuse before a known allocation spends the desktop/other-process reserve."""
    if additional_bytes < 0 or reserve_bytes < 0:
        raise ValueError("Host memory requirements must be nonnegative")
    available = state["host_available_bytes"]
    if available < additional_bytes + reserve_bytes:
        raise MemoryError(
            f"host_ram_reserve: {available / GIB:.2f} GiB available, "
            f"{additional_bytes / GIB:.2f} GiB additional, "
            f"{reserve_bytes / GIB:.2f} GiB reserved for desktop/other processes"
        )
