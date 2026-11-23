"""Validate paired receipts and print a Markdown timing table."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median


def read_run(path):
    data = json.loads((path / "result.json").read_text())
    if (data["case"], data["backend"]) != (path.parent.name, path.name):
        raise ValueError(f"Case or backend differs from the requested run in {path}")
    status = path / "returncode.txt"
    if status.exists() and status.read_text().strip() != "0":
        raise ValueError(f"Benchmark process failed in {path}")
    log = (path / "console.log").read_text()
    marker = (
        "BENCHMARK_STEEL_CUDA_GRAPH_ACTIVE"
        if data["backend"] == "steel"
        else "BENCHMARK_PHYSX_NATIVE_GPU"
    )
    if marker not in log or "dense constraint-row overflow" in log:
        raise ValueError(f"Backend or constraint capacity check failed in {path}")
    if data["cuda_graph"] != (data["backend"] == "steel"):
        raise ValueError(f"Unexpected graph status in {path}")
    with (path / "vram.csv").open(newline="") as file:
        samples = [
            (datetime.strptime(t.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(mib.strip()))
            for t, mib in csv.reader(file)
        ]
    windows = data["windows"]
    steady = [
        mib
        for timestamp, mib in samples
        if any(w["start_epoch_s"] <= timestamp <= w["start_epoch_s"] + w["wall_s"] for w in windows)
    ]
    if not steady:
        raise ValueError(f"No VRAM samples during the measurement windows in {path}")
    return data, max(steady)


def row(case, data, vram):
    windows = data["windows"]

    def med(key):
        return median(window[key] for window in windows)

    success = sum(w.get("success", False) for w in windows)
    outcomes = f"{success}/{len(windows)}" if "success" in windows[0] else "—"
    physics = f"{med('physics_ms_per_step'):.3f}" if windows[0]["physics_ms_per_step"] else "—"
    return (
        f"| {case} | {data['backend']} | {med('wall_s'):.3f} | "
        f"{med('environment_steps_per_s'):.0f} | {physics} | "
        f"{vram} | {outcomes} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    print("| Case | Backend | Wall s | Env steps/s | Physics ms/step | Peak VRAM MiB | Success |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    cases = args.cases or sorted(path.name for path in args.output.iterdir() if path.is_dir())
    for case in cases:
        physx, physx_vram = read_run(args.output / case / "physx")
        steel, steel_vram = read_run(args.output / case / "steel")
        if (
            physx["setup"] != steel["setup"]
            or physx["device_uuid"] != steel["device_uuid"]
            or [w["seed"] for w in physx["windows"]] != [w["seed"] for w in steel["windows"]]
        ):
            raise ValueError(f"Unpaired runs in {case}")
        print(row(case, physx, physx_vram))
        print(row(case, steel, steel_vram))


if __name__ == "__main__":
    main()
