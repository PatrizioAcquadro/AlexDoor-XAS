"""Targeted Point2Pose smoke/replay, initialized only from frozen 6.0B cues."""

import json
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.evaluation import quantiles, write_json
from alexdoor_xas.perception.ipc import unpack_array
from alexdoor_xas.recording.b1 import OBS_KEYS


def sensor_at(h5, row):
    return {key: h5[f"observations/{key}"][row] for key in OBS_KEYS}


def run_point2pose_smoke(
    path, recipe, output, models, *, seed_row=240, rows=12, seed=None, selection=None
):
    """Bounded CUDA smoke through the shared session; cached RGB-D is checked only."""
    from alexdoor_xas.perception.session import PerceptionSession

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with h5py.File(path, "r") as h5:
        calibration = json.loads(h5["metadata"].attrs["calibration"])
        observed = sensor_at(h5, seed_row)
        if seed:
            with np.load(seed) as cached:
                if not all(np.array_equal(observed[k], cached[k]) for k in OBS_KEYS):
                    raise ValueError("cached_seed_capture_mismatch")
        session = PerceptionSession(
            models, recipe, calibration, output, initialize_after_s=float(observed["time_s"])
        )
        try:
            if selection is not None:
                session.select(**selection)
            write_json(output / "runtime.json", session.runtime)
            for row in range(seed_row, min(seed_row + rows * 3, len(h5["observations/time_s"])), 3):
                packet = session.infer(sensor_at(h5, row))
                if not packet["objects"]:
                    raise ValueError(packet["identity"]["state"])
                masks_out = unpack_array(packet["masks"])
                np.save(output / f"sam3-{row}.npy", masks_out)
                if row == seed_row:
                    np.savez_compressed(output / "seed.npz", **observed, mask=masks_out[0])
                objects = [
                    {
                        k: unpack_array(v) if isinstance(v, dict) and "data" in v else v
                        for k, v in obj.items()
                    }
                    for obj in packet["objects"]
                ]
                write_json(output / f"objects-{row}.json", objects)
                records.append(
                    dict(
                        row=row,
                        frame=packet["frame"],
                        capture_s=packet["capture_s"],
                        latency_s=packet["latency_s"],
                        stages=packet["stages"],
                        identity=packet["identity"],
                        lost=any(o["lost"] for o in packet["objects"]),
                        torch_peak_bytes=packet["torch_peak_bytes"],
                        cuda_tsdf=all(o["tsdf_gpu"] for o in packet["objects"]),
                    )
                )
                write_json(output / "frames.json", records)
            latency = np.asarray([r["latency_s"] for r in records[1:]])
            report = dict(
                frames=len(records),
                latency_s=quantiles(latency),
                timely_tracking_fraction=float(
                    np.mean([r["latency_s"] <= 0.15 and not r["lost"] for r in records[1:]])
                ),
                latency_gate_passed=bool(len(latency) and np.quantile(latency, 0.95) <= 0.15),
                cuda_tsdf=all(r["cuda_tsdf"] for r in records),
                accuracy_validated=False,
                qualified=False,
                loaded_contact_admitted=False,
            )
            write_json(output / "smoke.json", report)
            return report
        except BaseException as error:
            write_json(
                output / "failure.json",
                dict(
                    error=f"{type(error).__name__}: {error}",
                    frames_completed=len(records),
                    qualified=False,
                ),
            )
            raise
        finally:
            session.close()
