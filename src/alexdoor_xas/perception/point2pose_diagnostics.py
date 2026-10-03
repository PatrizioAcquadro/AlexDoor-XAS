"""Targeted Point2Pose smoke/replay, initialized only from frozen 6.0B cues."""

import json
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.evaluation import quantiles, write_json
from alexdoor_xas.perception.geometry import surfaces
from alexdoor_xas.perception.point2pose_runtime import Point2PoseWorker
from alexdoor_xas.perception.point2pose_worker import unpack_array
from alexdoor_xas.perception.provider import ModelWorker
from alexdoor_xas.recording.b1 import OBS_KEYS


def sensor_at(h5, row):
    return {key: h5[f"observations/{key}"][row] for key in OBS_KEYS}


def automatic_seed(h5, row, models, config, output):
    sensor = sensor_at(h5, row)
    with (output / "automatic-candidates.log").open("w") as log:
        worker = ModelWorker(models, config, log)
        try:
            cue = worker.infer(sensor["rgb"])
        finally:
            worker.close()
    candidates = surfaces(cue, sensor, config)
    if not candidates:
        raise ValueError("no_automatic_rgbd_candidate")
    # Detection order is a diagnostic candidate, never a leaf ownership decision.
    candidate = candidates[0]
    np.savez_compressed(output / "seed.npz", **sensor, mask=candidate.observations[0].mask())
    write_json(
        output / "seed.json",
        dict(
            source_row=row,
            frame=int(sensor["frame"]),
            acquired_s=float(sensor["time_s"]),
            mask_index=candidate.observations[0].mask_index,
            candidates=len(candidates),
            selection="first automatic RGB-D component; diagnostic only",
            ownership_confirmed=False,
            loaded_contact_admitted=False,
        ),
    )
    return sensor, [candidate.observations[0].mask()]


def run_point2pose_smoke(path, recipe, output, models, *, seed_row=240, rows=12, seed=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        dict(
            mode="point2pose-smoke",
            episode=str(path),
            seed_row=seed_row,
            rows=rows,
            training_started=False,
            sealed_test_evaluated=False,
            qualified=False,
            useful_availability_min=0.95,
            position_limit_m=0.01,
            rotation_limit_deg=5,
            max_dynamic_age_s=0.15,
        ),
    )
    with h5py.File(path, "r") as h5:
        calibration = json.loads(h5["metadata"].attrs["calibration"])
        if seed is None:
            sensor, masks = automatic_seed(h5, seed_row, models, recipe.config, output)
        else:
            with np.load(seed) as cached:
                sensor = {key: cached[key] for key in OBS_KEYS}
                masks = [cached["mask"]]
            if not all(
                np.array_equal(sensor[key], sensor_at(h5, seed_row)[key]) for key in OBS_KEYS
            ):
                raise ValueError("cached_seed_capture_mismatch")
        worker = Point2PoseWorker(
            models / "point2pose",
            calibration["depth_interval_m"],
            recipe.config["plane_tolerance_m"],
            len(masks),
            output,
        )
        records = []
        try:
            write_json(output / "runtime.json", worker.runtime)
            for row in range(seed_row, min(seed_row + rows * 3, len(h5["observations/time_s"])), 3):
                result = worker.infer(sensor_at(h5, row), masks if row == seed_row else None)
                masks_out = unpack_array(result.pop("masks"))
                np.save(output / f"sam2-{row}.npy", masks_out)
                objects = []
                for obj in result.pop("objects"):
                    objects.append(
                        {
                            key: unpack_array(value)
                            if isinstance(value, dict) and "data" in value
                            else value
                            for key, value in obj.items()
                        }
                    )
                write_json(output / f"objects-{row}.json", objects)
                result.update(
                    row=row,
                    objects=[
                        dict(
                            object_id=o["object_id"],
                            lost=o["lost"],
                            correspondences=len(o["source_points"]),
                            tsdf_gpu=o["tsdf_gpu"],
                            tsdf_voxels=o["tsdf_voxels"],
                        )
                        for o in objects
                    ],
                )
                records.append(result)
                write_json(output / "frames.json", records)
            latency = [r["latency_s"] for r in records[1:]]
            useful = [
                r["latency_s"] <= 0.15
                and any(not o["lost"] and o["correspondences"] >= 5 for o in r["objects"])
                for r in records[1:]
            ]
            report = dict(
                frames=len(records),
                latency_s=quantiles(np.asarray(latency)),
                timely_tracking_fraction=float(np.mean(useful)),
                latency_gate_passed=bool(latency and np.quantile(latency, 0.95) <= 0.15),
                cuda_tsdf=all(o["tsdf_gpu"] for r in records for o in r["objects"]),
                torch_peak_bytes=max(r["torch_peak_bytes"] for r in records),
                accuracy_validated=False,
                qualified=False,
                loaded_contact_admitted=False,
            )
            write_json(output / "smoke.json", report)
            return report
        except Exception as error:
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
            worker.close()
