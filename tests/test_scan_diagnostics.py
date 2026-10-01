"""Scan-only inputs and retrospective comparison boundaries; no model execution."""

import json

import numpy as np
import pytest

from alexdoor_xas.perception.provider import CueEngine, GeometryProvider
from alexdoor_xas.perception.scan_diagnostics import compare_video_scan, diagnose_scan
from test_geometric_perception import EmptyWorker, recipe
from test_scan_fusion import camera_sensor


def test_scan_diagnostic_stops_at_25_and_releases_last_result_without_future_rgb(tmp_path):
    import h5py

    class RecordedWorker(EmptyWorker):
        def __init__(self):
            self.inputs = []

        def infer(self, rgb):
            self.inputs.append(int(rgb[0, 0, 0]))
            return dict(super().infer(rgb), scores=[], boxes=[])

    worker = RecordedWorker()
    provider = GeometryProvider(recipe(), CueEngine(worker, replay=True))
    path = tmp_path / "episode.hdf5"
    samples = [camera_sensor(t, 8 + i) for i, t in enumerate((0, 0.2, 25, 25.1))]
    for i, sample in enumerate(samples):
        sample["rgb"][:] = i + 1
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata").attrs["episode"] = json.dumps(
            dict(asset_id="door-2738468b94d74c5f", split="train", condition="nominal")
        )
        group = h5.create_group("observations")
        for key in samples[0]:
            group.create_dataset(key, data=np.array([s[key] for s in samples]))
        # Missing truth datasets remain irrelevant; this is purely observed geometry.
        h5.create_group("annotations")
    report = diagnose_scan(path, provider, tmp_path / "diagnostic")
    assert report["input_end_s"] == 25 and report["input_rows"] == 3
    assert report["final_available_s"] == pytest.approx(25.05)
    assert not report["offline_passed"] and not report["annotations_read"]
    assert worker.inputs == [1, 2, 3]
    assert not report["manipulation_images_read"]
    assert not (tmp_path / "diagnostic/captures/000011.png").exists()


def test_video_comparison_uses_same_captures_and_never_backdates_support(tmp_path):
    import h5py

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "captures").mkdir()
    source = tmp_path / "episode.hdf5"
    samples = [camera_sensor(t, i) for i, t in enumerate((0, 25, 25.1))]
    with h5py.File(source, "w") as h5:
        observations = h5.create_group("observations")
        for key in samples[0]:
            observations.create_dataset(key, data=np.array([s[key] for s in samples]))
    records = []
    from PIL import Image

    for row, sample in enumerate(samples[:2]):
        image, cue = f"captures/{row}.png", f"captures/{row}.npz"
        Image.fromarray(sample["rgb"]).save(baseline / image)
        np.savez_compressed(
            baseline / cue,
            boxes=[[60, 30, 180, 210]],
            tokens=np.ones((196, 384), np.float32),
            pixel_mapping=[1, 0, 0],
        )
        records.append(
            dict(row=row, frame=row, acquired_s=float(sample["time_s"]), image=image, cue=cue)
        )
    (baseline / "captures.json").write_text(json.dumps(records))

    class VideoWorker:
        def infer_video(self, images, prompts):
            assert images == [str(baseline / r["image"]) for r in records]
            assert prompts["frame_index"] == 0 and prompts["bounding_box_labels"] == [1]
            mask = np.zeros((240, 240), bool)
            mask[30:211, 60:181] = True
            return dict(
                mode="retrospective_video_diagnostic",
                latency_s=2,
                peak_memory_bytes=100,
                peak_reserved_bytes=200,
                frames=[
                    dict(
                        frame_index=i,
                        masks=[np.packbits(mask).tobytes()],
                        scores=[0.9],
                        object_ids=[1],
                    )
                    for i in range(2)
                ],
            )

    result = compare_video_scan(source, baseline, VideoWorker(), recipe().config)
    assert result["state"] == "complete" and result["mode"] == "retrospective_association_only"
    assert not result["qualified"] and not result["provider_replaced"]
    assert (baseline / "sam3-video/runtime.json").is_file()
    assert (baseline / "sam3-video/report.json").is_file()
    assert result["scan"]["objects"][0]["support"]["available_s"] == 27
