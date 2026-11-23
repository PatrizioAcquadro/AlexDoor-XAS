"""Scan-only inputs, observed references and completion timing; no model execution."""

import json

import numpy as np

from alexdoor_xas.perception.diagnostics.scan import diagnose_scan
from perception_helpers import camera_sensor


def test_scan_diagnostic_stops_at_25_and_releases_last_result_without_future_rgb(
    tmp_path, static_provider
):
    import h5py

    provider = static_provider
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
    assert report["final_available_s"] >= 25
    assert not report["offline_passed"] and not report["annotations_read"]
    assert provider.engine.worker.sam.inputs == [1, 2, 3]
    assert not report["manipulation_images_read"]
    captures = json.loads((tmp_path / "diagnostic/captures.json").read_text())
    # Gaps preserve completed captures; they do not reset the maintained session.
    assert [c["row"] for c in captures] == [0, 1, 2]
    assert captures[-1]["available_s"] == report["final_available_s"]
    assert (tmp_path / "diagnostic/scan-final.png").is_file()
