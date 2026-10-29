"""Scan-only inputs, observed references and completion timing; no model execution."""

import json

import numpy as np

from alexdoor_xas.perception.diagnostics.scan import diagnose_scan
from alexdoor_xas.perception.session import session_provider
from perception_helpers import EmptyWorker, camera_sensor, recipe


def test_scan_diagnostic_stops_at_25_and_releases_last_result_without_future_rgb(
    tmp_path, monkeypatch
):
    import h5py

    inputs = []

    class RecordedWorker(EmptyWorker):
        runtime = {}

        def infer(self, rgb, frame, time_s):
            inputs.append(int(rgb[0, 0, 0]))
            self.all_masks = np.empty((0, *rgb.shape[:2]), bool)
            return dict(
                super().infer(rgb),
                ids=[],
                scores=[],
                boxes=[],
                sam3=dict(frame=frame, time_s=time_s),
            )

        def close(self):
            pass

    monkeypatch.setattr(
        "alexdoor_xas.perception.sam3.runtime.Sam3Worker", lambda *args: RecordedWorker()
    )
    provider = session_provider(
        recipe(), "unused", {}, tmp_path / "session", replay=True, tracking=False
    )
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
    try:
        report = diagnose_scan(path, provider, tmp_path / "diagnostic")
    finally:
        provider.close()
    assert report["input_end_s"] == 25 and report["input_rows"] == 3
    assert 25 < report["final_available_s"] < 25.1
    assert not report["offline_passed"] and not report["annotations_read"]
    assert inputs == [1, 2, 3]
    assert not report["manipulation_images_read"]
    captures = json.loads((tmp_path / "diagnostic/captures.json").read_text())
    # Capture gaps never fabricate steps or refresh old support.
    assert [c["row"] for c in captures] == [0, 1, 2]
    assert captures[-1]["available_s"] == report["final_available_s"]
    assert (tmp_path / "diagnostic/scan-final.png").is_file()
