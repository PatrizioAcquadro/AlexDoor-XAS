"""Benchmark selection uses real eligibility, clocks and reset boundaries; no GPU work."""

import json
import runpy
from dataclasses import replace

import h5py
import numpy as np
import pytest
from test_temporal_batches import prepared_episode  # noqa: F401

from alexdoor_xas.dataset.temporal import TemporalEpisode


def test_selection_respects_masks_and_reads_no_images(request):
    path, config = request.getfixturevalue("prepared_episode")
    select = runpy.run_path("scripts/benchmark_temporal_pipeline.py")["select_workloads"]
    with TemporalEpisode(path, replace(config, history_s=0.2, image_offsets_s=None)) as episode:
        patterns, trace, unavailable = select(episode)
        eligible = set(episode.training_rows())
        assert episode.frames.reads == 0
        assert set(patterns) == {"distinct", "overlapping", "mixed"} and not unavailable
        assert all(len(rows) == 32 and set(rows) <= eligible for rows in patterns.values())
        assert len(set(patterns["distinct"])) == 32
        assert len(set(patterns["mixed"])) < 32
        np.testing.assert_array_equal(np.diff(patterns["overlapping"]), 1)
        assert trace["generation"] == 2 and set(trace["rows"]) <= eligible
        np.testing.assert_allclose(np.diff(trace["times"]), 6 * config.control_dt_s)
        assert not {100, 200, 300} & set(sum(patterns.values(), []) + trace["rows"])


def test_short_and_masked_sources_report_unavailable_workloads(request):
    path, config = request.getfixturevalue("prepared_episode")
    select = runpy.run_path("scripts/benchmark_temporal_pipeline.py")["select_workloads"]
    with TemporalEpisode(path, config) as episode:
        patterns, trace, unavailable = select(episode)
        assert patterns and trace is None and "incremental" in unavailable
        episode.eligibility[:] = False
        episode.eligibility[-1] = True
        patterns, trace, unavailable = select(episode)
        assert not patterns and trace is None
        assert set(unavailable) == {"distinct", "overlapping", "mixed", "incremental"}
        episode.eligibility[:] = False
        patterns, trace, unavailable = select(episode)
        assert not patterns and trace is None and len(unavailable) == 4
        assert unavailable["distinct"].endswith("found 0")
        assert episode.frames.reads == 0


def test_selection_does_not_bridge_reset_generations(request):
    path, config = request.getfixturevalue("prepared_episode")
    select = runpy.run_path("scripts/benchmark_temporal_pipeline.py")["select_workloads"]
    with TemporalEpisode(path, replace(config, history_s=0.2, image_offsets_s=None)) as episode:
        rows = episode.training_rows()
        episode.generations[rows[1::2]] += 1
        # Use the same admitted candidates to isolate the workload boundary rule.
        episode.training_rows = lambda: rows
        patterns, trace, unavailable = select(episode)
        assert set(patterns) == {"distinct"}
        assert trace is None and "overlapping" in unavailable


def test_final_test_is_rejected_before_gpu_access(request, tmp_path, monkeypatch):
    path, _ = request.getfixturevalue("prepared_episode")
    with h5py.File(path, "r+") as file:
        metadata = json.loads(file.attrs["episode"])
        metadata["split"] = "test"
        file.attrs["episode"] = json.dumps(metadata)
    module = runpy.run_path("scripts/benchmark_temporal_pipeline.py")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--source",
            "unused",
            "--prepared",
            str(path),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    def unexpected_gpu():
        pytest.fail("sealed-test refusal must precede GPU availability checks")

    monkeypatch.setattr(module["torch"].cuda, "is_available", unexpected_gpu)
    with pytest.raises(ValueError, match="sealed"):
        module["main"]()
    report = json.loads((tmp_path / "out/measurements.json").read_text())
    assert report["outcome"] == "failed" and report["optimizer_updates"] == 0
