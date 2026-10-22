"""Training balance and qualified native task completion boundaries."""

import json
import shutil

import h5py
import pytest
from test_pilot_evaluation import row
from test_temporal_batches import prepared_episode  # noqa: F401

from alexdoor_xas.policies.act import temporal_training
from alexdoor_xas.qualification.pilot_evaluation import PilotEvaluation
from alexdoor_xas.qualification.synthetic_probe import ProbeSetup


def test_door_sampling_ignores_duration_and_preserves_source_rows(
    request, tmp_path, monkeypatch
):
    path, cfg = request.getfixturevalue("prepared_episode")
    second = tmp_path / "short.hdf5"
    shutil.copyfile(path, second)
    with h5py.File(second, "r+") as file:
        file.attrs["episode"] = json.dumps(
            dict(episode_id="short", split="train", control_dt=1 / 60)
        )
    records = [
        dict(
            path=str(p),
            config=cfg.to_dict(),
            split="train",
            asset_id=door,
            episode_id=identity,
            strata=dict(push=rows),
        )
        for p, door, identity, rows in [
            (path, "long", "fixture", list(range(230, 299))),
            (second, "short", "short", [250]),
        ]
    ]
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(dict(records=records)))
    selected = []

    def capture(selections):
        selected.extend((episode.builder.episode_id, tuple(rows)) for episode, rows in selections)
        return {}

    monkeypatch.setattr(temporal_training, "mixed_temporal_batch", capture)
    data = temporal_training.TemporalTrainingSet(manifest)
    try:
        for seed in range(128):
            data.batch(seed, batch_size=8)
        counts = {
            name: sum(len(rows) for identity, rows in selected if name == identity)
            for name in ("fixture", "short")
        }
        assert 0.4 < counts["short"] / sum(counts.values()) < 0.6
        assert all(
            set(rows) <= ({250} if identity == "short" else set(range(230, 299)))
            for identity, rows in selected
        )
        with pytest.raises(ValueError, match="sealed"):
            temporal_training.TemporalTrainingSet(manifest, split="test")
    finally:
        data.close()


def test_four_native_samples_and_real_stabilization_required():
    score = PilotEvaluation(ProbeSetup(), 50, 1 / 60, native_samples=4, require_stabilization=True)

    def measured(t, released=False, **changes):
        return row(
            t,
            **(
                dict(
                    native_force_n=[0 if released else 1.0] * 4,
                    native_gap_m=[None if released else 0.0] * 4,
                    surface_separation_m=0.02 if released else 0.0,
                    commanded_withdrawal=released,
                    tool_position=[0, 0, 0],
                    door_speed_rad_s=0,
                    actual_joint_speed_peak_rad_s=0,
                )
                | changes
            ),
        )

    for tick in range(1, 211):
        score.update(measured(tick / 60))
    for tick in range(211, 231):
        score.update(measured(tick / 60, True))
    assert score.result()["released"] and not score.result()["task_success"]
    for tick in range(231, 351):
        score.update(measured(tick / 60, True))
    assert score.result()["stabilized"] and score.result()["task_success"]
    score.update(measured(351 / 60, True, actual_joint_speed_peak_rad_s=0.3))
    assert not score.result()["stabilized"] and not score.result()["task_success"]
    with pytest.raises(ValueError, match="native"):
        score.update(row(352 / 60))
