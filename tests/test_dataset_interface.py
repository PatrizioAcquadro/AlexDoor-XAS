"""Tests for A1-A4 loading, validation, normalization, and batching."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
    A4_PHASE_VOCAB,
    EE_DELTA_DIM,
)
from alexdoor_xas.dataset.export import export_datasets
from alexdoor_xas.dataset.loader import A4ChunkDataset, EpisodeDataset, obs_matrix
from alexdoor_xas.dataset.normalize import (
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path,
    save_norm_stats,
    validate_norm_stats,
)
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler
from alexdoor_xas.dataset.validate import (
    validate_a4_dataset,
    validate_dataset,
    validate_episode,
    validate_matched_action_space_datasets,
)
from conftest import make_episode

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)
pytestmark = requires_h5py

N_EPISODES = 4
OBS_KEYS = ("joint_pos", "joint_vel")


def _copy_dataset(src: Path, tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def _jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _export(tmp_root):
    return export_datasets(
        [make_episode(seed=seed) for seed in range(N_EPISODES)], tmp_root, version="v0"
    )


@pytest.fixture(scope="module")
def synthetic_exports(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="module")
def synthetic_a2(synthetic_exports) -> EpisodeDataset:
    return EpisodeDataset(synthetic_exports[A2_EE_DELTA])


def test_dataset_loads_records_with_stacked_arrays(synthetic_a2) -> None:
    assert len(synthetic_a2) == N_EPISODES
    assert synthetic_a2.action_space == A2_EE_DELTA
    assert synthetic_a2.action_dim == EE_DELTA_DIM
    record = synthetic_a2[0]
    assert record.n_steps > 0
    assert record.actions.shape == (record.n_steps, EE_DELTA_DIM)
    assert record.t.shape == (record.n_steps,)
    assert record.schema_version == "phase2.v2"
    assert record.success and record.termination_reason == "controller_done"
    assert "failure_label" not in record.__dataclass_fields__
    assert {"joint_pos", "joint_vel"} <= set(record.obs)
    assert "door_angle_rad" not in record.obs
    assert "object_state.door_angle_rad" in record.diagnostics
    assert "contact.inferred" in record.diagnostics
    assert synthetic_a2.by_id(record.episode_id) is record
    with pytest.raises(KeyError):
        synthetic_a2.by_id("no-such-episode")


def test_a1_dataset_has_joint_wide_actions(synthetic_exports) -> None:
    a1 = EpisodeDataset(synthetic_exports[A1_JOINT_DELTA])
    assert a1.action_dim == 7
    assert validate_dataset(a1).ok


def test_episode_ids_shared_across_action_spaces(synthetic_exports) -> None:
    ids = {
        space: sorted(EpisodeDataset(path).episode_ids)
        for space, path in synthetic_exports.items()
        if space != A4_OBJ_CENTRIC_CHUNK
    }
    a4_ids = sorted(A4ChunkDataset(synthetic_exports[A4_OBJ_CENTRIC_CHUNK]).episode_ids)
    reference = ids[A2_EE_DELTA]
    assert all(episode_ids == reference for episode_ids in ids.values())
    assert a4_ids == reference


def test_export_preserves_existing_versions_and_rejects_incomplete_recording(tmp_path):
    existing = _export(tmp_path)
    before = {p: p.read_bytes() for folder in existing.values() for p in folder.iterdir()}
    incomplete = make_episode()
    incomplete.extras.pop("action_door_frame")
    with pytest.raises(FileExistsError, match="already exists"):
        export_datasets([incomplete], tmp_path)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    with pytest.raises(ValueError, match="incomplete matched-action"):
        export_datasets([incomplete], tmp_path, version="next")
    assert not any(tmp_path.rglob("next"))


@pytest.mark.parametrize("failure", ["validation", "publication"])
def test_export_failure_leaves_no_partial_version(tmp_path, monkeypatch, failure):
    episode = make_episode()
    if failure == "validation":
        episode.extras["a4_chunks"][0]["duration_ticks"] = 0
    else:
        rename = Path.rename

        def fail_a3(path, target):
            if Path(target).parent.name == A3_OBJ_REL_EE_DELTA:
                raise OSError("publication interrupted")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_a3)
    with pytest.raises((ValueError, OSError)):
        export_datasets([episode], tmp_path)
    assert not any(tmp_path.rglob("v0"))
    assert not any(tmp_path.rglob(".export-*"))


def test_export_rejects_duplicate_ids_and_keeps_distinct_shared_prefixes(tmp_path):
    first, second = make_episode(1), make_episode(2)
    with pytest.raises(ValueError, match="ids must be unique"):
        export_datasets([first, first], tmp_path)
    first.meta = dataclasses.replace(first.meta, episode_id="samehead-one")
    second.meta = dataclasses.replace(second.meta, episode_id="samehead-two")
    paths = export_datasets([first, second], tmp_path)
    assert len(EpisodeDataset(paths[A2_EE_DELTA])) == 2


def test_observations_select_proprioception_in_explicit_order(synthetic_a2) -> None:
    record = synthetic_a2[0]
    obs = obs_matrix(record, OBS_KEYS)
    assert obs.shape == (record.n_steps, 14)
    np.testing.assert_array_equal(obs[:, :7], record.obs["joint_pos"])
    reversed_obs = obs_matrix(record, tuple(reversed(OBS_KEYS)))
    np.testing.assert_array_equal(reversed_obs[:, 7:], obs[:, :7])


@pytest.mark.parametrize("keys", [(), "core", ("joint_pos", "joint_pos")])
def test_observations_require_an_ordered_unique_selection(synthetic_a2, keys) -> None:
    with pytest.raises(ValueError, match="obs_keys"):
        obs_matrix(synthetic_a2[0], keys)


@pytest.mark.parametrize("key", ["door_angle_rad", "object_state.door_angle_rad", "contact.sensed"])
def test_diagnostics_cannot_be_selected_as_observations(synthetic_a2, key) -> None:
    with pytest.raises(ValueError, match="missing proprioceptive field"):
        obs_matrix(synthetic_a2[0], (key,))


def test_a4_dataset_parses_and_validates_chunks(synthetic_exports) -> None:
    a4 = A4ChunkDataset(synthetic_exports[A4_OBJ_CENTRIC_CHUNK])
    assert len(a4) == N_EPISODES
    record = a4[0]
    assert record.chunks and all(c.phase in A4_PHASE_VOCAB for c in record.chunks)
    assert validate_a4_dataset(a4).ok


def test_a4_validation_fails_closed_on_missing_outcome(synthetic_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(synthetic_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_missing_outcome")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["outcome"] = None
    _write_jsonl_records(jsonl, records)

    with pytest.raises(ValueError, match="outcome"):
        A4ChunkDataset(a4_dir)


def test_a4_validation_rejects_duplicate_ids(synthetic_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(synthetic_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_duplicate")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[1]["meta"]["episode_id"] = records[0]["meta"]["episode_id"]
    _write_jsonl_records(jsonl, records)

    result = validate_a4_dataset(A4ChunkDataset(a4_dir))

    assert not result.ok
    assert any("duplicate episode ids" in error for error in result.errors)


def test_a4_validation_rejects_bad_target_width(synthetic_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(synthetic_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_bad_target")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["chunks"][0]["contact_target_panel"] = [0.1, 0.2]
    _write_jsonl_records(jsonl, records)

    result = validate_a4_dataset(A4ChunkDataset(a4_dir))

    assert not result.ok
    assert any("contact_target_panel" in error for error in result.errors)


def test_validate_passes_on_exported_datasets(synthetic_a2, synthetic_exports) -> None:
    assert validate_dataset(synthetic_a2, A2_EE_DELTA).ok
    assert validate_dataset(EpisodeDataset(synthetic_exports[A3_OBJ_REL_EE_DELTA])).ok


def test_validate_catches_planted_defects(synthetic_a2) -> None:
    record = synthetic_a2[0]

    mismatch = validate_episode(record, expected_space=A3_OBJ_REL_EE_DELTA)
    assert any("does not match" in e for e in mismatch.errors)

    bad_actions = record.actions.copy()
    bad_actions[3, 0] = np.nan
    non_finite = validate_episode(dataclasses.replace(record, actions=bad_actions))
    assert any("non-finite action" in e for e in non_finite.errors)

    unknown = validate_episode(dataclasses.replace(record, schema_version="phase9.v9"))
    assert any("unknown schema_version" in e for e in unknown.errors)

    truncated = validate_episode(dataclasses.replace(record, t=record.t[:-2]))
    assert any("inconsistent step counts" in e for e in truncated.errors)


def test_validate_rejects_unknown_termination_reason(synthetic_a2) -> None:
    record = synthetic_a2[0]
    result = validate_episode(
        dataclasses.replace(record, success=False, termination_reason="novel_interpretation")
    )

    assert any("unknown termination_reason" in error for error in result.errors)


def test_validate_rejects_bad_timing_and_control_dt(synthetic_a2) -> None:
    record = synthetic_a2[0]
    bad_t = record.t.copy()
    bad_t[1] += record.meta["control_dt"] * 0.5
    result = validate_episode(dataclasses.replace(record, t=bad_t))
    assert any("timestamp deltas" in error for error in result.errors)

    bad_meta = dict(record.meta)
    bad_meta["control_dt"] = -0.01
    result = validate_episode(dataclasses.replace(record, meta=bad_meta))
    assert any("control_dt" in error for error in result.errors)


def test_validate_rejects_bad_contact_flags_and_sources(synthetic_a2) -> None:
    record = synthetic_a2[0]
    for change, message in (
        ({"sensed": 0.5}, "contact flag"),
        ({"source": ""}, "contact source"),
        ({"force_n": -1}, "force_n"),
    ):
        contact = {**record.buffer.steps[0].contact, **change}
        step = dataclasses.replace(record.buffer.steps[0], contact=contact)
        buffer = dataclasses.replace(record.buffer, steps=[step, *record.buffer.steps[1:]])
        result = validate_episode(dataclasses.replace(record, buffer=buffer))
        assert any(message in error for error in result.errors)


def test_validate_episode_reports_malformed_observation_shape(synthetic_a2) -> None:
    record = synthetic_a2[0]
    bad_obs = dict(record.obs)
    bad_obs["joint_pos"] = np.asarray(0.0)

    result = validate_episode(dataclasses.replace(record, obs=bad_obs))

    assert not result.ok
    assert any("obs 'joint_pos'" in error and "shape" in error for error in result.errors)


def test_numerical_validation_uses_explicit_limits_without_a_force_admission_default(synthetic_a2):
    record = synthetic_a2[0]
    step = dataclasses.replace(
        record.buffer.steps[0], contact={"source": "recorded_normal_force", "force_n": 250.0}
    )
    buffer = dataclasses.replace(record.buffer, steps=[step, *record.buffer.steps[1:]])
    assert validate_episode(dataclasses.replace(record, buffer=buffer)).ok
    obs = dict(record.obs)
    obs["joint_pos_target"] = np.full_like(obs["joint_pos_target"], 2.0)
    extras = {**record.extras, "joint_pos_limits": np.tile([-1.0, 1.0], (7, 1))}
    result = validate_episode(dataclasses.replace(record, obs=obs, extras=extras))
    assert any("targets exceed" in error for error in result.errors)


def test_validate_rejects_mislabeled_a3_actions(synthetic_exports) -> None:
    a3 = EpisodeDataset(synthetic_exports[A3_OBJ_REL_EE_DELTA])
    record = a3[0]
    bad_actions = record.actions.copy()
    bad_actions[0, 0] += 0.25

    result = validate_episode(dataclasses.replace(record, actions=bad_actions))

    assert any("action_door_frame" in error for error in result.errors)


def test_validate_dataset_reports_malformed_meta_and_action_rank(
    synthetic_exports, tmp_path
) -> None:
    dataset_dir = _copy_dataset(synthetic_exports[A2_EE_DELTA], tmp_path, "bad_meta")
    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["action_space"]
    meta_path.write_text(json.dumps(meta) + "\n")

    result = validate_dataset(EpisodeDataset(dataset_dir))
    assert not result.ok
    assert any("action_space" in error for error in result.errors)

    rank_dir = _copy_dataset(synthetic_exports[A2_EE_DELTA], tmp_path, "bad_action_rank")
    dataset = EpisodeDataset(rank_dir)
    n_steps = dataset[0].n_steps
    import h5py

    h5_path = sorted(rank_dir.glob("episode_*.hdf5"))[0]
    with h5py.File(h5_path, "r+") as h5:
        del h5["steps/action"]
        h5["steps"].create_dataset("action", data=np.zeros(n_steps))

    result = validate_dataset(EpisodeDataset(rank_dir))
    assert not result.ok
    assert any("rank 2" in error for error in result.errors)


def test_matched_action_space_validation_rejects_same_id_mismatched_content(
    synthetic_exports,
) -> None:
    a2 = EpisodeDataset(synthetic_exports[A2_EE_DELTA])
    a3 = EpisodeDataset(synthetic_exports[A3_OBJ_REL_EE_DELTA])
    a3.records = list(a3.records)
    bad_meta = dict(a3[0].meta)
    bad_meta["seed"] = int(bad_meta["seed"]) + 1000
    diagnostics = dict(a3[0].diagnostics)
    diagnostics["object_state.door_angle_rad"] = diagnostics["object_state.door_angle_rad"].copy()
    diagnostics["object_state.door_angle_rad"][0] += 1.0
    a3.records[0] = dataclasses.replace(a3[0], meta=bad_meta, diagnostics=diagnostics)

    result = validate_matched_action_space_datasets({A2_EE_DELTA: a2, A3_OBJ_REL_EE_DELTA: a3})

    assert not result.ok
    assert any("meta.seed differs" in error for error in result.errors)
    assert any("diagnostics field" in error for error in result.errors)


def test_norm_stats_roundtrip_and_positive_std(synthetic_a2, tmp_path) -> None:
    train_ids = synthetic_a2.episode_ids[:3]
    stats = compute_norm_stats(synthetic_a2, train_ids, OBS_KEYS)
    assert stats.action.dim == EE_DELTA_DIM
    assert stats.obs.dim == 14 and stats.obs_keys == OBS_KEYS
    assert (stats.action.std > 0.0).all()

    actions = synthetic_a2[0].actions
    np.testing.assert_allclose(
        stats.action.denormalize(stats.action.normalize(actions)), actions, atol=1e-9
    )

    path = save_norm_stats(norm_stats_path(tmp_path), stats)
    loaded = load_norm_stats(path)
    np.testing.assert_array_equal(loaded.action.mean, stats.action.mean)
    np.testing.assert_array_equal(loaded.obs.std, stats.obs.std)
    assert loaded.train_episode_ids == tuple(train_ids)
    assert validate_norm_stats(loaded, synthetic_a2, train_ids, OBS_KEYS) == []

    errors = validate_norm_stats(stats, synthetic_a2, train_ids, tuple(reversed(OBS_KEYS)))
    assert any("obs_keys" in error for error in errors)
    payload = json.loads(path.read_text())
    payload["obs_preset"] = payload.pop("obs_keys")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="obs_keys"):
        load_norm_stats(path)


def test_norm_stats_validation_rejects_stale_or_wrong_dimension_stats(synthetic_a2) -> None:
    train_ids = synthetic_a2.episode_ids[:3]
    stats = compute_norm_stats(synthetic_a2, train_ids, OBS_KEYS)

    stale_action = dataclasses.replace(
        stats.action,
        mean=stats.action.mean + 1.0,
    )
    stale = dataclasses.replace(stats, action=stale_action)
    assert any(
        "recomputed action mean" in error
        for error in validate_norm_stats(stale, synthetic_a2, train_ids, OBS_KEYS)
    )

    wrong_train = validate_norm_stats(stats, synthetic_a2, list(reversed(train_ids)), OBS_KEYS)
    assert any("train_episode_ids" in error for error in wrong_train)

    bad_action = dataclasses.replace(
        stats.action,
        mean=stats.action.mean[:-1],
        std=stats.action.std[:-1],
        min=stats.action.min[:-1],
        max=stats.action.max[:-1],
    )
    wrong_dim = dataclasses.replace(stats, action=bad_action)
    assert any(
        "action dim" in error
        for error in validate_norm_stats(wrong_dim, synthetic_a2, train_ids, OBS_KEYS)
    )


def test_chunk_sampler_windows_and_pads(synthetic_a2) -> None:
    horizon = 10
    sampler = ChunkSampler(synthetic_a2, obs_keys=OBS_KEYS, horizon=horizon)
    assert len(sampler) == sum(r.n_steps for r in synthetic_a2.records)
    assert sampler.obs_dim == 14 and sampler.action_dim == EE_DELTA_DIM

    record = synthetic_a2[0]
    mid = sampler.sample(5)
    np.testing.assert_array_equal(mid.actions, record.actions[5 : 5 + horizon])
    assert not mid.is_pad.any()
    np.testing.assert_array_equal(mid.obs, obs_matrix(record, OBS_KEYS)[5])

    last = sampler.sample(record.n_steps - 1)
    np.testing.assert_array_equal(last.actions[0], record.actions[-1])
    assert not last.is_pad[0] and last.is_pad[1:].all()
    assert (last.actions[1:] == 0.0).all()


def test_chunk_sampler_respects_split_restriction(synthetic_a2) -> None:
    chosen = [synthetic_a2.episode_ids[1]]
    sampler = ChunkSampler(synthetic_a2, obs_keys=OBS_KEYS, horizon=4, episode_ids=chosen)
    assert len(sampler) == sum(synthetic_a2.by_id(e).n_steps for e in chosen)
    np.testing.assert_array_equal(
        sampler.sample(0).actions[0], synthetic_a2.by_id(chosen[0]).actions[0]
    )


def test_batch_iterator_is_seeded_and_shaped(synthetic_a2) -> None:
    sampler = ChunkSampler(synthetic_a2, obs_keys=OBS_KEYS, horizon=8)
    iterator = BatchIterator(sampler, batch_size=16, seed=3)
    first_pass = list(iterator)
    second_pass = list(BatchIterator(sampler, batch_size=16, seed=3))
    assert len(first_pass) == len(iterator)
    for a, b in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(a["obs"], b["obs"])
        np.testing.assert_array_equal(a["actions"], b["actions"])
        np.testing.assert_array_equal(a["is_pad"], b["is_pad"])

    batch = first_pass[0]
    assert set(batch) == {"obs", "actions", "is_pad"}
    assert batch["obs"].shape == (16, 14)
    assert batch["actions"].shape == (16, 8, EE_DELTA_DIM)
    assert batch["is_pad"].shape == (16, 8) and batch["is_pad"].dtype == bool
    dropped = list(BatchIterator(sampler, batch_size=100, seed=0, drop_last=True))
    assert all(b["obs"].shape[0] == 100 for b in dropped)
