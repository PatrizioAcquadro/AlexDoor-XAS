"""Coverage and usable windows remain distinct; optional tau gaps are causal masks."""

import json

import h5py
import numpy as np
import pytest

from alexdoor_xas.dataset.temporal import raw_streams
from alexdoor_xas.policies.temporal import CausalObservationBuilder, TemporalConfig
from alexdoor_xas.recording.quality import review_native_collection, teacher_eligibility
from helpers import temporal_command, temporal_recorder


def make_collection(tmp_path, substeps=2):
    path = tmp_path / "complete.hdf5"
    log = temporal_recorder(path)
    names = ["inspect", "approach", "contact", "push", "hold", "release", "stabilize"]
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.1)
    log.file["metadata"].attrs["episode"] = json.dumps(
        dict(
            split="train",
            asset_id="fixture",
            episode_id="attempt-1",
            phase_names=names,
            observation_config=cfg.to_dict(),
            sim_dt=1 / (60 * substeps),
        )
    )
    log.file["metadata"].attrs["arm_effort_limits_nm"] = json.dumps([100] * 7)
    log.reset(0)
    for tick in range(169):
        t = tick / 60
        if tick:
            for offset in range(substeps - 1, 0, -1):
                native_time = t - offset / (60 * substeps)
                log.feedback(
                    np.ones(7),
                    acquired=native_time,
                    available=native_time,
                    healthy=True,
                    generation=0,
                )
        log.feedback(np.ones(7), acquired=t, available=t, healthy=True, generation=0)
        log.state(np.zeros(9), np.zeros(9), acquired=t, available=t, generation=0)
        if tick % 6 == 0:
            depth = np.ones((4, 6, 1), "f4")
            log.capture(
                np.zeros((4, 6, 3), "u1"),
                depth,
                np.ones_like(depth, bool),
                np.eye(3),
                np.eye(4),
                acquired=t,
                available=t,
                generation=0,
            )
        if tick < 168:
            temporal_command(log, t)
            log.annotate(
                dict(
                    phase=np.int8(tick // 24),
                    physical_safe=np.bool_(True),
                    phase_valid=np.bool_(True),
                    contact_valid=np.bool_(True),
                )
            )
    log.finish(dict(passed=True, released=True, stabilized=True, termination="fixture"))
    log.close()
    return path, cfg


@pytest.fixture
def collection(tmp_path):
    return make_collection(tmp_path)


@pytest.mark.parametrize("substeps", [2, 4])
def test_native_feedback_denominator_uses_recorded_physics_cadence(tmp_path, substeps):
    path, _ = make_collection(tmp_path, substeps)
    report = review_native_collection(path)
    assert report["expected_native_torque_samples"] == 1 + 168 * substeps
    assert report["native_torque_samples"] == report["expected_native_torque_samples"]
    assert report["native_feedback_coverage_complete"] and report["integrity_passed"]


def test_incomplete_transition_preserves_trusted_later_windows(collection):
    path, _ = collection
    full = review_native_collection(path)
    assert full["full_demonstration_coverage"] and full["data_usable"]
    with h5py.File(path, "r+") as file:
        metadata = json.loads(file["metadata"].attrs["episode"])
        metadata["observation_config"]["static_time_s"] = 0.4
        file["metadata"].attrs["episode"] = json.dumps(metadata)
    partial = review_native_collection(path)
    assert partial["complete"] and partial["integrity_passed"]
    assert partial["outcome"]["passed"] and partial["data_usable"]
    assert not partial["full_demonstration_coverage"]
    assert partial["missing_required_transitions"] == [
        dict(from_phase="inspect", to_phase="approach")
    ]
    assert partial["training_windows"] > 0


def test_torque_health_gap_keeps_masks_provenance_and_window_coverage(collection):
    path, cfg = collection
    baseline = review_native_collection(path)
    with h5py.File(path, "r+") as file:
        file["physics_feedback/healthy"][120] = False  # actual selected 1 s sample
        file["physics_feedback/torque_nm"][120] = np.nan
    gap = review_native_collection(path)
    assert gap["data_usable"] and gap["full_demonstration_coverage"]
    assert gap["training_windows"] == baseline["training_windows"]
    assert (
        0
        < gap["torque_support"]["windows_with_missing_past_or_current_tau"]
        < gap["training_windows"]
    )
    assert gap["torque_missing_or_unhealthy_samples"] == 1
    assert gap["torque_support"]["feedback_contract"]["source"] == "simulated_actuator_feedback"
    with h5py.File(path) as file:
        streams, *_ = raw_streams(file, cfg)
        snapshot, _ = CausalObservationBuilder(cfg, streams, episode_id="attempt-1").build_support(
            1, 0
        )
        assert not snapshot["obs"]["robot_valid"][-1, 2]
        np.testing.assert_array_equal(snapshot["obs"]["robot"][-1, 18:25], 0)
        assert np.isnan(file["physics_feedback/torque_nm"][120]).all()


def test_missing_unselected_substep_is_reported_without_blanket_rejection(collection):
    path, _ = collection
    with h5py.File(path, "r+") as file:
        group = file["physics_feedback"]
        for name in list(group):
            values = np.delete(group[name][:], 121, axis=0)  # between control slots
            del group[name]
            group.create_dataset(name, data=values)
    report = review_native_collection(path)
    assert report["integrity_passed"] and not report["native_feedback_coverage_complete"]
    assert report["data_usable"] and report["full_demonstration_coverage"]
    assert report["torque_support"]["missing_selected_slots"] == 0


def test_unsafe_ambiguous_labels_exclude_entire_crossing_chunks(collection):
    path, _ = collection
    with h5py.File(path, "r+") as file:
        file["evaluator/physical_safe"][70] = False
        file["evaluator/phase_valid"][90] = False
        eligible = teacher_eligibility(file)
        assert not eligible[70] and not eligible[90]
    report = review_native_collection(path)
    rows = [i for a, b in report["usable_window_ranges"] for i in range(a, b + 1)]
    assert report["unsafe_commands"] == 1
    assert not set(range(59, 71)).intersection(rows)
    assert not set(range(79, 91)).intersection(rows)
    with h5py.File(path, "r+") as file:
        outcome = json.loads(file["metadata"].attrs["outcome"])
        outcome["passed"] = False
        file["metadata"].attrs["outcome"] = json.dumps(outcome)
    assert not review_native_collection(path)["data_usable"]


def test_interrupted_terminal_support_never_auto_admits_expert(collection):
    path, _ = collection
    with h5py.File(path, "r+") as file:
        file.attrs["complete"] = False
        file["metadata"].attrs["terminal_support"] = False
        # Interrupted prefix with real final command and no fabricated post-state.
        group = file["control"]
        for name in list(group):
            values = group[name][:-1]
            del group[name]
            group.create_dataset(name, data=values)
    report = review_native_collection(path)
    assert report["integrity_passed"] and report["missing_terminal_states"] == 1
    assert not report["terminal_support"] and not report["complete"]
    assert not report["data_usable"] and not report["full_demonstration_coverage"]
    assert report["training_windows"] == 0


def test_declared_short_subtask_needs_qualification_but_not_release(collection):
    path, _ = collection
    with h5py.File(path, "r+") as file:
        # Complete contact-acquisition fixture, genuinely ending before push.
        for group_name, count in (
            ("commands", 72),
            ("evaluator", 72),
            ("control", 73),
            ("physics_feedback", 145),
            ("captures", 13),
        ):
            group = file[group_name]
            for name in list(group):
                values = group[name][:count]
                del group[name]
                group.create_dataset(name, data=values)
        metadata = json.loads(file["metadata"].attrs["episode"])
        metadata["subtask"] = dict(
            name="contact_acquisition", required_phases=["inspect", "approach", "contact"]
        )
        file["metadata"].attrs["episode"] = json.dumps(metadata)
        file["metadata"].attrs["outcome"] = json.dumps(
            dict(
                passed=True,
                subtask_success=True,
                subtask_qualified=False,
                released=False,
                stabilized=False,
                termination="subtask_fixture",
            )
        )
    unqualified = review_native_collection(path)
    assert unqualified["integrity_passed"] and unqualified["complete"]
    assert unqualified["subtask_success"] and not unqualified["subtask_qualified"]
    assert not unqualified["data_usable"] and not unqualified["full_task_success"]
    with h5py.File(path, "r+") as file:
        outcome = json.loads(file["metadata"].attrs["outcome"])
        outcome["subtask_qualified"] = True  # test-only declared review, no physical claim
        file["metadata"].attrs["outcome"] = json.dumps(outcome)
    qualified = review_native_collection(path)
    assert qualified["data_usable"] and qualified["required_scope_coverage"]
    assert not qualified["full_demonstration_coverage"] and not qualified["full_task_success"]
    assert not qualified["missing_required_transitions"]
    with h5py.File(path, "r+") as file:
        outcome["passed"] = False
        file["metadata"].attrs["outcome"] = json.dumps(outcome)
    assert not review_native_collection(path)["data_usable"]


def test_injected_commands_remain_causal_history_but_never_action_targets(collection):
    path, cfg = collection
    with h5py.File(path, "r+") as file:
        roles = np.zeros(168, "i1")
        roles[72:78] = 1
        roles[78:90] = 2
        file["evaluator"].create_dataset("action_role", data=roles)
        file["commands/applied_target"][72] = 0.25
        eligible = teacher_eligibility(file)
        assert not eligible[72:78].any()
        assert eligible[78:90].all()
        streams, *_ = raw_streams(file, cfg)
        assert set(streams) == {"images", "q", "dq", "tau", "target"}
        np.testing.assert_array_equal(streams["target"]["values"][72], 0.25)
        snap, _ = CausalObservationBuilder(cfg, streams, episode_id="attempt-1").build_support(
            73 / 60, 0
        )
        np.testing.assert_array_equal(snap["obs"]["robot"][-1, -7:], 0.25)
    report = review_native_collection(path)
    assert report["integrity_passed"]
    starts = {i for a, b in report["usable_window_ranges"] for i in range(a, b + 1)}
    assert not starts.intersection(range(61, 78))
    assert 78 in starts


def test_short_cleanup_is_recorded_but_excluded_from_expert_targets(collection):
    path, _ = collection
    with h5py.File(path, "r+") as file:
        metadata = json.loads(file["metadata"].attrs["episode"])
        metadata["subtask"] = dict(
            name="A",
            required_phases=["approach", "contact"],
            supervised_phases=["approach", "contact"],
        )
        file["metadata"].attrs["episode"] = json.dumps(metadata)
        outcome = json.loads(file["metadata"].attrs["outcome"])
        outcome.update(subtask_success=True, subtask_qualified=True)
        file["metadata"].attrs["outcome"] = json.dumps(outcome)
        eligible = teacher_eligibility(file)
        assert eligible[24:72].all()
        assert not eligible[72:].any()
        assert len(file["commands/accepted"]) == 168
    report = review_native_collection(path)
    assert report["required_scope_coverage"] and not report["full_task_success"]
    assert report["terminal_support"]
