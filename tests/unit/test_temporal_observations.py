"""Causality, missingness and reset semantics without model execution."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.policies.temporal import (
    CausalObservationBuilder,
    IncrementalObservationBuilder,
    TemporalConfig,
)


def streams(cfg, *, generation=2):
    times = np.arange(301) / 60
    camera = times[::6]

    def stream(t, value):
        return dict(
            acquired=t.copy(),
            available=t.copy(),
            generation=np.full(len(t), generation),
            values=value,
        )

    return dict(
        images=dict(
            **stream(
                camera,
                np.arange(len(camera), dtype="f4")[:, None, None, None]
                * np.ones((len(camera), 5, *cfg.image_size), "f4"),
            ),
            source_index=np.arange(len(camera)),
        ),
        q=stream(times, np.ones((len(times), 9), "f4") * times[:, None]),
        dq=stream(times, np.zeros((len(times), 9), "f4")),
        tau=dict(**stream(times, np.zeros((len(times), 7), "f4")), valid=np.ones(len(times), bool)),
        target=stream(times, np.ones((len(times), 7), "f4") * (10 + times[:, None])),
    )


def assert_snapshot_equal(a, b):
    for key in a["obs"]:
        np.testing.assert_array_equal(a["obs"][key], b["obs"][key], err_msg=key)
    for key in ("image_source_index", "image_times", "robot_times"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert a["ready"] == b["ready"]


def test_four_seconds_and_independent_sampling():
    cfg = TemporalConfig()
    assert cfg.image_count == 23 and cfg.state_count == 241
    assert replace(cfg, history_s=2, image_offsets_s=None).state_count == 121
    assert replace(cfg, history_s=6, image_offsets_s=None).image_count == 33
    with pytest.raises(ValueError):
        TemporalConfig(image_offsets_s=(0, 0.1))


def test_offline_incremental_startup_nominal_transition_terminal_and_bounded():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    offline = CausalObservationBuilder(cfg, raw, episode_id="same")
    from alexdoor_xas.assets.purdue import ARM_JOINTS

    online = IncrementalObservationBuilder(
        cfg,
        episode_id="same",
        generation=2,
        feedback_contract=dict(
            verified=True,
            joint_names=ARM_JOINTS,
            units="N m",
            sign_convention="positive_joint_coordinate",
            source="simulated_actuator_feedback",
            semantics="joint_output",
        ),
    )
    for tick in range(301):
        t = tick / 60
        # Native feedback/state before snapshot, current command afterwards.
        for key in ("q", "dq", "tau"):
            online.append(
                key, acquired=t, available=t, values=raw[key]["values"][tick], generation=2
            )
        if tick % 6 == 0:
            i = tick // 6
            online.append(
                "images",
                acquired=t,
                available=t,
                values=raw["images"]["values"][i],
                generation=2,
                source_index=i,
            )
        if tick in (0, 30, 42, 120, 240, 300):
            assert_snapshot_equal(offline.build(t, 2), online.build(t))
        online.append(
            "target", acquired=t, available=t, values=raw["target"]["values"][tick], generation=2
        )
    assert len(online.rows["q"]) < 260
    online.reset(3)
    empty = online.build(0)
    assert not empty["ready"] and not empty["obs"]["robot_valid"].any()
    with pytest.raises(ValueError, match="generation"):
        online.append("q", acquired=0, available=0, values=np.zeros(9), generation=2)


def test_future_late_current_command_and_privileged_fields_cannot_leak():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    builder = CausalObservationBuilder(cfg, raw, episode_id="e")
    before = builder.build(3, 2)
    # Privileged annotation is deliberately outside the stream allowlist.
    raw["q"]["privileged_phase"] = np.full(301, 123)
    for key in ("q", "dq", "tau", "target"):
        raw[key]["values"][181:] = 9999
    raw["target"]["values"][180] = -9999  # command at t cannot be its own input
    raw["images"]["values"][31:] = 9999
    assert_snapshot_equal(before, builder.build(3, 2))
    raw["q"]["available"][180] = 3.1  # unavailable at the saved decision
    delayed = builder.build(3, 2)
    assert delayed["obs"]["robot"][-1, 0] < 3
    assert delayed["ready"]  # previous tick still within justified freshness
    # Other generations never fill left padding or supply a static reference.
    assert not builder.build(3, 7)["obs"]["image_valid"].any()


def test_missing_feedback_is_not_zero_and_historical_gaps_do_not_refuse():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    builder = CausalObservationBuilder(cfg, raw, episode_id="e")
    present = builder.build(3, 2)
    assert present["obs"]["robot_valid"][-1, 2]
    raw["tau"]["valid"][:] = False
    raw["q"]["available"][60:90] += 4
    missing = builder.build(3, 2)
    assert missing["ready"]
    assert not missing["obs"]["robot_valid"][:, 2].any()
    np.testing.assert_array_equal(missing["obs"]["robot"][:, 18:25], 0)
    assert not missing["obs"]["robot_valid"][125, 0]


def test_clock_regression_requires_reset_and_explicit_domain():
    cfg = TemporalConfig(image_size=(2, 3))
    online = IncrementalObservationBuilder(cfg, episode_id="e")
    with pytest.raises(ValueError, match="clock"):
        online.append(
            "q",
            acquired=1,
            available=1,
            values=np.zeros(9),
            generation=0,
            clock_domain="device_unconverted",
        )
    with pytest.raises(ValueError, match="reset"):
        online.build(1)
    online.reset(1)
    online.append("q", acquired=0, available=0, values=np.zeros(9), generation=1)
    assert not online.build(0)["ready"]


def test_incremental_unverified_feedback_stays_masked():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0)
    online = IncrementalObservationBuilder(cfg, episode_id="hardware-unverified")
    online.append("tau", acquired=0, available=0, values=np.zeros(7), generation=0)
    snap = online.build(0)
    assert not snap["obs"]["robot_valid"][:, 2].any()


def test_invalid_historical_image_is_masked_but_current_image_is_required():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    raw["images"]["valid"] = np.ones(len(raw["images"]["acquired"]), bool)
    raw["images"]["valid"][10] = False
    builder = CausalObservationBuilder(cfg, raw, episode_id="e")
    snapshot = builder.build(3, 2)
    assert snapshot["ready"]
    masked = ~snapshot["obs"]["image_valid"]
    assert masked.any()
    np.testing.assert_array_equal(snapshot["obs"]["images"][masked], 0)
    raw["images"]["valid"][30] = False
    assert not builder.build(3, 2)["ready"]


def test_static_reference_cannot_clock_episode_progress_after_history_fill():
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    # Constant observations with identical relative support at t=4 and t=5;
    # reference acquisition remains at 0.5, so only its absolute age differs.
    for stream in raw.values():
        stream["values"][:] = 0.25
    builder = CausalObservationBuilder(cfg, raw, episode_id="same")
    a, b = builder.build(4 + 1 / 120, 2), builder.build(5 + 1 / 120, 2)
    assert a["ready"] and b["ready"]
    assert a["obs"]["image_valid"].all() and b["obs"]["image_valid"].all()
    for key in a["obs"]:
        np.testing.assert_allclose(a["obs"][key], b["obs"][key], atol=1e-6, rtol=0, err_msg=key)
    assert a["image_times"][0, 0] == b["image_times"][0, 0] == 0.5
    assert a["obs"]["image_timing"][0, 0] == -1
    assert a["obs"]["image_timing"][-1, 0] == pytest.approx(1 / 120)
    assert a["obs"]["image_timing"][1, 0] == pytest.approx(cfg.history_s + 1 / 120)
