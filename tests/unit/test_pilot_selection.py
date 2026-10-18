import numpy as np

from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.policies.pilot_selection import action_errors, update_selection

CFG = dict(
    minimum_relative_improvement=0.05,
    maximum_bucket_p95_increase_for_selection=0.10,
    persistent_regression_relative=0.10,
    persistent_bucket_p95_regression_relative=0.20,
    no_progress_patience=3,
    regression_patience=3,
)


def metric(score, release=1.0):
    return dict(primary=score, bucket_p95={"push": 1.0, "release": release})


def test_action_metrics_use_physical_commands_and_so3():
    a = np.zeros((1, 7))
    a[0, :2] = [3, 4]
    assert action_errors(A1_JOINT_DELTA, a, np.zeros_like(a))[0][0] == 5
    target = np.zeros((2, 6))
    pred = target.copy()
    pred[0, 0] = 0.02
    pred[1, 5] = np.deg2rad(10)
    np.testing.assert_allclose(action_errors(A2_EE_DELTA, pred, target)[0], [2, 2])
    pred[1, 5] = 2 * np.pi
    assert action_errors(A2_EE_DELTA, pred, target)[1]["rotation_rad"][1] < 1e-12


def test_better_average_cannot_hide_release_tail_regression():
    state = update_selection(None, metric(1), CFG)
    rejected = update_selection(state, metric(0.8, release=1.11), CFG)
    assert not rejected["accepted"] and rejected["best"]["primary"] == 1
    accepted = update_selection(rejected, metric(0.9), CFG)
    assert accepted["accepted"] and accepted["no_progress"] == 0


def test_three_unchanged_controls_stop_without_selecting_last():
    state = update_selection(None, metric(1), CFG)
    for n in range(1, 4):
        state = update_selection(state, metric(0.98), CFG)
        assert state["stop"] == (n == 3)
    assert state["best"]["primary"] == 1


def test_persistent_regression_is_counted_independently():
    state = update_selection(None, metric(1), CFG)
    for _ in range(3):
        state = update_selection(state, metric(1.2, release=1.3), CFG)
    assert state["regressions"] == 3 and state["stop"]
