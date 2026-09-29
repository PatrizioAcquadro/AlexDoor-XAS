"""Causal policy input assembly with numerical estimator fixtures, no model execution."""

from types import SimpleNamespace

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.model import DoorEstimate
from alexdoor_xas.policies.observations import B1Observer
from alexdoor_xas.recording.b1 import OBS_KEYS


def fake_estimator(binding):
    estimator = SimpleNamespace(config=binding.config, policy_encoding=True, reason="observed")

    def reset():
        estimator.encoding = None
        estimator.calls = []

    def update(obs):
        estimator.calls.append(obs)
        estimator.encoding = np.arange(binding.obs_dim - 18)
        return DoorEstimate(
            obs["time_s"],
            estimator.reason == "observed",
            estimator.reason,
            0.8,
            ObjectFrame(np.zeros(3), np.eye(3)),
            np.eye(3),
            0.0,
        )

    estimator.reset, estimator.update = reset, update
    return estimator


def sample(t, frame):
    result = dict.fromkeys(OBS_KEYS, None)
    result.update(time_s=t, frame=frame, joint_position=np.zeros(9), joint_velocity=np.zeros(9))
    return result


def test_privileged_metadata_cannot_enter_inputs_and_between_ticks_use_current_joints(b1_binding):
    estimator = fake_estimator(b1_binding)
    observer = B1Observer(estimator, b1_binding)
    obs = sample(0, 1)
    obs.update(asset_id="secret", theta_expert_d=2, annotations={"hinge": [8, 8, 8]})
    first = observer.update(obs)
    assert first.valid and set(estimator.calls[-1]) == set(OBS_KEYS)
    estimator.reason = "between_inference_ticks"
    obs = sample(1 / 60, 2)
    obs["joint_position"][:] = 0.25
    second = observer.update(obs)
    assert second.valid and second.estimate.timestamp_s == 0
    np.testing.assert_array_equal(second.features[:4], first.features[:4])
    np.testing.assert_array_equal(second.features[4:13], 0.25)
    assert not observer.update(sample(0.16, 3)).valid


def test_loss_reset_and_time_errors_never_reuse_old_encoding(b1_binding):
    estimator = fake_estimator(b1_binding)
    observer = B1Observer(estimator, b1_binding)
    assert observer.update(sample(0, 1)).valid
    estimator.reason = "low_confidence"
    assert not observer.update(sample(0.01, 2)).valid
    estimator.reason = "between_inference_ticks"
    assert not observer.update(sample(0.02, 3)).valid
    estimator.reason = "observed"
    assert observer.update(sample(0.1, 4)).valid
    assert observer.update(sample(0.09, 5)).reason == "nonmonotonic_observation"
    assert observer.last is None
    observer.reset()
    estimator.reason = "warming_up"
    assert not observer.update(sample(0, 1)).valid


def test_nonfinite_proprioception_invalidates_immediately(b1_binding):
    observer = B1Observer(fake_estimator(b1_binding), b1_binding)
    assert observer.update(sample(0, 1)).valid
    obs = sample(0.01, 2)
    obs["joint_velocity"][0] = np.nan
    assert observer.update(obs).reason == "invalid_proprioception"
    assert observer.last is None
