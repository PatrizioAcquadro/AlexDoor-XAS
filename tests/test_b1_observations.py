"""Causal policy input assembly with numerical estimator fixtures, no model execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import DoorEstimate
from alexdoor_xas.policies.observations import B1Observer
from alexdoor_xas.recording.b1 import OBS_KEYS


def fake_estimator(binding):
    estimator = SimpleNamespace(binding=binding, reason="observed")

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
            np.array([0.9, 2.0, 0.04]),
            np.array([0.0, 0.3, 1.09]),
            np.eye(3),
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


@pytest.mark.parametrize("field", ["offline_passed", "dynamic_passed", "frozen"])
def test_binding_rejects_unqualified_release(b1_binding, field):
    release = b1_binding.to_dict()
    release[field] = False
    with pytest.raises(ValueError, match="qualified and frozen"):
        type(b1_binding).from_dict(release)


def test_binding_verifies_artifacts_and_rejects_legacy(tmp_path, b1_binding):
    import hashlib

    path = tmp_path / "weights"
    path.write_bytes(b"selected artifact")
    release = b1_binding.to_dict()
    release["artifacts"] = {"visual": hashlib.sha256(path.read_bytes()).hexdigest()}
    binding = type(b1_binding).from_dict(release)
    binding.verify_artifacts({"visual": path})
    with pytest.raises(ValueError, match="names differ"):
        binding.verify_artifacts({})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        binding.verify_artifacts({"visual": path})
    release["schema"] = "b1.perception.release.v1"
    with pytest.raises(ValueError, match="qualified and frozen"):
        type(b1_binding).from_dict(release)


def test_unequal_visual_blocks_and_incompatible_provider(b1_binding):
    from alexdoor_xas.policies.observations import observation_columns

    release = b1_binding.to_dict()
    release["config"]["visual_dims"] = [1, 3]
    binding = type(b1_binding).from_dict(release)
    columns = observation_columns(np.arange(binding.obs_dim), binding)
    np.testing.assert_array_equal(columns["rgbd_static"], [0])
    np.testing.assert_array_equal(columns["rgbd_recent"], [1, 2, 3])
    with pytest.raises(ValueError, match="provider"):
        B1Observer(fake_estimator(b1_binding), binding)
    provider = fake_estimator(binding)
    observer = B1Observer(provider, binding)
    update = provider.update

    def wrong_width(sensor):
        estimate = update(sensor)
        provider.encoding = np.zeros(3)
        return estimate

    provider.update = wrong_width
    with pytest.raises(ValueError):
        observer.update(sample(0, 1))


def operational_provider(binding, count=1):
    from dataclasses import replace

    from alexdoor_xas.perception.contracts import OPERATIONAL_V1
    from test_operational_admission import operational_estimate

    recipe = SimpleNamespace(
        config=binding.config | {"geometry_profile": OPERATIONAL_V1}, obs_dim=binding.obs_dim
    )
    provider = SimpleNamespace(binding=recipe, generation=-1, between=False, old_result=None)

    def reset():
        provider.generation += 1
        provider.encoding = None
        provider.calls = []
        provider.selection_time = None

    def update(sensor):
        provider.calls.append(sensor)
        if provider.between:
            return DoorEstimate(sensor["time_s"], False, "between_inference_ticks")
        estimate = provider.old_result or operational_estimate(
            sensor["time_s"], provider.generation, count
        )
        if provider.selection_time is None:
            provider.selection_time = sensor["time_s"]
        state = estimate.operational
        estimate = replace(
            estimate,
            operational=replace(
                state, contact=replace(state.contact, selected_s=provider.selection_time)
            ),
        )
        provider.encoding = np.arange(recipe.obs_dim - 18)
        provider.last_estimate = estimate
        return estimate

    provider.reset, provider.update = reset, update
    return provider


def test_operational_features_qualification_and_truth_feedback_isolation(b1_binding):
    provider = operational_provider(b1_binding)
    observer = B1Observer(provider, provider.binding)
    sensor = sample(0, 1)
    sensor.update(robot_feedback={"torque": np.ones(7)}, annotations={"hinge": [7, 7, 7]})
    result = observer.update(sensor)
    assert result.features_available and result.valid and not result.estimate.valid
    assert set(provider.calls[-1]) == set(OBS_KEYS)
    assert len(result.features) == b1_binding.obs_dim
    provisional = operational_provider(b1_binding, count=2)
    result = B1Observer(provisional, provisional.binding).update(sample(0, 1))
    assert result.features_available and not result.valid and result.reason == "ambiguous_hinge"


def test_operational_cache_never_refreshes_time_or_crosses_episode_generations(b1_binding):
    provider = operational_provider(b1_binding)
    observer = B1Observer(provider, provider.binding)
    first = observer.update(sample(0, 1))
    assert first.valid
    provider.between = True
    assert observer.update(sample(0.1, 2)).valid
    assert not observer.update(sample(0.16, 3)).valid
    observer.reset()
    assert not observer.update(sample(0, 1)).features_available
    provider.between = False
    provider.old_result = first.estimate
    assert observer.update(sample(0.01, 2)).reason == "wrong_episode_generation"
    provider.old_result = None
    assert observer.update(sample(0.02, 3)).valid
    assert observer.update(sample(0.01, 4)).reason == "nonmonotonic_observation"
    assert observer.last is None and observer.last_contact is None


def test_release_v2_cannot_be_reinterpreted_as_operational(b1_binding):
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1

    release = b1_binding.to_dict()
    release["config"]["geometry_profile"] = OPERATIONAL_V1
    with pytest.raises(ValueError, match="only supports legacy"):
        type(b1_binding).from_dict(release)
    release = b1_binding.to_dict()
    release["geometry_profile"] = OPERATIONAL_V1
    with pytest.raises(ValueError, match="only supports legacy"):
        type(b1_binding).from_dict(release)
