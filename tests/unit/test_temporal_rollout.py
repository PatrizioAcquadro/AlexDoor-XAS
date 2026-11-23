"""Guarded expert rollouts retain real streams and explicitly reviewed targets."""

import json

import h5py
import numpy as np
import pytest

from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.dataset.temporal import raw_streams
from alexdoor_xas.policies.temporal import TemporalConfig


def test_synchronous_rollout_keeps_capture_feedback_and_delivered_target_clocks(tmp_path):
    path = tmp_path / "rollout.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "b1.policy_rollout.v1"
        meta = dict(
            split="train",
            clock_semantics="native_synchronous_simulation",
            joint_names=["neck", *reversed(ARM_JOINTS)],
        )
        f.create_group("metadata").attrs.update(
            episode=json.dumps(meta),
            feedback_contract=json.dumps(
                dict(
                    verified=True,
                    joint_names=list(ARM_JOINTS),
                    units="N m",
                    sign_convention="positive_joint_coordinate",
                    source="simulated_actuator_feedback",
                    semantics="joint_output",
                )
            ),
        )
        state = f.create_group("control")
        state["time_s"] = [0, 1 / 60, 2 / 60]
        state["generation"] = [2, 2, 2]
        state["joint_position"] = np.zeros((3, 9))
        state["joint_velocity"] = np.ones((3, 9))
        # Retain every actual capture, including an extra consumed frame.
        f["captures/time_s"] = [0, 1 / 60, 2 / 60]
        command = f.create_group("commands")
        command["time_s"] = [0, 1 / 60]
        command["joint_target"] = [np.arange(8), np.arange(8) + 1]
        feedback = f.create_group("physics_feedback")
        feedback["time_s"] = [0.005, 0.01, 0.015]
        feedback["available_s"] = [0.006, 0.012, 0.016]
        feedback["torque_nm"] = np.ones((3, 7))
        feedback["healthy"] = [True, False, True]
        feedback["present"] = [True, True, False]
        streams, _, times, generation = raw_streams(f, TemporalConfig())
        np.testing.assert_array_equal(streams["images"]["acquired"], times)
        np.testing.assert_array_equal(streams["tau"]["available"], [0.006, 0.012, 0.016])
        np.testing.assert_array_equal(streams["tau"]["valid"], [True, False, False])
        np.testing.assert_array_equal(streams["target"]["acquired"], [0, 1 / 60])
        np.testing.assert_array_equal(streams["target"]["values"][0], np.arange(7, 0, -1))
        np.testing.assert_array_equal(generation, [2, 2, 2])
        del f["metadata"].attrs["feedback_contract"]
        missing, _, _, _ = raw_streams(f, TemporalConfig())
        assert len(missing["tau"]["acquired"]) == 0
        meta.pop("clock_semantics")
        f["metadata"].attrs["episode"] = json.dumps(meta)
        with pytest.raises(ValueError, match="explicit synchronous"):
            raw_streams(f, TemporalConfig())
