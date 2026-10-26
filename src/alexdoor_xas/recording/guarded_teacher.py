"""Native teacher collection through the maintained observed A1 motion checks."""

import json

import numpy as np

from alexdoor_xas.recording.teacher import TemporalExpertRecorder


class GuardedTeacherRecorder(TemporalExpertRecorder):
    """Keep actual inspection history and admit the previewed controller target.

    Teacher poses propose commands. The observed runtime supplies all motion,
    contact and load admission; evaluator annotations never reach that runtime.
    """

    def __init__(self, *args, operational_recipe, robot_asset, **kwargs):
        super().__init__(*args, **kwargs)
        self.operational_recipe = operational_recipe
        self.robot_asset = robot_asset
        self.observer = None
        self.operational = None
        self.admitted_commands = 0
        self.guard_refusal = None
        self.brake_anchor = None

    def start(self, env, door, setup, phase="approach"):
        from alexdoor_xas.action.spaces import A1_JOINT_DELTA
        from alexdoor_xas.policies.encoding import pilot_binding
        from alexdoor_xas.policies.pilot_runtime import observed_pilot_runtime

        super().start(env, door, setup, phase)
        output = self.path.parent / "operational"
        output.mkdir()
        binding = pilot_binding(
            self.metadata["inspection"],
            depth_interval_m=(env.capture.near_m, env.capture.far_m),
            recipe="rgbd-conv-v1",
        )
        self.operational = observed_pilot_runtime(
            env,
            binding,
            self.robot_asset,
            setup,
            self.operational_recipe,
            output,
            A1_JOINT_DELTA,
        )
        self.observer, self.io, self.geometry, self.motion, _, _ = self.operational
        self.observer.reset()
        self.io.safety.reset()
        self.latest_observation = self.observer.update(self.io.observe())

    def _state(self):
        super()._state()
        if self.observer is not None:
            self.latest_observation = self.observer.update(self.io.observe())
            if self.phase == "inspect" and (reason := self.io.safety_reason("inspect")):
                self.guard_refusal = reason
                raise ValueError(reason)

    def execute(self, position, rotation, phase, *, joint_brake=False):
        import torch
        from scipy.spatial.transform import Rotation

        from alexdoor_xas.action.frames import ObjectFrame
        from alexdoor_xas.policies.observations import require_features
        from alexdoor_xas.policies.rollout import Command
        from alexdoor_xas.recording.b1_runtime import array

        if self.observer is None:
            raise RuntimeError("observed_teacher_not_started")
        try:
            if reason := self.io.safety_reason(None):
                raise ValueError(reason)
            observation = self.latest_observation
            require_features(observation, self.observer.binding)
            goal = ObjectFrame(np.asarray(position), np.asarray(rotation))
            q = array(self.env.capture.sample.joint_position)[0, :7]
            if joint_brake:
                if self.brake_anchor is None:
                    self.brake_anchor = q.copy()
                request = self.brake_anchor - q
                target, _, _ = self.env.preview_targets(
                    joint_delta=torch.as_tensor(
                        request, device=self.env.device, dtype=torch.float32
                    )[None]
                )
                command_kind, command_value = "joints", request
            else:
                self.brake_anchor = None
                target, _, _ = self.env.preview_targets(
                    world_goal=(
                        torch.as_tensor(position, device=self.env.device, dtype=torch.float32)[
                            None
                        ],
                        torch.as_tensor(
                            Rotation.from_matrix(rotation).as_quat(),
                            device=self.env.device,
                            dtype=torch.float32,
                        )[None],
                    )
                )
                command_kind, command_value = "pose", goal
            actual_delta = array(target)[0] - q
            admission = self.motion(observation, self.io.tool_pose(), actual_delta, 1)
            if not admission.admitted:
                raise ValueError(admission.reason or "teacher_motion_not_admitted")
            # IO independently checks that the pose controller preview equals
            # the admitted A1 endpoint before the actual setter/physics step.
            self.io.execute(Command(command_kind, command_value, admission=admission))
            self.admitted_commands += 1
            if self.io.response_violation:
                raise ValueError(self.io.response_violation)
        except (ValueError, RuntimeError) as error:
            self.guard_refusal = str(error)
            raise

    def close(self):
        super().close()
        if self.operational is not None:
            _, _, geometry, motion, load_log, contact_log = self.operational
            summary = dict(
                admitted_commands=self.admitted_commands,
                refusal=self.guard_refusal,
                evaluator_inputs=False,
                source="maintained_observed_A1_runtime_and_pose_controller_preview",
            )
            (self.path.parent / "operational-collection.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            motion.close()
            load_log.close()
            contact_log.close()
            geometry.close()
            self.operational = None
