"""Native teacher collection through the maintained observed A1 motion checks."""

import numpy as np

from alexdoor_xas.recording.teacher import TemporalExpertRecorder


class GuardedTeacherRecorder(TemporalExpertRecorder):
    """Keep actual inspection history and admit the previewed controller target.

    Teacher poses propose commands. The observed runtime supplies all motion,
    contact and load admission; evaluator annotations never reach that runtime.
    """

    def __init__(
        self, *args, operational_recipe, robot_asset, record_tracker_stream=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.operational_recipe = operational_recipe
        self.robot_asset = robot_asset
        self.observer = None
        self.operational = None
        self.admitted_commands = 0
        self.guard_refusal = None
        self.brake_anchor = None
        self.release_rotation = None
        self.record_tracker_stream = record_tracker_stream
        self.tracker_stream = None

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
        self.geometry.engine.interruption_check = getattr(env, "collection_check", None)
        self.io.safety.reset()
        if self.record_tracker_stream:
            from alexdoor_xas.recording.tracker_stream import TrackerStream

            self.tracker_stream = TrackerStream(output / "tracker-stream.hdf5")
        self._observe()

    def _observe(self):
        self.latest_observation = self.observer.update(self.io.observe())
        if self.tracker_stream is not None:
            self.tracker_stream.append(
                self.geometry,
                self.latest_observation.estimate,
                generation=self.env.generation,
                phase=self.phase,
            )

    def _state(self):
        super()._state()
        if self.observer is not None:
            self._observe()
            if self.phase == "inspect" and (reason := self.io.safety_reason("inspect")):
                self.guard_refusal = reason
                raise ValueError(reason)

    def execute(
        self,
        position,
        rotation,
        phase,
        *,
        joint_brake=False,
        normal_reference_only=False,
    ):
        import torch
        from scipy.spatial.transform import Rotation

        from alexdoor_xas.action.frames import ObjectFrame
        from alexdoor_xas.policies.observations import require_features
        from alexdoor_xas.policies.rollout import Command
        from alexdoor_xas.recording.b1_runtime import array

        if self.observer is None:
            raise RuntimeError("observed_teacher_not_started")
        try:
            if reason := self.io.safety_reason(None, defer_unloading_orientation=True):
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
                normal_options = {"normal_reference_only": True} if normal_reference_only else {}
                target, _, _ = self.env.preview_targets(
                    **normal_options,
                    world_goal=(
                        torch.as_tensor(position, device=self.env.device, dtype=torch.float32)[
                            None
                        ],
                        torch.as_tensor(
                            Rotation.from_matrix(rotation).as_quat(),
                            device=self.env.device,
                            dtype=torch.float32,
                        )[None],
                    ),
                )
                command_kind, command_value = "pose", goal
            actual_delta = array(target)[0] - q
            admission = self.motion(observation, self.io.tool_pose(), actual_delta, 1)
            if not admission.admitted:
                raise ValueError(admission.reason or "teacher_motion_not_admitted")
            stage = phase if phase in ("release", "stabilize") else None
            if (
                phase == "release"
                and self.release_rotation is not None
                and not np.array_equal(goal.rot, self.release_rotation)
            ):
                stage = "release_free"
            # IO independently checks that the pose controller preview equals
            # the admitted A1 endpoint before the actual setter/physics step.
            self.io.execute(
                Command(
                    command_kind,
                    command_value,
                    admission=admission,
                    stage=stage,
                    normal_reference_only=normal_reference_only and not joint_brake,
                )
            )
            self.admitted_commands += 1
            if self.io.response_violation:
                raise ValueError(self.io.response_violation)
            if phase == "release" and self.release_rotation is None:
                self.release_rotation = goal.rot.copy()
        except (ValueError, RuntimeError) as error:
            self.guard_refusal = str(error)
            error.probe_exception_phase = phase
            raise

    def close(self):
        super().close()
        if self.tracker_stream is not None:
            self.tracker_stream.close()
            self.tracker_stream = None
        if self.operational is not None:
            _, _, geometry, motion, load_log, contact_log = self.operational
            from alexdoor_xas.perception.evaluation import write_json

            summary = dict(
                admitted_commands=self.admitted_commands,
                refusal=self.guard_refusal,
                refusal_detail=self.io.safety.checks.last_failure,
                evaluator_inputs=False,
                source="maintained_observed_A1_runtime_and_bounded_controller_preview",
            )
            write_json(self.path.parent / "operational-collection.json", summary)
            from dataclasses import asdict

            tracking = geometry.tracking
            write_json(
                self.path.parent / "operational-perception-terminal.json",
                dict(
                    provider=geometry.diagnostics,
                    tracking=tracking.diagnostics,
                    selection=None if tracking.selection is None else asdict(tracking.selection),
                    candidates=[
                        dict(
                            candidate_id=c.candidate_id,
                            state=c.state,
                            observed_pixels=int(c.surface.observations[-1].mask().sum()),
                        )
                        for c in tracking.candidates
                    ],
                ),
            )
            motion.close()
            load_log.close()
            contact_log.close()
            geometry.close()
            self.operational = None
