"""Maintained teacher adapter: one set of native events feeds recording and ACT ingress."""

import json

import numpy as np

from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.dataset.temporal import preprocess_rgbd
from alexdoor_xas.policies.temporal import IncrementalObservationBuilder, TemporalConfig
from alexdoor_xas.recording.b1 import PHASES
from alexdoor_xas.recording.b1_runtime import ExpertRecorder, annotate_native_units, array
from alexdoor_xas.recording.temporal import TemporalRecorder


def simulated_feedback_contract(env):
    """Solver projection in resolved URDF coordinates; no hardware qualification."""
    if tuple(env.robot.joint_names[i] for i in env.arm_ids) != ARM_JOINTS:
        raise ValueError("Native torque joint order differs from ARM_JOINTS")
    return dict(
        verified=True,
        joint_names=list(ARM_JOINTS),
        units="N m",
        sign_convention="positive_joint_coordinate",
        source="simulated_actuator_feedback",
        semantics="joint_output",
        api="robot.root_view.get_dof_projected_joint_forces",
        interpretation="incoming joint spatial force projected along revolute DOF motion axis",
        hardware_verified=False,
    )


class TemporalExpertRecorder(ExpertRecorder):
    """Native synchronous simulation events, independent of evaluator/teacher features.

    State and request clocks are the episode's 60 Hz control boundary; feedback
    acquisition/availability is the actual simulated substep. Camera captures
    retain fresh native pixels at 10 Hz. No independent hardware time is inferred.
    """

    def __init__(
        self,
        path,
        metadata,
        *,
        interrupt_after_commands=None,
        image_compression=None,
        cache_temporal_frames=False,
    ):
        super().__init__(path, metadata, purpose="b1_partial_policy_pilot")
        self.interrupt_after_commands = interrupt_after_commands
        self.image_compression = image_compression
        if type(cache_temporal_frames) is not bool:
            raise ValueError("Temporal frame caching requires an explicit boolean")
        self.cache_temporal_frames = cache_temporal_frames
        self.env = None
        self.phase = None
        self.last_capture = -np.inf
        self.last_phase = None
        self.last_role = 0
        self.online_checks = []
        self.command_count = 0
        self.complete = False
        self.trace_log = None

    def start(self, env, door, setup, phase="approach"):
        self.env, self.door, self.setup, self.phase = env, door, setup, phase
        self.config = TemporalConfig(
            static_time_s=float(self.metadata.get("static_reference_time_s", 0.0))
            if self.metadata.get("intermediate_start") or "static_reference_time_s" in self.metadata
            else float(self.metadata["inspection"]["waypoints"][-1][0]),
            depth_interval_m=(env.capture.near_m, env.capture.far_m),
        )
        contract = simulated_feedback_contract(env)
        self.metadata.update(
            control_dt=env.step_dt,
            sim_dt=env.cfg.sim.dt,
            purpose="temporal_teacher_collection",
            observation_config=self.config.to_dict(),
            clock_semantics=(
                "synchronous_simulation_episode_time_acquisition_equals_availability; "
                "dispatch/application_is_controller_setter_not_motion_reached"
            ),
            full_joint_names=list(env.robot.joint_names),
            request_kinds={
                1: "joint_delta",
                2: "absolute_world_pose_xyz_xyzw",
                3: "world_pose_delta_xyz_rotvec_padded_zero",
            },
            phase_names=list(PHASES),
        )
        self.writer = TemporalRecorder(
            self.path,
            self.metadata,
            self.calibration,
            feedback_contract=contract,
            control_dt=env.step_dt,
            image_compression=self.image_compression,
        )
        self.writer.reset(env.generation)
        self.observations = IncrementalObservationBuilder(
            self.config,
            episode_id=self.metadata["episode_id"],
            generation=env.generation,
            feedback_contract=contract,
        )
        self.writer.file["metadata"].attrs["arm_effort_limits_nm"] = json.dumps(
            array(env.robot.data.joint_effort_limits)[0, env.arm_ids].tolist()
        )
        self.writer.file["metadata"].attrs["native_control_diagnostics"] = json.dumps(
            dict(
                q_dq="Simulated arm feedback at native acquisition time",
                target="Delivered position target for preceding native interval",
                computed_effort="PD plus feedforward before clipping; preceding native interval",
                applied_effort="Clipped commanded effort sent to solver; not measured joint output",
                gravity="Current-state model gravity, not completed-interval feedforward",
                policy_observation=False,
            )
        )
        # Privileged teacher diagnostics stay separate from raw observations.
        # Flush each completed transition so a later refusal cannot discard the
        # native force mean and pressure-feedback branch for the retained prefix.
        self.trace_log = self.path.with_name("teacher-trace.jsonl").open("x")
        self._state()
        annotate_native_units(
            self.writer.file, env.robot.cfg.spawn.asset_path, env.robot.joint_names
        )
        self._feedback(env.load_samples[-1])
        env.event_sink = self._event

    def _append(self, field, values, acquired, available, *, valid=True, source_index=-1):
        self.observations.append(
            field,
            values=values,
            acquired=acquired,
            available=available,
            generation=self.env.generation,
            valid=valid,
            source_index=source_index,
        )

    def _state(self):
        env = self.env
        sample = env.capture.sample
        t = float(sample.time_s)
        raw = dict(
            joint_position=array(sample.joint_position)[0],
            joint_velocity=array(sample.joint_velocity)[0],
        )
        p, quat = [array(v)[0] for v in env.tool_pose()]
        self.writer.state(
            raw["joint_position"],
            raw["joint_velocity"],
            acquired=t,
            available=t,
            generation=env.generation,
            full_joint_position=array(env.robot.data.joint_pos)[0],
            full_joint_velocity=array(env.robot.data.joint_vel)[0],
            full_joint_acceleration=array(env.robot.data.joint_acc)[0],
            tool_position=p,
            tool_quaternion_xyzw=quat,
            effort_limits_nm=array(env.robot.data.joint_effort_limits)[0, env.arm_ids],
            camera_frame=np.int64(sample.frame),
        )
        for key, name in (("q", "joint_position"), ("dq", "joint_velocity")):
            self._append(key, raw[name], t, t)
        if t - self.last_capture >= 0.1 - 1e-9:
            raw = self.observation(env)
            self.writer.capture(
                *(
                    raw[key]
                    for key in ("rgb", "depth_m", "valid_depth", "intrinsics", "camera_world")
                ),
                acquired=t,
                available=t,
                generation=env.generation,
                frame=int(raw["frame"]),
            )
            values = preprocess_rgbd(
                *(raw[key] for key in ("rgb", "depth_m", "valid_depth")),
                self.config,
                device=env.device,
            )
            if self.cache_temporal_frames:
                self.writer.cache_preprocessed(
                    values,
                    dict(
                        image_size=list(self.config.image_size),
                        depth_scale_m=self.config.depth_scale_m,
                        depth_interval_m=list(self.config.depth_interval_m),
                    ),
                    device=env.device,
                )
            self._append("images", values, t, t, source_index=self.writer.frame_index - 1)
            self.last_capture = t

    def _feedback(self, sample):
        f = sample.feedback
        healthy = bool(
            f.healthy
            and not f.reason
            and f.source == "simulated_actuator_feedback"
            and f.semantics == "joint_output"
            and f.joint_names == ARM_JOINTS
            and f.units == "N m"
            and f.sign_convention == "positive_joint_coordinate"
        )
        tau = np.full(7, np.nan) if f.torque_nm is None else f.torque_nm
        self.writer.feedback(
            tau,
            acquired=f.timestamp_s,
            available=f.available_s,
            healthy=healthy,
            generation=self.env.generation,
            diagnostic_state=dict(
                diagnostic_joint_position=sample.q,
                diagnostic_joint_velocity=sample.dq,
                diagnostic_position_target=array(self.env.targets)[0, self.env.arm_ids],
                diagnostic_velocity_target=array(self.env.robot.data.joint_vel_target)[
                    0, self.env.arm_ids
                ],
                diagnostic_computed_effort_nm=array(self.env.robot.data.computed_torque)[
                    0, self.env.arm_ids
                ],
                diagnostic_applied_effort_nm=array(self.env.robot.data.applied_torque)[
                    0, self.env.arm_ids
                ],
                diagnostic_gravity_feedforward_nm=array(
                    self.env.robot.data.gravity_compensation_forces
                )[0, self.env.arm_ids],
            ),
        )
        self._append("tau", tau, f.timestamp_s, f.available_s, valid=healthy)

    def _event(self, kind, value):
        env = self.env
        if kind == "feedback":
            self._feedback(value)
        elif kind == "state":
            self._state()
        elif kind == "command":
            t = float(value["request_s"])
            applied = array(env.targets)[0, env.arm_ids]
            requested = array(value["requested_target"])[0]
            self.writer.command(
                request_s=t,
                dispatch_s=t,
                applied_s=t,
                requested_target=requested,
                applied_target=applied,
                q_before=array(value["q_before"])[0],
                accepted=True,
                generation=env.generation,
                request_kind=value["request_kind"],
                request=array(value["request"])[0],
                clipped=bool(np.any(requested != applied)),
                neck_target=array(env.targets)[0, env.neck_ids],
                gripper_target=array(env.targets)[
                    0, [i for i, n in enumerate(env.robot.joint_names) if "WSG32" in n]
                ],
                effort_target_nm=array(env.robot.data.joint_effort_target)[0],
            )
            if self.command_count == 0:
                annotate_native_units(
                    self.writer.file, env.robot.cfg.spawn.asset_path, env.robot.joint_names
                )
            self._append("target", applied, t, t)
            self.command_count += 1
            if self.interrupt_after_commands == self.command_count:
                # Controller delivery really happened, but no physics/post-state
                # is claimed. The outer owner closes the prefix and verifies stop.
                raise KeyboardInterrupt("controlled_collection_interruption_after_dispatch")
        else:
            raise ValueError("Unknown native producer event")

    def transition(self, env, goal_p, goal_r, phase, trace):
        self.phase = phase
        row = dict(
            time_s=np.float64(env._time_s),
            generation=np.int64(env.generation),
            phase=np.int8(PHASES.index(phase)),
            action_role=np.int8(trace.get("action_role", 0)),
            recovery_valid=np.bool_(trace.get("recovery_valid", False)),
            physical_safe=np.bool_(trace.get("physical_safe", trace["valid"])),
            phase_valid=np.bool_(trace.get("phase_valid", trace["valid"])),
            contact_valid=np.bool_(trace["valid"]),
            loaded=np.bool_(trace.get("loaded", False)),
            angle=np.float64(array(env.door.data.joint_pos)[0, 0]),
            door_speed=np.float64(array(env.door.data.joint_vel)[0, 0]),
            force_n=np.float64(trace.get("force", 0)),
            material_error_m=np.float64(trace.get("material_error", 0)),
            separation_m=np.float64(trace.get("separation_m", np.nan)),
        )
        self.writer.annotate(row)
        from alexdoor_xas.perception.evaluation import json_safe

        self.trace_log.write(
            json.dumps(
                dict(
                    time_s=float(env._time_s),
                    generation=int(env.generation),
                    trace=json_safe(trace),
                ),
                allow_nan=False,
            )
            + "\n"
        )
        self.trace_log.flush()
        if phase != self.last_phase:
            self._snapshot(phase)
            self.last_phase = phase
        role = int(row["action_role"])
        if role != self.last_role:
            self._snapshot(f"role-{self.command_count}-{role}")
            self.last_role = role

    def _snapshot(self, label):
        snap = self.observations.build(self.env._time_s)
        path = self.path.parent / f"online-{label}.npz"
        np.savez_compressed(
            path,
            **snap["obs"],
            image_source_index=snap["image_source_index"],
            image_times=snap["image_times"],
            robot_times=snap["robot_times"],
            decision_time_s=snap["decision_time_s"],
            generation=snap["generation"],
            ready=snap["ready"],
        )
        self.online_checks.append(path.name)

    def finish(self, result):
        self._snapshot("terminal")
        self.writer.file["metadata"].attrs["camera_fk_error"] = json.dumps(
            dict(position_m=self.max_camera_error_m, rotation_rad=self.max_camera_error_rad)
        )
        self.writer.finish(dict(result, termination=result["stop_reason"]))
        self.complete = True

    def interrupt(self, outcome):
        if self.writer is not None:
            self.writer.interrupt(outcome)

    def close_recording(self):
        """Detach physical observation and flush data without waiting for model workers."""
        if self.env is not None:
            self.env.event_sink = None
        try:
            super().close()
        finally:
            if getattr(self, "trace_log", None) is not None:
                self.trace_log.close()
                self.trace_log = None

    def close(self):
        self.close_recording()
