"""Independent lossless streams for future complete causal A1 recordings.

This opt-in writer does not change the expert, collection lifecycle or controller.
The producer must keep recording pre-roll, release and real normal stabilization.
"""

import json

import numpy as np

from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.recording.rollout import RolloutWriter

SCHEMA = "b1.temporal-recording.v1"


class TemporalRecorder:
    def __init__(
        self,
        path,
        metadata,
        calibration,
        *,
        feedback_contract,
        control_dt=1 / 60,
        image_period_s=0.1,
    ):
        required = dict(
            joint_names=list(ARM_JOINTS), units="N m", sign_convention="positive_joint_coordinate"
        )
        if any(feedback_contract.get(key) != value for key, value in required.items()) or (
            type(feedback_contract.get("verified")) is not bool
            or not feedback_contract.get("source")
            or not feedback_contract.get("semantics")
        ):
            raise ValueError("Explicit feedback origin/order/units/sign/verification required")
        if metadata.get("split") not in ("train", "development"):
            raise ValueError("Recorder requires explicit non-test membership")
        self.writer = RolloutWriter(
            path, metadata, calibration, control_dt=control_dt, image_period_s=image_period_s
        )
        self.file = self.writer.file
        self.file.attrs["schema"] = SCHEMA
        self.file["metadata"].attrs.update(
            feedback_contract=json.dumps(feedback_contract),
            clock_domain="episode_monotonic",
            applied_semantics="controller_setpoint_delivered_or_acknowledged_not_position_reached",
            observed_joints=json.dumps(list(ARM_JOINTS + NECK_JOINTS)),
        )
        self.feedback_verified = feedback_contract["verified"]
        self.generation = None
        self.last = {}
        self.states = {}
        self.commands = {}
        self.last_q = None
        self.pending_command = False
        self.frame_index = 0

    def reset(self, generation):
        if self.generation is not None and generation <= self.generation:
            raise ValueError("Reset generation must increase")
        if self.pending_command:
            raise ValueError("Previous generation lacks terminal state")
        self.generation = int(generation)
        self.last.clear()
        self.last_q = None

    def _clock(self, group, acquired, available, generation, clock_domain):
        if (
            generation != self.generation
            or clock_domain != "episode_monotonic"
            or (
                not np.isfinite([acquired, available]).all()
                or not 0 <= acquired <= available
                or group in self.last
                and (acquired <= self.last[group][0] or available < self.last[group][1])
            )
        ):
            raise ValueError("Unverified, regressed or wrong-generation recording clock")
        self.last[group] = (float(acquired), float(available))
        return dict(
            acquired=np.float64(acquired),
            available=np.float64(available),
            generation=np.int64(generation),
        )

    def state(
        self,
        q,
        dq,
        *,
        acquired,
        available,
        generation,
        clock_domain="episode_monotonic",
        q_valid=True,
        dq_valid=True,
        **robot_state,
    ):
        q, dq = np.asarray(q, "f8"), np.asarray(dq, "f8")
        if q.shape != (9,) or dq.shape != (9,):
            raise ValueError("Expected observed9 measured state")
        row = self._clock("control", acquired, available, generation, clock_domain)
        if set(robot_state) & set(row) | set(robot_state) & {"joint_position", "joint_velocity"}:
            raise ValueError("Robot diagnostics cannot override observed state")
        self.writer.append(
            "control",
            dict(
                **row,
                joint_position=q,
                joint_velocity=dq,
                q_valid=np.bool_(q_valid and np.isfinite(q).all()),
                dq_valid=np.bool_(dq_valid and np.isfinite(dq).all()),
                **robot_state,
            ),
        )
        self.states[generation] = self.states.get(generation, 0) + 1
        self.last_q = q.copy()
        self.pending_command = False
        if self.states[generation] % 60 == 0:
            self.writer.flush()

    def capture(
        self,
        rgb,
        depth_m,
        valid_depth,
        intrinsics,
        camera_world,
        *,
        acquired,
        available,
        generation,
        clock_domain="episode_monotonic",
    ):
        rgb, depth, valid = map(np.asarray, (rgb, depth_m, valid_depth))
        if (
            depth.ndim != 3
            or depth.shape[-1] != 1
            or valid.shape != depth.shape
            or (rgb.shape != (*depth.shape[:2], 3) or rgb.dtype != np.uint8 or valid.dtype != bool)
        ):
            raise ValueError("Unaligned raw RGB-D")
        row = self._clock("captures", acquired, available, generation, clock_domain)
        self.writer.append(
            "captures",
            dict(
                **row,
                rgb=rgb,
                depth_m=depth,
                valid_depth=valid,
                intrinsics=np.asarray(intrinsics),
                camera_world=np.asarray(camera_world),
                source_index=np.int64(self.frame_index),
            ),
        )
        self.frame_index += 1

    def feedback(
        self,
        torque_nm,
        *,
        acquired,
        available,
        healthy,
        generation,
        clock_domain="episode_monotonic",
    ):
        values = np.asarray(torque_nm, "f8")
        if values.shape != (7,):
            raise ValueError("Expected native arm7 feedback")
        row = self._clock("physics_feedback", acquired, available, generation, clock_domain)
        self.writer.append(
            "physics_feedback",
            dict(
                **row,
                torque_nm=values,
                healthy=np.bool_(healthy and self.feedback_verified and np.isfinite(values).all()),
            ),
        )

    def command(
        self,
        *,
        request_s,
        dispatch_s,
        applied_s,
        requested_target,
        applied_target,
        q_before,
        accepted,
        generation,
        request_kind,
        request,
        requested_valid=True,
        clipped=False,
        neck_target,
        gripper_target,
        effort_target_nm,
    ):
        if generation != self.generation or self.last_q is None or self.pending_command:
            raise ValueError("Command must follow its own measured pre-state")
        q_before = np.asarray(q_before, "f8")
        if q_before.shape != (7,) or not np.array_equal(q_before, self.last_q[:7], equal_nan=True):
            raise ValueError("Command q_before differs from measured state")
        if not np.isfinite(request_s) or request_s < self.last["control"][1]:
            raise ValueError("Invalid request clock")
        if accepted:
            if (
                not np.isfinite([dispatch_s, applied_s]).all()
                or not request_s <= dispatch_s <= applied_s
            ):
                raise ValueError("Invalid command request/dispatch/application clocks")
        elif dispatch_s is not None or applied_s is not None or np.isfinite(applied_target).any():
            raise ValueError(
                "Refusal cannot have dispatch/application acknowledgement or applied target"
            )
        else:
            dispatch_s = applied_s = np.nan
        target, requested = np.asarray(applied_target, "f8"), np.asarray(requested_target, "f8")
        if (
            target.shape != (7,)
            or requested.shape != (7,)
            or (
                accepted
                and not np.isfinite(target).all()
                or requested_valid
                and not np.isfinite(requested).all()
            )
        ):
            raise ValueError("Invalid requested/applied target")
        self.writer.append(
            "commands",
            dict(
                request_s=np.float64(request_s),
                dispatch_s=np.float64(dispatch_s),
                applied_s=np.float64(applied_s),
                state_acquired_s=np.float64(self.last["control"][0]),
                requested_target=requested,
                applied_target=target,
                q_before=q_before,
                accepted=np.bool_(accepted),
                requested_valid=np.bool_(requested_valid),
                clipped=np.bool_(clipped),
                generation=np.int64(generation),
                request_kind=np.int64(request_kind),
                request=np.asarray(request, "f8"),
                neck_target=np.asarray(neck_target, "f8"),
                gripper_target=np.asarray(gripper_target, "f8"),
                effort_target_nm=np.asarray(effort_target_nm, "f8"),
            ),
        )
        self.commands[generation] = self.commands.get(generation, 0) + 1
        self.pending_command = True

    def annotate(self, values):
        """Evaluator/teacher data stay write-only and outside the observation adapter."""
        self.writer.append("evaluator", values)

    def interrupt(self, outcome):
        """Flush the real prefix, with explicit outcome but no invented terminal state."""
        self.file["metadata"].attrs["outcome"] = json.dumps(outcome, allow_nan=False)
        self.file["metadata"].attrs["terminal_support"] = not self.pending_command
        self.writer.flush()

    def finish(self, outcome):
        if (
            not isinstance(outcome.get("passed"), bool)
            or not outcome.get("termination")
            or (
                self.pending_command
                or not self.states
                or any(count != self.commands.get(gen, 0) + 1 for gen, count in self.states.items())
            )
        ):
            raise ValueError("Missing terminal observation or explicit outcome")
        self.file["metadata"].attrs["outcome"] = json.dumps(outcome, allow_nan=False)
        self.writer.flush()
        self.file.attrs["complete"] = True
        self.file.flush()

    def close(self):
        self.writer.close()
