"""Native simulation event ingress for temporal ACT, without evaluator features."""

import numpy as np

from alexdoor_xas.dataset.temporal import preprocess_rgbd
from alexdoor_xas.recording.b1_runtime import array


class NativeTemporalEvents:
    """Drain actual feedback and controller setter acknowledgments at each observation."""

    def __init__(self, env, config):
        self.env, self.config = env, config
        self.pending = []
        self.generation = None
        self.last_capture = self.last_state = -np.inf
        env.event_sink = self.event

    def event(self, kind, value):
        if self.generation != self.env.generation:
            self.pending.clear()
            self.generation = self.env.generation
            self.last_capture = self.last_state = -np.inf
        if kind == "command":
            t = float(value["request_s"])
            self.pending.append(
                dict(
                    field="target",
                    acquired=t,
                    available=t,
                    values=array(self.env.targets)[0, self.env.arm_ids],
                )
            )
        elif kind == "feedback":
            feedback = value.feedback
            self.pending.append(
                dict(
                    field="tau",
                    acquired=feedback.timestamp_s,
                    available=feedback.available_s,
                    values=np.full(7, np.nan) if feedback.torque_nm is None else feedback.torque_nm,
                    valid=feedback.healthy and not feedback.reason,
                )
            )
        elif kind != "state":
            raise ValueError("Unknown native simulation event")

    def __call__(self, sensor, observation):
        # Provider generation is the observation namespace. Pending events come
        # only from this physical reset, never from a source trajectory.
        if self.generation != self.env.generation:
            self.pending.clear()
            self.generation = self.env.generation
            self.last_capture = self.last_state = -np.inf
        t = float(sensor["time_s"])
        events, self.pending = self.pending, []
        if t > self.last_state:
            for field, key in (("q", "joint_position"), ("dq", "joint_velocity")):
                events.append(dict(field=field, acquired=t, available=t, values=array(sensor[key])))
            self.last_state = t
        if t - self.last_capture >= 0.1 - 1e-9:
            values = preprocess_rgbd(
                sensor["rgb"],
                sensor["depth_m"],
                sensor["valid_depth"],
                self.config,
                device=self.env.device,
            )
            events.append(
                dict(
                    field="images",
                    acquired=t,
                    available=t,
                    values=values,
                    source_index=int(sensor["frame"]),
                )
            )
            self.last_capture = t
        return [dict(event, generation=observation.generation) for event in events]
