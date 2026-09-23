"""Controlled synthetic-door probe. Simulator truth never enters policy adapters."""

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ProbeSetup:
    floor_pose: tuple = (-0.2, 0.3, 0.0)
    contact_fraction: float = 0.9
    contact_height: float = 1.2
    neck: tuple = (0.0, 0.0)
    initial_joints: dict = field(
        default_factory=lambda: {
            "LEFT_SHOULDER_X": 0.35,
            "LEFT_ELBOW_Y": -1.0,
            "RIGHT_SHOULDER_X": -0.35,
            "RIGHT_ELBOW_Y": -1.0,
        }
    )
    max_joint_speed: float = 0.5
    centering_gain: float = 2.0
    ik_damping: float = 0.01
    position_tolerance: float = 0.01
    material_drift_guard_m: float = 0.0025
    orientation_tolerance: float = float(np.deg2rad(5))
    approach_s: float = 6.0
    precontact_m: float = 0.03
    contact_s: float = 3.0
    release_s: float = 3.0
    horizon_s: float = 60.0
    sustain_s: float = 0.5
    hold_settle_s: float = 3.0
    hold_blend_s: float = 0.5
    angular_speed: float = float(np.deg2rad(5))
    lead_angle: float = float(np.deg2rad(0.3))
    compression_m: float = 0.003
    force_limit_n: float = 80.0
    soft_force_n: float = 50.0
    contact_load_guard_n: float = 0.10
    loaded_force_n: float = 0.02
    contact_force_window_s: float = 0.1
    contact_gap_tolerance_m: float = 0.0001
    stall_s: float = 3.0
    tie_deg: float = 0.5

    def __post_init__(self):
        positive = (
            self.max_joint_speed,
            self.centering_gain,
            self.ik_damping,
            self.position_tolerance,
            self.material_drift_guard_m,
            self.orientation_tolerance,
            self.approach_s,
            self.contact_s,
            self.release_s,
            self.horizon_s,
            self.sustain_s,
            self.hold_settle_s,
            self.hold_blend_s,
            self.angular_speed,
            self.lead_angle,
            self.compression_m,
            self.precontact_m,
            self.loaded_force_n,
            self.contact_force_window_s,
            self.contact_gap_tolerance_m,
            self.stall_s,
            self.tie_deg,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Probe tolerances, speeds and durations must be finite and positive")
        if (
            len(self.floor_pose) != 3
            or len(self.neck) != 2
            or not np.isfinite([*self.floor_pose, *self.neck]).all()
        ):
            raise ValueError("Invalid common floor or neck pose")
        if not 0 < self.contact_fraction < 1 or not 0 < self.contact_height < 2.1:
            raise ValueError("Contact lies outside the synthetic panel")
        if not self.loaded_force_n < self.soft_force_n < self.force_limit_n or not np.isfinite(
            self.force_limit_n
        ):
            raise ValueError("Require loaded < soft < hard finite force thresholds")
        if not self.loaded_force_n < self.contact_load_guard_n < self.soft_force_n:
            raise ValueError(
                "Contact-load guard must lie between loaded and upper force thresholds"
            )
        if self.hold_settle_s <= self.sustain_s:
            raise ValueError("Hold budget must include a complete sustain window")
        if self.material_drift_guard_m >= self.position_tolerance:
            raise ValueError("Material drift guard must reserve tracking margin for hold")
        if not np.isfinite(list(self.initial_joints.values())).all():
            raise ValueError("Non-finite ready joint pose")

    def to_dict(self):
        return asdict(self)


class SustainedAngle:
    """Lower angle in a contiguous valid-contact window, with real timestamps."""

    def __init__(self, duration):
        self.duration = duration
        self.samples = deque()
        self.maximum = None

    def update(self, time, angle, valid):
        if not valid:
            self.samples.clear()
            return
        if self.samples and time <= self.samples[-1][0]:
            raise ValueError("Probe times must increase")
        self.samples.append((time, angle))
        while len(self.samples) > 1 and time - self.samples[1][0] >= self.duration - 1e-9:
            self.samples.popleft()
        if time - self.samples[0][0] >= self.duration - 1e-9:
            value = min(a for _, a in self.samples)
            self.maximum = value if self.maximum is None else max(self.maximum, value)


class ContactLoad:
    """Causal mean of fixed-cadence force samples, gated by actual contact distance."""

    def __init__(self, duration, minimum_force, maximum_gap):
        self.duration = duration
        self.minimum_force = minimum_force
        self.maximum_gap = maximum_gap
        self.samples = deque()
        self.force = 0.0

    def update(self, time, force, gap):
        self.samples.append((time, force))
        while time - self.samples[0][0] >= self.duration - 1e-9:
            self.samples.popleft()
        self.force = sum(f for _, f in self.samples) / len(self.samples)
        return gap is not None and gap <= self.maximum_gap and self.force >= self.minimum_force


def rank_candidates(results, tie_deg):
    """Only complete four-case controlled candidates participate in minimax."""
    valid = [r for r in results if len(r["cases"]) == 4 and all(c["passed"] for c in r["cases"])]
    if not valid:
        return []
    remaining, ordered = list(valid), []
    while remaining:
        best = max(min(c["angle_deg"] for c in r["cases"]) for r in remaining)
        tied = [r for r in remaining if min(c["angle_deg"] for c in r["cases"]) >= best - tie_deg]
        tied.sort(
            key=lambda r: (
                -max(0.0, min(c["joint_margin"] for c in r["cases"])),
                max(c["peak_force_n"] for c in r["cases"]),
                -min(c["clearance_m"] for c in r["cases"]),
            )
        )
        ordered.extend(tied)
        remaining = [r for r in remaining if r not in tied]
    return ordered


def summarize_trials(trials):
    """Accept only resolved stops and repeatable limiting causes within two degrees."""
    resolved = {"kinematic_limit", "mechanical_stop", "safety_stop"}
    controlled = bool(trials) and all(
        r["passed"]
        and r["released"]
        and r["stop_reason"] in resolved
        and r["angle_deg"] is not None
        and np.isfinite(r["angle_deg"])
        for r in trials
    )
    causes = {(r["stop_reason"], r.get("safety_detail")) for r in trials}
    spread = (
        max(r["angle_deg"] for r in trials) - min(r["angle_deg"] for r in trials)
        if controlled
        else None
    )
    consistent = controlled and len(causes) == 1 and spread <= 2.0
    return dict(
        case=trials[0]["case"],
        passed=bool(consistent),
        meets_45_deg=bool(consistent and min(r["angle_deg"] for r in trials) >= 45.0),
        trials=trials,
        angle_deg=min(r["angle_deg"] for r in trials) if consistent else None,
        repeat_spread_deg=spread,
        joint_margin=min(r["joint_margin"] for r in trials),
        peak_force_n=max(r["peak_force_n"] for r in trials),
        clearance_m=min(r["clearance_m"] for r in trials),
    )


def run_probe(env, door, setup, output):
    import json

    import torch
    from scipy.spatial.transform import Rotation, Slerp

    from alexdoor_xas.envs.door_task.door_push_purdue_env import tensor

    output.mkdir(parents=True, exist_ok=True)
    (output / "setup.json").write_text(json.dumps(setup.to_dict(), indent=2) + "\n")
    env.reset()
    env.set_neck_target(setup.neck)
    env.sim.stage.Export(str(output / "scene.usda"))
    from alexdoor_xas.kinematics.purdue_chain import PurdueChain

    chain = PurdueChain(env.robot.cfg.spawn.asset_path, env.device)
    q_now = tensor(env.robot.data.joint_pos)[:, env.arm_ids]
    predicted, _ = chain.forward(q_now)
    root_p = tensor(env.robot.data.root_link_pos_w)[0].cpu().numpy()
    root_q = tensor(env.robot.data.root_link_quat_w)[0].cpu().numpy()
    root_r = Rotation.from_quat(root_q).as_matrix()
    actual_p, actual_q = [v[0].cpu().numpy() for v in env.tool_pose()]
    expected_p = root_p + root_r @ predicted[0, :3, 3].cpu().numpy()
    expected_r = root_r @ predicted[0, :3, :3].cpu().numpy()
    if (
        np.linalg.norm(actual_p - expected_p) > 0.001
        or Rotation.from_matrix(expected_r @ Rotation.from_quat(actual_q).as_matrix().T).magnitude()
        > 0.01
    ):
        raise RuntimeError("URDF screening disagrees with imported runtime kinematics")
    gripper_ids, _ = env.robot.find_joints(
        [
            "left_WSG32_JAW_OPENING",
            "left_WSG32_JAW_FOLLOWER",
            "right_WSG32_JAW_OPENING",
            "right_WSG32_JAW_FOLLOWER",
        ],
        preserve_order=True,
    )
    if len(gripper_ids) != 4:
        raise RuntimeError("Expected both closed WSG leader/follower pairs")
    zero = torch.zeros((1, 6), device=env.device)
    from alexdoor_xas.qualification.clearance import DoorClearance

    clearance = DoorClearance(env)
    min_clearance = float("inf")
    traces, raw, images = [], [], []
    distal = []
    for vertices in env.push_geometry.finger_vertices.values():
        face = vertices[np.abs(vertices[:, 0] - vertices[:, 0].max()) < 1e-6]
        distal.extend(face - env.push_geometry.translation)
    distal = np.asarray(distal)
    window = SustainedAngle(setup.sustain_s)
    contact_load = ContactLoad(
        setup.contact_force_window_s, setup.loaded_force_n, setup.contact_gap_tolerance_m
    )
    failure, reason, peak, min_margin = None, None, 0.0, 1.0
    dt = env.step_dt
    last_progress, last_angle = 0.0, 0.0
    missing_time = 0.0
    limit_evidence = None
    load_window = deque(maxlen=max(1, round(0.1 / dt)))
    safety_detail = None
    start_p, start_q = [v[0].cpu().numpy() for v in env.tool_pose()]

    def command(goal_p, goal_r, phase):
        nonlocal failure, peak, min_margin, min_clearance
        env.command_pose(goal_p, Rotation.from_matrix(goal_r).as_quat())
        env.step(zero)
        angle = float(tensor(env.door.data.joint_pos)[0, 0])
        speed = float(tensor(env.door.data.joint_vel)[0, 0])
        p, q = [v[0].cpu().numpy() for v in env.tool_pose()]
        actual_r = Rotation.from_quat(q).as_matrix()
        pe = float(np.linalg.norm(p - goal_p))
        re = float(Rotation.from_matrix(goal_r @ actual_r.T).magnitude())
        joints = tensor(env.robot.data.joint_pos)[0, env.arm_ids].cpu().numpy()
        limits = env.limits[0].cpu().numpy()
        margin = float(
            np.min(
                np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
                / (limits[:, 1] - limits[:, 0])
            )
        )
        closed_error = float(tensor(env.robot.data.joint_pos)[0, gripper_ids].abs().max())
        contacts = env.contact_history
        sample_forces = [
            sum(
                c["force_n"]
                for c in sample["contacts"]
                if c["category"] in ("positive", "negative")
            )
            for sample in contacts
        ]
        force = max(sample_forces)
        forbidden = any(sample["forbidden"] for sample in contacts)
        gaps = [
            min(
                (
                    c["separation_m"]
                    for c in sample["contacts"]
                    if c["category"] in ("positive", "negative")
                ),
                default=None,
            )
            for sample in contacts
        ]
        gap = max(gaps) if all(g is not None for g in gaps) else None
        time = len(traces) * dt
        loaded = contact_load.update(time, float(np.mean(sample_forces)), gap)
        peak, min_margin = max(peak, force), min(min_margin, margin)
        if not np.isfinite(np.r_[angle, speed, pe, re, joints, force]).all():
            failure = "invalid_physics"
        elif closed_error > 0.001:
            failure = "gripper_opened"
        elif np.any(joints < limits[:, 0] - 0.005) or np.any(joints > limits[:, 1] + 0.005):
            failure = "joint_limit_violation"
        elif forbidden:
            failure = "forbidden_contact"
        elif force > setup.force_limit_n:
            failure = "force_limit"
        material_p, material_r = door.contact_pose(
            angle, setup.contact_fraction, setup.contact_height
        )
        material_error = float(np.linalg.norm(p - material_p))
        orientation_error = float(Rotation.from_matrix(material_r @ actual_r.T).magnitude())
        footprint = ((distal @ actual_r.T + p) - door.hinge) @ material_r
        from_hinge = -door.sign * footprint[:, 1]
        footprint_inside = bool(
            np.all(
                (from_hinge >= 0.0)
                & (from_hinge <= door.width)
                & (footprint[:, 2] >= 0.01)
                & (footprint[:, 2] <= door.height + 0.01)
            )
        )
        valid = (
            loaded
            and footprint_inside
            and failure is None
            and material_error <= setup.position_tolerance
            and orientation_error <= setup.orientation_tolerance
        )
        min_clearance = min(min_clearance, clearance.measure(door, angle))
        visibility = None
        if env.capture is not None:
            from alexdoor_xas.qualification.visibility import measure_visibility

            visibility = measure_visibility(
                env, door, angle, setup.contact_fraction, setup.contact_height
            )
        if len(traces) % 300 == 0:
            print(
                f"{door.name} {phase} t={time:.1f}s angle={np.rad2deg(angle):.2f}deg "
                f"force={force:.3f}N error={pe:.4f}m",
                flush=True,
            )
        window.update(time, angle, valid and phase in ("push", "hold"))
        traces.append(
            dict(
                time=time,
                phase=phase,
                angle=angle,
                speed=speed,
                position_error=pe,
                orientation_error=re,
                material_error=material_error,
                footprint_inside=footprint_inside,
                closed_gripper_error_m=closed_error,
                force=force,
                filtered_force_n=contact_load.force,
                contact_gap_m=gap,
                loaded=loaded,
                valid=valid,
                joint_margin=margin,
                joints=joints.tolist(),
                tool_position=p.tolist(),
                tool_quaternion=q.tolist(),
                visibility=visibility,
            )
        )
        if any(sample["contacts"] for sample in contacts):
            raw.append(dict(time=time, samples=contacts))
        if env.capture is not None and len(traces) % 30 == 0:
            sample = env.capture.sample
            images.append(
                (len(traces), sample.rgb[0].cpu().numpy(), sample.depth_m[0].cpu().numpy())
            )
        return angle, pe, re, force, margin, loaded, valid

    pre_p, pre_r = door.contact_pose(
        0.0, setup.contact_fraction, setup.contact_height, -setup.precontact_m
    )
    slerp = Slerp([0, 1], Rotation.from_quat([start_q, Rotation.from_matrix(pre_r).as_quat()]))
    for tick in range(round(setup.approach_s / dt)):
        a = min(1.0, (tick + 1) * dt / (setup.approach_s * 0.7))
        blend = a * a * (3 - 2 * a)
        state = command(start_p + (pre_p - start_p) * blend, slerp(blend).as_matrix(), "approach")
        if failure:
            break
    if not failure and (
        state[1] > setup.position_tolerance or state[2] > setup.orientation_tolerance
    ):
        failure = "approach_tracking"
    if not failure:
        for tick in range(round(setup.contact_s / dt)):
            offset = -setup.precontact_m + (setup.precontact_m + setup.compression_m) * min(
                1.0, (tick + 1) * dt / (setup.contact_s * 0.7)
            )
            angle = float(tensor(env.door.data.joint_pos)[0, 0])
            state = command(
                *door.contact_pose(angle, setup.contact_fraction, setup.contact_height, offset),
                "contact",
            )
            if failure:
                break
        if not failure and not state[5]:
            failure = "no_loaded_contact"
    if not failure:
        reference = state[0]
        for tick in range(round(setup.horizon_s / dt)):
            angle = float(tensor(env.door.data.joint_pos)[0, 0])
            reference = min(
                reference + setup.angular_speed * dt, angle + setup.lead_angle, door.mechanical_stop
            )
            state = command(
                *door.contact_pose(
                    reference, setup.contact_fraction, setup.contact_height, setup.compression_m
                ),
                "push",
            )
            angle, pe, re, force, margin, loaded, valid = state
            elapsed = tick * dt
            if angle > last_angle + np.deg2rad(0.25):
                last_progress, last_angle = elapsed, angle
            missing_time = 0.0 if loaded else missing_time + dt
            if failure:
                break
            if margin < 0.002 and loaded:
                from alexdoor_xas.qualification.limits import constrained_step_evidence

                next_p, next_r = door.contact_pose(
                    angle + np.deg2rad(2), setup.contact_fraction, setup.contact_height
                )
                now_p, now_r = door.contact_pose(
                    angle, setup.contact_fraction, setup.contact_height
                )
                desired = np.r_[next_p - now_p, Rotation.from_matrix(next_r @ now_r.T).as_rotvec()]
                limit_evidence = constrained_step_evidence(
                    env.tool_jacobian()[0].cpu().numpy(),
                    tensor(env.robot.data.joint_pos)[0, env.arm_ids].cpu().numpy(),
                    env.limits[0].cpu().numpy(),
                    desired,
                )
                if limit_evidence["blocked"]:
                    reason = "kinematic_limit"
                    break
            if missing_time > 0.25:
                failure = "lost_contact"
                break
            if abs(angle - door.mechanical_stop) < np.deg2rad(0.5):
                reason = "mechanical_stop"
                break
            if (
                elapsed > 5.0
                and window.maximum is not None
                and (
                    traces[-1]["material_error"] > setup.material_drift_guard_m
                    or re > 0.5 * setup.orientation_tolerance
                )
            ):
                reason = "safety_stop"
                safety_detail = "tracking_margin"
                break
            load_window.append(force)
            if (
                elapsed > 5.0
                and window.maximum is not None
                and np.mean(load_window) < setup.contact_load_guard_n
            ):
                reason = "safety_stop"
                safety_detail = "declining_contact_load"
                break
            if force >= setup.soft_force_n:
                reason = "safety_stop"
                safety_detail = "high_normal_force"
                break
            if elapsed - last_progress > setup.stall_s:
                # Saturation alone cannot prove the required pose is unreachable.
                failure = "solver_tracking_stall"
                break
        else:
            failure = "timeout"
    held_angle = None
    if failure is None:
        held = SustainedAngle(setup.sustain_s)
        initial_lead = reference - state[0]
        for tick in range(round(setup.hold_settle_s / dt)):
            angle = float(tensor(env.door.data.joint_pos)[0, 0])
            blend = min(1.0, (tick + 1) * dt / setup.hold_blend_s)
            lead = initial_lead * (1.0 - blend * blend * (3.0 - 2.0 * blend))
            state = command(
                *door.contact_pose(
                    angle + lead,
                    setup.contact_fraction,
                    setup.contact_height,
                    setup.compression_m,
                ),
                "hold",
            )
            held.update(tick * dt, state[0], state[6])
            if failure:
                break
            if held.maximum is not None:
                held_angle = held.maximum
                break
        if held_angle is None:
            failure = failure or "invalid_hold"
    # Retrace a previously achieved pose behind the moving panel, releasing wrist limits.
    released = False
    if failure != "invalid_physics":
        final_angle = float(tensor(env.door.data.joint_pos)[0, 0])
        prior = [
            t
            for t in traces
            if t["phase"] in ("approach", "push") and t["angle"] <= final_angle - np.deg2rad(5)
        ]
        if prior:
            retreat_p = np.array(prior[-1]["tool_position"])
            retreat_q = np.array(prior[-1]["tool_quaternion"])
        else:
            retreat_p, retreat_q = pre_p, Rotation.from_matrix(pre_r).as_quat()
        current_p, current_q = [v[0].cpu().numpy() for v in env.tool_pose()]
        release_slerp = Slerp([0, 1], Rotation.from_quat([current_q, retreat_q]))
        for tick in range(round(setup.release_s / dt)):
            a = min(1.0, (tick + 1) * dt / (setup.release_s * 0.7))
            state = command(
                current_p + (retreat_p - current_p) * a, release_slerp(a).as_matrix(), "release"
            )
            if failure == "invalid_physics":
                break
        actual_p = env.tool_pose()[0][0].cpu().numpy()
        material_p, material_r = door.contact_pose(
            state[0], setup.contact_fraction, setup.contact_height
        )
        separation = float(np.dot(material_p - actual_p, material_r[:, 0]))
        released = not state[5] and separation >= 0.01 and state[1] <= setup.position_tolerance
    if not released:
        failure = failure or "unsafe_release"
    measured = held_angle if failure is None else window.maximum
    result = dict(
        case=door.name,
        passed=failure is None and window.maximum is not None,
        angle_deg=None if measured is None else float(np.rad2deg(measured)),
        stop_reason=failure or reason,
        limit_evidence=limit_evidence,
        safety_detail=safety_detail,
        peak_force_n=peak,
        joint_margin=min_margin,
        clearance_m=min_clearance,
        clearance_scope="robot_door_AABB_lower_bound_excluding_distal_panel_pairs",
        released=released,
        scope="synthetic_controlled_probe",
        force_components="normal_only",
        visibility=None
        if env.capture is None
        else dict(
            passed=all(t["visibility"]["passed"] for t in traces),
            failed_frames=sum(not t["visibility"]["passed"] for t in traces),
            checked_frames=len(traces),
            scope="geometric_panel_contact_surround_frame_observability",
        ),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trace.json").write_text(json.dumps(traces) + "\n")
    (output / "contacts.json").write_text(json.dumps(raw) + "\n")
    if images:
        from PIL import Image

        for tick, rgb, depth in images:
            Image.fromarray(rgb[..., :3]).save(output / f"rgb-{tick:05d}.png")
            np.save(output / f"depth-{tick:05d}.npy", depth)
    return result
