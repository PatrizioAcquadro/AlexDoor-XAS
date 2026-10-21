"""Shared controlled door probe. Simulator truth never enters policy adapters."""

import json
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
    door_mass_scale: float = 1.0
    door_hinge_damping: float | None = None
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
    preroll_s: float = 0.4
    stabilization_min_s: float = 2.0
    stabilization_max_s: float = 6.0
    stabilization_window_s: float = 0.5
    intermediate_preroll_s: float = 1.0
    horizon_s: float = 60.0
    sustain_s: float = 0.5
    hold_settle_s: float = 3.0
    hold_blend_s: float = 0.5
    angular_speed: float = float(np.deg2rad(5))
    lead_angle: float = float(np.deg2rad(0.3))
    compression_m: float = 0.003
    normal_admittance_m_n_s: float = 0.0005
    force_limit_n: float = 80.0
    soft_force_n: float = 50.0
    contact_load_guard_n: float = 0.10
    hold_force_n: float = 0.05
    loaded_force_n: float = 0.02
    contact_force_window_s: float = 0.1
    contact_gap_tolerance_m: float = 0.0001
    stall_s: float = 3.0
    tie_deg: float = 0.5  # Historical frozen setup field; no active ranking.

    def __post_init__(self):
        positive = (
            self.door_mass_scale,
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
            self.preroll_s,
            self.stabilization_min_s,
            self.stabilization_max_s,
            self.stabilization_window_s,
            self.intermediate_preroll_s,
            self.sustain_s,
            self.hold_settle_s,
            self.hold_blend_s,
            self.angular_speed,
            self.lead_angle,
            self.compression_m,
            self.normal_admittance_m_n_s,
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
        if not self.loaded_force_n < self.hold_force_n <= self.contact_load_guard_n:
            raise ValueError("Hold force target must preserve a positive loaded-contact margin")
        if (
            not 2 <= self.stabilization_min_s <= self.stabilization_max_s
            or self.intermediate_preroll_s <= self.sustain_s + self.contact_force_window_s
        ):
            raise ValueError("Require ordinary tail >=2s and fresh loaded pre-roll support")
        if self.hold_settle_s <= self.sustain_s:
            raise ValueError("Hold budget must include a complete sustain window")
        if self.material_drift_guard_m >= self.position_tolerance:
            raise ValueError("Material drift guard must reserve tracking margin for hold")
        if self.door_hinge_damping is not None and (
            not np.isfinite(self.door_hinge_damping) or self.door_hinge_damping < 0
        ):
            raise ValueError("Invalid simulated hinge damping")
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

    @property
    def current(self):
        """Minimum of the valid window ending at the most recent sample."""
        if self.samples and self.samples[-1][0] - self.samples[0][0] >= self.duration - 1e-9:
            return min(a for _, a in self.samples)
        return None

    def update(self, time, angle, valid):
        if not valid:
            self.samples.clear()
            return
        if self.samples and time <= self.samples[-1][0]:
            raise ValueError("Probe times must increase")
        self.samples.append((time, angle))
        while len(self.samples) > 1 and time - self.samples[1][0] >= self.duration - 1e-9:
            self.samples.popleft()
        if (value := self.current) is not None:
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


def hold_initial_lead(reference, angle, reference_step, lead_limit):
    """Carry the final commanded angular increment into the first hold endpoint.

    The incoming reference describes the last completed interval. Reusing it as
    the next endpoint pauses a moving unilateral push for one control tick.
    """
    return min(reference + max(0.0, reference_step) - angle, lead_limit)


def hold_reference(angle, initial_lead, mechanical_stop):
    """Material-following coordinates for pressure-regulated unilateral holding.

    This is not a new opening-angle setpoint. Load loss aborts the hold, so free
    coasting cannot become controlled progress by geometric following alone.
    """
    return min(angle + initial_lead, mechanical_stop)


def pressure_hold_pose(door, setup, position_angle, orientation_angle, compression):
    """Keep the pressure-position reference while aligning the contact faces."""
    position, _ = door.contact_pose(
        position_angle, setup.contact_fraction, setup.contact_height, compression
    )
    _, rotation = door.contact_pose(
        orientation_angle, setup.contact_fraction, setup.contact_height, compression
    )
    return position, rotation


def achieved_contact_hold_state(door, setup, row):
    """One measured position-to-force handoff; no state or gain is changed."""
    if not row["valid"] or not row["physical_safe"]:
        raise ValueError("Supported physical contact is required for hold handoff")
    angle = row["angle"]
    origin, rotation = door.contact_pose(angle, setup.contact_fraction, setup.contact_height)
    local = (np.asarray(row["tool_position"]) - origin) @ rotation
    bound = setup.position_tolerance - setup.material_drift_guard_m
    if abs(local[0]) > bound or np.linalg.norm(local[1:]) > bound:
        raise ValueError("Achieved contact exceeds existing controller offset bounds")
    tangent = np.array([0.0, local[1], local[2]])
    return angle, float(local[0]), tangent


def push_reference(reference, angle, setup, dt, mechanical_stop):
    """Advance on the reference clock; a coasting panel cannot advance the command."""
    return min(reference + setup.angular_speed * dt, angle + setup.lead_angle, mechanical_stop)


def tracking_reserve(samples, horizon):
    """Reserve hold time for a growing tracking error; ignore declining trends."""
    times, errors = np.asarray(samples, dtype=float).T
    centered = times - times.mean()
    denominator = np.dot(centered, centered)
    slope = np.dot(centered, errors) / denominator if denominator > 0 else 0.0
    return float(errors[-1] + max(0.0, slope) * horizon)


def tracking_margin_exhausted(samples, setup, *, orientation=False):
    """Stop sustained growth that threatens holding, not a stable servo bias."""
    guard = 0.5 * setup.orientation_tolerance if orientation else setup.material_drift_guard_m
    margin = (
        setup.orientation_tolerance
        if orientation
        else setup.position_tolerance - setup.material_drift_guard_m
    )
    if samples[-1][1] >= margin:
        return True
    recent = [s for s in samples if samples[-1][0] - s[0] <= setup.sustain_s + 1e-9]
    sustained = recent[-1][0] - recent[0][0] >= setup.sustain_s - 1e-9 and all(
        error > guard for _, error in recent
    )
    return sustained and tracking_reserve(recent, setup.hold_settle_s) >= margin


def conditioning_margin_exhausted(samples, setup, *, extra_contact_s=0.0):
    """Reserve hold time before entering the frozen DLS attenuation regime.

    At sigma=2*damping the weakest retained task-space gain is already 80%.
    This is a controller-derived planning guard, not a changed physical score
    or proof of global unreachability. Use a sustained trend, never a door angle.
    """
    if not np.isfinite(extra_contact_s) or extra_contact_s < 0:
        raise ValueError("Additional contact reserve must be finite and nonnegative")
    minimum = 2 * setup.ik_damping
    if samples[-1][1] <= minimum:
        return True
    if samples[-1][0] - samples[0][0] < setup.sustain_s - 1e-9:
        return False
    return (
        -tracking_reserve(
            [(t, -sigma) for t, sigma in samples],
            setup.hold_settle_s + setup.release_s + extra_contact_s,
        )
        <= minimum
    )


def normal_contact_support(compression, force, setup, dt, *, target_force=None, gap_m=None):
    """Bidirectional bounded force admittance, including removal of excess preload."""
    target_force = setup.contact_load_guard_n if target_force is None else target_force
    rate = setup.normal_admittance_m_n_s * (target_force - force)
    if gap_m is not None:
        if not np.isfinite(gap_m):
            raise ValueError("Invalid measured contact gap")
        if force < target_force and gap_m > 0:
            # Correct the measured separating displacement at the next control
            # boundary. The low-level servo still evolves physically; retain
            # normal rate/extent bounds and force-unloading priority.
            # Gap catch-up restores geometry; it must not replace the positive
            # force-error admittance that establishes load beyond first touch.
            rate += gap_m / dt
    rate_limit = setup.compression_m / setup.sustain_s
    increment = np.clip(rate, -rate_limit, rate_limit) * dt
    return float(
        np.clip(
            compression + increment,
            -setup.position_tolerance + setup.material_drift_guard_m,
            setup.position_tolerance - setup.material_drift_guard_m,
        )
    )


def tangential_compensation(offset, error, enabled, setup, dt):
    """Bounded integral correction of tracking bias along the leaf surface."""
    if not enabled:
        return offset.copy()
    error = np.asarray(error).copy()
    error[0] = 0.0  # Normal loading remains the separate compression controller.
    increment = error * dt / setup.sustain_s
    rate_bound = setup.compression_m * dt / setup.sustain_s
    increment *= min(1.0, rate_bound / max(np.linalg.norm(increment), 1e-12))
    corrected = offset + increment
    bound = setup.position_tolerance - setup.material_drift_guard_m
    return corrected * min(1.0, bound / max(np.linalg.norm(corrected), 1e-12))


def release_reference(start_p, start_r, retreat_p, retreat_r, elapsed, setup):
    """Withdraw along the fingers before rotating back toward an achieved pose."""
    from scipy.spatial.transform import Rotation, Slerp

    halfway = setup.release_s / 2
    withdrawn = start_p - setup.precontact_m * start_r[:, 0]
    if elapsed <= halfway:
        u = float(np.clip(elapsed / (halfway * 0.7), 0.0, 1.0))
        return start_p + (withdrawn - start_p) * u * u * (3 - 2 * u), start_r
    u = float(np.clip((elapsed - halfway) / (halfway * 0.7), 0.0, 1.0))
    blend = u * u * (3 - 2 * u)
    rotation = Slerp([0, 1], Rotation.from_matrix([start_r, retreat_r]))(blend).as_matrix()
    return withdrawn + (retreat_p - withdrawn) * blend, rotation


def summarize_trials(trials, *, required_count=None):
    """Accept only resolved stops and repeatable limiting causes within two degrees."""
    resolved = {"kinematic_limit", "mechanical_stop", "safety_stop"}
    controlled = (
        bool(trials)
        and (required_count is None or len(trials) == required_count)
        and all(
            r["passed"]
            and r["released"]
            and r["stop_reason"] in resolved
            and r["angle_deg"] is not None
            and np.isfinite(r["angle_deg"])
            for r in trials
        )
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


def prepare_probe(
    env, door, setup, output, recorder, inspection, intermediate=None, *, qualify_start_stop=False
):
    import json

    from scipy.spatial.transform import Rotation

    from alexdoor_xas.envs.door_task.door_push_purdue_env import tensor

    output.mkdir(parents=True, exist_ok=True)
    (output / "setup.json").write_text(json.dumps(setup.to_dict(), indent=2) + "\n")
    env.reset()
    env.set_neck_target(setup.neck)
    if intermediate is not None:
        from alexdoor_xas.qualification.intermediate import restore_hold_state

        audit = restore_hold_state(
            env,
            intermediate,
            setup,
            output / "intermediate-reset-physics.json",
            continue_controller=qualify_start_stop,
        )
        if qualify_start_stop:
            if audit["conditioning_contact_force_n"][-1] < setup.loaded_force_n:
                raise ValueError("loaded_start_contact_not_reconstructed")
            from alexdoor_xas.qualification.stop import verify_stop_continuation

            stopping = verify_stop_continuation(
                env,
                output / "start-stop.npz",
                force_limit_n=setup.force_limit_n,
            )
            diagnostic_s = getattr(env.cfg, "stop_diagnostic_continuation_s", 0.0)
            if diagnostic_s:
                stopping["two_second_gate_passed"] = stopping["passed"]
                stopping["diagnostic_continuation"] = verify_stop_continuation(
                    env,
                    output / "start-stop-diagnostic.npz",
                    duration_s=diagnostic_s,
                    force_limit_n=setup.force_limit_n,
                )
                stopping["passed"] &= stopping["diagnostic_continuation"]["passed"]
            (output / "start-stop.json").write_text(json.dumps(stopping, indent=2) + "\n")
            if not stopping["passed"]:
                raise ValueError("intermediate_start_stop_unqualified")
            env.reset()
            env.set_neck_target(setup.neck)
            audit = restore_hold_state(
                env,
                intermediate,
                setup,
                output / "intermediate-recording-reset-physics.json",
                continue_controller=qualify_start_stop,
            )
        if qualify_start_stop:
            if audit["conditioning_contact_force_n"][-1] < setup.loaded_force_n:
                raise ValueError("recording_start_contact_not_reconstructed")
            intermediate.update(audit["conditioned_controller"])
        (output / "intermediate-restore.json").write_text(json.dumps(audit, indent=2) + "\n")
    if recorder is not None:
        recorder.start(
            env,
            door,
            setup,
            phase="hold" if intermediate is not None else "inspect" if inspection else "approach",
        )
    if inspection and intermediate is None:
        from alexdoor_xas.perception.inspection import run_inspection

        audit = run_inspection(env, inspection, recorder)
        (output / "inspection.json").write_text(json.dumps(audit, indent=2) + "\n")
        # The static reference is acquired at the unchanged inspection endpoint.
        # Keep real parked-joint control for readiness plus one action horizon,
        # so a usable chunk can actually cross inspection -> approach.
        neck = list(inspection["waypoints"][-1][1:])
        pre_roll = dict(inspection, waypoints=[[0.0, *neck], [setup.preroll_s, *neck]])
        audit = run_inspection(env, pre_roll, recorder)
        (output / "preroll.json").write_text(json.dumps(audit, indent=2) + "\n")
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
    return chain, gripper_ids, root_p, root_r


def write_probe_artifacts(output, result, traces, contacts, images):
    import json

    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trace.json").write_text(json.dumps(traces) + "\n")
    (output / "contacts.json").write_text(json.dumps(contacts) + "\n")
    if images:
        from PIL import Image

        for tick, rgb, depth, _ in images:
            Image.fromarray(rgb[..., :3]).save(output / f"rgb-{tick:05d}.png")
            np.save(output / f"depth-{tick:05d}.npy", depth)
        (output / "camera.json").write_text(
            json.dumps({str(tick): camera for tick, _, _, camera in images}) + "\n"
        )


def run_probe(
    env,
    door,
    setup,
    output,
    *,
    recorder=None,
    capture_evidence=True,
    inspection=None,
    intermediate=None,
    campaign=None,
    task_hold=None,
):
    import torch
    from scipy.spatial.transform import Rotation, Slerp

    from alexdoor_xas.envs.door_task.door_push_purdue_env import tensor

    chain, gripper_ids, root_p, root_r = prepare_probe(
        env,
        door,
        setup,
        output,
        recorder,
        inspection,
        intermediate,
        qualify_start_stop=campaign is not None and intermediate is not None,
    )
    family = None if campaign is None else campaign["family"]
    correction = None
    if campaign is not None and campaign.get("variation", "").startswith("correction-"):
        from alexdoor_xas.qualification.collection_corrections import CollectionCorrection

        correction = CollectionCorrection(family, campaign["variation"], setup, env.step_dt)
    if family is not None:
        from alexdoor_xas.qualification.collection_phase import evaluate_phase_endpoint
    settle_c_hold = family == "C"
    hold_budget_s = setup.hold_settle_s * (2 if settle_c_hold else 1)
    if task_hold is not None:
        if family is not None or intermediate is not None:
            raise ValueError("Task-goal hold requires an ordinary complete episode")
        goal, maximum = task_hold["required_opening_deg"], task_hold["maximum_s"]
        if not np.isfinite([goal, maximum]).all() or not (
            goal > 0 and setup.hold_settle_s <= maximum <= 2 * setup.hold_settle_s
        ):
            raise ValueError("Invalid bounded task-goal hold")
        hold_budget_s = maximum
    late_states = deque(maxlen=round(5.0 / env.step_dt) + 1)
    zero = torch.zeros((1, 6), device=env.device)
    from alexdoor_xas.qualification.clearance import DoorClearance

    clearance = DoorClearance(env)
    min_clearance = float("inf")
    traces, raw, images = [], [], []
    distal = np.concatenate(env.push_geometry.distal_faces)
    window = SustainedAngle(setup.sustain_s)
    contact_load = ContactLoad(
        setup.contact_force_window_s, setup.loaded_force_n, setup.contact_gap_tolerance_m
    )
    failure, reason, peak, min_margin = None, None, 0.0, 1.0
    peak_joint_speed = 0.0
    dt = env.step_dt
    last_progress, last_angle = 0.0, 0.0
    missing_time = 0.0
    limit_evidence = None
    load_window = deque(maxlen=max(1, round(0.1 / dt)))
    safety_detail = None
    tracking = deque(maxlen=max(2, round(setup.sustain_s / dt) + 1))
    angular_tracking = deque(maxlen=tracking.maxlen)
    conditioning = deque(maxlen=tracking.maxlen)
    start_p, start_q = [v[0].cpu().numpy() for v in env.tool_pose()]
    compensation = np.zeros(3)
    normal_feedback = None

    def support_normal(compression, phase):
        nonlocal normal_feedback
        gap = traces[-1]["contact_gap_m"] if traces else None
        force = contact_load.force
        source = "contact_qualification_window"
        achieved_angle = correction is not None and correction.holding_achieved_angle()
        if (
            correction is not None
            and correction.supported_hold_handoffs
            and (phase == "hold" or achieved_angle)
        ):
            # After the position-to-force handoff, stale impact load must not
            # suppress catch-up after actual contact disappears. Keep the .1s
            # load window for qualification; control uses the preceding native
            # interval's measured average, with the same admittance and bounds.
            force = traces[-1]["native_mean_normal_force_n"]
            source = "preceding_native_interval_mean"
        normal_feedback = dict(force_n=force, source=source)
        if correction is not None and correction.position_return_active(phase):
            # The measured-origin transition already corrects position error.
            # Do not stack a second gap catch-up; keep force unloading active.
            gap = None
        return normal_contact_support(
            compression,
            force,
            setup,
            dt,
            target_force=setup.hold_force_n if phase == "hold" or achieved_angle else None,
            gap_m=gap,
        )

    def command(goal_p, goal_r, phase, *, injected_pose=None):
        nonlocal failure, peak, min_margin, min_clearance, compensation, peak_joint_speed
        correction_active = correction is not None and failure is None
        applied_compensation = (
            compensation.copy() if phase in ("contact", "push", "hold") else np.zeros(3)
        )
        goal_p = goal_p + goal_r @ applied_compensation
        if correction_active:
            gap_target = correction.material_gap_target(phase)
            if gap_target is not None:
                material_angle = float(tensor(env.door.data.joint_pos)[0, 0])
                injected_pose = door.contact_pose(
                    material_angle, setup.contact_fraction, setup.contact_height, -gap_target
                )
            if injected_pose is not None:
                injected_p, injected_r = injected_pose
                injected_pose = (injected_p + injected_r @ applied_compensation, injected_r)
            measured_p, measured_q = [v[0].cpu().numpy() for v in env.tool_pose()]
            goal_p, goal_r = correction.target(
                goal_p,
                goal_r,
                phase,
                measured_p,
                Rotation.from_quat(measured_q).as_matrix(),
                injected_pose=injected_pose,
            )
        if recorder is not None:
            recorder.phase = phase
        env.command_pose(goal_p, Rotation.from_matrix(goal_r).as_quat())
        env.step(zero)
        angle = float(tensor(env.door.data.joint_pos)[0, 0])
        speed = float(tensor(env.door.data.joint_vel)[0, 0])
        p, q = [v[0].cpu().numpy() for v in env.tool_pose()]
        actual_r = Rotation.from_quat(q).as_matrix()
        pe = float(np.linalg.norm(p - goal_p))
        re = float(Rotation.from_matrix(goal_r @ actual_r.T).magnitude())
        joints = tensor(env.robot.data.joint_pos)[0, env.arm_ids].cpu().numpy()
        actual_speed = max(float(np.max(np.abs(s.dq))) for s in env.load_samples)
        peak_joint_speed = max(peak_joint_speed, actual_speed)
        sigma = min(
            float(np.linalg.svd(s.jacobian_world, compute_uv=False)[-1]) for s in env.load_samples
        )
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
        violation = None
        if not np.isfinite(np.r_[angle, speed, pe, re, joints, force]).all():
            violation = "invalid_physics"
        elif closed_error > 0.001:
            violation = "gripper_opened"
        elif np.any(joints < limits[:, 0] - 0.005) or np.any(joints > limits[:, 1] + 0.005):
            violation = "joint_limit_violation"
        elif actual_speed > setup.max_joint_speed * 1.02:
            violation = "actual_joint_speed_limit"
            env.stop_motion()
        elif forbidden:
            violation = "forbidden_contact"
        elif force > setup.force_limit_n:
            violation = "force_limit"
        if violation is not None:
            failure = failure or violation
        material_p, material_r = door.contact_pose(
            angle, setup.contact_fraction, setup.contact_height
        )
        material_error = float(np.linalg.norm(p - material_p))
        orientation_error = float(Rotation.from_matrix(material_r @ actual_r.T).magnitude())
        footprint_inside = door.footprint_inside(distal @ actual_r.T + p, angle)
        physical_safe = violation is None and not env.motion_stopped
        valid = (
            loaded
            and footprint_inside
            and failure is None
            and material_error <= setup.position_tolerance
            and orientation_error <= setup.orientation_tolerance
        )
        if phase in ("contact", "push", "hold") and failure is None:
            compensation = tangential_compensation(
                compensation,
                (material_p - p) @ material_r,
                (phase == "contact" or loaded) and orientation_error <= setup.orientation_tolerance,
                setup,
                dt,
            )
        min_clearance = min(min_clearance, clearance.measure(door, angle))
        visibility = None
        if env.capture is not None and capture_evidence:
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
                material_orientation_error=orientation_error,
                material_error=material_error,
                tangential_compensation_m=applied_compensation.tolist(),
                footprint_inside=footprint_inside,
                closed_gripper_error_m=closed_error,
                force=force,
                native_mean_normal_force_n=float(np.mean(sample_forces)),
                filtered_force_n=contact_load.force,
                normal_control_feedback=normal_feedback
                if phase in ("contact", "push", "hold")
                else None,
                contact_gap_m=gap,
                loaded=loaded,
                valid=valid,
                physical_safe=physical_safe,
                physical_violation=violation,
                phase_valid=physical_safe and (valid if phase in ("push", "hold") else True),
                separation_m=float(np.dot(material_p - p, material_r[:, 0])),
                joint_margin=margin,
                actual_joint_speed_peak_rad_s=actual_speed,
                minimum_jacobian_singular_value=sigma,
                joints=joints.tolist(),
                tool_position=p.tolist(),
                tool_quaternion=q.tolist(),
                visibility=visibility,
            )
        )
        if any(sample["contacts"] for sample in contacts):
            raw.append(dict(time=time, samples=contacts))
        if correction_active:
            # Preserve the role of the action already delivered, including a
            # command whose physical feedback first raises the failure latch.
            correction.observe(traces[-1])
        if recorder is not None:
            recorder.transition(env, goal_p, goal_r, phase, traces[-1])
        if env.capture is not None and capture_evidence and len(traces) % 30 == 0:
            sample = env.capture.sample
            camera = env.camera.data
            images.append(
                (
                    len(traces),
                    sample.rgb[0].cpu().numpy(),
                    sample.depth_m[0].cpu().numpy(),
                    dict(
                        position_m=tensor(camera.pos_w)[0].cpu().tolist(),
                        quaternion_ros_xyzw=tensor(camera.quat_w_ros)[0].cpu().tolist(),
                        intrinsics=tensor(camera.intrinsic_matrices)[0].cpu().tolist(),
                    ),
                )
            )
        return angle, pe, re, force, margin, loaded, valid

    pre_p, pre_r = door.contact_pose(
        0.0, setup.contact_fraction, setup.contact_height, -setup.precontact_m
    )
    if intermediate is None:
        slerp = Slerp([0, 1], Rotation.from_quat([start_q, Rotation.from_matrix(pre_r).as_quat()]))
        for tick in range(round(setup.approach_s / dt)):
            a = min(1.0, (tick + 1) * dt / (setup.approach_s * 0.7))
            blend = a * a * (3 - 2 * a)
            state = command(
                start_p + (pre_p - start_p) * blend, slerp(blend).as_matrix(), "approach"
            )
            if failure:
                break
        if not failure and (
            state[1] > setup.position_tolerance or state[2] > setup.orientation_tolerance
        ):
            failure = "approach_tracking"
        compression = None
        if not failure:
            for tick in range(round(setup.contact_s / dt)):
                offset = -setup.precontact_m + (setup.precontact_m + setup.compression_m) * min(
                    1.0, (tick + 1) * dt / (setup.contact_s * 0.7)
                )
                if compression is not None:
                    offset = support_normal(compression, "contact")
                angle = float(tensor(env.door.data.joint_pos)[0, 0])
                state = command(
                    *door.contact_pose(angle, setup.contact_fraction, setup.contact_height, offset),
                    "contact",
                )
                # Freeze the ramp at acquired load; the remaining contact interval
                # establishes light loading rather than injecting 3 mm of preload.
                if state[5] or compression is not None:
                    compression = offset
                traces[-1]["contact_compression_m"] = offset
                if failure:
                    break
            if not failure and not state[5]:
                failure = "no_loaded_contact"
        if not failure and family == "A":
            reason = "short_acquisition_endpoint"
        if not failure and family != "A":
            reference = state[0]
            reference_step = 0.0
            compression = setup.compression_m if compression is None else compression
            for tick in range(round(setup.horizon_s / dt)):
                angle = float(tensor(env.door.data.joint_pos)[0, 0])
                previous_reference = reference
                reference = push_reference(
                    reference,
                    angle,
                    setup,
                    dt,
                    door.mechanical_stop,
                )
                reference_step = reference - previous_reference
                compression = support_normal(compression, "push")
                advance = (
                    None
                    if correction is None
                    else correction.injected_reference_advance(reference_step)
                )
                state = command(
                    *door.contact_pose(
                        reference, setup.contact_fraction, setup.contact_height, compression
                    ),
                    "push",
                    injected_pose=None
                    if advance is None
                    else door.contact_pose(
                        reference + advance,
                        setup.contact_fraction,
                        setup.contact_height,
                        compression,
                    ),
                )
                traces[-1]["push_compression_m"] = compression
                if family is None and hasattr(recorder, "observations"):
                    from alexdoor_xas.qualification.intermediate import capture_phase_state

                    late_states.append(
                        capture_phase_state(
                            env,
                            phase="push",
                            compression=compression,
                            compensation=compensation,
                            retreat_p=pre_p,
                            retreat_r=pre_r,
                            initial_lead=hold_initial_lead(
                                reference, state[0], reference_step, setup.lead_angle
                            ),
                            reference_angle=reference,
                            source=dict(
                                asset_id=door.name,
                                episode_id=recorder.metadata["episode_id"],
                                split=recorder.metadata["split"],
                                setup=setup.to_dict(),
                            ),
                        )
                    )
                if (
                    family == "B"
                    and not failure
                    and evaluate_phase_endpoint(traces, family, setup)["passed"]
                    and (
                        correction is None
                        or (correction.injected_commands and not correction.pending)
                    )
                ):
                    reason = "short_controlled_opening_endpoint"
                    break
                angle, pe, re, force, margin, loaded, valid = state
                tracking.append((tick * dt, traces[-1]["material_error"]))
                angular_tracking.append((tick * dt, traces[-1]["material_orientation_error"]))
                predicted_error = tracking_reserve(tracking, setup.hold_settle_s)
                traces[-1]["predicted_hold_error_m"] = predicted_error
                traces[-1]["predicted_hold_orientation_rad"] = tracking_reserve(
                    angular_tracking, setup.hold_settle_s
                )
                elapsed = tick * dt
                conditioning.append((elapsed, traces[-1]["minimum_jacobian_singular_value"]))
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
                    desired = np.r_[
                        next_p - now_p, Rotation.from_matrix(next_r @ now_r.T).as_rotvec()
                    ]
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
                    if correction is not None and correction.permit_recontact(
                        traces[-1], missing_time
                    ):
                        # Do not cancel the injected error with an independent
                        # compression ramp. The reported corrective transition
                        # and ordinary contact feedback own physical recontact.
                        traces[-1]["bounded_recontact"] = True
                    else:
                        failure = "lost_contact"
                        break
                if abs(angle - door.mechanical_stop) < np.deg2rad(0.5):
                    reason = "mechanical_stop"
                    break
                if (
                    elapsed > setup.sustain_s
                    and window.maximum is not None
                    and conditioning_margin_exhausted(conditioning, setup)
                ):
                    reason, safety_detail = "safety_stop", "conditioning_margin"
                    break
                hold_blocked = False
                if elapsed > 5.0 and (
                    tick % max(1, round(setup.contact_force_window_s / dt)) == 0
                    or predicted_error > setup.material_drift_guard_m
                ):
                    # A push-only contact cannot brake an inertially coasting panel.
                    # Check future reach even while present tracking is still accurate.
                    endpoint = min(
                        reference + max(0.0, traces[-1]["speed"]) * setup.hold_settle_s,
                        door.mechanical_stop,
                    )
                    end_p, end_r = door.contact_pose(
                        endpoint, setup.contact_fraction, setup.contact_height, compression
                    )
                    _, end_pe, end_re = chain.solve(
                        chain.array((end_p - root_p) @ root_r)[None],
                        chain.array(root_r.T @ end_r)[None],
                        tensor(env.robot.data.joint_pos)[:, env.arm_ids],
                    )
                    traces[-1]["hold_endpoint_error_m"] = float(end_pe[0])
                    traces[-1]["hold_endpoint_orientation_rad"] = float(end_re[0])
                    hold_blocked = (
                        float(end_pe[0]) > setup.material_drift_guard_m
                        or float(end_re[0]) > 0.5 * setup.orientation_tolerance
                    )
                if (
                    elapsed > 5.0
                    and window.maximum is not None
                    and (
                        tracking_margin_exhausted(tracking, setup)
                        or hold_blocked
                        or tracking_margin_exhausted(angular_tracking, setup, orientation=True)
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
                    and compression >= setup.position_tolerance - setup.material_drift_guard_m
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
        if failure is None and family != "A":
            held = SustainedAngle(setup.sustain_s)
            initial_lead = hold_initial_lead(reference, state[0], reference_step, setup.lead_angle)
            missing_time = 0.0
            for tick in range(round(hold_budget_s / dt)):
                compression = support_normal(compression, "hold")
                angle = float(tensor(env.door.data.joint_pos)[0, 0])
                target = hold_reference(angle, initial_lead, door.mechanical_stop)
                state = command(
                    *door.contact_pose(
                        target, setup.contact_fraction, setup.contact_height, compression
                    ),
                    "hold",
                )
                traces[-1]["hold_compression_m"] = compression
                held.update(tick * dt, state[0], state[6])
                missing_time = 0.0 if state[5] else missing_time + dt
                if missing_time > 0.25:
                    failure = failure or "lost_hold_contact"
                    env.stop_motion()
                if failure:
                    break
                if task_hold is not None:
                    from alexdoor_xas.qualification.collection_phase import opening_hold_complete

                    if opening_hold_complete(
                        held.current,
                        (tick + 1) * dt,
                        setup.hold_settle_s,
                        np.deg2rad(task_hold["required_opening_deg"]),
                    ):
                        break
            held_angle = held.current if failure is None else None
            if held_angle is None:
                failure = failure or "invalid_hold"
    elif str(intermediate.get("phase", "hold")) == "push":
        # A genuine late-push source is physically restored under the same model.
        # The same measured exit guards choose hold; duration is only a budget.
        reason = "short_late_push_hold"
        compression = float(intermediate["compression"])
        compensation = np.array(intermediate["compensation"], copy=True)
        reference = float(intermediate["reference_angle"])
        reference_step = 0.0
        for tick in range(round(15.0 / dt)):
            angle = float(tensor(env.door.data.joint_pos)[0, 0])
            previous = reference
            settling_pace = correction is not None and correction.holding_achieved_angle()
            if not settling_pace:
                reference = push_reference(
                    reference,
                    angle,
                    setup,
                    dt,
                    door.mechanical_stop,
                )
            reference_step = reference - previous
            compression = support_normal(compression, "push")
            advance = (
                None
                if correction is None
                else correction.injected_reference_advance(reference_step)
            )
            state = command(
                *pressure_hold_pose(
                    door, setup, reference, angle if settling_pace else reference, compression
                ),
                "push",
                injected_pose=None
                if advance is None
                else door.contact_pose(
                    reference + advance, setup.contact_fraction, setup.contact_height, compression
                ),
            )
            traces[-1]["push_compression_m"] = compression
            if correction is not None and correction.needs_supported_hold_handoff(traces[-1]):
                reference, compression, compensation = achieved_contact_hold_state(
                    door, setup, traces[-1]
                )
                reference_step = 0.0
                correction.accept_supported_hold_handoff()
                traces[-1]["supported_hold_handoff"] = dict(
                    reference_angle=reference,
                    compression_m=compression,
                    tangential_compensation_m=compensation.tolist(),
                    scope="Achieved-contact force handoff; C2 settles before ordinary hold",
                )
                reason = "corrected_late_push_to_hold"
                if not correction.holding_achieved_angle():
                    break
            if correction is not None and correction.holding_achieved_angle():
                settling_pace = True
                traces[-1]["corrective_settling"] = dict(
                    achieved_angle=reference,
                    pending=correction.pending,
                    scope="Fixed achieved goal; original bounded correction recontact applies",
                )
                if not correction.pending:
                    reason = "corrected_late_push_to_hold"
                    break
            missing_time = 0.0 if state[5] else missing_time + dt
            if missing_time > 0.25:
                if correction is not None and correction.permit_recontact(traces[-1], missing_time):
                    traces[-1]["bounded_recontact"] = True
                else:
                    failure = failure or "lost_contact"
            if failure:
                break
            elapsed = tick * dt
            if family == "D" and campaign.get("opening_goal_deg") is not None:
                from alexdoor_xas.qualification.collection_phase import release_preparation_ready

                required_history_s = float(campaign["pre_release_history_s"])
                if release_preparation_ready(
                    traces,
                    setup,
                    opening_goal_rad=np.deg2rad(campaign["opening_goal_deg"]),
                    required_history_s=required_history_s,
                    dt=dt,
                ):
                    traces[-1]["release_preparation"] = dict(
                        opening_goal_deg=campaign["opening_goal_deg"],
                        fresh_motion_s=len(traces) * dt,
                        ordinary_hold_s=setup.hold_settle_s,
                        required_pre_release_history_s=required_history_s,
                    )
                    reason = "supported_task_opening_goal_before_release"
                    break
            tracking.append((elapsed, traces[-1]["material_error"]))
            angular_tracking.append((elapsed, traces[-1]["material_orientation_error"]))
            conditioning.append((elapsed, traces[-1]["minimum_jacobian_singular_value"]))
            if abs(state[0] - door.mechanical_stop) < np.deg2rad(0.5):
                reason = "mechanical_stop"
                break
            # The forward-opening forecast chooses when to stop advancing.
            # A captured corrective goal is already stationary; its original
            # native physical/IK guards and bounded recontact still apply.
            if window.maximum is not None and elapsed > setup.sustain_s and not settling_pace:
                extra_contact_s = (
                    0.0 if correction is None else correction.additional_contact_reserve_s()
                )
                traces[-1]["additional_contact_reserve_s"] = extra_contact_s
                if conditioning_margin_exhausted(
                    conditioning, setup, extra_contact_s=extra_contact_s
                ):
                    reason, safety_detail = "safety_stop", "conditioning_margin"
                    break
                endpoint_angle = min(
                    reference
                    + max(0.0, traces[-1]["speed"]) * (setup.hold_settle_s + extra_contact_s),
                    door.mechanical_stop,
                )
                end_p, end_r = door.contact_pose(
                    endpoint_angle,
                    setup.contact_fraction,
                    setup.contact_height,
                    compression,
                )
                _, end_pe, end_re = chain.solve(
                    chain.array((end_p - root_p) @ root_r)[None],
                    chain.array(root_r.T @ end_r)[None],
                    tensor(env.robot.data.joint_pos)[:, env.arm_ids],
                )
                if (
                    tracking_margin_exhausted(tracking, setup)
                    or tracking_margin_exhausted(angular_tracking, setup, orientation=True)
                    or float(end_pe[0]) > setup.material_drift_guard_m
                    or float(end_re[0]) > 0.5 * setup.orientation_tolerance
                ):
                    reason, safety_detail = "safety_stop", "tracking_margin"
                    break
            if state[3] >= setup.soft_force_n:
                reason, safety_detail = "safety_stop", "high_normal_force"
                break
        else:
            failure = "short_push_exit_unresolved"
        if not failure and family == "C" and len(traces) * dt < 4.0:
            failure = "insufficient_fresh_push_history"
        held = SustainedAngle(setup.sustain_s)
        initial_lead = hold_initial_lead(reference, state[0], reference_step, setup.lead_angle)
        missing_time = 0.0
        if not failure:
            for tick in range(round(hold_budget_s / dt)):
                compression = support_normal(compression, "hold")
                angle = float(tensor(env.door.data.joint_pos)[0, 0])
                # C2 corrects excess progression by holding the achieved angle,
                # rather than advancing its goal with the disturbed panel.
                # Normal force/position support and all physical guards remain active.
                hold_angle = (
                    reference
                    if correction is not None and correction.holding_achieved_angle()
                    else hold_reference(angle, initial_lead, door.mechanical_stop)
                )
                state = command(
                    *pressure_hold_pose(
                        door,
                        setup,
                        hold_angle,
                        angle
                        if correction is not None and correction.holding_achieved_angle()
                        else hold_angle,
                        compression,
                    ),
                    "hold",
                )
                held.update(tick * dt, state[0], state[6])
                missing_time = 0.0 if state[5] else missing_time + dt
                if missing_time > 0.25:
                    failure = failure or "lost_hold_contact"
                    env.stop_motion()
                if failure:
                    break
                if settle_c_hold:
                    from alexdoor_xas.qualification.collection_phase import correction_hold_complete

                    if (correction is None or not correction.pending) and correction_hold_complete(
                        traces, setup, dt
                    ):
                        break
            if settle_c_hold and not failure:
                from alexdoor_xas.qualification.collection_phase import correction_hold_complete

                if not correction_hold_complete(traces, setup, dt):
                    failure = "unsettled_hold_budget_exhausted"
        held_angle = held.current if not failure else None
        if held_angle is None:
            failure = failure or "invalid_hold"
    else:
        reason = "intermediate_hold_release"
        initial_lead = float(intermediate["initial_lead"])
        compression = float(intermediate["compression"])
        compensation = np.array(intermediate["compensation"], copy=True)
        held = SustainedAngle(setup.sustain_s)
        for tick in range(round(setup.intermediate_preroll_s / dt)):
            compression = support_normal(compression, "hold")
            angle = float(tensor(env.door.data.joint_pos)[0, 0])
            state = command(
                *door.contact_pose(
                    hold_reference(
                        angle, float(intermediate["initial_lead"]), door.mechanical_stop
                    ),
                    setup.contact_fraction,
                    setup.contact_height,
                    compression,
                ),
                "hold",
            )
            held.update(tick * dt, state[0], state[6])
            if failure:
                break
        held_angle = held.current if failure is None else None
        if held_angle is None:
            failure = failure or "intermediate_loaded_preroll_unqualified"
        (output / "intermediate-qualification.json").write_text(
            json.dumps(
                dict(
                    passed=held_angle is not None,
                    preroll_s=len(traces) * dt,
                    held_angle_rad=held_angle,
                    fresh_generation=env.generation,
                    contact_valid_samples=sum(t["valid"] for t in traces),
                    total_samples=len(traces),
                ),
                indent=2,
            )
            + "\n"
        )
    if family == "D" and campaign.get("opening_goal_deg") is not None:
        if held_angle is None or np.rad2deg(held_angle) < campaign["opening_goal_deg"]:
            failure = failure or "required_opening_not_reached_before_release"

    # Clear raised relief before rotating toward a pose that releases wrist limits.
    released = False
    if failure != "invalid_physics" and not env.motion_stopped:
        final_angle = float(tensor(env.door.data.joint_pos)[0, 0])
        from alexdoor_xas.qualification.collection_phase import preceding_retreat_row

        prior = preceding_retreat_row(traces, final_angle)
        if intermediate is not None:
            retreat_p = np.array(intermediate["retreat_p"])
            retreat_q = Rotation.from_matrix(intermediate["retreat_r"]).as_quat()
        elif prior is not None:
            retreat_p = np.array(prior["tool_position"])
            retreat_q = np.array(prior["tool_quaternion"])
        else:
            retreat_p, retreat_q = pre_p, Rotation.from_matrix(pre_r).as_quat()
        if held_angle is not None and intermediate is None and hasattr(recorder, "observations"):
            if late_states:
                late_state = late_states[0]
                late_state.update(
                    retreat_p=retreat_p, retreat_r=Rotation.from_quat(retreat_q).as_matrix()
                )
                np.savez_compressed(output / "late-push-state.npz", **late_state)
            from alexdoor_xas.qualification.intermediate import save_hold_state

            save_hold_state(
                output / "hold-state.npz",
                env,
                compression=compression,
                compensation=compensation,
                initial_lead=initial_lead,
                retreat_p=retreat_p,
                retreat_r=Rotation.from_quat(retreat_q).as_matrix(),
                source=dict(
                    asset_id=door.name,
                    episode_id=None if recorder is None else recorder.metadata["episode_id"],
                    split=None if recorder is None else recorder.metadata["split"],
                    setup=setup.to_dict(),
                ),
            )
        current_p, current_q = [v[0].cpu().numpy() for v in env.tool_pose()]
        release_slerp = Slerp([0, 1], Rotation.from_quat([current_q, retreat_q]))
        release_pause_started = False
        for tick in range(round(setup.release_s / dt)):
            a = min(1.0, (tick + 1) * dt / (setup.release_s * 0.7))
            goal_p, goal_r = current_p + (retreat_p - current_p) * a, release_slerp(a).as_matrix()
            if held_angle is not None or family == "A":
                goal_p, goal_r = release_reference(
                    current_p,
                    Rotation.from_quat(current_q).as_matrix(),
                    retreat_p,
                    Rotation.from_quat(retreat_q).as_matrix(),
                    (tick + 1) * dt,
                    setup,
                )
            injected_pose = None
            pause_handoff = None
            if correction is not None and failure is None and correction.holds_release_contact():
                if not release_pause_started:
                    # The declared error is continued support, not another push.
                    # Capture achieved offsets once and remove the old angular
                    # opening lead; physics, gains and velocities keep evolving.
                    _, compression, compensation = achieved_contact_hold_state(
                        door, setup, traces[-1]
                    )
                    initial_lead = 0.0
                    release_pause_started = True
                    pause_handoff = dict(
                        compression_m=compression,
                        tangential_compensation_m=compensation.tolist(),
                        opening_lead_rad=initial_lead,
                        scope="One achieved-contact handoff to excluded force-support pause",
                    )
                # The erroneous action is continued real contact, not a world
                # pose that the moving panel can leave without expert withdrawal.
                compression = normal_contact_support(
                    compression,
                    contact_load.force,
                    setup,
                    dt,
                    target_force=setup.hold_force_n,
                    gap_m=traces[-1]["contact_gap_m"],
                )
                angle = float(tensor(env.door.data.joint_pos)[0, 0])
                injected_p, injected_r = door.contact_pose(
                    hold_reference(angle, initial_lead, door.mechanical_stop),
                    setup.contact_fraction,
                    setup.contact_height,
                    compression,
                )
                injected_pose = (injected_p + injected_r @ compensation, injected_r)
            state = command(goal_p, goal_r, "release", injected_pose=injected_pose)
            if pause_handoff is not None:
                traces[-1]["release_contact_pause_handoff"] = pause_handoff
            if failure == "invalid_physics" or env.motion_stopped:
                break
        actual_p = env.tool_pose()[0][0].cpu().numpy()
        material_p, material_r = door.contact_pose(
            state[0], setup.contact_fraction, setup.contact_height
        )
        separation = float(np.dot(material_p - actual_p, material_r[:, 0]))
        released = not state[5] and separation >= 0.01 and state[1] <= setup.position_tolerance
    if not released:
        failure = failure or "unsafe_release"
    stabilized = False
    tail = []
    if released and not env.motion_stopped:
        for tick in range(round(setup.stabilization_max_s / dt)):
            state = command(goal_p, goal_r, "stabilize")
            tail.append(traces[-1])
            supported = tail[-round(setup.stabilization_window_s / dt) :]
            # Allow one ordinary reference step over the settle window, and
            # half the existing angular tie per window. No stillness/open-angle gate.
            joint_speed_limit = setup.max_joint_speed * dt / setup.stabilization_window_s
            door_speed_limit = np.deg2rad(setup.tie_deg) / setup.stabilization_window_s
            stabilized = bool(
                (tick + 1) * dt >= setup.stabilization_min_s
                and len(supported) * dt >= setup.stabilization_window_s - 1e-9
                and all(
                    t["physical_safe"] and not t["loaded"] and t["separation_m"] >= 0.01
                    for t in supported
                )
                and max(t["actual_joint_speed_peak_rad_s"] for t in supported) <= joint_speed_limit
                and max(abs(t["speed"]) for t in supported) <= door_speed_limit
                and np.linalg.norm(np.ptp([t["tool_position"] for t in supported], axis=0))
                <= setup.material_drift_guard_m
            )
            if not traces[-1]["physical_safe"] or env.motion_stopped or stabilized:
                break
        if not stabilized:
            failure = failure or "stabilization_budget_exhausted"
    measured = window.maximum
    result = dict(
        case=door.name,
        passed=failure is None and (window.maximum is not None or family == "A"),
        angle_deg=None if measured is None else float(np.rad2deg(measured)),
        hold_angle_deg=None if held_angle is None else float(np.rad2deg(held_angle)),
        hold_control_duration_s=sum(row["phase"] == "hold" for row in traces) * dt,
        hold_duration_budget_s=hold_budget_s,
        stop_reason=failure or reason,
        limit_evidence=limit_evidence,
        safety_detail=safety_detail,
        physical_violations=sorted(
            {t["physical_violation"] for t in traces if t["physical_violation"]}
        ),
        peak_force_n=peak,
        peak_actual_joint_speed_rad_s=peak_joint_speed,
        joint_margin=min_margin,
        clearance_m=min_clearance,
        clearance_scope="robot_door_AABB_lower_bound_excluding_distal_panel_pairs",
        released=released,
        stabilized=stabilized,
        stabilization_duration_s=len(tail) * dt,
        stabilization_budget_s=setup.stabilization_max_s,
        stabilization_joint_speed_limit_rad_s=setup.max_joint_speed
        * dt
        / setup.stabilization_window_s,
        stabilization_door_speed_limit_rad_s=float(
            np.deg2rad(setup.tie_deg) / setup.stabilization_window_s
        ),
        intermediate_start=intermediate is not None,
        scope="controlled_door_probe",
        force_components="normal_only",
        visibility=None
        if env.capture is None or not capture_evidence
        else dict(
            passed=all(t["visibility"]["passed"] for t in traces),
            failed_frames=sum(not t["visibility"]["passed"] for t in traces),
            checked_frames=len(traces),
            scope="geometric_panel_contact_surround_frame_observability",
        ),
    )
    if family is not None:
        from alexdoor_xas.qualification.collection_phase import qualify_phase_result

        result.update(
            qualify_phase_result(
                traces, family, setup, result, None if correction is None else correction.summary()
            )
        )
    write_probe_artifacts(output, result, traces, raw, images)
    return result
