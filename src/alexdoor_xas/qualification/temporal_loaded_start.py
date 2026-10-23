"""Qualified loaded-state initialization for explicitly separate ACT diagnostics."""

import json
from pathlib import Path

import numpy as np

from alexdoor_xas.recording.b1_runtime import array


def _source_step(env, setup, state):
    """Advance the maintained material-following controller during initialization."""
    import torch
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.qualification.synthetic_probe import (
        hold_reference,
        normal_contact_support,
        push_reference,
        tangential_compensation,
    )

    door = env.cfg.prepared_door
    phase = str(state.get("phase", "hold"))
    forces = [
        sum(c["force_n"] for c in row["contacts"] if c["category"] in ("positive", "negative"))
        for row in env.contact_history
    ]
    gaps = [
        min(
            (
                c["separation_m"]
                for c in row["contacts"]
                if c["category"] in ("positive", "negative")
            ),
            default=None,
        )
        for row in env.contact_history
    ]
    gap = max(gaps) if all(g is not None for g in gaps) else None
    compression = normal_contact_support(
        float(state["compression"]),
        max(forces),
        setup,
        env.step_dt,
        target_force=setup.hold_force_n if phase == "hold" else setup.contact_load_guard_n,
        gap_m=gap,
    )
    angle = float(array(env.door.data.joint_pos)[0, 0])
    reference = (
        push_reference(
            float(state.get("reference_angle", 0)), angle, setup, env.step_dt, door.mechanical_stop
        )
        if phase == "push"
        else hold_reference(angle, float(state["initial_lead"]), door.mechanical_stop)
    )
    position, rotation = door.contact_pose(
        reference, setup.contact_fraction, setup.contact_height, compression
    )
    position = position + rotation @ np.asarray(state["compensation"])
    env.command_pose(position, Rotation.from_matrix(rotation).as_quat())
    env.step(torch.zeros((1, 6), device=env.device))
    force = max(
        sum(c["force_n"] for c in row["contacts"] if c["category"] in ("positive", "negative"))
        for row in env.contact_history
    )
    material, frame = door.contact_pose(
        float(array(env.door.data.joint_pos)[0, 0]), setup.contact_fraction, setup.contact_height
    )
    state.update(
        compression=compression,
        reference_angle=reference,
        compensation=tangential_compensation(
            np.asarray(state["compensation"]),
            (material - array(env.tool_pose()[0])[0]) @ frame,
            force >= setup.loaded_force_n,
            setup,
            env.step_dt,
        ),
    )
    return ObjectFrame(position, rotation)


def loaded_phase_startup(env, setup, state_path, *, asset_id, output):
    """Return a startup producer; source truth is used only before policy control.

    Reuse the maintained state reconstruction and independent start-stop checks.
    No old observation prefix is attached. Fresh pre-roll uses the maintained
    source controller and ends before any ACT manipulation command.
    """
    state_path, output = Path(state_path), Path(output)
    with np.load(state_path, allow_pickle=False) as source:
        state = {key: source[key].copy() for key in source.files}
    metadata = json.loads(str(state["metadata"]))
    report = json.loads((state_path.parent.parent / "report.json").read_text())
    if (
        metadata["asset_id"] != asset_id
        or metadata["split"] != "train"
        or report["case"] != asset_id
        or report["passed"] is not True
    ):
        raise ValueError("Loaded diagnostic requires a qualified matching training source")
    for key in (
        "floor_pose",
        "contact_fraction",
        "contact_height",
        "max_joint_speed",
        "force_limit_n",
        "position_tolerance",
        "orientation_tolerance",
    ):
        if not np.array_equal(metadata["setup"][key], setup.to_dict()[key]):
            raise ValueError("Loaded diagnostic source/setup mismatch: " + key)

    def startup(observer, io):
        from alexdoor_xas.qualification.synthetic_probe import ContactLoad, prepare_probe

        if metadata["episode_id"] not in observer.execution.policy.stats.train_episode_ids:
            raise ValueError("Loaded source is outside selected training data")
        if len(state["stiffness"]) != len(env._motion_impedance) or any(
            not np.array_equal(saved_k, array(stiffness))
            or not np.array_equal(saved_d, array(damping))
            for (_, _, stiffness, damping), saved_k, saved_d in zip(
                env._motion_impedance, state["stiffness"], state["damping"], strict=True
            )
        ):
            raise ValueError("Loaded source would change qualified normal-motion gains")
        current = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        prepare_probe(
            env,
            env.cfg.prepared_door,
            setup,
            output,
            None,
            None,
            current,
            qualify_start_stop=True,
        )
        observer.reset()
        io.safety.reset()
        io.parked_tool = io.tool_pose()
        io.parked_joints = array(env.targets)[0, env.arm_ids].copy()
        observation = observer.update(io.observe())
        load = ContactLoad(
            setup.contact_force_window_s,
            setup.loaded_force_n,
            setup.contact_gap_tolerance_m,
        )
        rows, sustained_since = [], None
        load_source = getattr(io, "load_source", None)
        load_reason = "missing_explicit_contact_selection" if load_source is not None else ""
        reason = ""
        for _ in range(round(setup.intermediate_preroll_s / io.dt)):
            if reason := io.safety_reason("inspect"):
                break
            from alexdoor_xas.policies.rollout import Command

            io.initializing_phase = True
            try:
                goal = _source_step(env, setup, current)
                io.step_recorder(Command("pose", goal, "hold"), goal)
            finally:
                io.initializing_phase = False
            observation = observer.update(io.observe())
            local = getattr(getattr(observation, "estimate", None), "local", None)
            if load_source is not None:
                load_reason = "missing_explicit_contact_selection"
                if local is not None and local.selection is not None:
                    # Acquisition is still initializing. Preserve unavailable samples
                    # through the existing warmup; require current support at handoff.
                    try:
                        measured = load_source(io, io.robot_feedback())
                        io.safety.checks.contact_monitor.contact_load(measured, observation.time_s)
                    except ValueError as error:
                        load_reason = str(error)
                    else:
                        load_reason = ""
            force = [
                sum(
                    c["force_n"] for c in r["contacts"] if c["category"] in ("positive", "negative")
                )
                for r in env.contact_history
            ]
            gaps = [
                min(
                    (
                        c["separation_m"]
                        for c in r["contacts"]
                        if c["category"] in ("positive", "negative")
                    ),
                    default=None,
                )
                for r in env.contact_history
            ]
            gap = max(gaps) if all(g is not None for g in gaps) else None
            loaded = load.update(observation.time_s, float(np.mean(force)), gap)
            physical_ok = (
                not any(r["forbidden"] for r in env.contact_history)
                and max(force) <= setup.force_limit_n
                and max(float(np.max(abs(s.dq))) for s in env.load_samples)
                <= setup.max_joint_speed * 1.02
            )
            sustained_since = (
                (observation.time_s if sustained_since is None else sustained_since)
                if loaded and physical_ok
                else None
            )
            rows.append(
                dict(
                    time_s=observation.time_s,
                    loaded=loaded,
                    force_n=force,
                    gap_m=gap,
                    physical_ok=physical_ok,
                    operational_load_ready=not load_reason,
                    operational_load_reason=load_reason,
                )
            )
            if not physical_ok:
                reason = "unsafe_loaded_diagnostic_preroll"
                break
        if not reason:
            reason = io.safety_reason("inspect")
        if not reason and load_reason:
            reason = "loaded_diagnostic_observed_load_unavailable: " + load_reason
        if not reason and (
            sustained_since is None or observation.time_s - sustained_since < setup.sustain_s - 1e-9
        ):
            reason = "loaded_diagnostic_preroll_not_sustained"
        (output / "fresh-loaded-preroll.json").write_text(
            json.dumps(
                dict(
                    source=str(state_path),
                    source_episode=metadata["episode_id"],
                    scope="initialization_only_not_learned_task_completion",
                    old_history_imported=False,
                    preroll_controller="qualified_source_initialization_only",
                    rows=rows,
                    passed=not reason,
                    reason=reason,
                ),
                indent=2,
            )
            + "\n"
        )
        return observation, reason

    return startup
