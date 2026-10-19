"""Native teacher recording integrity and prospective phase-specific supervision."""

import json

import h5py
import numpy as np

from alexdoor_xas.recording.temporal import SCHEMA


def teacher_eligibility(file):
    """Annotations review targets; they never enter observation streams.

    A successful collection is required. Contact/load validity gates push/hold,
    while qualified acquisition, withdrawal and ordinary tail use phase validity.
    Interrupted/failed prefixes remain diagnostic, including their real actions.
    """
    commands, evaluator = file["commands"], file["evaluator"]
    n = len(commands.get("accepted", ()))
    result = np.zeros(n, bool)
    outcome = json.loads(file["metadata"].attrs.get("outcome", "{}"))
    if (
        not file.attrs.get("complete", False)
        or not outcome.get("passed")
        or not outcome.get("released")
        or not outcome.get("stabilized")
    ):
        return result
    if len(evaluator.get("phase_valid", ())) != n:
        raise ValueError("Every expert outcome requires its own phase-quality annotation")
    result[:] = (
        commands["accepted"][:] & evaluator["physical_safe"][:] & evaluator["phase_valid"][:]
    )
    metadata = json.loads(file["metadata"].attrs["episode"])
    names = metadata.get("phase_names", [])
    if "contact" in names and "phase" in evaluator and "contact_valid" in evaluator:
        contact = np.flatnonzero(evaluator["phase"][:] == names.index("contact"))
        valid = evaluator["contact_valid"][:][contact]
        # Unloaded acquisition is valid before contact is established. Once
        # acquired, weak/ambiguous contact rows remain diagnostic until valid again.
        acquired = np.maximum.accumulate(valid)
        result[contact] &= ~acquired | valid
    cfg = metadata.get("observation_config")
    if cfg is not None:
        result &= commands["state_acquired_s"][:] >= cfg["static_time_s"] + cfg["warmup_s"] - 1e-9
    return result


def validate_native_recording(path):
    """Read only; incomplete sequences are recoverable evidence, never completed experts."""
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema") != SCHEMA:
            raise ValueError("Expected native temporal recording")
        metadata = json.loads(file["metadata"].attrs["episode"])
        if metadata.get("split") not in ("train", "development"):
            raise ValueError("Final test remains sealed")
        state, command, feedback, captures = (
            file[k] for k in ("control", "commands", "physics_feedback", "captures")
        )
        n, count = len(command.get("accepted", ())), len(state.get("acquired", ()))
        errors = []

        def require(condition, reason):
            if not condition:
                errors.append(reason)

        for group in (state, feedback, captures):
            if "acquired" not in group:
                continue
            a, v, g = (group[k][:] for k in ("acquired", "available", "generation"))
            require(
                bool(np.isfinite(a).all() and np.isfinite(v).all() and np.all(v >= a)),
                "invalid_stream_clocks",
            )
            for gen in np.unique(g):
                require(bool(np.all(np.diff(a[g == gen]) > 0)), "regressed_acquisition")
        complete = bool(file.attrs["complete"])
        require(count == n + 1 if complete else count in (n, n + 1), "state_command_alignment")
        if n:
            times = command["state_acquired_s"][:]
            require(bool(np.array_equal(times, state["acquired"][:n])), "command_prestate_clock")
            require(
                bool(
                    np.array_equal(
                        command["q_before"][:], state["joint_position"][:n, :7], equal_nan=True
                    )
                ),
                "command_q_before",
            )
            accepted = command["accepted"][:]
            dispatch, applied = command["dispatch_s"][:], command["applied_s"][:]
            require(
                bool(
                    np.all(np.isfinite(dispatch[accepted]))
                    and np.all(applied[accepted] >= dispatch[accepted])
                ),
                "unacknowledged_accepted_command",
            )
            require(
                bool(
                    np.isnan(dispatch[~accepted]).all()
                    and np.isnan(applied[~accepted]).all()
                    and np.isnan(command["applied_target"][:][~accepted]).all()
                ),
                "fabricated_refusal_ack",
            )
            require(
                bool(
                    np.array_equal(
                        command["clipped"][:][accepted],
                        np.any(
                            command["requested_target"][:][accepted]
                            != command["applied_target"][:][accepted],
                            axis=1,
                        ),
                    )
                ),
                "clipping_diagnostic",
            )
        if count > 1:
            require(
                bool(
                    np.allclose(
                        np.diff(state["acquired"][:]),
                        file["metadata"].attrs["control_dt"],
                        atol=1e-9,
                        rtol=0,
                    )
                ),
                "control_cadence",
            )
        if len(captures.get("acquired", ())) > 1:
            require(bool(np.all(np.diff(captures["acquired"][:]) <= 0.1 + 1e-9)), "rgbd_below_10hz")
        if "frame" in captures:
            frames, generations = captures["frame"][:], captures["generation"][:]
            for gen in np.unique(generations):
                known = frames[generations == gen]
                if np.all(known >= 0):
                    require(bool(np.all(np.diff(known) > 0)), "repeated_native_capture_frame")
        expected_feedback = 1 + 2 * (count - 1)
        require(
            len(feedback.get("acquired", ())) == expected_feedback,
            "missing_native_substep_feedback",
        )
        tau = feedback["torque_nm"][:]
        healthy = feedback["healthy"][:] & np.isfinite(tau).all(axis=1)
        limits = np.asarray(json.loads(file["metadata"].attrs["arm_effort_limits_nm"]))
        eligible = teacher_eligibility(file)
        return dict(
            integrity_passed=not errors,
            errors=errors,
            complete=complete,
            states=count,
            commands=n,
            missing_terminal_states=max(0, n + 1 - count),
            captures=len(captures.get("acquired", ())),
            native_torque_samples=len(tau),
            expected_native_torque_samples=expected_feedback,
            torque_healthy_samples=int(healthy.sum()),
            torque_missing_or_unhealthy_samples=int((~healthy).sum()),
            torque_finite_samples_per_joint=np.isfinite(tau).sum(axis=0).tolist(),
            torque_abs_peak_nm=np.nanmax(abs(tau), axis=0).tolist(),
            torque_near_configured_effort_limit_samples=(abs(tau) >= 0.99 * limits)
            .sum(axis=0)
            .tolist(),
            torque_saturation_status=(
                "No saturation flag exposed; limit comparison is diagnostic "
                "for projected solver output, not hardware saturation certification"
            ),
            eligible_commands=int(eligible.sum()),
            diagnostic_commands=int(n - eligible.sum()),
            duration_s=float(state["acquired"][-1]),
            source_group=metadata.get("family", metadata["asset_id"]),
            outcome=json.loads(file["metadata"].attrs.get("outcome", "{}")),
        )


def review_native_collection(path):
    """Collection gate reuses the exact prepared sampler readiness/window rules."""
    from alexdoor_xas.dataset.temporal import raw_streams, training_decisions
    from alexdoor_xas.policies.temporal import CausalObservationBuilder, TemporalConfig

    report = validate_native_recording(path)
    with h5py.File(path, "r") as file:
        eligible = teacher_eligibility(file)
        if not report["integrity_passed"] or not report["complete"] or not eligible.any():
            return dict(report, data_usable=False, training_windows=0, transitions=[])
        metadata = json.loads(file["metadata"].attrs["episode"])
        cfg = TemporalConfig(**metadata["observation_config"])
        streams, _, times, generations = raw_streams(file, cfg)
        builder = CausalObservationBuilder(cfg, streams, episode_id=metadata["episode_id"])
        commands = file["commands"]
        rows = training_decisions(
            builder,
            times,
            generations,
            commands["state_acquired_s"][:],
            commands["generation"][:],
            eligible,
        )
        phase, names = file["evaluator/phase"][:], metadata["phase_names"]
        transitions = []
        for i in np.flatnonzero(phase[1:] != phase[:-1]) + 1:
            crossings = rows[(rows < i) & (rows + 12 > i)]
            transitions.append(
                dict(
                    from_phase=names[int(phase[i - 1])],
                    to_phase=names[int(phase[i])],
                    command_row=int(i),
                    valid_crossing_windows=len(crossings),
                )
            )
        usable = bool(
            len(rows)
            and all(t["valid_crossing_windows"] for t in transitions)
            and report["torque_missing_or_unhealthy_samples"] == 0
        )
        return dict(report, data_usable=usable, training_windows=len(rows), transitions=transitions)
