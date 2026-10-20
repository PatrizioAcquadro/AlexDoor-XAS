"""Native teacher recording integrity and prospective phase-specific supervision."""

import json

import h5py
import numpy as np

from alexdoor_xas.recording.temporal import SCHEMA


def teacher_eligibility(file):
    """Annotations review targets; they never enter observation streams.

    A successful complete collection, or explicitly qualified named subtask,
    is required. Contact/load validity gates push/hold,
    while qualified acquisition, withdrawal and ordinary tail use phase validity.
    Interrupted/failed prefixes remain diagnostic, including their real actions.
    """
    commands, evaluator = file["commands"], file["evaluator"]
    n = len(commands.get("accepted", ()))
    result = np.zeros(n, bool)
    outcome = json.loads(file["metadata"].attrs.get("outcome", "{}"))
    metadata = json.loads(file["metadata"].attrs["episode"])
    subtask = metadata.get("subtask")
    if subtask is not None:
        # A named, physically qualified subtask is not a failed full-task attempt.
        # Merely setting a phase list or recording a prefix does not qualify it.
        if not subtask.get("name") or not subtask.get("required_phases"):
            raise ValueError("Subtask requires a name and declared phase sequence")
        qualified = (
            outcome.get("subtask_success") is True and outcome.get("subtask_qualified") is True
        )
    else:
        qualified = outcome.get("released") and outcome.get("stabilized")
    if not file.attrs.get("complete", False) or not outcome.get("passed") or not qualified:
        return result
    if len(evaluator.get("phase_valid", ())) != n:
        raise ValueError("Every expert outcome requires its own phase-quality annotation")
    result[:] = (
        commands["accepted"][:] & evaluator["physical_safe"][:] & evaluator["phase_valid"][:]
    )
    if "action_role" in evaluator:
        # Injected errors remain actual delivered-target observation history, but
        # cannot be expert targets. The window builder rejects crossing chunks.
        result &= np.isin(evaluator["action_role"][:], (0, 2))
    names = metadata.get("phase_names", [])
    if subtask is not None and "supervised_phases" in subtask:
        supervised = [names.index(name) for name in subtask["supervised_phases"]]
        result &= np.isin(evaluator["phase"][:], supervised)
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
        control_dt = float(file["metadata"].attrs["control_dt"])
        # Historical recordings predate explicit sim_dt metadata and used two
        # native samples per control interval. New recordings declare their rate.
        sim_dt = float(metadata.get("sim_dt", control_dt / 2))
        if not np.isfinite(sim_dt) or sim_dt <= 0:
            raise ValueError("Invalid recorded physics cadence")
        substeps = control_dt / sim_dt
        if not np.isfinite(substeps) or substeps < 1 or not np.isclose(substeps, round(substeps)):
            raise ValueError("Nonintegral recorded physics cadence")
        expected_feedback = 1 + round(substeps) * (count - 1)
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
            torque_abs_peak_nm=[
                float(abs(column[np.isfinite(column)]).max()) if np.isfinite(column).any() else None
                for column in tau.T
            ],
            native_torque_missing_samples=max(0, expected_feedback - len(tau)),
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
            native_feedback_coverage_complete=len(tau) == expected_feedback,
            source_group=metadata.get("family", metadata.get("asset_id")),
            physical_safe_commands=int(file["evaluator/physical_safe"][:].sum()),
            unsafe_commands=int((~file["evaluator/physical_safe"][:]).sum()),
            terminal_support=bool(file["metadata"].attrs.get("terminal_support", complete)),
            outcome=json.loads(file["metadata"].attrs.get("outcome", "{}")),
        )


def row_ranges(rows):
    """Inclusive ranges for an explicit row set, without huge report arrays."""
    rows = np.asarray(rows, dtype="i8")
    if not len(rows):
        return []
    return [
        [int(group[0]), int(group[-1])]
        for group in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
    ]


def review_native_collection(path):
    """Separate trusted window candidates from required full demonstration coverage.

    Reporting a candidate does not admit a new source. Preparation still requires
    an explicit command mask. Failed/interrupted sources get no automatic labels.
    Optional measured torque is masked using the same causal support rules as ACT.
    """
    from alexdoor_xas.dataset.temporal import raw_streams, training_decisions
    from alexdoor_xas.policies.temporal import (
        CausalObservationBuilder,
        TemporalConfig,
        select_support,
    )

    report = validate_native_recording(path)
    with h5py.File(path, "r") as file:
        eligible = teacher_eligibility(file)
        metadata = json.loads(file["metadata"].attrs["episode"])
        cfg = TemporalConfig(**metadata["observation_config"])
        streams, _, times, generations = raw_streams(file, cfg)
        builder = CausalObservationBuilder(cfg, streams, episode_id=metadata["episode_id"])
        commands = file["commands"]
        rows = (
            training_decisions(
                builder,
                times,
                generations,
                commands["state_acquired_s"][:],
                commands["generation"][:],
                eligible,
            )
            if report["integrity_passed"]
            else np.empty(0, "i8")
        )
        phase, names = file["evaluator/phase"][:], metadata["phase_names"]
        transitions = []
        for i in np.flatnonzero(phase[1:] != phase[:-1]) + 1:
            # Recording integrity guarantees one command per decision before terminal.
            crossings = rows[(rows < i) & (rows + 12 > i)]
            transitions.append(
                dict(
                    from_phase=names[int(phase[i - 1])],
                    to_phase=names[int(phase[i])],
                    command_row=int(i),
                    valid_crossing_windows=len(crossings),
                )
            )
        subtask = metadata.get("subtask")
        required = (
            subtask["required_phases"]
            if subtask is not None
            else ["hold", "release", "stabilize"]
            if metadata.get("intermediate_start")
            else ["inspect", "approach", "contact", "push", "hold", "release", "stabilize"]
        )
        if (
            not required
            or len(set(required)) != len(required)
            or (subtask is not None and any(p not in names for p in required))
        ):
            raise ValueError("Required phases must be a nonempty unique recorded phase sequence")
        coverage = []
        for name in required:
            commands_in_phase = np.flatnonzero(phase == names.index(name)) if name in names else []
            ready_rows = np.intersect1d(rows, commands_in_phase)
            coverage.append(
                dict(
                    phase=name,
                    commands=len(commands_in_phase),
                    usable_window_starts=len(ready_rows),
                )
            )
        missing = [
            dict(from_phase=a, to_phase=b)
            for a, b in zip(required[:-1], required[1:], strict=True)
            if not any(
                t["from_phase"] == a and t["to_phase"] == b and t["valid_crossing_windows"]
                for t in transitions
            )
        ]
        full = bool(len(rows) and all(c["usable_window_starts"] for c in coverage) and not missing)
        # A raw missing substep does not imply a missing selected token. Conversely,
        # one long gap may affect many overlapping histories. Count actual support.
        affected, unsupported_slots = [], 0
        tau = streams["tau"]
        for row in rows:
            t, gen = times[row], generations[row]
            slots = t - np.arange(cfg.state_count - 1, -1, -1) / cfg.state_hz
            _, valid, _, _ = select_support(tau, slots, t, gen, cfg.state_gap_s, shape=(7,))
            absent = (slots >= 0) & ~valid
            if absent.any():
                affected.append(row)
                unsupported_slots += int(absent.sum())
        return dict(
            report,
            data_usable=bool(len(rows)),
            required_scope_coverage=full,
            full_demonstration_coverage=bool(
                full and subtask is None and not metadata.get("intermediate_start")
            ),
            full_task_success=bool(
                subtask is None
                and not metadata.get("intermediate_start")
                and report["outcome"].get("passed")
                and report["outcome"].get("released")
                and report["outcome"].get("stabilized")
            ),
            subtask_name=None if subtask is None else subtask["name"],
            subtask_success=bool(report["outcome"].get("subtask_success", False)),
            subtask_qualified=bool(report["outcome"].get("subtask_qualified", False)),
            training_windows=len(rows),
            usable_window_ranges=row_ranges(rows),
            reviewed_command_ranges=row_ranges(np.flatnonzero(eligible)),
            supervision_admission="Explicit source/command-mask review required; candidates only",
            phase_coverage=coverage,
            transitions=transitions,
            missing_required_transitions=missing,
            torque_support=dict(
                feedback_contract=json.loads(file["metadata"].attrs["feedback_contract"]),
                windows_with_missing_past_or_current_tau=len(affected),
                affected_window_ranges=row_ranges(affected),
                missing_selected_slots=unsupported_slots,
                treatment="Masked optional measurements; no replacement or blanket rejection",
            ),
        )
