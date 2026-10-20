"""Physical short-episode endpoints, evaluated from measured rows, outside policy inputs."""

import numpy as np


def sustained_rows(rows, duration, predicate):
    """Return a real contiguous support window, or no evidence."""
    window = []
    for row in rows:
        if not predicate(row):
            window = []
            continue
        if window and row["time"] - window[-1]["time"] > 1 / 60 + 1e-8:
            window = []
        window.append(row)
        while len(window) > 1 and row["time"] - window[1]["time"] >= duration - 1e-9:
            window.pop(0)
        if row["time"] - window[0]["time"] >= duration - 1e-9:
            return window
    return []


def evaluate_phase_endpoint(rows, family, setup):
    """Phase labels select the interval; actual contact/motion must prove its endpoint.

    Completion of withdrawal, ordinary stabilization and independent safety stop
    are separate required gates applied by the recording owner.
    """
    if family not in ("A", "B", "C", "D"):
        raise ValueError("Unknown collection family")

    def supported(row):
        return bool(
            row["physical_safe"]
            and row["loaded"]
            and row["footprint_inside"]
            and row["material_error"] <= setup.position_tolerance
            and row["material_orientation_error"] <= setup.orientation_tolerance
        )

    acquisition = [r for r in rows if r["phase"] == "contact"]
    acquired = sustained_rows(acquisition, setup.sustain_s, supported)
    evidence = []
    if family == "A":
        evidence = acquired
    elif family == "B" and acquired:
        origin = acquired[0]["angle"]
        evidence = sustained_rows(
            [r for r in rows if r["phase"] == "push"],
            setup.sustain_s,
            lambda r: supported(r) and r["angle"] >= origin + np.deg2rad(1),
        )
    elif family == "C":
        push = [r for r in rows if r["phase"] == "push"]
        hold = [r for r in rows if r["phase"] == "hold"]
        moving = bool(push and max(r["angle"] for r in push) - push[0]["angle"] >= np.deg2rad(0.5))
        if moving:
            evidence = sustained_rows(
                hold,
                setup.sustain_s,
                lambda r: (
                    supported(r)
                    and abs(r["speed"]) <= np.deg2rad(setup.tie_deg) / setup.stabilization_window_s
                ),
            )
            if (
                evidence
                and np.linalg.norm(np.ptp([r["tool_position"] for r in evidence], axis=0))
                > setup.material_drift_guard_m
            ):
                evidence = []
    elif family == "D":
        had_contact = any(supported(r) for r in rows)
        if had_contact:
            evidence = sustained_rows(
                [r for r in rows if r["phase"] == "stabilize"],
                setup.stabilization_window_s,
                lambda r: r["physical_safe"] and not r["loaded"] and r["separation_m"] >= 0.01,
            )
    return dict(
        passed=bool(evidence),
        family=family,
        clock="probe_relative_completed_control_row_time",
        support_start_s=None if not evidence else evidence[0]["time"],
        support_end_s=None if not evidence else evidence[-1]["time"],
        supported_samples=len(evidence),
        acquisition_supported_samples=len(acquired),
        minimum_supported_angle_deg=None
        if not evidence
        else float(np.rad2deg(min(r["angle"] for r in evidence))),
        scope="measured_subtask_endpoint_requires_separate_release_tail_and_stop",
    )


def correction_hold_complete(rows, setup, dt):
    """Finish a real hold only after its minimum and current settling window."""
    hold = [row for row in rows if row["phase"] == "hold"]
    if len(hold) * dt < setup.hold_settle_s - 1e-9:
        return False
    cutoff = hold[-1]["time"] - setup.sustain_s - 1e-9
    # Keep the genuine push evidence, but an earlier passing hold window cannot
    # excuse later residual motion, load loss, or unsafe feedback.
    current = [row for row in rows if row["phase"] == "push"]
    current += [row for row in hold if row["time"] >= cutoff]
    return evaluate_phase_endpoint(current, "C", setup)["passed"]


def qualify_phase_result(rows, family, setup, probe_result, correction=None):
    """Apply the declared subtask contract without granting complete-task success.

    B ends at sustained initial opening. Its excluded cleanup still executes all
    hold physics/guards, withdrawal and stabilization; a missing final hold window
    is diagnostic rather than an additional B endpoint. Other failures stay failed.
    Independent stopping remains a separate gate owned by the recording caller.
    """
    endpoint = evaluate_phase_endpoint(rows, family, setup)
    if family == "D":
        before_release = [row for row in rows if row["phase"] in ("push", "hold")]
        endpoint["fresh_pre_release_history_s"] = len(before_release) / 60
        endpoint["passed"] &= len(before_release) / 60 >= 4.0
    corrected = correction is None or (
        correction["injected_commands"] > 0
        and correction["corrective_commands"] > 0
        and not correction["pending_correction"]
    )
    cleanup_only = bool(
        family == "B"
        and endpoint["passed"]
        and corrected
        and probe_result["stop_reason"] == "invalid_hold"
        and not probe_result["physical_violations"]
        and rows
        and all(row["physical_safe"] for row in rows)
        and probe_result["released"]
        and probe_result["stabilized"]
    )
    passed = bool(
        (probe_result["passed"] or cleanup_only)
        and endpoint["passed"]
        and probe_result["released"]
        and probe_result["stabilized"]
        and corrected
    )
    result = dict(
        passed=passed,
        subtask_success=passed,
        subtask_qualified=passed,
        full_task_success=False,
        phase_endpoint=endpoint,
        correction=correction,
        scope="short_subtask_not_complete_task",
    )
    if cleanup_only:
        result.update(
            stop_reason="short_controlled_opening_endpoint",
            cleanup_hold_review=dict(
                original_probe_passed=probe_result["passed"],
                original_stop_reason=probe_result["stop_reason"],
                final_hold_window_qualified=False,
                hold_rows_preserved=sum(row["phase"] == "hold" for row in rows),
                scope="B endpoint passed; final cleanup hold is not the B task endpoint",
            ),
        )
    return result
