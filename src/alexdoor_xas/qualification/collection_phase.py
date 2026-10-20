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
