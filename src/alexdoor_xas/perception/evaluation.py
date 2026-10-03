"""Reusable scientific error metrics and evaluator-only contact correspondence."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.frames import ObjectFrame, rot_z

LIMITS = dict(
    hinge_origin_m=0.01,
    hinge_rotation_deg=5.0,
    dimensions_m=0.01,
    contact_local_m=0.01,
    contact_rotation_local_deg=5.0,
    contact_position_m=0.01,
    contact_rotation_deg=5.0,
    panel_rotation_deg=5.0,
    signed_angle_deg=5.0,
)
COMPONENTS = (*LIMITS, "width_m", "height_m", "thickness_m")


@dataclass(frozen=True)
class MaterialContactReference:
    """Evaluator-only association at selection; independent surface query remains 6.0E work."""

    selection_id: str
    selected_s: float
    leaf_local_pose: ObjectFrame
    footprint_leaf: tuple[np.ndarray, np.ndarray]

    def require_selection(self, selection):
        if (selection.selection_id, selection.selected_s) != (self.selection_id, self.selected_s):
            raise ValueError("contact_correspondence_requires_new_surface_query")


def rotation_error(a, b):
    if a is None or np.asarray(a).shape != (3, 3) or not np.isfinite(a).all():
        return np.nan
    return float(np.rad2deg(Rotation.from_matrix(np.asarray(a) @ b.T).magnitude()))


def position_error(a, b):
    if a is None or np.asarray(a).shape != np.asarray(b).shape or not np.isfinite(a).all():
        return np.nan
    return float(np.linalg.norm(np.asarray(a) - b))


def state_errors(estimate, truth):
    errors = dict.fromkeys(COMPONENTS, np.nan)
    frame = estimate.frame
    if frame is not None:
        errors["hinge_origin_m"] = position_error(frame.origin, truth["hinge_origin"])
        errors["hinge_rotation_deg"] = rotation_error(frame.rot, truth["hinge_rotation"])
    if estimate.signed_angle is not None:
        delta = estimate.signed_angle - truth["signed_angle"]
        errors["signed_angle_deg"] = float(
            abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))))
        )
    true_panel = truth["hinge_rotation"] @ rot_z(truth["signed_angle"])
    errors["panel_rotation_deg"] = rotation_error(estimate.panel_rotation, true_panel)
    if estimate.dimensions is not None:
        delta = abs(estimate.dimensions - truth["dimensions"])
        if np.isfinite(delta).all():
            errors["dimensions_m"] = float(delta.max())
        errors.update(zip(("width_m", "height_m", "thickness_m"), delta, strict=True))
    errors["contact_position_m"] = position_error(
        estimate.contact_position, truth["contact_position"]
    )
    errors["contact_rotation_deg"] = rotation_error(
        estimate.contact_rotation, truth["contact_rotation"]
    )
    if frame is not None and estimate.panel_rotation is not None:
        if estimate.contact_position is not None:
            local = estimate.panel_rotation.T @ (estimate.contact_position - frame.origin)
            errors["contact_local_m"] = position_error(local, truth["contact_local"])
        if estimate.contact_rotation is not None:
            local_r = estimate.panel_rotation.T @ estimate.contact_rotation
            errors["contact_rotation_local_deg"] = rotation_error(
                local_r, truth["contact_rotation_local"]
            )
    return np.array([errors[k] for k in COMPONENTS], dtype=float)


def quantiles(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, p50=None, p95=None, maximum=None)
    return dict(
        n=len(values),
        p50=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(values.max()),
    )


def score(errors, accepted, reasons, selection, causes=None):
    indices = np.flatnonzero(selection)
    errors, accepted = errors[indices], accepted[indices]
    counts = Counter(str(reasons[i]) for i in indices)
    limits = np.array(list(LIMITS.values()))
    accurate = np.isfinite(errors[:, : len(limits)]).all(1) & (
        errors[:, : len(limits)] <= limits
    ).all(1)
    coverage = float(accepted.mean()) if len(indices) else 0.0
    precision = float(accurate[accepted].mean()) if accepted.any() else 0.0
    all_errors = {k: quantiles(errors[:, i]) for i, k in enumerate(COMPONENTS)}
    accepted_errors = {k: quantiles(errors[accepted, i]) for i, k in enumerate(COMPONENTS)}
    passed = (
        coverage >= 0.95
        and precision >= 0.95
        and all(
            accepted_errors[k]["p95"] is not None and accepted_errors[k]["p95"] <= limit
            for k, limit in LIMITS.items()
        )
    )
    return dict(
        observations=len(indices),
        accepted=int(accepted.sum()),
        valid_coverage=coverage,
        accepted_state_precision=precision,
        accurate_coverage=float(accurate.mean()) if len(indices) else 0,
        offline_passed=passed,
        reasons=dict(counts),
        rejection_causes=dict(
            Counter(
                cause
                for i, keep in zip(indices, accepted, strict=True)
                if not keep
                for cause in (
                    str(causes[i]).split("|") if causes is not None else [str(reasons[i])]
                )
                if cause
            )
        ),
        errors_all_finite=all_errors,
        errors_rejected={k: quantiles(errors[~accepted, i]) for i, k in enumerate(COMPONENTS)},
        errors_accepted=accepted_errors,
    )


def json_safe(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
