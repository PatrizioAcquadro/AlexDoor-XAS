"""Evaluator-only sensitivity analysis; registration and operational gates stay fixed."""

import numpy as np

from alexdoor_xas.perception.evaluation import quantiles

POSITION_BOUNDS_M = (0.010, 0.015, 0.020)
ROTATION_BOUND_DEG = 5.0


def gaps(rows, available):
    """Include leading gaps and censored missing tails, using original capture times."""
    intervals, start = [], None
    for i, supported in enumerate(available):
        if not supported and start is None:
            start = i
        if supported and start is not None:
            intervals.append(
                dict(
                    first_s=rows[start]["time_s"],
                    recovery_s=rows[i]["time_s"],
                    samples=i - start,
                    observed_s=rows[i]["time_s"] - rows[start]["time_s"],
                    censored=False,
                )
            )
            start = None
    if start is not None:
        intervals.append(
            dict(
                first_s=rows[start]["time_s"],
                last_s=rows[-1]["time_s"],
                samples=len(rows) - start,
                observed_s=rows[-1]["time_s"] - rows[start]["time_s"],
                censored=True,
            )
        )
    return dict(
        intervals=intervals,
        maximum_observed_s=max((g["observed_s"] for g in intervals), default=0),
        terminal=intervals[-1] if intervals and intervals[-1]["censored"] else None,
        recoveries=sum(not g["censored"] for g in intervals),
    )


def error_distribution(rows):
    values = np.asarray([r["capture_error"] for r in rows], dtype=float).reshape(-1, 2)
    if not len(values):
        return dict(
            position_m=quantiles(np.empty(0)),
            rotation_deg=quantiles(np.empty(0)),
            position_peak=None,
            rotation_peak=None,
        )
    peak = int(np.argmax(values[:, 0]))
    rotation_peak = int(np.argmax(values[:, 1]))
    return dict(
        position_m=quantiles(values[:, 0]),
        rotation_deg=quantiles(values[:, 1]),
        position_p99_m=float(np.quantile(values[:, 0], 0.99)),
        rotation_p99_deg=float(np.quantile(values[:, 1], 0.99)),
        position_peak=dict(time_s=rows[peak]["time_s"], error_m=float(values[peak, 0])),
        rotation_peak=dict(
            time_s=rows[rotation_peak]["time_s"], error_deg=float(values[rotation_peak, 1])
        ),
        position_histogram_mm=dict(
            edges=[0, 5, 10, 15, 20, 50, 100, 1000, "infinity"],
            counts=np.histogram(values[:, 0] * 1000, [0, 5, 10, 15, 20, 50, 100, 1000, np.inf])[
                0
            ].tolist(),
        ),
    )


def evaluate_bounds(rows):
    """Correct accepted/scheduled, correct/accepted and chronological correct-pose gaps.

    The seed is excluded from correctness and errors, but retained in the original
    scheduled denominator. Missing output is unavailable at every bound.
    """
    accepted = [r for r in rows if r.get("integration_accepted") and not r.get("is_seed")]
    finite = [r for r in rows if r.get("native_pose_valid") and not r.get("is_seed")]
    result = dict(
        scheduled=len(rows),
        accepted_nonseed=len(accepted),
        accepted_availability=len(accepted) / len(rows) if rows else 0,
        native_tracking_nonseed=sum(
            bool(r.get("native_tracking") and not r.get("is_seed")) for r in rows
        ),
        native_lost_nonseed=sum(bool(r.get("native_lost") and not r.get("is_seed")) for r in rows),
        rotation_bound_deg=ROTATION_BOUND_DEG,
        accepted_errors=error_distribution(accepted),
        all_finite_errors=error_distribution(finite),
        thresholds={},
    )
    for bound in POSITION_BOUNDS_M:
        flags = [
            bool(
                r.get("integration_accepted")
                and not r.get("is_seed")
                and r.get("capture_error") is not None
                and r["capture_error"][0] <= bound
                and r["capture_error"][1] <= ROTATION_BOUND_DEG
            )
            for r in rows
        ]
        correct = sum(flags)
        tail = (
            [i for i, r in enumerate(rows) if r["time_s"] >= rows[-1]["time_s"] - 5] if rows else []
        )
        result["thresholds"][str(round(bound * 1000))] = dict(
            correct=correct,
            wrong_accepted=len(accepted) - correct,
            availability=correct / len(rows) if rows else 0,
            accepted_precision=correct / len(accepted) if accepted else None,
            continuity=gaps(rows, flags),
            last_5s=dict(
                scheduled=len(tail),
                correct=sum(flags[i] for i in tail),
                wrong_accepted=sum(
                    rows[i].get("integration_accepted", False) and not flags[i] for i in tail
                ),
            ),
        )
    return result
