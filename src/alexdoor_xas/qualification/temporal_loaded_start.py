"""Qualified loaded-state initialization for explicitly separate ACT diagnostics."""

import json
from pathlib import Path

import numpy as np

from alexdoor_xas.recording.b1_runtime import array


def loaded_phase_startup(env, setup, state_path, *, asset_id, output):
    """Return a startup producer; source truth is used only before policy control.

    Reuse the maintained state reconstruction and independent start-stop checks.
    No old observation prefix is attached. The fresh pre-roll uses the same held
    measured joint pose as qualified short intermediate episodes.
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
        prepare_probe(
            env,
            env.cfg.prepared_door,
            setup,
            output,
            None,
            None,
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in state.items()
            },
            qualify_start_stop=True,
        )
        observer.reset()
        io.safety.reset()
        io.parked_tool = io.tool_pose()
        io.parked_joints = array(env.capture.sample.joint_position)[0, :7].copy()
        observation = observer.update(io.observe())
        load = ContactLoad(
            setup.contact_force_window_s,
            setup.loaded_force_n,
            setup.contact_gap_tolerance_m,
        )
        rows, sustained_since = [], None
        reason = ""
        for _ in range(round(setup.intermediate_preroll_s / io.dt)):
            if reason := io.safety_reason("inspect"):
                break
            io.hold_step()
            observation = observer.update(io.observe())
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
                )
            )
            if not physical_ok:
                reason = "unsafe_loaded_diagnostic_preroll"
                break
        if not reason:
            reason = io.safety_reason("inspect")
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
