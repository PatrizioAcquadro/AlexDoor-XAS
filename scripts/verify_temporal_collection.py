#!/usr/bin/env python
"""Bounded native collection audit and actual-data CUDA backward, without updates."""

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def verify(source, output):
    import h5py
    import numpy as np
    import torch

    from alexdoor_xas.dataset.sampling import collate_torch
    from alexdoor_xas.dataset.temporal import TemporalEpisode, prepare_temporal
    from alexdoor_xas.policies.act.model import act_loss
    from alexdoor_xas.policies.act.temporal import (
        TemporalActPolicy,
        fit_temporal_stats,
        normalize_temporal_batch,
        temporal_model,
    )
    from alexdoor_xas.policies.act.train import _to_device
    from alexdoor_xas.policies.temporal import CausalObservationBuilder, TemporalConfig
    from alexdoor_xas.recording.quality import teacher_eligibility, validate_native_recording

    started = perf_counter()
    report = validate_native_recording(source)
    assert report["integrity_passed"], report["errors"]
    with h5py.File(source, "r") as file:
        metadata = json.loads(file["metadata"].attrs["episode"])
        cfg = TemporalConfig(**metadata["observation_config"])
        eligible = teacher_eligibility(file)
        phase = file["evaluator/phase"][:]
        phase_names = metadata["phase_names"]
        feedback = file["physics_feedback"]
        traces = {}
        for name in ("contact", "hold", "release"):
            where = np.flatnonzero(phase == phase_names.index(name))
            if len(where):
                t = float(file["evaluator/time_s"][where[0]])
                use = np.flatnonzero(abs(feedback["acquired"][:] - t) <= 0.5)
                traces[name] = dict(
                    time_s=feedback["acquired"][:][use].tolist(),
                    torque_nm=feedback["torque_nm"][:][use].tolist(),
                )
        report["representative_torque_traces"] = traces
        diagnostic = not report["complete"] or not eligible.any()
        if not diagnostic:
            assert metadata["split"] == "train", "This check fits statistics on training data only"
    prepared = output / (source.parent.name + "-240x384.hdf5")
    prepare_temporal(source, prepared, cfg, eligible=eligible, device="cuda:0")
    if diagnostic:
        with TemporalEpisode(prepared, cfg) as episode:
            rows = episode.training_rows()
            assert not len(rows) and not episode.eligibility.any()
            assert len(episode.times) == report["states"]
            assert len(episode.action_times) == report["commands"]
            assert not episode.sample(len(episode.times) - 1)["supervision_eligible"]
            report["prefix_recovery"] = dict(
                prepared=str(prepared),
                actual_states=len(episode.times),
                actual_commands=len(episode.action_times),
                no_invented_terminal=True,
                source_complete=bool(episode.file.attrs["source_complete"]),
                source_terminal_support=bool(episode.file.attrs["source_terminal_support"]),
                training_windows=0,
            )
        report.update(
            data_usable=False,
            training_windows=0,
            validation_wall_s=perf_counter() - started,
            validation_process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        )
        return report
    with TemporalEpisode(prepared, cfg) as episode:
        rows = episode.training_rows(horizon=12)
        report["prepared"] = str(prepared)
        report["training_windows"] = len(rows)
        agreement = []
        for path in sorted(source.parent.glob("online-*.npz")):
            saved = np.load(path, allow_pickle=False)
            snap = episode.builder.build(float(saved["decision_time_s"]), int(saved["generation"]))
            assert snap["ready"] == bool(saved["ready"])
            for key, values in snap["obs"].items():
                np.testing.assert_allclose(
                    values, saved[key], rtol=0, atol=1e-6, err_msg=path.name + ":" + key
                )
            for key in ("image_source_index", "image_times", "robot_times"):
                np.testing.assert_array_equal(snap[key], saved[key], err_msg=path.name + ":" + key)
            agreement.append(path.name)
        report["offline_incremental_agreement"] = agreement
        transitions = []
        selected = []
        for i in np.flatnonzero(phase[1:] != phase[:-1]) + 1:
            crossing = rows[(rows < i) & (rows + 12 > i)]
            transitions.append(
                dict(
                    from_phase=phase_names[int(phase[i - 1])],
                    to_phase=phase_names[int(phase[i])],
                    command_row=int(i),
                    valid_crossing_windows=len(crossing),
                )
            )
            if len(crossing):
                selected.append(int(crossing[len(crossing) // 2]))
        report["transitions"] = transitions
        report["all_transitions_covered"] = all(t["valid_crossing_windows"] for t in transitions)
        report["data_usable"] = bool(report["all_transitions_covered"] and len(rows))
        assert len(rows), "No usable actual-data training windows"
        selected = list(dict.fromkeys([*selected, int(rows[-1])]))
        stats = fit_temporal_stats([episode])
        model = temporal_model(cfg).cuda()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        # One bounded actual-data batch contains transition and final-tail rows.
        torch.cuda.reset_peak_memory_stats()
        batch = _to_device(
            collate_torch(normalize_temporal_batch(episode.batch(selected), stats)), "cuda:0"
        )
        batch["obs"]["robot"].requires_grad_(True)
        model.train()
        torch.cuda.synchronize()
        begin = perf_counter()
        predicted, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])
        loss = act_loss(predicted, batch["actions"], batch["is_pad"], mu, logvar, 1)["loss"]
        loss.backward()
        torch.cuda.synchronize()
        elapsed = perf_counter() - begin
        assert torch.isfinite(loss) and torch.isfinite(predicted).all()
        for module in (model.obs_proj.images, model.obs_proj.temporal):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
        grad = batch["obs"]["robot"].grad
        assert (
            grad.isfinite().all()
            and grad[..., :18].abs().sum() > 0
            and grad[..., 18:25].abs().sum() > 0
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        report["cuda_backward"] = dict(
            batch_size=len(selected),
            rows=selected,
            wall_s=elapsed,
            finite=True,
            image_temporal_state_torque_gradients=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            optimizer_updates=0,
            weights_unchanged=True,
        )
        model.eval()
        policy = TemporalActPolicy(model, stats)
        gen = int(episode.generations[rows[-1]])
        policy.reset(episode.builder.episode_id, gen)
        snap = episode.sample(rows[-1])
        t = snap["decision_time_s"]
        tau = episode.builder.streams["tau"]
        # Alter a copied measured stream, never the recording or prepared store.
        altered_tau = dict(tau, values=np.asarray(tau["values"]).copy())
        idx = np.flatnonzero(
            (tau["generation"] == gen) & (tau["acquired"] <= t) & (tau["available"] <= t)
        )[-1]
        altered_tau["values"][idx] += 1
        changed = CausalObservationBuilder(
            cfg,
            dict(episode.builder.streams, tau=altered_tau),
            episode_id=episode.builder.episode_id,
        ).build(t, gen)
        np.testing.assert_allclose(
            changed["obs"]["robot"][-1, 18:25] - snap["obs"]["robot"][-1, 18:25], 1, atol=1e-5
        )
        a = normalize_temporal_batch(dict(obs=snap["obs"], actions=np.zeros((12, 7))), stats)[
            "obs"
        ]["robot"]
        b = normalize_temporal_batch(dict(obs=changed["obs"], actions=np.zeros((12, 7))), stats)[
            "obs"
        ]["robot"]
        assert np.any(a[:, 18:25] != b[:, 18:25])
        missing_tau = dict(
            acquired=np.empty(0),
            available=np.empty(0),
            generation=np.empty(0, "i8"),
            values=np.empty((0, 7)),
            valid=np.empty(0, bool),
        )
        missing = CausalObservationBuilder(
            cfg,
            dict(episode.builder.streams, tau=missing_tau),
            episode_id=episode.builder.episode_id,
        ).build(t, gen)
        assert not missing["obs"]["robot_valid"][:, 2].any()
        np.testing.assert_array_equal(missing["obs"]["robot"][:, 18:25], 0)
        reference = policy.predict(missing, reuse=False)
        missing["obs"]["robot"][:, 18:25] = 1e6
        np.testing.assert_array_equal(policy.predict(missing, reuse=False), reference)
        report["torque_tensor_checks"] = dict(
            valid_measured_change_reaches_normalized_act_input=True,
            missing_history_masked=True,
            invalid_values_do_not_affect_prediction=True,
        )
    report["validation_wall_s"] = perf_counter() - started
    report["validation_process_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required; no CPU fallback")
    reports = {}
    try:
        for source in args.recordings:
            print(f"Verifying {source}", flush=True)
            reports[source.parent.name] = verify(source, args.output)
            (args.output / "report.json").write_text(
                json.dumps(reports, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in reports[source.parent.name].items()
                        if k not in ("representative_torque_traces", "outcome")
                    }
                ),
                flush=True,
            )
    except BaseException as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error=repr(error), completed=list(reports)), indent=2) + "\n"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
