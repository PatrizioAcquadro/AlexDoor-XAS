#!/usr/bin/env python3
"""Bounded temporal input delivery and incremental inference; no optimizer updates.

Select eligible workloads from a prepared train/development episode. Report
unavailable workloads explicitly. Keep equivalence checks outside latency trials.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.assets.purdue import ARM_JOINTS  # noqa: E402
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats  # noqa: E402
from alexdoor_xas.dataset.sampling import collate_torch  # noqa: E402
from alexdoor_xas.dataset.temporal import TemporalEpisode, raw_streams  # noqa: E402
from alexdoor_xas.policies.act.model import act_loss  # noqa: E402
from alexdoor_xas.policies.act.temporal import (  # noqa: E402
    OBS_KEYS,
    TemporalActPolicy,
    normalize_temporal_batch,
    temporal_model,
)
from alexdoor_xas.policies.common.data import prefetch_batches  # noqa: E402
from alexdoor_xas.policies.common.data import to_device as _to_device  # noqa: E402
from alexdoor_xas.policies.temporal import (  # noqa: E402
    IncrementalObservationBuilder,
    TemporalConfig,
)


def summary(values):
    return dict(
        n=len(values),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        min_ms=min(values),
        max_ms=max(values),
        values_ms=values,
    )


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def select_workloads(episode):
    """Select real eligible rows without decoding images or crossing a reset/gap."""
    batch_size, trace_queries, replan_ticks = 32, 151, 6
    rows = episode.training_rows()
    patterns, unavailable = {}, {}
    if len(rows) >= batch_size:
        patterns["distinct"] = rows[np.linspace(0, len(rows) - 1, batch_size, dtype=int)].tolist()
    else:
        unavailable["distinct"] = f"Need {batch_size} distinct eligible rows; found {len(rows)}"
    cuts = (
        np.flatnonzero(
            (np.diff(rows) != 1)
            | (np.diff(episode.generations[rows]) != 0)
            | ~np.isclose(
                np.diff(episode.times[rows]), episode.config.control_dt_s, atol=1e-8, rtol=0
            )
        )
        + 1
    )
    run = max(np.split(rows, cuts), key=len)
    if len(run) >= batch_size:
        start = (len(run) - batch_size) // 2
        patterns["overlapping"] = run[start : start + batch_size].tolist()
    else:
        unavailable["overlapping"] = (
            f"Need {batch_size} consecutive eligible rows; found {len(run)}"
        )
    if "distinct" in patterns and "overlapping" in patterns:
        patterns["mixed"] = (
            patterns["distinct"][:8]
            + patterns["overlapping"][:20]
            + [patterns["overlapping"][0]] * 2
            + [patterns["distinct"][-1]] * 2
        )
    else:
        unavailable["mixed"] = "Distinct and overlapping workloads are required"
    trace_rows = run[::replan_ticks][:trace_queries]
    fill_queries = int(
        np.ceil(episode.config.history_s / (replan_ticks * episode.config.control_dt_s))
    )
    trace = None
    if len(trace_rows) > fill_queries + 1:
        trace = dict(
            rows=trace_rows.tolist(),
            times=episode.times[trace_rows].tolist(),
            generation=int(episode.generations[trace_rows[0]]),
        )
    else:
        unavailable["incremental"] = (
            "Insufficient consecutive history for cache fill and steady queries"
        )
    return patterns, trace, unavailable


def diagnostic_stats():
    # Declared fixed diagnostic transform; never fit unreviewed expert labels.
    def identity(width):
        return NormStats(np.zeros(width), np.ones(width), np.zeros(width), np.ones(width), 1)

    obs = identity(32)
    obs.mean[18:25], obs.std[18:25] = 0.2, 0.3
    return DatasetNormStats(identity(7), obs, OBS_KEYS, ("diagnostic_only",), "A1_joint_delta")


def check_budget():
    if torch.cuda.max_memory_allocated() > 18 * 1024**3:
        raise RuntimeError("18 GiB allocated VRAM budget exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > 8 * 1024:
        raise RuntimeError("8 GiB peak RSS budget exceeded")


def measure_batches(episode, patterns, model, stats, repeats):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    baseline = torch.cuda.memory_allocated()
    # Materialize the real two FP32 Adam moments without calling optimizer.step.
    # This measures resident state, not optimizer update time or update peak.
    for parameter in model.parameters():
        optimizer.state[parameter].update(
            step=torch.tensor(0.0),
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )
    resident = torch.cuda.memory_allocated() - baseline
    output = dict(
        optimizer=dict(
            parameters_mib=parameters / 1024**2,
            measured_resident_moments_mib=resident / 1024**2,
            estimated_foreach_temporary_mib=parameters / 1024**2,
            update_peak_measured=False,
            updates=0,
        )
    )

    def backward(batch):
        model.zero_grad(set_to_none=True)
        predicted, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])
        act_loss(predicted, batch["actions"], batch["is_pad"], mu, logvar, 1)["loss"].backward()

    # Same workload, first call separate. No optimizer update or checkpoint change.
    host = normalize_temporal_batch(episode.batch(next(iter(patterns.values()))), stats)
    batch = _to_device(collate_torch(host), "cuda")
    _, first = timed(lambda batch=batch: backward(batch))
    del host, batch
    output["first_forward_backward_ms"] = first
    torch.cuda.reset_peak_memory_stats()
    for name, rows in patterns.items():
        # Equivalence, including normalization, is deliberately untimed.
        batch = normalize_temporal_batch(episode.batch(rows), stats)
        for i, row in enumerate(rows):
            sample = episode.sample(row)
            reference = normalize_temporal_batch(
                dict(
                    obs={key: value[None] for key, value in sample["obs"].items()},
                    actions=sample["actions"][None],
                    is_pad=sample["is_pad"][None],
                ),
                stats,
            )
            for key in batch["obs"]:
                np.testing.assert_array_equal(batch["obs"][key][i], reference["obs"][key][0])
            np.testing.assert_array_equal(batch["actions"][i], reference["actions"][0])
            np.testing.assert_array_equal(batch["is_pad"][i], reference["is_pad"][0])
        del sample, reference, batch

        def batches(rows=rows):
            for _ in range(repeats):
                yield normalize_temporal_batch(episode.batch(rows), stats)

        delivery, transfer, gpu, cycle = [], [], [], []
        start = time.perf_counter()
        previous = start
        iterator = prefetch_batches(batches())
        try:
            for host in iterator:
                arrived = time.perf_counter()
                delivery.append((arrived - previous) * 1000)
                batch, elapsed = timed(lambda host=host: _to_device(collate_torch(host), "cuda"))
                transfer.append(elapsed)
                _, elapsed = timed(lambda batch=batch: backward(batch))
                gpu.append(elapsed)
                end = time.perf_counter()
                cycle.append((end - previous) * 1000)
                previous = end
                check_budget()
                del batch, host
        finally:
            iterator.close()
        wall = (time.perf_counter() - start) * 1000
        output[name] = dict(
            delivery_wait=summary(delivery),
            transfer=summary(transfer),
            forward_backward=summary(gpu),
            complete_cycle=summary(cycle),
            total_wall_ms=wall,
            amortized_ms=wall / repeats,
            first_cycle_ms=cycle[0],
            steady_cycle=summary(cycle[1:]),
            equivalence="exact, including order/duplicates/targets/masks/normalization",
        )
    output["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    output["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    output["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    del optimizer
    model.zero_grad(set_to_none=True)
    return output


def measure_incremental(source, config, model, stats, repeats, trace, *, legacy_synchronous):
    model.eval()
    policy = TemporalActPolicy(model, stats)
    fill_queries = int(np.ceil(config.history_s / (6 * config.control_dt_s)))
    generation = trace["generation"]
    trace = np.asarray(trace["times"])
    output = dict(trace_times_s=trace.tolist(), generation=generation, runs=[], equivalence=[])
    contract = dict(
        verified=True,
        joint_names=ARM_JOINTS,
        units="N m",
        sign_convention="positive_joint_coordinate",
        source="simulated_actuator_feedback",
        semantics="joint_output",
    )
    with h5py.File(source, "r", rdcc_nbytes=1024**2) as file:
        streams, _, _, _ = raw_streams(file, config, legacy_synchronous=legacy_synchronous)
        events = []
        for field, stream in streams.items():
            a = stream["acquired"]
            same_generation = stream["generation"] == generation
            keep = (
                same_generation
                & (a >= trace[0] - config.history_s - config.image_gap_s)
                & (a <= trace[-1])
            )
            if field == "images":
                static = np.flatnonzero(
                    same_generation
                    & (a <= config.static_time_s)
                    & (stream["available"] <= trace[0])
                )
                if not len(static):
                    raise ValueError("Raw source lacks the selected generation's static image")
                keep[static[-1]] = True
            if field == "target":
                carry = np.flatnonzero(
                    same_generation & (a < trace[0] - config.history_s - config.image_gap_s)
                )
                if len(carry):
                    keep[carry[-1]] = True
            events.extend(
                (float(stream["available"][i]), field, int(i)) for i in np.flatnonzero(keep)
            )
        events.sort()

        def replay(reuse, *, compare=False):
            online = IncrementalObservationBuilder(
                config, episode_id="saved-trace", generation=generation, feedback_contract=contract
            )
            policy.reset("saved-trace", generation)
            position, durations, errors, counts, failures = 0, [], [], [], []
            hits_before, misses_before = model.obs_proj.cache_hits, model.obs_proj.cache_misses

            def query(t):
                nonlocal position
                while position < len(events) and events[position][0] <= t:
                    _, field, index = events[position]
                    stream = streams[field]
                    value = (
                        stream["values"][[index]][0]
                        if field == "images"
                        else stream["values"][index]
                    )
                    online.append(
                        field,
                        acquired=stream["acquired"][index],
                        available=stream["available"][index],
                        generation=int(stream["generation"][index]),
                        source_index=int(stream["source_index"][index])
                        if field == "images"
                        else -1,
                        values=value,
                        valid=bool(stream["valid"][index]) if "valid" in stream else True,
                    )
                    position += 1
                snapshot = online.build(float(t))
                prediction = policy.predict(snapshot, now_s=float(t), reuse=reuse)
                return snapshot, prediction

            # Explicit startup: native history ingest/preprocess and first prediction.
            (_, prediction), fill = timed(lambda: query(trace[0]))
            for t in trace[1:]:
                if compare:
                    snapshot, reused = query(t)
                    full = policy.predict(snapshot, now_s=float(t), reuse=False)
                    difference = float(np.max(np.abs(full - reused)))
                    errors.append(difference)
                    mismatches = ~np.isclose(full, reused, atol=1e-6, rtol=1e-5)
                    if mismatches.any():
                        failures.append(
                            dict(
                                time_s=float(t),
                                elements=int(mismatches.sum()),
                                max_absolute_error=difference,
                            )
                        )
                else:
                    (_, prediction), ms = timed(lambda t=t: query(t))
                    durations.append(ms)
                    counts.append(
                        dict(
                            hits=model.obs_proj.cache_hits - hits_before,
                            misses=model.obs_proj.cache_misses - misses_before,
                        )
                    )
                check_budget()
            if compare:
                return dict(
                    passed=not failures,
                    failures=failures,
                    failing_queries=len(failures),
                    compared_elements=len(errors) * 12 * 7,
                    max_absolute_error=max(errors),
                    queries=len(errors),
                    comparison="untimed; separate from latency runs",
                    criterion="allclose atol=1e-6 rtol=1e-5 (existing CUDA regression tolerance)",
                )
            return dict(
                reuse=reuse,
                history_fill_first_prediction_ms=fill,
                cache_fill_after_first=summary(durations[:fill_queries]),
                steady=summary(durations[fill_queries:]),
                counts=counts,
                feature_cache_entries=len(model.obs_proj.cache),
            )

        # No correctness work in latency loops; same source/trace for every trial.
        output["equivalence"] = replay(True, compare=True)
        for trial in range(repeats):
            for reuse in (False, True) if trial % 2 == 0 else (True, False):
                result = replay(reuse)
                result["trial"] = trial
                output["runs"].append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prepared", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inference-only", action="store_true")
    mode.add_argument("--batches-only", action="store_true")
    mode.add_argument("--equivalence-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        source=str(args.source.resolve()),
        prepared=str(args.prepared.resolve()),
        optimizer_updates=0,
        physics_steps=0,
        final_test_opened=False,
        includes=(
            "batch join/read/copy/normalize, one-batch prefetch, H2D, forward/backward; "
            "inference native decode/preprocess/ingress/build/normalize/"
            "transfers/ACT/action return"
        ),
    )
    start = time.perf_counter()
    try:
        if not 3 <= args.repeats <= 12:
            raise ValueError("Bounded repeats must be between 3 and 12")
        with h5py.File(args.prepared, "r") as file:
            preprocessing = json.loads(file.attrs["preprocessing"])
            metadata = json.loads(file.attrs["episode"])
            legacy_synchronous = bool(file.attrs.get("legacy_synchronous", False))
            recorded_source = file.attrs.get("source")
        if recorded_source is not None and Path(recorded_source).resolve() != args.source.resolve():
            raise ValueError("Raw source differs from the prepared episode")
        config = TemporalConfig(**metadata.get("observation_config", preprocessing))
        result["config"] = config.to_dict()
        with TemporalEpisode(args.prepared, config) as episode:
            patterns, trace, unavailable = select_workloads(episode)
            result.update(patterns=patterns, trace=trace, unavailable=unavailable)
            batches_requested = not (args.inference_only or args.equivalence_only)
            inference_requested = not args.batches_only
            if not ((batches_requested and patterns) or (inference_requested and trace)):
                result["outcome"] = "unavailable"
                return
            if not torch.cuda.is_available():
                raise RuntimeError("Actual CUDA required")
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            model, stats = temporal_model(config).cuda(), diagnostic_stats()
            if batches_requested and patterns:
                result["batch"] = measure_batches(episode, patterns, model, stats, args.repeats)
        if inference_requested and trace:
            result["incremental"] = measure_incremental(
                args.source,
                config,
                model,
                stats,
                0 if args.equivalence_only else 3,
                trace,
                legacy_synchronous=legacy_synchronous,
            )
        result["outcome"] = "completed_with_unavailable_workloads" if unavailable else "completed"
    except BaseException as error:
        result.update(outcome="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - start
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_initialized() else None
        )
        result["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_initialized() else None
        )
        (args.output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
