#!/usr/bin/env python3
"""Bounded saved-input temporal ACT costs; no optimizer, physics or selection.

Run with the project Python and CUDA. Source must be train/development, never
final test. Each resolution is a fresh process for meaningful peak RSS/VRAM.
Outputs are exclusive to preserve unsuccessful attempts.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from alexdoor_xas.dataset.sampling import collate_torch
from alexdoor_xas.dataset.temporal import (
    TemporalEpisode,
    prepare_temporal,
    preprocess_rgbd,
    raw_streams,
)
from alexdoor_xas.policies.act.model import act_loss
from alexdoor_xas.policies.act.temporal import temporal_inference_tensors, temporal_model
from alexdoor_xas.policies.common.data import to_device as _to_device
from alexdoor_xas.policies.temporal import (
    IncrementalObservationBuilder,
    TemporalConfig,
    causal_indices,
)

CASES = [0, 1500, 1512, 1976, 2040, 4500, 8189, 8369, 8729]
STEADY = [4500 + 6 * i for i in range(12)]


def rss_mib():
    with open("/proc/self/status") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024


def summary(values):
    return dict(
        samples=len(values),
        min_ms=min(values),
        p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        max_ms=max(values),
        values_ms=values,
    )


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def support_keys(snapshot):
    return [
        (snapshot["episode_id"], snapshot["generation"], int(i))
        for i in snapshot["image_source_index"]
    ]


def verify_incremental(ep):
    streams = ep.builder.streams
    from alexdoor_xas.assets.purdue import ARM_JOINTS

    online = IncrementalObservationBuilder(
        ep.config,
        episode_id=ep.builder.episode_id,
        generation=int(ep.generations[0]),
        feedback_contract=dict(
            verified=True,
            joint_names=ARM_JOINTS,
            units="N m",
            sign_convention="positive_joint_coordinate",
            source="simulated_actuator_feedback",
            semantics="joint_output",
        ),
    )
    events = []
    for key, stream in streams.items():
        for i, available in enumerate(stream["available"]):
            events.append((available, key, i))
    events.sort()
    position, result = 0, []
    for row in CASES:
        t = ep.times[row]
        while position < len(events) and events[position][0] <= t:
            _, key, i = events[position]
            stream = streams[key]
            value = stream["values"][[i]][0] if key == "images" else stream["values"][i]
            online.append(
                key,
                acquired=stream["acquired"][i],
                available=stream["available"][i],
                generation=int(stream["generation"][i]),
                values=value,
                source_index=int(stream["source_index"][i]) if key == "images" else -1,
                valid=bool(stream["valid"][i]) if "valid" in stream else True,
            )
            position += 1
        a, b = ep.sample(row), online.build(t)
        for name in a["obs"]:
            np.testing.assert_array_equal(
                a["obs"][name], b["obs"][name], err_msg=f"row={row}, {name}"
            )
        for name in ("image_source_index", "image_times", "robot_times"):
            np.testing.assert_array_equal(a[name], b[name], err_msg=f"row={row}, {name}")
        assert a["ready"] == b["ready"]
        result.append(
            dict(
                row=row,
                time_s=float(t),
                ready=a["ready"],
                images=int(a["obs"]["image_valid"].sum()),
                targets_padding=int(a["is_pad"].sum()),
            )
        )
    return result


def measure_inference(source, prepared, output, cfg, *, with_backward=False):
    """Complete host-input path after a measured change; preserve original runs."""
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True, exist_ok=False)
    ep = TemporalEpisode(prepared, cfg)
    result = dict(
        prepared=str(prepared),
        config=cfg.to_dict(),
        optimizer_updates=0,
        includes=(
            "saved native read/decode + GPU preprocessing/D2H + join/load + "
            "required H2D + CNN/cache + temporal ACT + action D2H"
        ),
        samples=12,
        attempts=[],
    )
    begin = time.perf_counter()
    try:
        model, elapsed = timed(lambda: temporal_model(cfg).cuda().eval())
        result["model_startup_ms"] = elapsed
        with h5py.File(source, "r", rdcc_nbytes=1024**2) as raw, torch.inference_mode():

            def encode(row):
                return preprocess_rgbd(
                    *(raw[f"observations/{key}"][row] for key in ("rgb", "depth_m", "valid_depth")),
                    cfg,
                )

            first = ep.sample(STEADY[0])
            unique = np.unique(first["image_source_index"])
            _, fill_ms = timed(lambda: [encode(int(i)) for i in unique if i >= 0])
            result["native_history_fill_ms"] = fill_ms
            # Startup is measured independently; steady loops retain each first
            # cache fill too, with all values reported rather than discarded.
            tensor = temporal_inference_tensors(
                {k: v[None] for k, v in first["obs"].items()}, "cuda", reuse=False
            )
            _, cold = timed(lambda: model.predict(tensor))
            result["first_forward_ms"] = cold
            for reuse in (False, True):
                model.obs_proj.clear_cache()
                total, cpu, preproc, transfer, net, errors = [], [], [], [], [], []
                for row in STEADY:
                    start = time.perf_counter()
                    snap, load_ms = timed(lambda row=row: ep.sample(row))
                    frame, pp_ms = timed(
                        lambda snap=snap: encode(int(snap["image_source_index"][-1]))
                    )
                    np.testing.assert_array_equal(frame, snap["obs"]["images"][-1])
                    host = {k: v[None] for k, v in snap["obs"].items()}
                    tensor, tr_ms = timed(
                        lambda host=host, reuse=reuse: temporal_inference_tensors(
                            host, "cuda", reuse=reuse
                        )
                    )
                    if reuse:
                        tensor["cache_keys"] = support_keys(snap)
                    prediction, net_ms = timed(
                        lambda tensor=tensor: model.predict(tensor)[0].cpu().numpy()
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    total.append(elapsed)
                    cpu.append(load_ms)
                    preproc.append(pp_ms)
                    transfer.append(tr_ms)
                    net.append(net_ms)
                    if reuse:
                        full_tensor = temporal_inference_tensors(host, "cuda", reuse=False)
                        full = model.predict(full_tensor)[0].cpu().numpy()
                        errors.append(float(np.max(np.abs(full - prediction))))
                        np.testing.assert_allclose(prediction, full, atol=2e-6, rtol=2e-5)
                result["reuse" if reuse else "full"] = dict(
                    total=summary(total),
                    join_load=summary(cpu),
                    native_preprocessing=summary(preproc),
                    transfer=summary(transfer),
                    network_cache_and_action_d2h=summary(net),
                    max_equivalence_error=max(errors) if errors else None,
                    cache_entries=len(model.obs_proj.cache),
                    cache_capacity=model.obs_proj.cache_capacity,
                )
        if with_backward:
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            probe = _to_device(collate_torch(ep.batch([4500], diagnostics=True)), "cuda")
            torch.cuda.reset_peak_memory_stats()

            def backward(batch):
                model.zero_grad(set_to_none=True)
                prediction, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])
                act_loss(prediction, batch["actions"], batch["is_pad"], mu, logvar, 1)[
                    "loss"
                ].backward()

            _, probe_ms = timed(lambda: backward(probe))
            peak = torch.cuda.max_memory_allocated() / 1024**2
            parameters = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            estimate = max(0, peak - 3 * parameters) * 32 + 3 * parameters
            result["batch1_probe"] = dict(
                ms=probe_ms, allocated_mib=peak, batch32_estimate_mib=estimate
            )
            free, _ = torch.cuda.mem_get_info()
            if estimate > min(18 * 1024, free / 1024**2 * 0.85):
                raise RuntimeError("Measured probe exceeds batch32 budget")
            model.zero_grad(set_to_none=True)
            del probe
            rows = ([1976, 2040, 4500, 8189, 8369] * 7)[:32]
            host, loading = timed(lambda: ep.batch(rows, diagnostics=True))
            if rss_mib() > 8 * 1024:
                raise RuntimeError("RSS budget exceeded before batch32")
            batch, transfer = timed(lambda: _to_device(collate_torch(host), "cuda"))
            result["loading_batch32_ms"], result["batch32_transfer_ms"] = loading, transfer
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(3):
                _, elapsed = timed(lambda: backward(batch))
                durations.append(elapsed)
                if torch.cuda.max_memory_allocated() > 18 * 1024**3:
                    raise RuntimeError("VRAM budget exceeded")
            result["forward_backward_batch32"] = summary(durations)
            result["visual_gradient"] = any(
                p.grad is not None and bool(p.grad.abs().sum() > 0)
                for p in model.obs_proj.images.parameters()
            )
            result["temporal_gradient"] = any(
                p.grad is not None and bool(p.grad.abs().sum() > 0)
                for p in model.obs_proj.temporal.parameters()
            )
            result["current_state_gradient"] = bool(
                model.obs_proj.readout.weight.grad[:, -18:].abs().sum() > 0
            )
        result["attempts"].append(dict(outcome="completed"))
    except BaseException as error:
        result["attempts"].append(dict(outcome="failed", error=repr(error)))
        raise
    finally:
        result["elapsed_s"] = time.perf_counter() - begin
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        (output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
        ep.close()


def write_detail_sheet(source, stores, destination):
    """Compare identical acquired frames to native RGB; diagnostic crop only."""
    cases = (1976, 2040, 8189, 8369)
    labels = ("contact acquisition", "contact to push", "push to hold", "hold to release")
    crop = (550, 220, 820, 420)
    tile = (540, 400)
    canvas = Image.new("RGB", (tile[0] * 4, (tile[1] + 32) * 4 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Native acquired RGB", "60 x 96", "120 x 192", "240 x 384")):
        draw.text((column * tile[0] + 8, 8), label, fill="black")
    metrics = []
    with h5py.File(source, "r") as raw:
        for row, (case, label) in enumerate(zip(cases, labels, strict=True)):
            snapshots = [ep.sample(case) for ep in stores]
            sources = [int(sample["image_source_index"][-1]) for sample in snapshots]
            assert len(set(sources)) == 1
            native = raw["observations/rgb"][sources[0]]
            baseline = Image.fromarray(native)
            images = [baseline]
            for sample in snapshots:
                values = sample["obs"]["images"][-1, :3].transpose(1, 2, 0)
                images.append(
                    Image.fromarray((values * 255).round().astype("u1")).resize(
                        baseline.size, Image.Resampling.NEAREST
                    )
                )
            for column, image in enumerate(images):
                y = 32 + row * (tile[1] + 32)
                draw.text(
                    (column * tile[0] + 8, y), f"{label}; source row {sources[0]}", fill="black"
                )
                canvas.paste(
                    image.crop(crop).resize(tile, Image.Resampling.NEAREST),
                    (column * tile[0], y + 24),
                )
                if column:
                    difference = np.abs(
                        np.asarray(image.crop(crop), dtype="f4")
                        - np.asarray(baseline.crop(crop), dtype="f4")
                    )
                    metrics.append(
                        dict(
                            case=case,
                            acquired_source_row=sources[0],
                            size=stores[column - 1].config.image_size,
                            rgb_crop_mae_u8=float(difference.mean()),
                        )
                    )
    canvas.save(destination)
    destination.with_suffix(".json").write_text(
        json.dumps(
            dict(
                diagnostic_only=True,
                crop_native_pixels=crop,
                metrics=metrics,
                interpretation=(
                    "Reconstruction detail; no contact detection, learning "
                    "or task performance measurement"
                ),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=[60, 120, 240], required=True)
    parser.add_argument(
        "--prepared", type=Path, help="Reuse frames; measure inference only unless --with-backward"
    )
    parser.add_argument(
        "--with-backward",
        action="store_true",
        help="Include three batch32 backward passes; no optimizer",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = dict(
        source=str(args.source.resolve()),
        prepared=str(args.prepared),
        budget=dict(vram_mib=18 * 1024, rss_mib=8 * 1024, wall_s=900),
        optimizer_updates=0,
        physics_steps=0,
        attempts=[],
    )
    started = time.perf_counter()
    try:
        assert torch.cuda.is_available(), "Actual compatible CUDA GPU required"
        torch.set_num_threads(4)
        with h5py.File(args.source, "r", rdcc_nbytes=1024**2) as source:
            metadata = json.loads(source["metadata"].attrs["episode"])
            if metadata.get("split") not in ("train", "development"):
                raise ValueError("Final test remains sealed")
            interval = json.loads(source["metadata"].attrs["calibration"])["depth_interval_m"]
            cfg = TemporalConfig(
                image_size=(args.height, args.height * 8 // 5), depth_interval_m=interval
            )
            selected = set()
            if args.prepared is None:
                streams, _, times, generations = raw_streams(source, cfg, legacy_synchronous=True)
                for row in sorted(set(CASES + STEADY)):
                    t = times[row]
                    slots = np.r_[min(t, cfg.static_time_s), t - np.array(cfg.image_offsets_s)]
                    ids = causal_indices(
                        streams["images"], slots, t, generations[row], cfg.image_gap_s
                    )
                    selected.update(streams["images"]["source_index"][ids[ids >= 0]].tolist())
        store = args.prepared
        if store is None:
            store = args.output / "frames.hdf5"
            _, preparation_ms = timed(
                lambda: prepare_temporal(
                    args.source,
                    store,
                    cfg,
                    image_rows=np.array(sorted(selected), "i8"),
                    legacy_synchronous=True,
                )
            )
            run["preparation"] = dict(
                wall_ms=preparation_ms,
                frames=len(selected),
                bytes=store.stat().st_size,
                includes="native read/decode, GPU pooling/transfers, frame-store write",
            )
        with TemporalEpisode(store, cfg) as ep:
            run["offline_incremental"] = verify_incremental(ep)
            run["saved_frame_indices"] = ep.builder.streams["images"]["source_index"].tolist()
        if time.perf_counter() - started > 900:
            raise RuntimeError("Measurement wall budget exhausted before model checks")
        measure_inference(
            args.source,
            store,
            args.output / "costs",
            cfg,
            with_backward=args.with_backward or args.prepared is None,
        )
        run["attempts"].append(dict(outcome="completed"))
    except BaseException as error:
        run["attempts"].append(dict(outcome="failed", error=repr(error)))
        raise
    finally:
        run["elapsed_s"] = time.perf_counter() - started
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
