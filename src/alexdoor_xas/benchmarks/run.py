"""Run one configured workload on one backend using the Kit Python runtime."""

import argparse
import importlib
import json
import os
import traceback
from pathlib import Path
from time import perf_counter, time


def load_factory(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def episode(fixture, cfg, case, seed, recording, synchronize):
    from collections import defaultdict

    buckets = defaultdict(float)

    def measure(label, operation, *args):
        start = perf_counter()
        value = operation(*args)
        synchronize()
        buckets[label] += perf_counter() - start
        return value

    physics_hz, control_hz, frame_hz = (
        cfg[key] for key in ("physics_hz", "control_hz", "frame_hz")
    )
    if physics_hz % control_hz or control_hz % frame_hz:
        raise ValueError("Use integer ratios between physics, control and frame frequencies")
    per_control = physics_hz // control_hz
    controls = round(case["duration_s"] * control_hz)
    rgbd = case.get("rgbd", False)
    native_tick = getattr(fixture, "control_tick", None)
    physics_dt = 1 / physics_hz
    synchronize()
    start_epoch = time()
    start = perf_counter()
    measure("reset", fixture.reset, seed)
    first_step = fixture.sim.get_physics_step_count()
    for index in range(controls):
        if native_tick is None:
            measure("control", fixture.control, index / control_hz)
            for _ in range(per_control):
                measure("control", fixture.dispatch)
                measure("physics", fixture.sim.step, False)
                measure("control", fixture.advance, physics_dt)
        else:
            native_tick(index / control_hz, per_control, physics_dt, measure)
        measure("evaluation_sampling", fixture.observe, (index + 1) / control_hz)
        if rgbd and (index + 1) % (control_hz // frame_hz) == 0:
            frame = measure("rendering_and_readback", fixture.capture)
            detection = measure("perception", fixture.perceive, frame)
            measure("recording", fixture.record, frame, detection, (index + 1) / control_hz)
    if rgbd:
        measure("recording", fixture.finish_recording, recording)
    result = measure("evaluation", fixture.evaluate)
    total = perf_counter() - start
    steps = fixture.sim.get_physics_step_count() - first_step
    if steps != controls * per_control:
        raise RuntimeError(f"Expected {controls * per_control} physics steps, observed {steps}")
    return dict(
        seed=seed,
        start_epoch_s=start_epoch,
        wall_s=total,
        breakdown_s=dict(buckets),
        physics_steps=steps,
        physics_ms_per_step=1000 * buckets["physics"] / steps if "physics" in buckets else None,
        steps_per_s=steps / total,
        environment_steps_per_s=case["envs"] * steps / total,
        unallocated_s=total - sum(buckets.values()),
        **result,
    )


def run(args, cfg, case, app_seconds):
    import torch
    from isaaclab import sim as sim_utils
    from isaaclab.sim import SimulationContext

    from isaac_physics_profile import steel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device to the benchmark")
    scene_start = perf_counter()
    roots = [f"/World/envs/env_{i}" for i in range(case["envs"])]
    physics = None
    if args.backend == "steel":
        site = Path(os.environ["NEWTON_SITE"]).resolve()
        physics = steel.config(
            site, roots, measure_contact=case.get("measure_contact", case.get("task") == "block")
        )
    sim_kwargs = {} if physics is None else {"physics": physics}
    simulation_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1 / cfg["physics_hz"],
        render_interval=cfg["physics_hz"] // cfg["frame_hz"],
        **sim_kwargs,
    )
    factory = load_factory(case.get("factory", cfg["factory"]))
    fixture = factory(simulation_cfg, cfg, case, args.backend, roots)
    fixture.start()
    sim = fixture.sim
    manager = sim.physics_manager
    if args.backend == "physx":
        scenes = [prim for prim in sim.stage.Traverse() if prim.GetTypeName() == "PhysicsScene"]
        if (
            manager.__name__ != "PhysxManager"
            or len(scenes) != 1
            or not scenes[0].GetAttribute("physxScene:enableGPUDynamics").Get()
        ):
            raise RuntimeError("Native PhysX GPU dynamics were not enabled")
        print("BENCHMARK_PHYSX_NATIVE_GPU", flush=True)
    if case.get("rgbd"):
        for _ in range(8):
            fixture.capture()
    scene_seconds = perf_counter() - scene_start
    warmups = []
    for index in range(cfg["warmups"]):
        path = args.output / f"warmup-{index + 1}.npz" if case.get("rgbd") else None
        warmups.append(
            episode(fixture, cfg, case, cfg["seed"] + index, path, torch.cuda.synchronize)
        )
    graph = getattr(manager, "_graph", None) is not None
    if args.backend == "steel" and not graph:
        raise RuntimeError("Steel requires a captured CUDA Graph after warm-up")
    if graph:
        print("BENCHMARK_STEEL_CUDA_GRAPH_ACTIVE", flush=True)
    windows = []
    for repeat in range(cfg["repeats"]):
        recording = args.output / f"window-{repeat + 1}.npz" if case.get("rgbd") else None
        result = episode(
            fixture, cfg, case, cfg["seed"] + repeat, recording, torch.cuda.synchronize
        )
        windows.append(result)
        print(
            "BENCHMARK_WINDOW",
            repeat + 1,
            round(result["wall_s"], 3),
            result.get("success"),
            flush=True,
        )
    receipt = dict(
        backend=args.backend,
        case=args.case,
        cuda_graph=graph,
        device=torch.cuda.get_device_name(),
        device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
        app_start_s=app_seconds,
        scene_start_s=scene_seconds,
        setup=cfg,
        warmup=warmups,
        windows=windows,
    )
    if args.backend == "steel":
        import newton
        import warp

        receipt.update(newton_version=newton.__version__, warp_version=warp.__version__)
        if manager._graph is None:
            raise RuntimeError("Steel CUDA Graph was lost during measurement")
    SimulationContext.clear_instance()
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("physx", "steel"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    case = cfg["cases"][args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError(args.output / "result.json")
    from isaaclab.app import AppLauncher

    t = perf_counter()
    app = AppLauncher(headless=True, enable_cameras=case.get("rgbd", False), device="cuda:0").app
    app_seconds = perf_counter() - t
    try:
        if args.backend == "steel":
            from isaac_physics_profile.steel import install_wheel

            install_wheel(os.environ["NEWTON_SITE"])
        run(args, cfg, case, app_seconds)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
