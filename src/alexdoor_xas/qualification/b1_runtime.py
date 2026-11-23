"""Launch the maintained B1 collection and qualification runtime."""

import json
import os
from contextlib import contextmanager, nullcontext

import numpy as np

from .door_geometry import PreparedDoor
from .expert import configure_physics_substeps, probe_config
from .synthetic_probe import ProbeSetup

NATIVE_PROFILE = "newton_steel_warp"


def native_selected():
    return os.environ.get("ISAAC_PHYSICS_PROFILE") == NATIVE_PROFILE


def launch_native_steel(cfg):
    """Configure the Purdue scene using the pinned Isaac Lab solver and renderer."""
    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import NewtonCfg, SteelSolverCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    cfg.physics_profile = None
    cfg.native_steel = True
    cfg.urdf_cache_dir = cfg.urdf_cache_dir or os.environ["ALEXDOOR_USD_CACHE"]
    cfg.sim.physics = NewtonCfg(
        solver_cfg=SteelSolverCfg(
            iterations=16, enable_joint_limits=True, articulation_constraint_max=512
        ),
        use_cuda_graph=True,
    )
    return launch_simulation(
        (cfg, NewtonWarpRendererCfg()),
        {"headless": True, "device": cfg.sim.device, "visualizer": "none"},
    )


def load_probe(folder, record, setup_path, diagnostic_path, *, seed=None):
    recipe = json.loads((folder / "recipe.json").read_text())
    setup_values = json.loads(setup_path.read_text())
    stop, pitch, substeps = None, 0.0, None
    if diagnostic_path is not None:
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic["asset_id"] != folder.name:
            raise ValueError("Diagnostic setup belongs to a different door")
        stop = float(diagnostic["hinge_stop_deg"])
        if not 0 < stop < min(45, record["mechanical_limit_deg"]):
            raise ValueError("A bounded diagnostic requires a hinge stop below 45 degrees")
        record = {**record, "mechanical_limit_deg": stop}
        setup_values.update(diagnostic["probe"])
        pitch = float(diagnostic["camera_mount_pitch_rad"])
        substeps = diagnostic["physics_substeps"]
        seed = diagnostic["seed"]
    return (
        PreparedDoor(folder, record, recipe),
        ProbeSetup(**setup_values),
        dict(stop=stop, pitch=pitch, substeps=substeps, seed=seed),
    )


@contextmanager
def launch_b1(door, setup, args, *, stop=None, pitch=0.0, substeps=None, seed=None):
    """Create the same Purdue environment with the selected launcher and renderer."""
    native = native_selected()
    if native:
        cfg = probe_config(door, setup, args.device)
        runtime = launch_native_steel(cfg)
    else:
        from isaaclab.app import AppLauncher

        options = (
            vars(args)
            if hasattr(args, "headless")
            else {
                "headless": True,
                "device": args.device,
                "width": 1280,
                "height": 720,
            }
        )
        # Kit has to start before the legacy environment configuration is imported.
        app = AppLauncher({**options, "enable_cameras": True}).app
        cfg = probe_config(door, setup, args.device)
        runtime = nullcontext(app)

    cfg.diagnostic_hinge_stop_deg = stop
    cfg.camera_mount_pitch_rad = pitch
    if substeps is not None:
        configure_physics_substeps(cfg, substeps)
    if seed is not None:
        cfg.seed = seed

    with runtime:
        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv

        env = DoorPushPurdueEnv(cfg)
        try:
            if native:
                print(json.dumps(native_state(env)), flush=True)
            yield env
        finally:
            env.close()


def native_state(env, *, stepped=False):
    import torch
    from isaaclab_newton.physics import SteelSolverCfg

    backend = env.sim.backend_manager.backend
    renderer = env.camera._renderer
    if not isinstance(env.cfg.sim.physics.solver_cfg, SteelSolverCfg):
        raise RuntimeError("SteelSolverCfg is required")
    state = dict(
        device=torch.cuda.get_device_name(env.device),
        solver=type(backend.solver_ops).__name__,
        renderer=type(renderer).__name__,
        graphs=sorted(str(key) for key in backend.graphs),
        physics_steps=env.sim.get_physics_step_count(),
        contact_capacity_checks=env._native_contacts.snapshot.capacity_checks,
    )

    if stepped and (
        not state["graphs"] or state["contact_capacity_checks"] < state["physics_steps"]
    ):
        raise RuntimeError("Steel CUDA Graph or per-step capacity check missing")
    return state


def light_parameters(condition, seed):
    if condition == "nominal":
        return dict(intensity_scale=1.0, color=[1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed)
    return dict(
        intensity_scale=float(rng.uniform(0.8, 1.2)),
        color=rng.uniform(0.9, 1.0, 3).tolist(),
    )


def configure_native_light(env, condition, seed):
    """Vary the direction of the tiled renderer's scene light."""
    direction = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3)
    lighting = dict(model="newton_warp_directional", direction=direction.tolist())
    if condition == "nominal":
        return lighting

    import warp as wp

    # The tiled camera accepts a direction for its white directional light.
    # Rotate it in azimuth while keeping its elevation and ambient light fixed.
    rng = np.random.default_rng(seed)
    azimuth_delta = float(rng.choice([-1, 1]) * rng.uniform(25, 45))
    angle = np.deg2rad(azimuth_delta)
    cosine, sine = np.cos(angle), np.sin(angle)
    direction[:2] = [[cosine, -sine], [sine, cosine]] @ direction[:2]
    renderer = env.camera._renderer
    renderer.newton_sensor.utils.create_default_light(
        direction=wp.vec3f(*direction), enable_shadows=renderer.cfg.enable_shadows
    )
    return dict(
        model=lighting["model"], direction=direction.tolist(), azimuth_delta_deg=azimuth_delta
    )
