"""Select Steel in Isaac Lab beta2 and expose solved contact forces."""

import json
import os
import pickle
import subprocess
import sys
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from importlib.util import LazyLoader
from pathlib import Path
from tempfile import TemporaryDirectory

_IMPORT_COMMAND = ("/usr/bin/python3", str(Path(__file__).with_name("steel_import.py")))
_OPTIONAL_NEWTON_MODULES = {"newton.controllers", "newton.ik", "newton.viewer"}


@contextmanager
def prepare_import_worker(site):
    """Load Newton and OpenUSD while Kit constructs the scene."""
    with subprocess.Popen(
        [*_IMPORT_COMMAND, "--stdin"],
        env=dict(os.environ, PYTHONPATH=str(site)),
        stdin=subprocess.PIPE,
        text=True,
    ) as worker:
        try:
            yield worker
        finally:
            if worker.poll() is None:
                worker.kill()


def install_wheel(site, *, defer_optional=False):
    site = Path(site).resolve(strict=True)
    if not (site / "newton/__init__.py").is_file():
        raise ValueError("NEWTON_SITE must contain the Newton wheel and its OpenUSD dependencies")

    class WheelFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] != "newton":
                return None
            parent = site.joinpath(*fullname.split(".")[:-1])
            spec = PathFinder.find_spec(fullname, [str(parent)])
            if spec is None:
                raise ModuleNotFoundError(fullname)
            if defer_optional and fullname in _OPTIONAL_NEWTON_MODULES:
                # Isaac RTX and AlexDoor control have their own implementations.
                # Load these Newton APIs on first use of their public exports.
                spec.loader = LazyLoader(spec.loader)
            return spec

    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] == "newton":
            # Bypass LazyLoader's attribute hook so repeat installation does
            # not initialize unused modules.
            origin = object.__getattribute__(module, "__dict__").get("__file__")
            if origin and not Path(origin).resolve().is_relative_to(site / "newton"):
                raise RuntimeError(f"Kit initialized another Newton wheel: {origin}")
    sys.meta_path.insert(0, WheelFinder())


def config(
    site,
    env_roots,
    *,
    measure_contact,
    constraint_capacity=None,
    velocity_limits=False,
    rigid_gap=None,
    contact_view=None,
    import_worker=None,
):
    import newton
    import warp as wp
    from isaaclab.sim.utils import get_current_stage
    from isaaclab.utils.configclass import configclass
    from newton.selection import ArticulationView

    # Isaac Lab 3.0.0-beta2 targets earlier Newton tensor names.
    if not hasattr(newton.solvers, "SolverNotifyFlags"):
        newton.solvers.SolverNotifyFlags = newton.ModelFlags
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg, NewtonSolverCfg

    get_attribute = ArticulationView.get_attribute
    aliases = {"joint_target_pos": "joint_target_q", "joint_target_vel": "joint_target_qd"}

    def compatible_get(view, name, source, *args, **kwargs):
        return get_attribute(view, aliases.get(name, name), source, *args, **kwargs)

    ArticulationView.get_attribute = compatible_get

    class NewtonSteelManager(NewtonManager):
        @classmethod
        def instantiate_builder_from_stage(cls):
            stage = get_current_stage()
            with TemporaryDirectory(prefix="steel-benchmark-") as scratch:
                usd = Path(scratch) / "stage.usdc"
                out = Path(scratch) / "builder.pkl"
                if not stage.Export(str(usd)):
                    raise RuntimeError("Could not export the common fixture scene")
                args = [str(usd), str(out), "default" if rigid_gap is None else str(rigid_gap)]
                args.extend(env_roots)
                if (
                    import_worker is not None
                    and import_worker.stdin is not None
                    and not import_worker.stdin.closed
                ):
                    import_worker.communicate(json.dumps(args), timeout=60)
                    if import_worker.returncode != 0:
                        raise subprocess.CalledProcessError(
                            import_worker.returncode, import_worker.args
                        )
                else:
                    subprocess.run(
                        [*_IMPORT_COMMAND, *args],
                        env=dict(os.environ, PYTHONPATH=str(site)),
                        timeout=60,
                        check=True,
                    )
                data, values = pickle.loads(out.read_bytes())
            builder = cls.create_builder()
            if values.keys() != builder.custom_attributes.keys():
                raise RuntimeError("Newton importer and Kit disagree on model attributes")
            vars(builder).update(data)
            for name, value in values.items():
                builder.custom_attributes[name].values = value
            NewtonManager._num_envs = max(1, len(env_roots))
            cls.set_builder(builder)

        @classmethod
        def _build_solver(cls, model, solver_cfg):
            NewtonManager._solver = newton.solvers.SolverSteel(
                model,
                iterations=16,
                velocity_iterations=4,
                articulation_constraint_max=constraint_capacity
                or (128 if cls.measure_contact else 32),
                enable_joint_limits=True,
                enable_joint_velocity_limits=velocity_limits,
                enable_joint_friction=True,
                enable_sleeping=False,
            )
            NewtonManager._use_single_state = False
            NewtonManager._needs_collision_pipeline = True

        @classmethod
        def _initialize_contacts(cls):
            super()._initialize_contacts()
            if cls.measure_contact:
                cls.observables = cls._solver.observables(
                    [cls._solver.ObservableKind.CONTACT_F],
                    contacts=cls._contacts,
                )
            if contact_view is not None:
                cls.contact_poses = wp.clone(cls._state_0.body_q)
                cls.contact_sim_view = contact_view(cls)

        @classmethod
        def _solver_specific_clear(cls):
            super()._solver_specific_clear()
            cls.observables = None
            if contact_view is not None:
                cls.contact_poses = None
                cls.contact_sim_view = None

        @classmethod
        def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
            if contact_view is not None:
                wp.copy(cls.contact_poses, state_0.body_q)
            cls._solver.step(
                state_0,
                state_1,
                control,
                contacts,
                substep_dt,
                observables=cls.observables,
            )

    # Isaac Lab resolves the configured manager by module path.
    globals()["NewtonSteelManager"] = NewtonSteelManager
    NewtonSteelManager.observables = None
    NewtonSteelManager.measure_contact = measure_contact
    from .feedback import install_gravity

    install_gravity(NewtonSteelManager)

    @configclass
    class SteelSolverCfg(NewtonSolverCfg):
        class_type: str = "isaac_physics_profile.steel:NewtonSteelManager"
        solver_type: str = "steel"

    return NewtonCfg(solver_cfg=SteelSolverCfg(), num_substeps=1, use_cuda_graph=True)
