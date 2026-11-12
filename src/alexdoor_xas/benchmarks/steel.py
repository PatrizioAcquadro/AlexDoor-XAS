"""Select Steel in Isaac Lab beta2 and expose solved contact forces."""

import os
import pickle
import subprocess
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path
from tempfile import TemporaryDirectory


def install_wheel(site):
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
            return spec

    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] == "newton" and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(site / "newton"):
                raise RuntimeError(f"Kit initialized another Newton wheel: {module.__file__}")
    sys.meta_path.insert(0, WheelFinder())


def config(site, env_roots, *, measure_contact):
    import newton
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
                # The local importer writes its result into this private directory.
                env = dict(os.environ, PYTHONPATH=str(site))
                subprocess.run(
                    [
                        "/usr/bin/python3",
                        str(Path(__file__).with_name("steel_import.py")),
                        str(usd),
                        str(out),
                        *env_roots,
                    ],
                    env=env,
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
            NewtonManager._num_envs = len(env_roots)
            cls.set_builder(builder)

        @classmethod
        def _build_solver(cls, model, solver_cfg):
            NewtonManager._solver = newton.solvers.SolverSteel(
                model,
                iterations=16,
                velocity_iterations=4,
                articulation_constraint_max=128 if cls.measure_contact else 32,
                enable_joint_limits=True,
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

        @classmethod
        def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
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

    @configclass
    class SteelSolverCfg(NewtonSolverCfg):
        class_type: str = "alexdoor_xas.benchmarks.steel:NewtonSteelManager"
        solver_type: str = "steel"

    return NewtonCfg(solver_cfg=SteelSolverCfg(), num_substeps=1, use_cuda_graph=True)
