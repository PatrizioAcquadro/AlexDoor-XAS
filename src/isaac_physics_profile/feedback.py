"""Solved Steel forces, joint reactions and rigid-link dynamics for Isaac Lab."""

from pathlib import PurePosixPath

import numpy as np


def install_gravity(manager):
    import warp as wp
    from isaaclab.utils.warp import ProxyArray
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData

    def gravity(data):
        view = data._root_view
        if not hasattr(data, "_steel_gravity_indices"):
            model = manager._model
            articulation_starts = model.articulation_start.numpy()
            dof_starts = model.joint_qd_start.numpy()
            labels = [PurePosixPath(path).name for path in model.joint_label]
            indices = []
            for group in view.articulation_ids.numpy():
                if len(group) != 1 or group[0] < 0:
                    raise RuntimeError("Steel gravity requires one articulation per environment")
                start, end = articulation_starts[group[0] : group[0] + 2]
                joint_ids = {labels[i]: i for i in range(start, end)}
                indices.append([dof_starts[joint_ids[name]] for name in view.joint_names])
            import torch

            data._steel_gravity_indices = torch.tensor(np.asarray(indices), device=data.device)
            data._steel_gravity_values = wp.zeros(model.joint_dof_count, device=data.device)
        data._ensure_fk_fresh()
        view.eval_inverse_dynamics_passive(
            manager._state_0,
            gravity_force=data._steel_gravity_values,
        )
        values = wp.to_torch(data._steel_gravity_values)[data._steel_gravity_indices].contiguous()
        return ProxyArray(wp.from_torch(values, requires_grad=False))

    ArticulationData.gravity_compensation_forces = property(gravity)


class SteelJointLoad:
    """Project rigid-body reaction wrenches and match their inverse dynamics."""

    def __init__(self, robot, state, reactions):
        import torch
        import warp as wp

        self.data = data = robot.data
        self.state, self.reactions = state, reactions
        self.view = view = robot.root_view
        model = view.model
        groups = view.articulation_ids.numpy()
        if groups.shape != (1, 1) or groups[0, 0] < 0:
            raise RuntimeError("Steel joint load requires one fixed-base articulation")
        group = groups[0, 0]
        start, end = model.articulation_start.numpy()[group : group + 2]
        labels = [PurePosixPath(path).name for path in model.joint_label]
        joints = {labels[i]: i for i in range(start, end)}
        selected = [joints[name] for name in robot.joint_names]
        starts = model.joint_qd_start.numpy()
        if np.any(np.diff(starts)[selected] != 1):
            raise RuntimeError("Steel joint load requires scalar DOFs")
        children = model.joint_child.numpy()[selected]
        links = robot.body_names
        jac_rows = [links.index(PurePosixPath(model.body_label[i]).name) - 1 for i in children]
        if min(jac_rows) < 0:
            raise RuntimeError("Steel joint load could not resolve all moving links")
        self.dofs = torch.as_tensor(starts[selected], device=data.device)
        self.children = torch.as_tensor(children, device=data.device)
        self.jac_rows = torch.as_tensor(jac_rows, device=data.device)
        self.columns = torch.arange(len(selected), device=data.device)
        self.coriolis = wp.zeros(model.joint_dof_count, device=data.device)

    def read(self):
        import warp as wp

        data = self.data
        # The public Jacobian read refreshes link kinematics before inverse dynamics.
        jacobian = data.body_com_jacobian_w.torch[0, self.jac_rows]
        self.view.eval_inverse_dynamics_passive(self.state(), coriolis_force=self.coriolis)
        reaction = wp.to_torch(self.reactions())[self.children]
        torque = (reaction * jacobian[self.columns, :, self.columns]).sum(dim=-1)
        acceleration = data.joint_acc.torch[0]
        # BODY_PARENT_F omits rotor armature, which remains live in the solver.
        inertial = data.mass_matrix.torch[0] @ acceleration
        expected = (
            inertial
            - data.joint_armature.torch[0] * acceleration
            + data.gravity_compensation_forces.torch[0]
            + wp.to_torch(self.coriolis)[self.dofs]
        )
        return torque, expected, acceleration
