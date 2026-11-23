"""Solved Steel forces, joint reactions and rigid-link dynamics for Isaac Lab."""

from pathlib import PurePosixPath

import numpy as np


class SteelJointLoad:
    """Project rigid-body reaction wrenches and match their inverse dynamics."""

    def __init__(self, robot, backend):
        import torch
        import warp as wp

        self.data = data = robot.data
        self.backend = backend
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
        state = self.backend.state_0
        self.view.eval_inverse_dynamics_passive(state, coriolis_force=self.coriolis)
        reaction = wp.to_torch(state.body_parent_f)[self.children]
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
