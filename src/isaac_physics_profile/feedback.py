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

    def __init__(self, manager, data):
        import torch
        import warp as wp

        self.manager, self.data = manager, data
        model = manager._model
        view = data._root_view
        groups = view.articulation_ids.numpy()
        if groups.shape != (1, 1) or groups[0, 0] < 0:
            raise RuntimeError("Steel joint load requires one fixed-base articulation")
        group = groups[0, 0]
        start, end = model.articulation_start.numpy()[group : group + 2]
        labels = [PurePosixPath(path).name for path in model.joint_label]
        joints = {labels[i]: i for i in range(start, end)}
        selected = [joints[name] for name in view.joint_names]
        starts = model.joint_qd_start.numpy()
        if np.any(np.diff(starts)[selected] != 1):
            raise RuntimeError("Steel joint load requires scalar DOFs")
        children = model.joint_child.numpy()[selected]
        links = [PurePosixPath(path).name for path in view.link_names]
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

        data, manager = self.data, self.manager
        data._ensure_fk_fresh()
        data._root_view.eval_inverse_dynamics_passive(
            manager._state_0, coriolis_force=self.coriolis
        )
        reaction = wp.to_torch(manager.observables.body_parent_f)[self.children]
        jacobian = data.body_com_jacobian_w.torch[0, self.jac_rows]
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


class SteelContactReader:
    """World-space normal force on each body from its named partner."""

    def __init__(self, sim):
        self.sim = sim
        self.manager = sim.physics_manager
        model = self.manager._model
        self.ids = {path: i for i, path in enumerate(model.body_label)}
        self.count = model.body_count
        self.shape_owner = model.shape_body.numpy()
        self.shape_owner[self.shape_owner < 0] = self.count
        self.step = -1
        self.matrix = np.zeros((self.count + 1, self.count + 1, 3), dtype=np.float32)

    def _refresh(self):
        step = self.sim.get_physics_step_count()
        if self.step == step:
            return
        self.manager._solver.check_constraint_capacity()
        contacts = self.manager._contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count >= contacts.rigid_contact_max:
            raise RuntimeError("Steel contact buffer is saturated")
        shape_a = contacts.rigid_contact_shape0.numpy()[:count]
        shape_b = contacts.rigid_contact_shape1.numpy()[:count]
        if np.any(shape_a < 0) or np.any(shape_b < 0):
            raise RuntimeError("Steel returned an invalid contact owner")
        a = self.shape_owner[shape_a]
        b = self.shape_owner[shape_b]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        force = self.manager.observables.contact_f.numpy()[:count, :3]
        normal_force = np.minimum((force * normal).sum(axis=1), 0)[:, None] * normal
        if not np.isfinite(normal_force).all():
            raise RuntimeError("Steel returned nonfinite contact force")
        self.matrix.fill(0)
        np.add.at(self.matrix, (a, b), normal_force)
        np.add.at(self.matrix, (b, a), -normal_force)
        self.step = step

    def pairs(self, body, partners):
        self._refresh()
        return self.matrix[self.ids[body], [self.ids[path] for path in partners]]

    def net(self, bodies):
        self._refresh()
        return self.matrix[[self.ids[path] for path in bodies]].sum(axis=1)
