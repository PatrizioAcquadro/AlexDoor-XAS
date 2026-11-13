"""Solved Steel normal forces and rigid-link gravity for Isaac Lab consumers."""

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
