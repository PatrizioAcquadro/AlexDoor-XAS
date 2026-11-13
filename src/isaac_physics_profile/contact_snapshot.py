"""Read solved Steel contact geometry once per Isaac Lab physics step."""

import numpy as np

_CORE_STATUS = {"disabled_tree_joint", "joint_friction_invalid", "constraint_overflow"}


class SteelContactSnapshot:
    """Share a native contact sample between all views of the same solver step."""

    def __init__(self, manager, shape_body, shape_owner, *, check_capacity=False):
        self.manager = manager
        self.shape_body = shape_body
        self.shape_owner = shape_owner
        self.check_capacity = check_capacity
        self._step = None
        self._sources = ()
        self._host = ()
        self._arrays = ()

    def invalidate(self):
        """Refresh on the next read even if the physics step count stayed fixed."""
        self._step = None

    def _check_status(self, status):
        # The native checker owns optional status semantics and failure diagnostics.
        # Core statuses are sticky flags, so all zeros are exactly its fast path.
        if status is None or any(buffer.numpy().any() for buffer in status):
            self.manager._solver.check_constraint_capacity()

    def _refresh(self):
        import warp as wp
        from scipy.spatial.transform import Rotation

        manager = self.manager
        contacts = manager._contacts
        geometry = (
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            manager.observables.contact_f,
            manager.contact_poses,
        )
        status = manager._solver._status_sources() if self.check_capacity else {}
        core_status = status.keys() == _CORE_STATUS
        sources = (*geometry, *status.values()) if core_status else geometry
        step = manager._sim.get_physics_step_count()
        if len(sources) != len(self._sources) or any(
            a is not b for a, b in zip(sources, self._sources, strict=True)
        ):
            self._host = tuple(wp.empty_like(array, device="cpu", pinned=True) for array in sources)
            self._sources = sources
            self._step = None
        if self._step == step:
            return
        device = sources[0].device
        stream = wp.get_stream(device)
        for host, source in zip(self._host, sources, strict=True):
            wp.copy(host, source, stream=stream)
        wp.synchronize_stream(stream)
        if self.check_capacity:
            self._check_status(self._host[len(geometry) :] if core_status else None)
        count = int(self._host[0].numpy()[0])
        if not 0 <= count < contacts.rigid_contact_max:
            raise RuntimeError("Steel contact buffer saturated or invalid")
        shape0, shape1, p0, p1, normal, margin0, margin1, force = (
            host.numpy()[:count] for host in self._host[1 : len(geometry) - 1]
        )
        poses = self._host[len(geometry) - 1].numpy()
        if any(np.any((s < 0) | (s >= len(self.shape_owner))) for s in (shape0, shape1)):
            raise RuntimeError("Steel contact has an invalid shape owner")
        p0, p1 = p0.copy(), p1.copy()
        for points, shapes in ((p0, shape0), (p1, shape1)):
            bodies = self.shape_body[shapes]
            valid = bodies >= 0
            if np.any(valid):
                pose = poses[bodies[valid]]
                points[valid] = Rotation.from_quat(pose[:, 3:]).apply(points[valid]) + pose[:, :3]
        force = force[:, :3]
        separation = np.einsum("ij,ij->i", p1 - p0, normal) - margin0 - margin1
        midpoint = (p0 + p1 + (margin0 - margin1)[:, None] * normal) * 0.5
        normal_force = -np.einsum("ij,ij->i", force, normal)
        if not np.isfinite(np.c_[normal_force, separation, midpoint, force]).all():
            raise RuntimeError("Invalid Steel contact observation")
        if np.any(normal_force < -1e-4):
            raise RuntimeError("Steel solved contact normal force has the wrong sign")
        normal_force = np.maximum(normal_force, 0)
        tangent = force + normal_force[:, None] * normal
        self._arrays = (
            self.shape_owner[shape0],
            self.shape_owner[shape1],
            normal_force,
            midpoint,
            normal,
            separation,
            tangent,
        )
        self._step = step

    def rows(self, dt, owners, filters, capacity):
        if not np.isclose(dt, self.manager._solver_dt):
            raise RuntimeError("Contact time step differs from the solved Steel interval")
        self._refresh()
        a, b, force, point, normal, separation, tangent = self._arrays
        contacts = len(a)
        sides = np.concatenate((a, b))
        partners = np.concatenate((b, a))
        matches = sides[None, :] == owners[:, None]
        if filters is not None:
            matches &= partners[None, :] == filters[:, None]
        # Nonzero walks the matrix by sensor, then side, then original contact.
        sensor, slot = np.nonzero(matches)
        if len(slot) >= capacity:
            raise RuntimeError("Purdue contact readout capacity exceeded")
        counts = np.bincount(sensor, minlength=len(owners)).astype(np.uint32)
        starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.uint32)
        indices = slot % contacts if contacts else slot
        sign = 2 * (slot >= contacts).astype(np.int64) - 1
        values = (
            force[indices],
            point[indices],
            normal[indices] * sign[:, None],
            separation[indices],
            partners[slot],
            tangent[indices] * -sign[:, None],
        )
        # Consumers use one unused sentinel element to distinguish full buffers.
        return [np.pad(v, [(0, 1), *[(0, 0)] * (v.ndim - 1)]) for v in values], counts, starts
