"""Expose Steel contacts through the Purdue contact observer interface."""

import newton
import numpy as np
import warp as wp

from .steel_contact_snapshot import SteelContactSnapshot


class BodyView:
    def __init__(self, adapter, paths):
        self.adapter = adapter
        self.prim_paths = list(paths)
        self.indices = [adapter.body_ids[p] for p in paths]

    def get_transforms(self):
        poses = self.adapter.manager._state_0.body_q.numpy()[self.indices]
        return wp.array(poses, dtype=wp.float32, device="cpu")


class ContactView:
    def __init__(self, adapter, paths, max_contact_data_count):
        self.adapter = adapter
        self.sensor_paths = list(paths)
        self.capacity = max_contact_data_count
        self.owner_indices = np.asarray([adapter.owner_ids[path] for path in paths])

    def get_other_actor_paths_from_ids(self, actor_ids):
        return [self.adapter.owners[int(i)] for i in actor_ids.numpy()]

    def get_raw_contact_data(self, dt):
        (force, point, normal, separation, other, _), counts, starts = self.adapter.snapshot.rows(
            dt, self.owner_indices, None, self.capacity
        )
        arrays = (force, point, normal, separation, counts, starts, other.astype(np.uint64))
        return tuple(wp.array(value, device="cpu") for value in arrays)


class SteelPurdueViews:
    def __init__(self, manager):
        self.manager = manager
        model = manager._model
        self.shape_body = model.shape_body.numpy()
        self.body_ids = {p: i for i, p in enumerate(model.body_label)}
        self.owners = list(model.body_label)
        self.shape_owner = self.shape_body.copy()
        for i in np.flatnonzero(self.shape_body < 0):
            self.shape_owner[i] = len(self.owners)
            name = model.shape_label[i]
            if name == "/World/Ground/GroundPlane/CollisionPlane":
                if int(model.shape_type.numpy()[i]) != int(newton.GeoType.PLANE):
                    raise RuntimeError("Purdue ground owner must resolve to a static plane")
                # PhysX reports the same scene plane under its parent actor path.
                name = "/World/Ground/CollisionPlane"
            self.owners.append(name)
        self.owner_ids = {p: i for i, p in enumerate(self.owners)}
        self.snapshot = SteelContactSnapshot(
            manager, self.shape_body, self.shape_owner, check_capacity=True
        )

    def reset(self):
        """Clear solved state in place for the single AlexDoor environment."""
        manager = self.manager
        solver = manager._solver
        # Report sticky failures before reset clears them. Preserve the buffers bound to the graph.
        solver.check_constraint_capacity()
        solver.reset(manager._state_0)
        manager._contacts.clear()
        self.snapshot.invalidate()

    def create_rigid_body_view(self, paths):
        return BodyView(self, paths)

    def create_rigid_contact_view(self, paths, *, max_contact_data_count):
        return ContactView(self, paths, max_contact_data_count)
