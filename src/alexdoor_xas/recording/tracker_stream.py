"""Lossless capture of the observations actually consumed by the tracker."""

import json
from dataclasses import asdict

import h5py
import numpy as np

from alexdoor_xas.perception.evaluation import json_safe
from alexdoor_xas.recording.b1 import OBS_KEYS


class TrackerStream:
    def __init__(self, path):
        self.file = h5py.File(path, "x")
        self.file.attrs.update(schema="alexdoor.tracker-stream.v1", rows=0)
        self.observations = self.file.create_group("observations")
        self.masks = self.file.create_dataset(
            "packed_masks", (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
        )
        self.metadata = self.file.create_dataset(
            "publication", (0,), maxshape=(None,), dtype=h5py.string_dtype()
        )
        self.last_frame = -1
        self.rows = 0

    def append(self, provider, estimate, *, generation, phase):
        if provider.last_publication is None:
            return
        captured, result = provider.last_publication
        frame = int(captured["frame"])
        if frame == self.last_frame:
            return
        if frame < self.last_frame:
            raise ValueError("out_of_order_tracker_publication")
        cue = result["cue"]
        local = estimate.local
        patches = (
            []
            if local is None
            else [
                dict(
                    patch_id=patch.patch_id,
                    reason=patch.reason,
                    support=None if patch.pose_support is None else asdict(patch.pose_support),
                    normal=None if patch.world_pose is None else patch.world_pose.rot[:, 0],
                    normal_bound_rad=patch.normal_rotation_bound_rad,
                    alignment=None
                    if patch.normal_alignment is None
                    else asdict(patch.normal_alignment),
                )
                for patch in local.patches
            ]
        )
        tracking = provider.tracking.diagnostics
        metadata = json_safe(
            dict(
                frame=frame,
                generation=generation,
                phase=phase,
                acquired_s=float(captured["time_s"]),
                available_s=float(cue["available_s"]),
                mask_ids=cue["ids"],
                mask_shape=cue["shape"],
                patches=patches,
                local_fit=tracking.get("local_surface_constraint"),
                local_certificate=tracking.get("local_certificate"),
            )
        )
        for key in OBS_KEYS:
            value = np.asarray(captured[key])
            if key not in self.observations:
                self.observations.create_dataset(
                    key,
                    (0, *value.shape),
                    maxshape=(None, *value.shape),
                    dtype=value.dtype,
                    **(
                        dict(chunks=(1, *value.shape), compression="lzf", shuffle=key == "depth_m")
                        if value.ndim >= 3
                        else {}
                    ),
                )
            dataset = self.observations[key]
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = value
        self.masks.resize(self.rows + 1, axis=0)
        self.masks[self.rows] = np.frombuffer(b"".join(cue["masks"]), dtype=np.uint8)
        self.metadata.resize(self.rows + 1, axis=0)
        self.metadata[self.rows] = json.dumps(metadata, allow_nan=False)
        self.last_frame = frame
        self.rows += 1
        if self.rows == 1 or self.rows % 30 == 0:
            self.flush()

    def flush(self):
        self.file.attrs.modify("rows", self.rows)
        self.file.flush()

    def close(self):
        try:
            self.flush()
        finally:
            self.file.close()
