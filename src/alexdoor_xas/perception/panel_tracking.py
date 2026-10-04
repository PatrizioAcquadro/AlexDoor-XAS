"""Point2Pose adapter: independent rigid candidates and panel-fixed material zones."""

import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import (
    MAX_DYNAMIC_AGE_S,
    FieldSupport,
    LocalContactSelection,
    LocalMaterialState,
    LocalPatchState,
    validate_local_transition,
)
from alexdoor_xas.perception.geometry import contact_frame, project, similar_surface
from alexdoor_xas.perception.material_zone import (
    BoundSource,
    ContactBudget,
    contact_depth_supported,
    object_frame,
    pose_matrix,
    query_slip,
    transported_zone,
)
from alexdoor_xas.perception.observed_articulation import fit_motion_axis
from alexdoor_xas.perception.point2pose_seed import candidate_references
from alexdoor_xas.perception.point2pose_worker import unpack_array
from alexdoor_xas.perception.scan import object_members, registered_support, surface_support


@dataclass
class PanelCandidate:
    candidate_id: str
    surface: object
    initial_world: np.ndarray
    map_from_object: np.ndarray
    object_from_zone: np.ndarray
    geometry: FieldSupport
    pose: np.ndarray | None = None
    support: FieldSupport | None = None
    ownership: str = "ambiguous_leaf_candidate"
    registration_position_m: float | None = None
    registration_rotation_rad: float | None = None
    state: str = "initializing"
    registration_origin_map: np.ndarray | None = None
    camera_world: np.ndarray | None = None


def metric_registration_bound(source, current, transform, depth_error_m):
    """Conditioned residual/depth envelope, not the dimensionless TAPIR uncertainty.

    This is a diagnostic correspondence bound; hardware sensor/calibration errors
    remain separate and unmeasured unless the caller supplies them explicitly.
    """
    if len(source) < 5 or source.shape != current.shape:
        return None, None
    relative = source - source.mean(0)
    singular = np.linalg.svd(relative, compute_uv=False)
    # A planar distributed cloud observes rotation; a line does not.
    lever = float(singular[1] / np.sqrt(len(source)))
    if lever <= 2 * depth_error_m:
        return None, None
    residual = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - current, axis=1)
    # The initial-map depth belongs to local_geometry in the composite budget.
    # Include only this acquisition's depth and registration residual here.
    error = float(residual.max() + depth_error_m)
    return error, float(2 * np.arcsin(min(1.0, error / (2 * lever))))


def measured_registration(obj, depth_error_m):
    """The integration's measured inlier support, shared with offline diagnosis."""
    source, current = obj["source_points"], obj["current_points"]
    inliers = obj["inliers"]
    measured = obj.get("measured_correspondences", np.zeros(len(source), bool))
    if inliers.dtype == bool and len(inliers) == len(source):
        source, current, measured = source[inliers], current[inliers], measured[inliers]
    source, current = source[measured], current[measured]
    p, a = metric_registration_bound(source, current, obj["camera_from_map"], depth_error_m)
    return source, p, a


class PanelTracking:
    def __init__(
        self,
        engine,
        config,
        *,
        calibration=None,
        tool_fk=None,
        distal_faces=None,
        calibration_bound=None,
        relative_speed_bound=None,
    ):
        self.engine, self.config = engine, config
        self.calibration, self.tool_fk, self.distal_faces = calibration, tool_fk, distal_faces
        self.calibration_bound = calibration_bound
        self.relative_speed_bound = relative_speed_bound
        self.before_start = None
        self.generation = -1
        self.reset()

    def reset(self, generation=None):
        self.generation = self.generation + 1 if generation is None else generation
        self.candidates, self.selection = [], None
        self.last_submitted = self.last_result = None
        self.selection_counter = 0
        self.motion_samples = {}
        self.diagnostics = dict(
            state="acquiring_automatic_candidates", loaded_contact_admitted=False
        )
        self.engine.reset()

    def initialize(self, candidates, sensor, *, start_s=None):
        if self.candidates:
            return False
        processing_started = time.perf_counter()
        roots = [
            s
            for s in candidates
            if abs(s.normal[2]) < 0.3
            and np.ptp(s.points @ s.basis, axis=0)[1] > 0.2
            and np.ptp(s.points @ s.basis, axis=0)[2] > 0.3
        ]
        identifiers = {id(s): f"candidate-{i}" for i, s in enumerate(roots)}
        eligible = []
        for surface in roots:
            observation = surface.observations[0]
            if observation.frame != int(sensor["frame"]) or observation.acquired_s != float(
                sensor["time_s"]
            ):
                raise ValueError("unsynchronized_candidate_initialization")
            try:
                candidate_references(
                    observation.mask(),
                    sensor["depth_m"],
                    sensor["valid_depth"],
                    sensor["intrinsics"],
                    self.config["plane_tolerance_m"],
                )
            except ValueError as error:
                self.diagnostics.setdefault("initialization_rejections", []).append(
                    dict(
                        candidate_id=identifiers[id(surface)],
                        frame=int(sensor["frame"]),
                        reason=str(error),
                    )
                )
            else:
                eligible.append(surface)
        roots = eligible
        unique = []
        for surface in roots:
            duplicate = any(
                similar_surface(old, surface, self.config)
                and similar_surface(surface, old, self.config)
                and registered_support(old, surface.observations[0])
                and registered_support(surface, old.observations[0])
                for old in unique
            )
            if not duplicate:
                unique.append(surface)
        self.diagnostics["redundant_observations"] = len(roots) - len(unique)
        roots = unique
        groups = [(s, object_members(s, candidates, self.config)) for s in roots]
        roots = [
            s
            for s, _ in groups
            if not any(
                s is not other and any(s is member for member in members)
                for other, members in groups
            )
        ]
        masks = []
        for surface in roots:
            observation = surface.observations[0]
            if observation.frame != int(sensor["frame"]) or observation.acquired_s != float(
                sensor["time_s"]
            ):
                raise ValueError("unsynchronized_candidate_initialization")
            mask = observation.mask()
            # The reference is a measured point. No panel size/area identifies a leaf.
            world = pose_matrix(
                ObjectFrame(surface.points[len(surface.points) // 2], contact_frame(surface.normal))
            )
            self.candidates.append(
                PanelCandidate(
                    identifiers[id(surface)],
                    deepcopy(surface),
                    world,
                    np.linalg.inv(sensor["camera_world"]) @ world,
                    np.eye(4),
                    surface_support(surface, self.generation, self.config),
                )
            )
            masks.append(mask)
        if not masks:
            return False
        if self.before_start is not None:
            self.before_start()
        started = max(o.available_s for s in roots for o in s.observations)
        self.engine.submit(
            sensor,
            masks,
            start_s=max(started, start_s or started) + time.perf_counter() - processing_started,
        )
        self.last_submitted = int(sensor["frame"])
        return True

    def select_zone(self, patch_id, now, *, source):
        if source not in ("diagnostic", "policy") or patch_id not in [
            c.candidate_id for c in self.candidates
        ]:
            raise ValueError("invalid_explicit_zone_selection")
        self.selection_counter += 1
        selection = LocalContactSelection(
            f"zone-{self.generation}-{self.selection_counter}",
            patch_id,
            float(now),
            source,
            self.selection.selection_id if self.selection else None,
        )
        validate_local_transition(self.selection, selection)
        self.selection = selection

    def consume(self, captured, result, ready):
        started = time.perf_counter()
        if len(result["objects"]) != len(self.candidates):
            raise ValueError("point2pose_candidate_identity_count_changed")
        self.last_result = result
        for candidate, raw in zip(self.candidates, result["objects"], strict=True):
            obj = {
                k: unpack_array(v) if isinstance(v, dict) and "data" in v else v
                for k, v in raw.items()
            }
            transform = obj["camera_from_map"]
            # Distributed native references estimate candidate pose. The uniform
            # contact face does not need its own visual features or point tracks.
            source, p, a = measured_registration(obj, self.config["plane_tolerance_m"])
            self.diagnostics.setdefault("support_by_candidate", {})[candidate.candidate_id] = dict(
                measured_candidate_pairs=len(source),
                native_lost=bool(obj["lost"]),
                registration_position_m=p,
                registration_rotation_rad=a,
            )
            if obj["lost"] or p is None or a is None:
                candidate.state = "tracking_lost"
                continue  # Frozen native pose never renews acquisition/support.
            candidate.pose = transported_zone(
                captured["camera_world"],
                transform,
                candidate.map_from_object,
                candidate.object_from_zone,
            )
            candidate.registration_position_m, candidate.registration_rotation_rad = p, a
            candidate.registration_origin_map = source.mean(0)
            candidate.camera_world = np.array(captured["camera_world"], copy=True)
            candidate.support = FieldSupport(
                float(captured["time_s"]), ready, float(captured["time_s"]), self.generation, p, a
            )
            candidate.state = "tracked"
            samples = self.motion_samples.setdefault(candidate.candidate_id, deque(maxlen=32))
            if not samples or np.linalg.norm(candidate.pose[:3, 3] - samples[-1][0][:3, 3]) > p:
                samples.append((candidate.pose.copy(), p, a))
            axis = fit_motion_axis(
                candidate.initial_world, list(samples), self.config.get("floor_z_m", 0.0)
            )
            if axis is not None and candidate.ownership == "observed_moving_rigid_candidate":
                axis.update(
                    acquired_s=candidate.support.acquired_s,
                    supported_s=candidate.support.supported_s,
                    available_s=candidate.support.available_s,
                    generation=self.generation,
                )
                self.diagnostics.setdefault("observed_articulation", {})[candidate.candidate_id] = (
                    axis
                )
            # Verify root geometry independently of SAM2's visual membership.
            # Tangential handle motion on a fixed plane cannot certify the leaf.
            initial_n, current_n = candidate.initial_world[:3, 0], candidate.pose[:3, 0]
            plane_angle = float(np.arccos(np.clip(initial_n @ current_n, -1, 1)))
            normal_travel = abs(
                float(initial_n @ (candidate.pose[:3, 3] - candidate.initial_world[:3, 3]))
            )
            consistent = self.root_consistent(candidate, captured)
            self.diagnostics["support_by_candidate"][candidate.candidate_id][
                "root_geometry_consistent"
            ] = consistent
            if consistent and (normal_travel > 2 * p or plane_angle > 2 * a):
                candidate.ownership = "observed_moving_rigid_candidate"
        result["adapter_latency_s"] = time.perf_counter() - started
        ready += result["adapter_latency_s"]
        for candidate in self.candidates:
            if candidate.state == "tracked":
                candidate.support = replace(candidate.support, available_s=ready)
                axis = self.diagnostics.get("observed_articulation", {}).get(candidate.candidate_id)
                if axis is not None:
                    axis["available_s"] = ready
        self.diagnostics.update(
            state="tracked"
            if any(c.state == "tracked" for c in self.candidates)
            else "tracking_lost",
            latency_s=ready - float(captured["time_s"]),
            inference_latency_s=result["latency_s"],
            captured_s=result["capture_s"],
            available_s=ready,
            candidates=len(self.candidates),
        )

    def root_consistent(self, candidate, sensor):
        motion = candidate.pose @ np.linalg.inv(candidate.initial_world)
        points = candidate.surface.points[:: max(1, len(candidate.surface.points) // 64)]
        points = points @ motion[:3, :3].T + motion[:3, 3]
        uv, z = project(points, sensor["intrinsics"], sensor["camera_world"])
        uv = np.rint(uv).astype(int)
        h, w = sensor["rgb"].shape[:2]
        good = (z > 0) & (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
        rows = np.flatnonzero(good)
        measured = sensor["depth_m"][uv[rows, 1], uv[rows, 0], 0]
        good[rows] &= (
            sensor["valid_depth"][uv[rows, 1], uv[rows, 0], 0]
            & np.isfinite(measured)
            & (abs(measured - z[rows]) <= 2 * self.config["plane_tolerance_m"])
        )
        if good.sum() < 5:
            return False
        spread = np.linalg.svd(points[good] - points[good].mean(0), compute_uv=False)
        return bool(spread[1] / np.sqrt(good.sum()) > 2 * self.config["plane_tolerance_m"])

    def budget(self, candidate, now, distal_faces=None, tool_world=None):
        faces = self.distal_faces if distal_faces is None else distal_faces
        # Each radius is measured from its actual rotation origin to the fingers.
        finger_r = None if faces is None else max(np.linalg.norm(f, axis=1).max() for f in faces)
        registration_origin = candidate.registration_origin_map
        origin_o = (
            None
            if registration_origin is None
            else (np.linalg.inv(candidate.map_from_object) @ np.r_[registration_origin, 1.0])[:3]
        )
        tool_o = candidate.object_from_zone[:3, 3]
        local_r = finger_r
        if tool_world is not None and candidate.pose is not None:
            relative = np.linalg.inv(candidate.pose) @ pose_matrix(tool_world)
            tool_o = (candidate.object_from_zone @ relative)[:3, 3]
            local_r = None if finger_r is None else np.linalg.norm(relative[:3, 3]) + finger_r
        object_r = (
            None
            if finger_r is None or origin_o is None
            else (np.linalg.norm(tool_o - origin_o) + finger_r)
        )
        camera_r = (
            None
            if finger_r is None
            else (
                np.linalg.norm((candidate.map_from_object @ candidate.object_from_zone)[:3, 3])
                + finger_r
            )
        )
        calibration = self.calibration_bound or (None, None)
        if tool_world is not None and candidate.camera_world is not None and finger_r is not None:
            camera_r = (
                np.linalg.norm(
                    (np.linalg.inv(candidate.camera_world) @ pose_matrix(tool_world))[:3, 3]
                )
                + finger_r
            )
        age = None if candidate.support is None else max(0.0, now - candidate.support.supported_s)
        speed = self.relative_speed_bound or (None, None)
        temporal_p = (
            0.0 if age == 0 else None if age is None or speed[0] is None else age * speed[0]
        )
        temporal_a = (
            0.0 if age == 0 else None if age is None or speed[1] is None else age * speed[1]
        )
        return ContactBudget(
            (
                BoundSource(
                    "local_geometry",
                    candidate.geometry.position_bound_m,
                    candidate.geometry.rotation_bound_rad,
                    None if local_r is None or object_r is None else max(local_r, object_r),
                ),
                BoundSource(
                    "dynamic_pose",
                    candidate.registration_position_m,
                    candidate.registration_rotation_rad,
                    object_r,
                ),
                BoundSource("calibration_fk", *calibration, camera_r),
                BoundSource("temporal_growth", temporal_p, temporal_a, object_r),
            )
        )

    def local_state(self, now, sensor=None):
        patches = []
        for candidate in self.candidates:
            fresh = (
                candidate.state == "tracked"
                and candidate.support is not None
                and candidate.support.available_s <= now
                and 0 <= now - candidate.support.supported_s <= MAX_DYNAMIC_AGE_S + 1e-9
            )
            reason = (
                "ambiguous_leaf_ownership"
                if fresh and candidate.ownership == "ambiguous_leaf_candidate"
                else ""
            )
            support = candidate.support if fresh else None
            identity = (
                None
                if support is None
                else FieldSupport(
                    support.acquired_s,
                    support.available_s,
                    support.supported_s,
                    self.generation,
                    reason=reason,
                )
            )
            tool = None
            if (
                fresh
                and sensor is not None
                and self.tool_fk is not None
                and self.calibration is not None
            ):
                tool = self.tool_fk(sensor["joint_position"], self.calibration)
            budget = self.budget(candidate, now, tool_world=tool)
            pose_support = (
                None
                if support is None
                else FieldSupport(
                    support.acquired_s,
                    support.available_s,
                    support.supported_s,
                    self.generation,
                    budget.displacement_m,
                    budget.rotation_rad,
                )
            )
            clearance = (None, None)
            if (
                fresh
                and sensor is not None
                and self.tool_fk is not None
                and self.calibration is not None
                and self.distal_faces is not None
                and not reason
            ):
                slip = query_slip(
                    candidate.surface,
                    candidate.initial_world,
                    candidate.pose,
                    tool,
                    self.distal_faces,
                    budget,
                    self.config,
                )
                clearance = slip.finger_clearance_m
                self.diagnostics["tool_relative_zone"] = vars(slip)
                if not contact_depth_supported(
                    sensor, candidate.pose, tool, self.distal_faces, budget.displacement_m
                ):
                    clearance = (None, None)
                    reason = "missing_current_contact_geometry"
            patches.append(
                LocalPatchState(
                    candidate.candidate_id,
                    object_frame(candidate.pose) if fresh else None,
                    candidate.geometry,
                    identity,
                    pose_support,
                    clearance,
                    reason
                    or (
                        ""
                        if fresh
                        else candidate.state
                        if candidate.state == "tracking_lost"
                        else "stale_dynamic_support"
                    ),
                    tracking_state=candidate.state if fresh else "unavailable",
                    ownership=candidate.ownership,
                    uncertainty_sources=tuple(
                        (s.name, s.position_m, s.rotation_rad, s.lever_m) for s in budget.sources
                    ),
                )
            )
            if not fresh:
                self.diagnostics.get("observed_articulation", {}).pop(candidate.candidate_id, None)
        reference = self.selection.patch_id if self.selection else None
        return LocalMaterialState(self.generation, reference, tuple(patches), self.selection)

    def update(self, sensor):
        now = float(sensor["time_s"])
        frame = int(sensor["frame"])
        if self.candidates and frame != self.last_submitted:
            self.engine.submit(sensor)
            self.last_submitted = frame
        if event := self.engine.poll(now):
            captured, _, result, ready = event
            self.consume(captured, result, max(ready, now))
        return self.local_state(now, sensor)
