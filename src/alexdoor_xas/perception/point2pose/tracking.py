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
from alexdoor_xas.perception.geometry import (
    contact_frame,
    deproject,
    project,
    similar_surface,
    surfaces,
)
from alexdoor_xas.perception.ipc import unpack_array
from alexdoor_xas.perception.local_surface import SurfacePose, constrained_surface_pose
from alexdoor_xas.perception.material_zone import (
    BoundSource,
    ContactBudget,
    contact_depth_supported,
    object_frame,
    pose_matrix,
    query_slip,
    transported_zone,
)
from alexdoor_xas.perception.observed_articulation import (
    MotionObservation,
    StaticReference,
    current_angle,
    fit_motion_axis,
)
from alexdoor_xas.perception.point2pose.seed import candidate_references, verify_candidate
from alexdoor_xas.perception.scan import (
    footprint_support,
    object_members,
    observed_material_surface,
    registered_support,
    static_material_witnesses,
    supported_material_zone,
    surface_support,
)


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
    static_reference: StaticReference | None = None
    sam3_id: int | None = None
    p2p_object_id: int | None = None
    material_recovery_unverified: bool = False
    static_material_witnesses: tuple = ()
    local_constraint: SurfacePose | None = None
    local_constraint_reason: str = "local_surface_not_processed"
    local_surface: object | None = None
    material_cover: object | None = None


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
        constraint = config.get("local_surface_constraint")
        if constraint is not None and (
            not isinstance(constraint, dict)
            or set(constraint) != {"point_error_m", "normal_model_bound_rad"}
            or not np.isfinite(list(constraint.values())).all()
            or constraint["point_error_m"] <= 0
            or constraint["normal_model_bound_rad"] < 0
        ):
            raise ValueError("invalid_local_surface_constraint_recipe")
        self.calibration, self.tool_fk, self.distal_faces = calibration, tool_fk, distal_faces
        self.calibration_bound = calibration_bound
        self.relative_speed_bound = relative_speed_bound
        self.generation = -1
        self.reset()

    def reset(self, generation=None):
        self.generation = self.generation + 1 if generation is None else generation
        self.candidates, self.selection = [], None
        self.last_submitted = self.last_result = None
        self.selection_counter = 0
        self.initialization_options = []
        self.motion_samples = {}
        self.diagnostics = dict(
            state="acquiring_automatic_candidates", loaded_contact_admitted=False
        )
        self.engine.reset()

    def eligible_surfaces(self, candidates, sensor, whole_leaf_mask=None):
        """Measured seed alternatives, without starting P2P or resolving ownership."""
        self.diagnostics["seed_state"] = "insufficient_geometry"
        self.diagnostics["initialization_rejections"] = []
        self.initialization_options = []
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
                references = candidate_references(
                    observation.mask(),
                    sensor["depth_m"],
                    sensor["valid_depth"],
                    sensor["intrinsics"],
                    self.config["plane_tolerance_m"],
                )
                if whole_leaf_mask is not None:
                    verify_candidate(observation.mask(), whole_leaf_mask, references)
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
        self.initialization_options = roots
        return roots, identifiers

    def initialize(
        self, candidates, sensor, *, start_s=None, whole_leaf_mask=None, allow_ambiguous_seed=False
    ):
        if self.candidates:
            return False
        processing_started = time.perf_counter()
        if whole_leaf_mask is not None and whole_leaf_mask.shape != sensor["rgb"].shape[:2]:
            raise ValueError("whole_leaf_mask_shape_changed")
        roots, identifiers = self.eligible_surfaces(candidates, sensor, whole_leaf_mask)
        if whole_leaf_mask is not None:
            # One semantic leaf identity can have several measured plane faces.
            # The first eligible face supplies a zone; the native object receives
            # the unchanged whole-leaf mask, without another segmentation.
            if len(roots) > 1 and not allow_ambiguous_seed:
                self.diagnostics["seed_state"] = "ambiguous_geometry"
                return False
            roots = roots[:1]
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
            self.candidates[-1].sam3_id = observation.semantic_id
            self.candidates[-1].p2p_object_id = len(self.candidates) - 1
            self.candidates[-1].static_reference = StaticReference.observed(
                world, self.candidates[-1].geometry
            )
            masks.append(mask if whole_leaf_mask is None else whole_leaf_mask)
        if not masks:
            return False
        started = max(o.available_s for s in roots for o in s.observations)
        self.engine.submit(
            sensor,
            masks,
            start_s=max(started, start_s or started) + time.perf_counter() - processing_started,
        )
        self.last_submitted = int(sensor["frame"])
        return True

    def adopt_candidates(self, candidates, sensor, scan_memory=None, *, available_s=None):
        """Attach finite support at its actual publication time; preserve acquisition."""
        available_s = float(sensor["time_s"]) if available_s is None else float(available_s)
        if not np.isfinite(available_s) or available_s < float(sensor["time_s"]):
            raise ValueError("invalid_material_adoption_clock")
        self.candidates = candidates
        for candidate in candidates:
            candidate.static_material_witnesses = ()
            if candidate.ownership == "observed_rigid_material":
                candidate.ownership = "ambiguous_leaf_candidate"
        if self.distal_faces is None:
            return
        for candidate in candidates:
            if candidate.geometry.acquired_s != float(sensor["time_s"]):
                raise ValueError("unsynchronized_material_zone_initialization")
            zone = supported_material_zone(candidate.surface, self.distal_faces, self.config)
            self.diagnostics.setdefault("initial_material_zone", {})[candidate.candidate_id] = (
                "maximum_observed_two_finger_margin"
                if zone is not None
                else "no_supported_finite_material_zone"
            )
            if zone is None:
                continue
            if scan_memory is not None and self.calibration_bound is not None:
                p, a = self.calibration_bound
                if p is not None and a is not None:

                    def witnesses(zone, candidate=candidate, p=p, a=a):
                        radius = np.linalg.norm(zone.origin - sensor["camera_world"][:3, 3])
                        error = candidate.geometry.position_bound_m + p + 2 * radius * np.sin(a / 2)
                        return static_material_witnesses(
                            scan_memory.observed_surfaces,
                            zone,
                            self.distal_faces,
                            self.config,
                            candidate.sam3_id,
                            error,
                        )

                    # A broad current-only zone can hide an independently
                    # supported material anchor elsewhere on the same surface.
                    # Prefer such an observed anchor; never manufacture views
                    # or use task geometry to select it. If none exists, keep
                    # the original ambiguous diagnostic candidate.
                    if len(witnesses(zone)) < 3:
                        supported = supported_material_zone(
                            candidate.surface,
                            self.distal_faces,
                            self.config,
                            support_filter=lambda pose, witnesses=witnesses: (
                                len(witnesses(pose)) >= 3
                            ),
                        )
                        if supported is not None:
                            zone = supported
            candidate.initial_world = pose_matrix(zone)
            candidate.map_from_object = (
                np.linalg.inv(sensor["camera_world"]) @ candidate.initial_world
            )
            candidate.object_from_zone = np.eye(4)
            candidate.static_reference = StaticReference.observed(
                candidate.initial_world, candidate.geometry
            )
            if scan_memory is not None and self.calibration_bound is not None:
                p, a = self.calibration_bound
                if p is not None and a is not None:
                    radius = np.linalg.norm(zone.origin - sensor["camera_world"][:3, 3])
                    error = candidate.geometry.position_bound_m + p + 2 * radius * np.sin(a / 2)
                    candidate.static_material_witnesses = static_material_witnesses(
                        scan_memory.observed_surfaces,
                        zone,
                        self.distal_faces,
                        self.config,
                        candidate.sam3_id,
                        error,
                    )
                    if any(
                        w["available_s"] > available_s for w in candidate.static_material_witnesses
                    ):
                        raise ValueError("future_static_material_witness")
            self.diagnostics.setdefault("static_material_witnesses", {})[candidate.candidate_id] = (
                candidate.static_material_witnesses
            )
            if scan_memory is not None:
                candidate.local_surface = observed_material_surface(
                    candidate.surface, scan_memory.surfaces, candidate.sam3_id, self.config
                )
                if max(o.available_s for o in candidate.local_surface.observations) > available_s:
                    raise ValueError("future_local_material_surface")

    def attach_material_cover(self, views):
        """Frozen observed inspection relief; no runtime pose/identity replacement."""
        from alexdoor_xas.perception.material_zone import MaterialCover

        if self.config.get("material_depth_cover") is None:
            return
        for candidate in self.candidates:
            candidate.material_cover = MaterialCover.from_inspection(
                candidate.surface,
                views,
                candidate.sam3_id,
                self.config,
                generation=self.generation,
            )
            self.diagnostics.setdefault("material_cover", {})[candidate.candidate_id] = (
                candidate.material_cover.diagnostics
            )

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

    def consume(self, captured, result, ready, *, clock=None):
        started = time.perf_counter()
        if len(result["objects"]) != len(self.candidates):
            raise ValueError("point2pose_candidate_identity_count_changed")
        expected_ids = [
            c.p2p_object_id if c.p2p_object_id is not None else i
            for i, c in enumerate(self.candidates)
        ]
        if [obj["object_id"] for obj in result["objects"]] != expected_ids:
            raise ValueError("point2pose_candidate_identity_changed")
        self.last_result = result
        for candidate, raw in zip(self.candidates, result["objects"], strict=True):
            obj = {
                k: unpack_array(v) if isinstance(v, dict) and "data" in v else v
                for k, v in raw.items()
            }
            if result.get("semantic_absent", False):
                candidate.material_recovery_unverified = True
            transform = obj["camera_from_map"]
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                candidate.state = "tracking_lost"
                continue
            # Distributed native references estimate candidate pose. The uniform
            # contact face does not need its own visual features or point tracks.
            source, p, a = measured_registration(obj, self.config["plane_tolerance_m"])
            self.diagnostics.setdefault("support_by_candidate", {})[candidate.candidate_id] = dict(
                measured_candidate_pairs=len(source),
                native_lost=bool(obj["lost"]),
                registration_position_m=p,
                registration_rotation_rad=a,
            )
            if obj["lost"] or p is None or a is None or result.get("semantic_absent", False):
                candidate.state = "tracking_lost"
                articulation = self.diagnostics.setdefault("observed_articulation", {}).get(
                    candidate.candidate_id
                )
                if articulation is not None:
                    articulation["angle"] = None
                    articulation["current_reason"] = "tracking_lost"
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
            initial_camera = candidate.initial_world @ np.linalg.inv(candidate.map_from_object)
            anchor = (initial_camera @ np.r_[source.mean(0), 1])[:3]
            observation = MotionObservation(candidate.pose.copy(), candidate.support, anchor)
            if not samples or np.linalg.norm(candidate.pose[:3, 3] - samples[-1].pose[:3, 3]) > p:
                samples.append(observation)
            fit_detail = {}
            axis = fit_motion_axis(
                candidate.static_reference,
                list(samples),
                self.config.get("floor_z_m", 0.0),
                floor_bound_m=self.config.get("floor_bound_m"),
                diagnostics=fit_detail,
            )
            self.diagnostics.setdefault("observed_articulation", {})[candidate.candidate_id] = dict(
                static_reference=dict(
                    rotation=candidate.static_reference.rotation.copy(),
                    support=vars(candidate.static_reference.support),
                    closed_by_protocol=candidate.static_reference.closed_by_protocol,
                ),
                hinge=axis,
                angle=current_angle(candidate.static_reference, observation),
                fit=fit_detail,
                ownership=candidate.ownership,
                operational_reference_published=False,
            )
            # Verify root geometry independently of semantic mask membership.
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
            elif (
                candidate.ownership != "observed_moving_rigid_candidate"
                and consistent
                and self.selection is not None
                and self.selection.patch_id == candidate.candidate_id
                and len(candidate.static_material_witnesses) >= 3
            ):
                candidate.ownership = "observed_rigid_material"
            elif candidate.ownership == "observed_rigid_material" and not consistent:
                candidate.ownership = "ambiguous_leaf_candidate"
        result["adapter_latency_s"] = time.perf_counter() - started
        ready = (
            max(ready, float(clock())) if clock is not None else ready + result["adapter_latency_s"]
        )
        for candidate in self.candidates:
            if candidate.state == "tracked":
                candidate.support = replace(candidate.support, available_s=ready)
                axis = self.diagnostics.get("observed_articulation", {}).get(candidate.candidate_id)
                if axis is not None and axis.get("angle") is not None:
                    axis["angle"]["available_s"] = ready
                    # Hinge support retains the times of its actual motion samples.
                    samples = self.motion_samples[candidate.candidate_id]
                    if samples and samples[-1].support.supported_s == float(captured["time_s"]):
                        samples[-1] = replace(samples[-1], support=candidate.support)
                    if axis["hinge"] is not None:
                        axis["hinge"]["available_s"] = max(axis["hinge"]["available_s"], ready)
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

    def constrain_local(self, captured, result):
        """Constrain only A1/A2 local pose; native/A3/A4 diagnostics stay unchanged."""
        config = self.config.get("local_surface_constraint")
        if config is None:
            return
        self.diagnostics["local_surface_query"] = dict(
            expanded="local_surface_expansion_reason" in result,
            reason=result.get("local_surface_expansion_reason"),
            surfaces=len(result.get("local_surfaces") or []),
        )
        for candidate, raw in zip(self.candidates, result["objects"], strict=True):
            candidate.local_constraint = None
            normal_points = None
            try:
                if candidate.state != "tracked" or candidate.material_recovery_unverified:
                    raise ValueError("unavailable_native_material_identity")
                normal_points = self.reobserved_material_points(candidate, captured)
                choices = [
                    s
                    for s in (result.get("local_surfaces") or [])
                    if s.observations[0].semantic_id == candidate.sam3_id and abs(s.normal[2]) < 0.3
                ]
                if not choices:
                    raise ValueError("missing_or_ambiguous_current_surface")
                obj = {
                    k: unpack_array(v) if isinstance(v, dict) and "data" in v else v
                    for k, v in raw.items()
                }
                measured = obj["measured_correspondences"] & obj["inliers"]
                consistent, failures = [], []
                # A newly larger visible panel face need not be the original
                # material plane. Apply the existing depth/rigidity consistency
                # check before ranking current observed support by area.
                for current in sorted(
                    choices, key=lambda s: int(s.observations[0].mask().sum()), reverse=True
                ):
                    try:
                        observation = current.observations[0]
                        if observation.frame != int(
                            captured["frame"]
                        ) or observation.acquired_s != float(captured["time_s"]):
                            raise ValueError("unsynchronized_local_surface")
                        constraint = constrained_surface_pose(
                            candidate.surface,
                            current,
                            candidate.initial_world @ np.linalg.inv(candidate.map_from_object),
                            captured["camera_world"],
                            obj["source_points"][measured],
                            obj["current_points"][measured],
                            config["point_error_m"],
                            current_normal_points=normal_points,
                            map_anchor=(candidate.map_from_object @ candidate.object_from_zone)[
                                :3, 3
                            ],
                        )
                        local = self._constrained_candidate(candidate, constraint)
                        # Identity belongs to the original measured seed.
                        # A coverage atlas contains other partial views whose
                        # occlusion today cannot invalidate that seed's match.
                        if not self.root_consistent(
                            replace(local, surface=candidate.surface), captured
                        ):
                            raise ValueError("local_surface_material_inconsistent")
                        consistent.append((int(observation.mask().sum()), constraint))
                    except (ValueError, KeyError) as error:
                        failures.append(str(error))
                if not consistent:
                    raise ValueError(failures[0])
                areas = [area for area, _ in consistent]
                if areas.count(max(areas)) != 1:
                    raise ValueError("missing_or_ambiguous_current_surface")
                candidate.local_constraint = consistent[int(np.argmax(areas))][1]
                candidate.local_constraint_reason = ""
            except (ValueError, KeyError) as error:
                if (
                    candidate.state == "tracked"
                    and not candidate.material_recovery_unverified
                    and result.get("local_surfaces_dominant", False)
                    and result.get("cue") is not None
                ):
                    # Relief may hide the selected material from the first-plane
                    # query. Reuse every current measured component, retaining
                    # the same conditioning, identity and ambiguity checks.
                    self.diagnostics["local_surface_query_expansions"] = (
                        self.diagnostics.get("local_surface_query_expansions", 0) + 1
                    )
                    result["local_surfaces"] = surfaces(result["cue"], captured, self.config)
                    result["local_surfaces_dominant"] = False
                    result["local_surface_expansion_reason"] = str(error)
                    return self.constrain_local(captured, result)
                candidate.local_constraint_reason = str(error)
            self.diagnostics.setdefault("local_surface_constraint", {})[candidate.candidate_id] = {
                "available": candidate.local_constraint is not None,
                "reason": candidate.local_constraint_reason,
                "native_pose_unchanged": True,
                "reobserved_normal_points": 0 if normal_points is None else len(normal_points),
                "normal_rotation_bound_rad": None
                if candidate.local_constraint is None
                else candidate.local_constraint.normal_rotation_bound_rad,
            }

    @staticmethod
    def _constrained_candidate(candidate, constraint):
        return replace(
            candidate,
            pose=transported_zone(
                candidate.camera_world,
                constraint.camera_from_map,
                candidate.map_from_object,
                candidate.object_from_zone,
            ),
            registration_origin_map=constraint.registration_origin_map,
            registration_position_m=constraint.position_bound_m,
            registration_rotation_rad=constraint.rotation_bound_rad,
            surface=candidate.local_surface or candidate.surface,
        )

    def reobserved_material_points(self, candidate, sensor):
        """Fresh depth on the tracked seed; no other semantic component or old XYZ."""
        motion = candidate.pose @ np.linalg.inv(candidate.initial_world)
        predicted = candidate.surface.points @ motion[:3, :3].T + motion[:3, 3]
        uv, z = project(predicted, sensor["intrinsics"], sensor["camera_world"])
        good = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
        uv, z = np.rint(uv[good]).astype(int), z[good]
        h, w = sensor["rgb"].shape[:2]
        good = (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
        uv, z = uv[good], z[good]
        measured = sensor["depth_m"][uv[:, 1], uv[:, 0], 0]
        good = (
            sensor["valid_depth"][uv[:, 1], uv[:, 0], 0]
            & np.isfinite(measured)
            & (abs(measured - z) <= 2 * self.config["plane_tolerance_m"])
        )
        pixels = np.unique(uv[good], axis=0)
        return deproject(
            sensor["depth_m"][pixels[:, 1], pixels[:, 0], 0],
            pixels,
            sensor["intrinsics"],
            sensor["camera_world"],
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
        radius = None if faces is None else max(np.linalg.norm(f, axis=1).max() for f in faces)
        return self._point_budget(candidate, now, radius, tool_world)

    def frame_budget(self, candidate, now):
        """Dynamic frame-origin error; local geometry is published separately.

        Do not publish a complete fingertip displacement as frame translation
        and then charge the same geometry/rotation again in action admission.
        """
        complete = self._point_budget(candidate, now, 0.0, None)
        return ContactBudget(tuple(s for s in complete.sources if s.name != "local_geometry"))

    def point_budget_query(self, candidate, now):
        """Discard this per-admission query before candidate state changes."""
        inverses = {}

        def inverse(matrix):
            key = id(matrix)
            if key not in inverses:
                inverses[key] = np.linalg.inv(matrix)
            return inverses[key]

        def query(radius, tool_world):
            # Preserve the original finite-face radius arithmetic exactly.
            finger_r = np.linalg.norm(np.array([[radius, 0.0, 0.0]]), axis=1).max()
            return self._point_budget(candidate, now, finger_r, tool_world, _inverse=inverse)

        return query

    def _point_budget(self, candidate, now, finger_r, tool_world, *, _inverse=np.linalg.inv):
        # Each angular lever starts at its actual registration/calibration origin.
        registration_origin = candidate.registration_origin_map
        origin_o = (
            None
            if registration_origin is None
            else (_inverse(candidate.map_from_object) @ np.r_[registration_origin, 1.0])[:3]
        )
        tool_o = candidate.object_from_zone[:3, 3]
        local_r = finger_r
        if tool_world is not None and candidate.pose is not None:
            relative = _inverse(candidate.pose) @ pose_matrix(tool_world)
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
                np.linalg.norm((_inverse(candidate.camera_world) @ pose_matrix(tool_world))[:3, 3])
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
            constraint_reason = ""
            constraint = None
            if self.config.get("local_surface_constraint") is not None:
                constraint = candidate.local_constraint
                if constraint is None:
                    constraint_reason = candidate.local_constraint_reason
                else:
                    candidate = self._constrained_candidate(candidate, constraint)
            fresh = (
                not constraint_reason
                and candidate.state == "tracked"
                and candidate.support is not None
                and candidate.support.available_s <= now
                and 0 <= now - candidate.support.supported_s <= MAX_DYNAMIC_AGE_S + 1e-9
            )
            reason = (
                constraint_reason
                if constraint_reason
                else "material_recovery_unverified"
                if candidate.material_recovery_unverified
                else "ambiguous_leaf_ownership"
                if fresh and candidate.ownership == "ambiguous_leaf_candidate"
                else ""
            )
            support = candidate.support if fresh else None
            identity = (
                None
                if support is None or candidate.material_recovery_unverified
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
            budget = self.budget(
                candidate, now, tool_world=object_frame(candidate.pose) if fresh else None
            )
            frame_budget = self.frame_budget(candidate, now)
            pose_support = (
                None
                if support is None
                else FieldSupport(
                    support.acquired_s,
                    support.available_s,
                    support.supported_s,
                    self.generation,
                    frame_budget.displacement_m,
                    frame_budget.rotation_rad,
                )
            )
            clearance = (None, None)
            if fresh and self.distal_faces is not None and not reason:
                # This field describes the selected material zone, including
                # while the robot is parked away from it. The action producer
                # separately checks the actual tool footprints and full sweep.
                selected = query_slip(
                    candidate.surface,
                    candidate.initial_world,
                    candidate.pose,
                    object_frame(candidate.pose),
                    self.distal_faces,
                    budget,
                    self.config,
                )
                # Publish measured material geometry separately from dynamic
                # error. Admission charges the complete pose/calibration bounds
                # against the actual action's footprint and sweep exactly once.
                measured = footprint_support(
                    candidate.surface,
                    object_frame(candidate.initial_world),
                    self.distal_faces,
                    self.config,
                )
                clearance = measured.finger_clearance_m
                self.diagnostics["selected_material_zone"] = vars(selected)
                if tool is not None:
                    tool_budget = self.budget(candidate, now, tool_world=tool)
                    slip = query_slip(
                        candidate.surface,
                        candidate.initial_world,
                        candidate.pose,
                        tool,
                        self.distal_faces,
                        tool_budget,
                        self.config,
                    )
                    self.diagnostics["tool_relative_zone"] = vars(slip)
                    # Transported material geometry and direct pixel support are
                    # different statements; neither measures a hidden interface.
                    visible = contact_depth_supported(
                        sensor, candidate.pose, tool, self.distal_faces, tool_budget.displacement_m
                    )
                    self.diagnostics.setdefault("contact_visibility", {})[
                        candidate.candidate_id
                    ] = dict(
                        direct_surface_depth_supported=visible,
                        contact_directly_measured=False,
                        reason="current_depth_compatible"
                        if visible
                        else "interface_unobserved_or_occluded",
                        contact_admitted=False,
                    )
            normal_rotation_bound = None
            calibration_angle = (self.calibration_bound or (None, None))[1]
            angular_speed = (self.relative_speed_bound or (None, None))[1]
            age = None if support is None else max(0.0, now - support.supported_s)
            angular_growth = (
                0.0
                if age == 0
                else None
                if age is None or angular_speed is None
                else age * angular_speed
            )
            if (
                constraint is not None
                and fresh
                and calibration_angle is not None
                and angular_growth is not None
                and np.isfinite([calibration_angle, angular_growth]).all()
                and min(calibration_angle, angular_growth) >= 0
            ):
                normal_rotation_bound = (
                    constraint.normal_rotation_bound_rad
                    + self.config["local_surface_constraint"]["normal_model_bound_rad"]
                    + calibration_angle
                    + angular_growth
                )
            normal_position_bound = None
            calibration_position = (self.calibration_bound or (None, None))[0]
            linear_speed = (self.relative_speed_bound or (None, None))[0]
            linear_growth = (
                0.0
                if age == 0
                else None
                if age is None or linear_speed is None
                else age * linear_speed
            )
            if (
                normal_rotation_bound is not None
                and calibration_position is not None
                and linear_growth is not None
                and np.isfinite([calibration_position, linear_growth]).all()
                and min(calibration_position, linear_growth) >= 0
            ):
                camera_lever = np.linalg.norm(candidate.pose[:3, 3] - candidate.camera_world[:3, 3])
                normal_position_bound = (
                    constraint.normal_position_bound_m
                    + 2
                    * constraint.normal_anchor_lever_m
                    * np.sin(
                        min(
                            self.config["local_surface_constraint"]["normal_model_bound_rad"], np.pi
                        )
                        / 2
                    )
                    + calibration_position
                    + 2 * camera_lever * np.sin(min(calibration_angle, np.pi) / 2)
                    + linear_growth
                )
            normal_sources = ()
            if normal_position_bound is not None:
                normal_sources = (
                    (
                        "measured_plane",
                        constraint.normal_point_bound_m,
                        constraint.normal_rotation_bound_rad,
                        tuple(constraint.normal_origin_world.tolist()),
                    ),
                    (
                        "normal_model",
                        0.0,
                        self.config["local_surface_constraint"]["normal_model_bound_rad"],
                        tuple(constraint.normal_origin_world.tolist()),
                    ),
                    (
                        "calibration_fk",
                        calibration_position,
                        calibration_angle,
                        tuple(candidate.camera_world[:3, 3].tolist()),
                    ),
                    (
                        "temporal_growth",
                        linear_growth,
                        angular_growth,
                        tuple(candidate.pose[:3, 3]),
                    ),
                )
            pose_sources = ()
            if (
                fresh
                and candidate.registration_origin_map is not None
                and candidate.camera_world is not None
            ):
                registration_world = (
                    candidate.pose
                    @ np.linalg.inv(candidate.map_from_object @ candidate.object_from_zone)
                    @ np.r_[candidate.registration_origin_map, 1.0]
                )[:3]
                pose_sources = tuple(
                    (
                        s.name,
                        s.position_m,
                        s.rotation_rad,
                        s.lever_m,
                        tuple(candidate.camera_world[:3, 3].tolist())
                        if s.name == "calibration_fk"
                        else tuple(registration_world.tolist()),
                    )
                    for s in frame_budget.sources
                )
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
                    tracking_state="tracking" if fresh else "unavailable",
                    ownership=candidate.ownership,
                    normal_rotation_bound_rad=normal_rotation_bound,
                    normal_position_bound_m=normal_position_bound,
                    normal_uncertainty_sources=normal_sources,
                    pose_temporal_position_m=next(
                        (s.position_m for s in frame_budget.sources if s.name == "temporal_growth"),
                        None,
                    ),
                    pose_uncertainty_sources=pose_sources,
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
