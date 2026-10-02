"""Axis-free observed material tracks; explicit selection never chooses a better contact."""

from dataclasses import replace

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import (
    ContactSelection,
    FieldSupport,
    HingeHypothesis,
    LocalMaterialState,
    LocalPatchState,
    validate_local_transition,
)
from alexdoor_xas.perception.geometry import (
    contact_frame,
    deproject,
    hinge_from_motion,
    project,
    rigid_fit,
)
from alexdoor_xas.perception.scan import footprint_support
from alexdoor_xas.perception.tracking import PixelMotionTracker, sample_points


def local_mask(sensor, pose, radius, tolerance):
    """Measured neighborhood only; a plane elsewhere cannot supply this patch."""
    depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
    corners = np.array([[0, y, z] for y in (-radius, radius) for z in (-radius, radius)])
    pixels, optical = project(
        corners @ pose.rot.T + pose.origin, sensor["intrinsics"], sensor["camera_world"]
    )
    mask = np.zeros(depth.shape, bool)
    if not np.isfinite(pixels).all() or (optical <= 0).any():
        return mask
    lo = np.maximum(np.floor(pixels.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(pixels.max(0)).astype(int) + 3, depth.shape[::-1])
    mask[lo[1] : hi[1], lo[0] : hi[0]] = True
    v, u = np.nonzero(mask & valid & np.isfinite(depth) & (depth > 0))
    points = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
    local = (points - pose.origin) @ pose.rot
    keep = (abs(local[:, 0]) < tolerance) & (abs(local[:, 1:]) < radius).all(1)
    mask[:] = False
    mask[v[keep], u[keep]] = True
    return mask


def motion_bounds(source, residual, depth_error):
    """Rigid-fit sensitivity, not a confidence score or an empirical residual alone."""
    singular = np.linalg.svd(source - source.mean(0), compute_uv=False)
    span = singular[1] / np.sqrt(len(source)) if len(singular) > 1 else 0
    position = 2 * depth_error + residual
    rotation = 2 * np.arcsin(min(1.0, position / max(span, 1e-12)))
    return float(position), float(rotation)


class MaterialTrack:
    def __init__(self, patch_id, hint, config, generation, covers):
        self.patch_id, self.hint = patch_id, np.asarray(hint, float)
        if self.hint.shape != (3,) or not np.isfinite(self.hint).all():
            raise ValueError("Invalid diagnostic patch hint")
        self.config, self.generation, self.covers = config, generation, covers
        self.tracker = PixelMotionTracker()
        self.reference = self.closed_pose = self.pose = None
        self.anchor = None
        self.geometry = self.identity = self.dynamic = None
        self.clearance = (None, None)
        self.motions = []
        self.reason = "acquiring_material"
        self.last_observed_frame = None
        self.losses = self.recoveries = 0
        self.observed = False
        self.motion = None
        self.orb_reference = None
        self.next_reacquisition_s = 0.0

    def observe(self, surface, sensor, available_s):
        """Initialize from an available local measured face, not retrospective fusion."""
        if self.reference is not None:
            return
        normal = surface.normal.copy()
        if normal[0] < 0:
            normal *= -1
        rotation = contact_frame(normal)
        origin = self.hint.copy()
        if abs(normal[0]) < 0.5:
            return
        # Only tangent hint coordinates are supplied; normal position comes from current depth.
        offset = surface.offset * (1 if surface.normal @ normal > 0 else -1)
        origin[0] = (offset - origin[1:] @ normal[1:]) / normal[0]
        pose = ObjectFrame(origin, rotation)
        footprint = footprint_support(surface, pose, self.covers, self.config)
        if not footprint.supported:
            return
        radius = self.config.get("material_radius_m", 0.08)
        points = surface.points
        local = (points - origin) @ rotation
        keep = (abs(local[:, 1:]) < radius).all(1)
        if keep.sum() < 12:
            return
        self.reference = replace(
            surface,
            points=points[keep],
            extent_points=points[keep],
            anchors=np.empty((0, 3)),
            features=surface.features[:0],
        )
        self.closed_pose = self.pose = pose
        self.clearance = footprint.finger_clearance_m
        acquired = float(sensor["time_s"])
        error = self.config["plane_tolerance_m"] + surface.residual_m
        span = np.ptp(local[keep, 1:], axis=0).min()
        self.geometry = FieldSupport(
            acquired,
            available_s,
            acquired,
            self.generation,
            error,
            float(np.arctan2(2 * error, max(span, 1e-9))),
        )
        self.reason = "awaiting_current_material_matches"

    def _remember(self):
        self.anchor = tuple(
            np.array(v, copy=True)
            for v in (
                self.tracker.gray,
                self.tracker.pixels,
                self.tracker.source,
                self.tracker.source_uncertainty,
            )
        )
        import cv2

        orb = cv2.ORB_create(nfeatures=500, edgeThreshold=8, fastThreshold=10, patchSize=15)
        mask = local_mask(
            self.seed_sensor,
            self.closed_pose,
            self.config.get("material_radius_m", 0.08),
            2 * self.config["plane_tolerance_m"],
        ).astype(np.uint8)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8)) * 255
        keys, descriptors = orb.detectAndCompute(self.tracker.gray, mask)
        self.orb_reference = (keys, descriptors)

    def membership_mask(self, sensor, motion=None):
        """Current measured points must map into the original observed material raster."""
        r, t = (np.eye(3), np.zeros(3)) if motion is None else motion[:2]
        pose = ObjectFrame(r @ self.closed_pose.origin + t, r @ self.closed_pose.rot)
        mask = local_mask(
            sensor,
            pose,
            self.config.get("material_radius_m", 0.08),
            2 * self.config["plane_tolerance_m"],
        )
        v, u = np.nonzero(mask)
        source = (
            deproject(
                sensor["depth_m"][v, u, 0],
                np.c_[u, v],
                sensor["intrinsics"],
                sensor["camera_world"],
            )
            - t
        ) @ r
        verified = np.zeros(len(source), bool)
        for observation in self.reference.observations:
            pixels, z = project(source, observation.intrinsics, observation.camera_world)
            xy = np.rint(pixels).astype(int)
            h, w = observation.shape
            inside = (z > 0) & (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
            verified[inside] |= observation.mask()[xy[inside, 1], xy[inside, 0]]
        mask[v[~verified], u[~verified]] = False
        return mask

    def _reacquire(self, sensor):
        """Mutual distinctive RGB matches to immutable material, verified by metric rigidity."""
        import cv2

        keys, descriptors = self.orb_reference
        if descriptors is None or len(keys) < 12:
            return False
        gray = cv2.cvtColor(sensor["rgb"], cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=8, fastThreshold=10, patchSize=15)
        current, desc = orb.detectAndCompute(gray, None)
        if desc is None or len(current) < 12:
            return False
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        forward = matcher.knnMatch(descriptors, desc, k=2)
        backward = matcher.match(desc, descriptors)
        matches = [
            a
            for pair in forward
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < 0.7 * b.distance and backward[a.trainIdx].trainIdx == a.queryIdx
        ]
        if len(matches) < 12:
            return False
        original = np.array([keys[m.queryIdx].pt for m in matches])
        following = np.array([current[m.trainIdx].pt for m in matches])
        # Original keypoints get material coordinates from the originally captured depth.
        indices, source = sample_points(self.seed_sensor, original)
        following = following[indices]
        indices, target = sample_points(sensor, following)
        source, following = source[indices], following[indices]
        if len(source) < 12:
            return False
        best = None
        rng = np.random.default_rng(int(sensor["frame"]))
        for _ in range(32):
            take = rng.choice(len(source), 4, replace=False)
            fitted = rigid_fit(source[take], target[take])
            if fitted is None:
                continue
            r, t, _ = fitted
            keep = np.linalg.norm(source @ r.T + t - target, axis=1) < 0.004
            if keep.sum() >= 12 and (best is None or keep.sum() > best.sum()):
                best = keep
        if best is None or best.mean() < 0.7:
            return False
        fit = rigid_fit(source[best], target[best])
        if fit is None or fit[2] > 0.004:
            return False
        self.tracker.gray, self.tracker.pixels, self.tracker.source = (
            gray,
            following[best],
            source[best],
        )
        self.tracker.source_uncertainty = np.zeros(best.sum())
        self.tracker.diagnostics.update(
            pixel_motion="reacquired_original_material", tracked_pixels=int(best.sum())
        )
        self.motion = fit
        return True

    def update(self, sensor):
        if self.reference is None:
            return
        if self.geometry.available_s > float(sensor["time_s"]):
            self.reason = "awaiting_geometry_completion"
            return
        previous_motion = self.motion
        self.motion = None
        acquired = float(sensor["time_s"])
        was_observed = self.observed
        if self.tracker.gray is None and self.anchor is not None:
            matched = False
            if acquired >= self.next_reacquisition_s:
                self.next_reacquisition_s = acquired + self.config["semantic_period_s"]
                matched = self._reacquire(sensor)
            motion = self.motion if matched else None
        else:
            initial = self.anchor is None
            mask = self.membership_mask(sensor, previous_motion)
            motion = self.tracker.update(
                sensor, self.reference, material_mask=mask, allow_reseed=True
            )
            if initial and self.tracker.gray is not None:
                self.seed_sensor = {k: np.array(v, copy=True) for k, v in sensor.items()}
                self._remember()
        self.observed = motion is not None
        if motion is None:
            if was_observed:
                self.losses += 1
            self.reason = (
                self.tracker.diagnostics["pixel_motion"]
                if self.anchor is None
                else ("lost_material_matches")
            )
            return
        r, t, residual = motion
        source = self.tracker.source
        position, rotation = motion_bounds(source, residual, self.config["plane_tolerance_m"])
        self.pose = ObjectFrame(r @ self.closed_pose.origin + t, r @ self.closed_pose.rot)
        self.motion = motion
        support = FieldSupport(
            self.geometry.acquired_s,
            acquired,
            acquired,
            self.generation,
            position + self.geometry.position_bound_m,
            rotation + self.geometry.rotation_bound_rad,
        )
        self.identity = self.dynamic = support
        self.reason = "tracked_material"
        if not was_observed and self.last_observed_frame is not None:
            self.recoveries += 1
        self.last_observed_frame = int(sensor["frame"])
        self.motions = (self.motions + [(r, t, position)])[-100:]

    def state(self):
        return LocalPatchState(
            self.patch_id,
            self.pose,
            self.geometry,
            self.identity,
            self.dynamic,
            self.clearance,
            self.reason,
        )

    def invalidate(self, reason):
        self.tracker.reset()
        self.observed = False
        self.reason = reason


class MaterialTracker:
    def __init__(self, config, generation, covers):
        self.config, self.generation, self.covers = config, generation, covers
        self.tracks = {}
        self.reference_id = None
        self.selection = None
        self.rigid_links = set()
        self.link_evidence = {}
        self.hypotheses = ()
        self.angle = self.angle_support = None

    def request(self, patch_id, tangent_hint, *, reference=False):
        if not patch_id or patch_id in self.tracks:
            raise ValueError("Duplicate or missing material patch ID")
        self.tracks[patch_id] = MaterialTrack(
            patch_id, tangent_hint, self.config, self.generation, self.covers
        )
        if reference:
            self.reference_id = patch_id

    def select(self, selection, now):
        if selection.patch_id not in self.tracks or selection.source not in (
            "diagnostic",
            "policy",
        ):
            raise ValueError("Unknown patch or action source")
        if (
            not selection.selection_id
            or not np.isfinite(selection.selected_s)
            or not (0 <= selection.selected_s <= now)
        ):
            raise ValueError("unavailable_contact_selection")
        track = self.tracks[selection.patch_id]
        if track.geometry is None:
            raise ValueError("unobserved_selected_patch")
        track.geometry.require(now, self.generation)
        validate_local_transition(self.selection, selection)
        self.selection = selection

    def observe(self, candidates, sensor, available_s):
        for track in self.tracks.values():
            if track.reference is not None:
                continue
            matching = []
            for surface in candidates:
                normal = surface.normal
                if abs(normal[0]) < 0.5:
                    continue
                position = track.hint.copy()
                position[0] = (surface.offset - position[1:] @ normal[1:]) / normal[0]
                pixels, depth = project(
                    position[None], sensor["intrinsics"], sensor["camera_world"]
                )
                u, v = np.rint(pixels[0]).astype(int)
                h, w = sensor["depth_m"].shape[:2]
                if (
                    depth[0] > 0
                    and 0 <= u < w
                    and 0 <= v < h
                    and any(o.mask()[v, u] for o in surface.observations)
                ):
                    matching.append(surface)
            # Overlapping proposals of the same physical plane do not establish a distinct identity.
            equivalent = matching and all(
                abs(abs(s.normal @ matching[0].normal) - 1) < 1e-4
                and abs(s.offset * np.sign(s.normal @ matching[0].normal) - matching[0].offset)
                <= self.config["plane_tolerance_m"]
                for s in matching
            )
            if equivalent:
                track.observe(matching[0], sensor, available_s)

    def update(self, sensor):
        for track in self.tracks.values():
            track.update(sensor)
        now = float(sensor["time_s"])
        tracks = list(self.tracks.values())
        for i, first in enumerate(tracks):
            for second in tracks[i + 1 :]:
                key = tuple(sorted((first.patch_id, second.patch_id)))
                if not first.observed or not second.observed:
                    continue
                a, b = first.motion, second.motion
                angle = abs(np.arctan2(a[0][1, 0], a[0][0, 0]))
                # Static coplanarity is not proof of mobile ownership. Require informative motion.
                coherent = angle >= np.deg2rad(self.config["motion_min_angle_deg"])
                coherent &= (
                    angle > first.dynamic.rotation_bound_rad + second.dynamic.rotation_bound_rad
                )
                coherent &= np.linalg.norm(a[0] - b[0]) <= 0.01
                coherent &= np.linalg.norm(a[1] - b[1]) <= 0.01
                count = self.link_evidence.get(key, 0)
                self.link_evidence[key] = count + 1 if coherent else 0
                if coherent and self.link_evidence[key] >= 3:
                    self.rigid_links.add(key)
                elif not coherent and angle >= np.deg2rad(self.config["motion_min_angle_deg"]):
                    self.rigid_links.discard(key)
        # A verified link can update pose, never the original geometric observation.
        for target in tracks:
            if target.observed or target.geometry is None:
                continue
            donors = [
                d
                for d in tracks
                if d.observed and tuple(sorted((d.patch_id, target.patch_id))) in self.rigid_links
            ]
            if len(donors) != 1:
                continue
            donor = donors[0]
            r, t, _ = donor.motion
            target.pose = ObjectFrame(r @ target.closed_pose.origin + t, r @ target.closed_pose.rot)
            lever = np.linalg.norm(target.closed_pose.origin - donor.closed_pose.origin)
            extra = 2 * lever * np.sin(min(np.pi, donor.dynamic.rotation_bound_rad) / 2)
            target.identity = donor.identity
            target.dynamic = replace(
                donor.dynamic,
                acquired_s=target.geometry.acquired_s,
                position_bound_m=donor.dynamic.position_bound_m + extra,
            )
            target.reason = "verified_rigid_transfer"
        return self.state(now)

    def state(self, now):
        hypotheses, angle, angle_support = self.hypotheses, self.angle, self.angle_support
        reference = self.tracks.get(self.reference_id)
        if reference is not None and reference.observed:
            r = reference.motion[0]
            angle = float(np.arctan2(r[1, 0], r[0, 0]))
            angle_support = reference.dynamic
            fitted = hinge_from_motion(
                reference.motions,
                np.deg2rad(self.config["motion_min_angle_deg"]),
                self.config["floor_z_m"],
            )
            if fitted is not None:
                origin, bound = fitted
                support = replace(
                    reference.dynamic,
                    position_bound_m=bound,
                    rotation_bound_rad=reference.dynamic.rotation_bound_rad,
                )
                hypotheses = (
                    HingeHypothesis(
                        "motion-axis",
                        ObjectFrame(origin, reference.reference.basis),
                        self.config["floor_z_m"],
                        support,
                    ),
                )
                if self.hypotheses:
                    previous = self.hypotheses[0]
                    if np.linalg.norm(origin - previous.frame.origin) > (
                        bound + previous.support.position_bound_m
                    ):
                        compatible = any(
                            np.linalg.norm(origin - h.frame.origin)
                            <= bound + h.support.position_bound_m
                            for h in self.hypotheses
                        )
                        hypotheses = (
                            self.hypotheses
                            if compatible
                            else self.hypotheses
                            + (
                                replace(
                                    hypotheses[0],
                                    hypothesis_id=f"motion-axis-{len(self.hypotheses)}",
                                ),
                            )
                        )
                    elif len(self.hypotheses) > 1:
                        hypotheses = self.hypotheses
            self.hypotheses = hypotheses
            self.angle, self.angle_support = angle, angle_support
        return LocalMaterialState(
            self.generation,
            self.reference_id,
            tuple(t.state() for t in self.tracks.values()),
            self.selection,
            hypotheses,
            angle,
            angle_support,
        )

    def invalidate(self, reason):
        for track in self.tracks.values():
            track.invalidate(reason)
        self.rigid_links.clear()
        self.link_evidence.clear()

    def operational_contact(self, state, panel, now):
        """Represent the exact explicit material selection in a supported panel frame."""
        from alexdoor_xas.perception.contracts import validate_local_contact

        patch = validate_local_contact(state, now)
        local = ObjectFrame(
            panel.rot.T @ (patch.world_pose.origin - panel.origin),
            panel.rot.T @ patch.world_pose.rot,
        )
        selected = state.selection
        return ContactSelection(
            selected.selection_id,
            selected.patch_id,
            selected.selected_s,
            local,
            patch.world_pose,
            patch.geometry_support,
            patch.pose_support,
            selected.predecessor_id,
        )
