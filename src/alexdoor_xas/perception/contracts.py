"""Observed door geometry in world coordinates; independent of any estimator."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame

LEGACY_FULL_STATE = "legacy-full-state"
OPERATIONAL_V1 = "operational-v1"
MAX_DYNAMIC_AGE_S = 0.15
POSITION_LIMIT_M = 0.01
ROTATION_LIMIT_RAD = np.deg2rad(5.0)
TIME_ROUNDOFF_S = 1e-9  # nanosecond clock resolution; no control-tick freshness grace
POSE_ROUNDOFF = 32 * np.finfo(np.float32).eps  # accumulated runtime tensor roundoff


def geometry_profile(binding):
    profile = binding.config.get("geometry_profile", LEGACY_FULL_STATE)
    if profile not in (LEGACY_FULL_STATE, OPERATIONAL_V1):
        raise ValueError("Unknown geometry profile")
    return profile


@dataclass(frozen=True)
class FieldSupport:
    """Observed support; None bounds are unmeasured, never confidence-derived.

    Bounds are absolute conservative errors in meters/radians. Static support
    has no age-only expiry. Prediction cannot advance supported_s.
    """

    acquired_s: float
    available_s: float
    supported_s: float
    generation: int
    position_bound_m: float | None = None
    rotation_bound_rad: float | None = None
    reason: str = ""

    def require(self, now, generation, *, dynamic=False, max_age=MAX_DYNAMIC_AGE_S):
        times = [self.acquired_s, self.supported_s, self.available_s, now]
        if (
            self.reason
            or type(self.generation) is not int
            or self.generation < 0
            or self.generation != generation
            or not np.isfinite(times).all()
            or not 0 <= self.acquired_s <= self.supported_s <= self.available_s <= now
        ):
            raise ValueError(self.reason or "unavailable_field_support")
        if dynamic and not 0 <= now - self.supported_s <= (
            min(max_age, MAX_DYNAMIC_AGE_S) + TIME_ROUNDOFF_S
        ):
            raise ValueError("stale_dynamic_support")

    def require_bounds(self, *, position=False, rotation=False, qualification=False):
        for needed, value, limit in (
            (position, self.position_bound_m, POSITION_LIMIT_M),
            (rotation, self.rotation_bound_rad, ROTATION_LIMIT_RAD),
        ):
            if needed and (
                value is None
                or not np.isfinite(value)
                or value < 0
                or qualification
                and value > limit + POSE_ROUNDOFF
            ):
                raise ValueError("missing_or_excessive_field_bound")


@dataclass(frozen=True)
class HingeHypothesis:
    hypothesis_id: str
    frame: ObjectFrame  # closed orientation, z axis and calibrated floor intersection
    floor_z_m: float
    support: FieldSupport


@dataclass(frozen=True)
class ContactSelection:
    """Observed material patch; evaluator correspondence is deliberately absent.

    local_pose uses the reference hypothesis's current panel frame. A changed
    target needs a new selection_id and an explicit predecessor_id.
    """

    selection_id: str
    patch_id: str
    selected_s: float
    local_pose: ObjectFrame
    world_pose: ObjectFrame
    local_support: FieldSupport
    world_support: FieldSupport
    predecessor_id: str | None = None


@dataclass(frozen=True)
class OperationalState:
    leaf_id: str  # estimator-owned identity, never a prepared asset ID
    generation: int
    tracking_state: str
    identity_support: FieldSupport
    hypotheses: tuple[HingeHypothesis, ...]
    reference_id: str | None
    angle_support: FieldSupport
    panel_support: FieldSupport
    contact: ContactSelection | None
    dimensions_support: FieldSupport | None = None


@dataclass(frozen=True)
class LocalPatchState:
    """A material reference is independent of both a hinge and a chosen contact."""

    patch_id: str
    world_pose: ObjectFrame | None
    geometry_support: FieldSupport | None
    identity_support: FieldSupport | None
    pose_support: FieldSupport | None
    finger_clearance_m: tuple[float | None, float | None] = (None, None)
    reason: str = "acquiring_material"


@dataclass(frozen=True)
class LocalContactSelection:
    selection_id: str
    patch_id: str
    selected_s: float
    source: str
    predecessor_id: str | None = None


@dataclass(frozen=True)
class LocalMaterialState:
    generation: int
    reference_id: str | None
    patches: tuple[LocalPatchState, ...]
    selection: LocalContactSelection | None = None
    hypotheses: tuple[HingeHypothesis, ...] = ()
    signed_angle: float | None = None
    angle_support: FieldSupport | None = None


def validate_local_contact(state, now, *, generation=None):
    """Local geometric support only; never action/load admission or qualification."""
    from alexdoor_xas.action.b1 import checked_pose

    generation = state.generation if generation is None else generation
    if type(state.generation) is not int or state.generation < 0 or state.generation != generation:
        raise ValueError("wrong_episode_generation")
    selection = state.selection
    if selection is None or selection.source not in ("diagnostic", "policy"):
        raise ValueError("missing_explicit_contact_selection")
    if (
        not selection.selection_id
        or not np.isfinite(selection.selected_s)
        or not (0 <= selection.selected_s <= now)
    ):
        raise ValueError("unavailable_contact_selection")
    patches = [p for p in state.patches if p.patch_id == selection.patch_id]
    if len(patches) != 1:
        raise ValueError("missing_or_ambiguous_selected_patch")
    patch = patches[0]
    for support, dynamic in (
        (patch.geometry_support, False),
        (patch.identity_support, True),
        (patch.pose_support, True),
    ):
        if support is None:
            raise ValueError(patch.reason or "missing_material_support")
        support.require(now, generation, dynamic=dynamic)
    patch.geometry_support.require_bounds(position=True, rotation=True)
    patch.pose_support.require_bounds(position=True, rotation=True)
    checked_pose(patch.world_pose)
    if any(c is None or not np.isfinite(c) or c <= 0 for c in patch.finger_clearance_m):
        raise ValueError("insufficient_observed_footprint")
    return patch


def validate_local_transition(previous, current):
    if previous is None:
        if current.predecessor_id is not None:
            raise ValueError("unexpected_contact_predecessor")
    elif current.selection_id == previous.selection_id:
        if current != previous:
            raise ValueError("contact_reselected_without_transition")
    elif (
        current.predecessor_id != previous.selection_id or current.selected_s <= previous.selected_s
    ):
        raise ValueError("missing_contact_transition")


@dataclass(frozen=True)
class GeometryAdmission:
    qualified: bool
    reason: str


@dataclass(frozen=True)
class RobotFeedback:
    """Separate monitor channel; reported actuator signal is not external force.

    Positive torque follows the named joint's positive URDF coordinate. The
    producer must declare origin/semantics; no command-to-measurement conversion.
    """

    timestamp_s: float
    available_s: float
    joint_names: tuple[str, ...]
    torque_nm: np.ndarray | None
    source: str
    semantics: str
    healthy: bool
    reason: str = ""
    units: str = "N m"
    sign_convention: str = "positive_joint_coordinate"

    @classmethod
    def unavailable(cls, time_s, joint_names, reason="force_feedback_unavailable"):
        return cls(
            time_s, time_s, tuple(joint_names), None, "unavailable", "unknown", False, reason
        )


@dataclass(frozen=True)
class DoorEstimate:
    timestamp_s: float
    valid: bool
    reason: str
    confidence: float = 0.0
    frame: ObjectFrame | None = None
    panel_rotation: np.ndarray | None = None
    signed_angle: float | None = None
    dimensions: np.ndarray | None = None
    contact_position: np.ndarray | None = None
    contact_rotation: np.ndarray | None = None
    operational: OperationalState | None = None
    local: LocalMaterialState | None = None

    def fresh(self, now_s, max_age_s=0.15):
        return self.valid and 0 <= now_s - self.timestamp_s <= max_age_s + TIME_ROUNDOFF_S


def validate_complete(estimate):
    """Reject incomplete or inconsistent geometry before a controller consumes it."""
    from alexdoor_xas.action.b1 import checked_pose, finite_vector, panel_pose

    checked_pose(estimate.frame)
    panel = panel_pose(estimate.frame, estimate.signed_angle)
    rotation = np.asarray(estimate.panel_rotation)
    if rotation.shape != (3, 3) or not np.allclose(rotation, panel.rot, atol=1e-5):
        raise ValueError("Inconsistent observed panel rotation")
    if np.any(finite_vector(estimate.dimensions, 3) <= 0):
        raise ValueError("Nonpositive observed dimensions")
    checked_pose(ObjectFrame(estimate.contact_position, estimate.contact_rotation))
    if not np.isfinite(estimate.confidence) or not 0 <= estimate.confidence <= 1:
        raise ValueError("Invalid observed confidence")


def same_pose(a, b):
    from alexdoor_xas.action.b1 import checked_pose

    checked_pose(a)
    checked_pose(b)
    return np.allclose(a.origin, b.origin, atol=1e-8, rtol=0) and np.allclose(
        a.rot, b.rot, atol=1e-8, rtol=0
    )


def validate_reference(estimate, now, max_age=MAX_DYNAMIC_AGE_S, *, generation=None):
    """Validate observed support even with multiple credible hinge hypotheses."""
    from alexdoor_xas.action.b1 import checked_pose, panel_pose

    state = estimate.operational
    if state is None:
        raise ValueError("missing_operational_state")
    generation = state.generation if generation is None else generation
    if type(state.generation) is not int or state.generation < 0 or state.generation != generation:
        raise ValueError("wrong_episode_generation")
    if not np.isfinite(max_age) or max_age <= 0:
        raise ValueError("invalid_dynamic_timeout")
    if not state.leaf_id or state.tracking_state not in ("tracking", "ambiguous"):
        raise ValueError("unresolved_leaf_identity")
    state.identity_support.require(now, generation, dynamic=True, max_age=max_age)
    hypotheses = state.hypotheses
    ids = [h.hypothesis_id for h in hypotheses]
    if not ids or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid_hinge_hypotheses")
    for hypothesis in hypotheses:
        checked_pose(hypothesis.frame)
        hypothesis.support.require(now, generation)
        hypothesis.support.require_bounds(position=True, rotation=True)
        if not np.isfinite(hypothesis.floor_z_m) or not np.isclose(
            hypothesis.frame.origin[2], hypothesis.floor_z_m, atol=1e-8, rtol=0
        ):
            raise ValueError("unsupported_floor_intersection")
    reference = next((h for h in hypotheses if h.hypothesis_id == state.reference_id), None)
    if reference is None:
        raise ValueError("missing_supported_reference")
    if not same_pose(estimate.frame, reference.frame):
        raise ValueError("inconsistent_reference_frame")
    state.angle_support.require(now, generation, dynamic=True, max_age=max_age)
    state.angle_support.require_bounds(rotation=True)
    state.panel_support.require(now, generation, dynamic=True, max_age=max_age)
    state.panel_support.require_bounds(rotation=True)
    panel = panel_pose(reference.frame, estimate.signed_angle)
    if np.asarray(estimate.panel_rotation).shape != (3, 3) or not np.allclose(
        estimate.panel_rotation, panel.rot, atol=1e-8, rtol=0
    ):
        raise ValueError("inconsistent_panel_rotation")
    contact = state.contact
    if contact is None or not contact.selection_id or not contact.patch_id:
        raise ValueError("missing_observed_contact")
    if not np.isfinite(contact.selected_s) or not 0 <= contact.selected_s <= now:
        raise ValueError("unavailable_contact_selection")
    for support, dynamic in ((contact.local_support, False), (contact.world_support, True)):
        support.require(now, generation, dynamic=dynamic, max_age=max_age)
        support.require_bounds(position=True, rotation=True)
    checked_pose(contact.local_pose)
    checked_pose(contact.world_pose)
    expected = ObjectFrame(
        panel.point_to_world(contact.local_pose.origin), panel.rot @ contact.local_pose.rot
    )
    if not same_pose(expected, contact.world_pose) or not same_pose(
        contact.world_pose, ObjectFrame(estimate.contact_position, estimate.contact_rotation)
    ):
        raise ValueError("inconsistent_contact_correspondence")
    return reference


def validate_geometry(estimate, now, max_age, *, profile=LEGACY_FULL_STATE, generation=None):
    if profile == LEGACY_FULL_STATE:
        if not estimate.fresh(now, max_age):
            raise ValueError(f"Unavailable/stale observed geometry: {estimate.reason}")
        validate_complete(estimate)
    elif profile == OPERATIONAL_V1:
        validate_reference(estimate, now, max_age, generation=generation)
        if len(estimate.operational.hypotheses) != 1:
            raise ValueError("ambiguous_hinge")
        if estimate.operational.tracking_state != "tracking":
            raise ValueError("provisional_geometry")
        state = estimate.operational
        state.hypotheses[0].support.require_bounds(position=True, rotation=True, qualification=True)
        for support in (state.angle_support, state.panel_support):
            support.require_bounds(rotation=True, qualification=True)
        for support in (state.contact.local_support, state.contact.world_support):
            support.require_bounds(position=True, rotation=True, qualification=True)
    else:
        raise ValueError("Unknown geometry profile")


def geometric_admission(estimate, now, max_age=MAX_DYNAMIC_AGE_S, **kwargs):
    """Support check; independent offline/dynamic evidence still qualifies a release."""
    try:
        validate_geometry(estimate, now, max_age, **kwargs)
    except (ValueError, TypeError, AttributeError) as error:
        return GeometryAdmission(False, str(error))
    return GeometryAdmission(True, "supported_geometry")


def validate_contact_transition(previous, current):
    """A new material target must declare its predecessor; tracking does not reselect it."""
    if previous is None:
        if current.predecessor_id is not None:
            raise ValueError("unexpected_contact_predecessor")
    elif current.selection_id == previous.selection_id:
        if (current.patch_id, current.selected_s, current.predecessor_id) != (
            previous.patch_id,
            previous.selected_s,
            previous.predecessor_id,
        ):
            raise ValueError("contact_reselected_without_transition")
    elif (
        current.predecessor_id != previous.selection_id or current.selected_s <= previous.selected_s
    ):
        raise ValueError("missing_contact_transition")
