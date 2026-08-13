"""Pure contracts and measurements for Phase 4.1 door qualification."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SCHEMA = "alexdoor.phase4_1_assets.v1"
ACCEPTED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
HANDEDNESSES = {"left", "right"}
SOURCE_FORMAT_PRIORITY = ("usd", "usdz", "glb", "gltf", "blend", "fbx", "obj")
MIN_WIDTH_M = 0.65
MAX_WIDTH_M = 1.20
MIN_HEIGHT_M = 1.80
MAX_HEIGHT_M = 2.40
MIN_THICKNESS_M = 0.025
MAX_THICKNESS_M = 0.10
MAX_TRIANGLES = 250_000
MAX_SOURCE_VERTICES = 1_000_000
MAX_CONNECTED_COMPONENTS = 512
MAX_TEXTURE_EDGE_PX = 4096
PANEL_MASS_KG = 25.0
HINGE_DAMPING_NM_S_RAD = 4.0
HINGE_LIMIT_DEG = (0.0, 90.0)
FRICTION = 0.5
RESTITUTION = 0.0
SUSTAIN_TICKS = 30
OPEN_ANGLE_DEG = 45.0
REPEAT_SUSTAINED_TOL_DEG = 2.0
REPEAT_CURVE_TOL_DEG = 3.0
FORCE_LIMIT_N = 200.0


class QualificationError(ValueError):
    """A candidate, result, or manifest violates the frozen Phase 4.1 contract."""


@dataclass(frozen=True)
class DoorDimensions:
    """Canonical panel dimensions in meters."""

    width_m: float
    height_m: float
    thickness_m: float

    def validate(self) -> None:
        values = (self.width_m, self.height_m, self.thickness_m)
        if not all(math.isfinite(value) for value in values):
            raise QualificationError("door dimensions must be finite")
        if not MIN_WIDTH_M <= self.width_m <= MAX_WIDTH_M:
            raise QualificationError(f"door width outside [{MIN_WIDTH_M}, {MAX_WIDTH_M}] m")
        if not MIN_HEIGHT_M <= self.height_m <= MAX_HEIGHT_M:
            raise QualificationError(f"door height outside [{MIN_HEIGHT_M}, {MAX_HEIGHT_M}] m")
        if not MIN_THICKNESS_M <= self.thickness_m <= MAX_THICKNESS_M:
            raise QualificationError(
                f"door thickness outside [{MIN_THICKNESS_M}, {MAX_THICKNESS_M}] m"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DoorDimensions:
        result = cls(
            width_m=float(value["width_m"]),
            height_m=float(value["height_m"]),
            thickness_m=float(value["thickness_m"]),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, float]:
        return {
            "width_m": self.width_m,
            "height_m": self.height_m,
            "thickness_m": self.thickness_m,
        }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable byte representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cuboid_inertia_kg_m2(
    dimensions: DoorDimensions, mass_kg: float = PANEL_MASS_KG
) -> tuple[float, float, float]:
    """Return diagonal inertia for X=thickness, Y=width, Z=height."""
    dimensions.validate()
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise QualificationError("mass must be finite and positive")
    x = dimensions.thickness_m
    y = dimensions.width_m
    z = dimensions.height_m
    return (
        mass_kg * (y * y + z * z) / 12.0,
        mass_kg * (x * x + z * z) / 12.0,
        mass_kg * (x * x + y * y) / 12.0,
    )


def geometry_fingerprint(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Fingerprint triangle geometry independently of materials and rigid/uniform transforms.

    Sorted, scale-normalized triangle edge lengths and areas make the digest invariant
    to vertex order, translation, rotation, reflection, and uniform scale.  Reflection
    invariance intentionally makes mirrored copies collide with their source identity.
    """
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise QualificationError("vertices must have shape (N, 3), N >= 3")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
        raise QualificationError("faces must have shape (M, 3), M >= 1")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise QualificationError("face indices are outside the vertex array")
    if not np.isfinite(points).all():
        raise QualificationError("geometry contains non-finite vertices")
    tri = points[triangles]
    edges = np.stack(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ),
        axis=1,
    )
    positive = edges[edges > 1e-12]
    if not positive.size:
        raise QualificationError("geometry is degenerate")
    scale = float(np.median(positive))
    edge_rows = np.sort(np.round(edges / scale, 7), axis=1)
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ) / (2.0 * scale * scale)
    descriptors = np.column_stack((edge_rows, np.round(area, 7)))
    descriptors = descriptors[np.lexsort(descriptors.T[::-1])]
    header = np.asarray([len(points), len(triangles)], dtype="<i8").tobytes()
    return hashlib.sha256(header + descriptors.astype("<f8").tobytes()).hexdigest()


def connected_face_components(
    faces: np.ndarray,
    vertex_count: int,
    *,
    max_components: int = MAX_CONNECTED_COMPONENTS,
) -> list[np.ndarray]:
    """Return face-index groups using bounded-memory vertex union-find.

    ``trimesh.Trimesh.split`` may build an unexpectedly large adjacency graph for
    malformed imported meshes.  The qualification path only needs topological
    components, so this implementation keeps memory linear in vertices and faces and
    rejects pathological inputs before constructing any component meshes.
    """
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise QualificationError("faces must have shape (M, 3), M >= 1")
    if len(triangles) > MAX_TRIANGLES:
        raise QualificationError(f"source mesh exceeds {MAX_TRIANGLES} triangles")
    if not 3 <= vertex_count <= MAX_SOURCE_VERTICES:
        raise QualificationError(
            f"source vertex count must be in [3, {MAX_SOURCE_VERTICES}]"
        )
    if triangles.min() < 0 or triangles.max() >= vertex_count:
        raise QualificationError("face indices are outside the vertex array")
    if not 1 <= max_components <= MAX_CONNECTED_COMPONENTS:
        raise QualificationError(
            f"max_components must be in [1, {MAX_CONNECTED_COMPONENTS}]"
        )

    parent = np.arange(vertex_count, dtype=np.int32)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        root_a = find(first)
        root_b = find(second)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for first, second, third in triangles:
        union(int(first), int(second))
        union(int(second), int(third))

    roots = np.fromiter(
        (find(int(face[0])) for face in triangles),
        dtype=np.int32,
        count=len(triangles),
    )
    unique_roots = np.unique(roots)
    if len(unique_roots) > max_components:
        raise QualificationError(
            f"source mesh has {len(unique_roots)} connected components; "
            f"limit is {max_components}"
        )
    order = np.argsort(roots, kind="stable")
    sorted_roots = roots[order]
    boundaries = np.flatnonzero(sorted_roots[1:] != sorted_roots[:-1]) + 1
    return [group.copy() for group in np.split(order, boundaries)]


def connected_mesh_face_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    relative_weld_tolerance: float = 1e-9,
    max_components: int = MAX_CONNECTED_COMPONENTS,
) -> list[np.ndarray]:
    """Group mesh faces after deterministic position-only vertex welding.

    USD/glTF conversion commonly duplicates vertices at normals, UV, and material
    seams.  Position welding is used only for connectivity discovery; original
    vertices, faces, normals, UVs, and materials remain untouched in output meshes.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise QualificationError("vertices must have shape (N, 3)")
    if not 3 <= len(points) <= MAX_SOURCE_VERTICES:
        raise QualificationError(
            f"source vertex count must be in [3, {MAX_SOURCE_VERTICES}]"
        )
    if not np.isfinite(points).all():
        raise QualificationError("geometry contains non-finite vertices")
    if not math.isfinite(relative_weld_tolerance) or not 0.0 < relative_weld_tolerance <= 1e-6:
        raise QualificationError("relative weld tolerance must be in (0, 1e-6]")
    span = np.ptp(points, axis=0)
    scale = float(np.max(span))
    if scale <= 0.0:
        raise QualificationError("geometry is degenerate")
    tolerance = max(scale * relative_weld_tolerance, np.finfo(np.float64).eps)
    quantized = np.rint((points - points.min(axis=0)) / tolerance).astype(np.int64)
    _, welded_indices = np.unique(quantized, axis=0, return_inverse=True)
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise QualificationError("faces must have shape (M, 3)")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise QualificationError("face indices are outside the vertex array")
    welded_faces = welded_indices[triangles]
    return connected_face_components(
        welded_faces,
        int(welded_indices.max()) + 1,
        max_components=max_components,
    )


def maximum_sustained_angle_deg(
    angles_rad: Sequence[float], window_ticks: int = SUSTAIN_TICKS
) -> float:
    """Compute ``max(min(window))`` for a fixed-width angle trace."""
    values = np.degrees(np.asarray(angles_rad, dtype=np.float64).reshape(-1))
    if window_ticks <= 0:
        raise QualificationError("sustained-angle window must be positive")
    if len(values) < window_ticks or not np.isfinite(values).all():
        raise QualificationError("angle trace is finite and at least one window long")
    minima = np.minimum.reduce(
        [values[offset : len(values) - window_ticks + offset + 1] for offset in range(window_ticks)]
    )
    return float(np.max(minima))


def align_angle_curves_deg(
    first_rad: Sequence[float], second_rad: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Align 60 Hz traces and hold the last valid sample of the shorter trace."""
    first = np.degrees(np.asarray(first_rad, dtype=np.float64).reshape(-1))
    second = np.degrees(np.asarray(second_rad, dtype=np.float64).reshape(-1))
    if not len(first) or not len(second):
        raise QualificationError("repeatability traces must not be empty")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise QualificationError("repeatability traces must be finite")
    size = max(len(first), len(second))
    return (
        np.pad(first, (0, size - len(first)), constant_values=first[-1]),
        np.pad(second, (0, size - len(second)), constant_values=second[-1]),
    )


def repeatability_metrics(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the required clean rollout pair."""
    first_curve, second_curve = align_angle_curves_deg(
        first["angle_curve_rad"], second["angle_curve_rad"]
    )
    sustained_a = maximum_sustained_angle_deg(first["angle_curve_rad"])
    sustained_b = maximum_sustained_angle_deg(second["angle_curve_rad"])
    same_outcome = bool(first["passed"]) == bool(second["passed"])
    same_termination = first["termination"] == second["termination"]
    sustained_diff = abs(sustained_a - sustained_b)
    curve_error = float(np.max(np.abs(first_curve - second_curve)))
    passed = (
        same_outcome
        and same_termination
        and sustained_diff <= REPEAT_SUSTAINED_TOL_DEG
        and curve_error <= REPEAT_CURVE_TOL_DEG
    )
    return {
        "passed": passed,
        "same_outcome": same_outcome,
        "same_termination": same_termination,
        "first_max_sustained_deg": sustained_a,
        "second_max_sustained_deg": sustained_b,
        "max_sustained_difference_deg": sustained_diff,
        "max_time_aligned_curve_error_deg": curve_error,
        "alignment_hz": 60,
        "shorter_trace_extension": "last_valid_angle",
    }


def validate_diagnostic_count(pair_passed: bool, diagnostic_rollouts: Sequence[Any]) -> None:
    """Require no diagnostics after a passing pair and exactly three after a failed pair."""
    expected = 0 if pair_passed else 3
    if len(diagnostic_rollouts) != expected:
        raise QualificationError(
            f"repeatability pair passed={pair_passed} requires {expected} diagnostics, "
            f"got {len(diagnostic_rollouts)}"
        )


def bootstrap_n_qual(
    per_door_maxima_deg: Mapping[str, Sequence[float]],
    *,
    seed: int = 4101,
    replicas: int = 10_000,
    candidates: range = range(10, 201),
    error_limit_deg: float = 2.0,
) -> dict[str, Any]:
    """Choose the smallest common sample count meeting the frozen q10 error rule."""
    if seed != 4101 or replicas != 10_000:
        raise QualificationError("Phase 4.1 bootstrap requires seed 4101 and 10,000 replicas")
    observations: dict[str, np.ndarray] = {}
    for door_id, values in sorted(per_door_maxima_deg.items()):
        sample = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(sample) < 2 or not np.isfinite(sample).all():
            raise QualificationError(f"{door_id} needs at least two finite rollout maxima")
        observations[door_id] = sample
    if not observations:
        raise QualificationError("bootstrap requires at least one door")

    rng = np.random.default_rng(seed)
    error_by_n: dict[str, float] = {}
    selected: int | None = None
    for n in candidates:
        errors: list[np.ndarray] = []
        for values in observations.values():
            truth = float(np.quantile(values, 0.10, method="linear"))
            indices = rng.integers(0, len(values), size=(replicas, n))
            estimates = np.quantile(values[indices], 0.10, axis=1, method="linear")
            errors.append(np.abs(estimates - truth))
        p95 = float(np.quantile(np.concatenate(errors), 0.95, method="higher"))
        error_by_n[str(n)] = p95
        if selected is None and p95 <= error_limit_deg:
            selected = n
            break
    if selected is None:
        raise QualificationError("no n_qual in 10..200 meets the 2 degree bootstrap rule")
    return {
        "recommended_n_qual": selected,
        "seed": seed,
        "replicas": replicas,
        "quantile": 0.10,
        "confidence_quantile": 0.95,
        "absolute_error_limit_deg": error_limit_deg,
        "p95_absolute_error_deg_by_n": error_by_n,
    }


def validate_remote_candidate(
    record: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]] = ()
) -> None:
    """Apply every frozen pre-download gate represented by the worklist schema."""
    required = {
        "slot",
        "source_url",
        "source_uid",
        "author",
        "license",
        "license_url",
        "attribution",
        "dependencies",
        "selected_format",
        "archive_size_bytes",
        "reported_triangles",
        "reported_texture_max_px",
        "reported_dimensions_m",
        "door_type",
        "handedness",
        "frame_panel_separable",
        "visual_duplicate_check",
        "ownership_dispute_check",
        "custom_terms_check",
        "retrieval_date",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise QualificationError(f"remote record is missing fields: {missing}")
    if record["license"] not in ACCEPTED_LICENSES:
        raise QualificationError(f"ineligible license: {record['license']!r}")
    if not str(record["license_url"]).startswith("https://creativecommons.org/"):
        raise QualificationError("license evidence must be a canonical Creative Commons URL")
    if record["handedness"] not in HANDEDNESSES:
        raise QualificationError("handedness must be left or right")
    if str(record["selected_format"]).lower() not in SOURCE_FORMAT_PRIORITY:
        raise QualificationError("selected format is unsupported")
    if record["door_type"] not in {"interior", "exterior", "industrial"}:
        raise QualificationError("candidate is not a full-size single-leaf push door")
    if record["frame_panel_separable"] is not True:
        raise QualificationError("frame and panel are not visibly separable")
    for field in ("visual_duplicate_check", "ownership_dispute_check", "custom_terms_check"):
        if record[field] != "pass":
            raise QualificationError(f"remote gate failed: {field}={record[field]!r}")
    if record["archive_size_bytes"] is not None and int(record["archive_size_bytes"]) <= 0:
        raise QualificationError("reported archive size must be positive")
    triangles = record["reported_triangles"]
    if triangles is not None and int(triangles) > MAX_TRIANGLES:
        raise QualificationError("reported triangle count exceeds 250,000")
    texture = record["reported_texture_max_px"]
    if texture is not None and int(texture) > MAX_TEXTURE_EDGE_PX:
        raise QualificationError("reported texture resolution exceeds 4K")
    dimensions = record["reported_dimensions_m"]
    if dimensions is not None:
        DoorDimensions.from_mapping(dimensions)
    dependencies = record["dependencies"]
    if not isinstance(dependencies, list) or any(
        dependency.get("license") not in ACCEPTED_LICENSES for dependency in dependencies
    ):
        raise QualificationError("every redistributed dependency must have CC0 or CC BY 4.0")
    if any(
        other.get("source_uid") == record["source_uid"]
        or other.get("source_url") == record["source_url"]
        for other in accepted
    ):
        raise QualificationError("source UID/URL duplicates an accepted candidate")


def validate_manifest(data: Mapping[str, Any], *, require_complete: bool = True) -> None:
    """Validate the tracked Phase 4.1 manifest and its completion gate."""
    if data.get("schema") != SCHEMA:
        raise QualificationError(f"manifest schema must be {SCHEMA!r}")
    records = data.get("assets")
    if not isinstance(records, list):
        raise QualificationError("manifest assets must be a list")
    if require_complete and len(records) != 24:
        raise QualificationError(f"complete manifest needs exactly 24 assets, got {len(records)}")
    uids: set[str] = set()
    urls: set[str] = set()
    fingerprints: set[str] = set()
    handedness: Counter[str] = Counter()
    forbidden_split_keys = {"split", "train", "development", "dev", "test", "theta_primary"}
    for index, record in enumerate(records):
        overlap = forbidden_split_keys & record.keys()
        if overlap:
            raise QualificationError(f"asset {index} contains Subphase 4.2 fields: {overlap}")
        validate_remote_candidate(record)
        DoorDimensions.from_mapping(record["normalized"]["dimensions_m"])
        if int(record["normalized"]["triangles"]) > MAX_TRIANGLES:
            raise QualificationError(f"asset {index} exceeds the triangle limit")
        if int(record["normalized"]["texture_max_px"]) > MAX_TEXTURE_EDGE_PX:
            raise QualificationError(f"asset {index} exceeds the texture limit")
        for key, values in (
            ("source_uid", uids),
            ("source_url", urls),
            ("geometry_fingerprint", fingerprints),
        ):
            value = str(record[key])
            if value in values:
                raise QualificationError(f"duplicate {key}: {value}")
            values.add(value)
        handedness[str(record["handedness"])] += 1
        if require_complete:
            for gate in ("static", "physics", "nominal", "repeatability"):
                if record["qualification"][gate].get("passed") is not True:
                    raise QualificationError(f"asset {index} has not passed {gate}")
            if record.get("final_status") != "provisional_for_phase4_2":
                raise QualificationError(f"asset {index} has invalid final status")
            for digest_field in ("source_sha256", "geometry_fingerprint", "evidence_sha256"):
                if len(str(record[digest_field])) != 64:
                    raise QualificationError(f"asset {index} has invalid {digest_field}")
    if require_complete and handedness != Counter({"left": 12, "right": 12}):
        raise QualificationError(f"handedness must be 12/12, got {dict(handedness)}")
    if require_complete:
        recommendation = data.get("n_qual_recommendation", {})
        if not 10 <= int(recommendation.get("recommended_n_qual", 0)) <= 200:
            raise QualificationError("manifest lacks one valid n_qual recommendation")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def dump_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def handedness_sign(handedness: Literal["left", "right"] | str) -> float:
    if handedness not in HANDEDNESSES:
        raise QualificationError(f"unknown handedness: {handedness!r}")
    return 1.0 if handedness == "left" else -1.0


__all__ = [
    "ACCEPTED_LICENSES",
    "DoorDimensions",
    "FORCE_LIMIT_N",
    "QualificationError",
    "SCHEMA",
    "align_angle_curves_deg",
    "bootstrap_n_qual",
    "canonical_sha256",
    "cuboid_inertia_kg_m2",
    "dump_json",
    "geometry_fingerprint",
    "handedness_sign",
    "load_json",
    "maximum_sustained_angle_deg",
    "repeatability_metrics",
    "sha256_file",
    "validate_manifest",
    "validate_diagnostic_count",
    "validate_remote_candidate",
]
