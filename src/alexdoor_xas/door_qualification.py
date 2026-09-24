"""Reusable door preparation contracts and raw angle measurements.

These helpers do not implement B1 expert qualification or corpus completion.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ACCEPTED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
HANDEDNESSES = {"left", "right"}
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


class QualificationError(ValueError):
    """A door preparation input or measurement is invalid."""


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
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / (
        2.0 * scale * scale
    )
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
        raise QualificationError(f"source vertex count must be in [3, {MAX_SOURCE_VERTICES}]")
    if triangles.min() < 0 or triangles.max() >= vertex_count:
        raise QualificationError("face indices are outside the vertex array")
    if not 1 <= max_components <= MAX_CONNECTED_COMPONENTS:
        raise QualificationError(f"max_components must be in [1, {MAX_CONNECTED_COMPONENTS}]")

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
            f"source mesh has {len(unique_roots)} connected components; limit is {max_components}"
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
        raise QualificationError(f"source vertex count must be in [3, {MAX_SOURCE_VERTICES}]")
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


def handedness_sign(handedness: Literal["left", "right"] | str) -> float:
    if handedness not in HANDEDNESSES:
        raise QualificationError(f"unknown handedness: {handedness!r}")
    return 1.0 if handedness == "left" else -1.0


__all__ = [
    "ACCEPTED_LICENSES",
    "DoorDimensions",
    "QualificationError",
    "cuboid_inertia_kg_m2",
    "geometry_fingerprint",
    "handedness_sign",
    "sha256_file",
]
