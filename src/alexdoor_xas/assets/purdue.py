"""Task-owned Purdue joint order and collision-derived push geometry."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_JOINTS = (
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Y",
)
NECK_JOINTS = ("NECK_Z", "NECK_Y")
PUSH_PARENT = "right_WSG32_BASE_LINK"
FINGER_LINKS = tuple(f"right_WSG32_{side}_UMI_V1_FINGER_LINK" for side in ("NEGATIVE", "POSITIVE"))


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """URDF origin in meters, with intrinsic roll/pitch/yaw."""
    out = np.eye(4)
    if origin is None:
        return out
    out[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    r, p, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    out[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return out


def read_stl_vertices(path: Path) -> np.ndarray:
    """Read the canonical binary or ASCII STL without a mesh dependency."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + count * 50:
        triangles = np.array(
            [struct.unpack_from("<9f", data, 84 + index * 50 + 12) for index in range(count)]
        ).reshape(-1, 3)
    else:
        triangles = np.array(
            [
                [float(v) for v in line.split()[1:]]
                for line in data.decode().splitlines()
                if line.strip().startswith("vertex ")
            ]
        )
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.isfinite(triangles).all():
        raise ValueError(f"Invalid collision STL: {path}")
    return triangles


@dataclass(frozen=True)
class PushGeometry:
    translation: np.ndarray
    wrist_from_base: np.ndarray
    finger_vertices: dict[str, np.ndarray]


def derive_push_geometry(urdf: str | Path) -> PushGeometry:
    """Average the closed fingers' forward support-face centers in the WSG base."""
    path = Path(urdf)
    root = ET.parse(path).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}

    def to_base(link):
        if link == PUSH_PARENT:
            return np.eye(4)
        joint = parents[link]
        return to_base(joint.find("parent").get("link")) @ origin_matrix(joint.find("origin"))

    vertices = {}
    supports = []
    for link in FINGER_LINKS:
        element = root.find(f"link[@name='{link}']")
        collision = next(
            c
            for c in element.findall("collision")
            if c.get("name", "").endswith("UMI_V1_CONTACT_CONVEX")
        )
        mesh = collision.find("geometry/mesh")
        points = read_stl_vertices(path.parent / mesh.get("filename"))
        points *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        transform = to_base(link) @ origin_matrix(collision.find("origin"))
        points = points @ transform[:3, :3].T + transform[:3, 3]
        vertices[link] = points
        face = points[np.abs(points[:, 0] - points[:, 0].max()) < 1e-6]
        supports.append((face.min(axis=0) + face.max(axis=0)) / 2)
    mount = root.find("joint[@name='RIGHT_WSG32_MOUNT']")
    return PushGeometry(np.mean(supports, axis=0), origin_matrix(mount.find("origin")), vertices)
