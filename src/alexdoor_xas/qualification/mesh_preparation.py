"""Reviewed surface separation and backing for static graphics assets."""

import numpy as np
import trimesh

from .preparation import require


def prepare_components(components, recipe):
    """Keep the remainder at its source index; append each selected surface."""
    result = [mesh.copy() for mesh in components]
    for split in recipe.get("component_splits", []):
        index = split["component"]
        require(type(index) is int and 0 <= index < len(result), "Invalid split component")
        require(
            bool(split.get("review")), "Review the modeled frame/leaf boundary before splitting"
        )
        bounds = split["bounds"]
        require(bool(bounds) and set(bounds) <= set("xyz"), "Specify inspected-coordinate bounds")
        mesh = result[index]
        selected = np.ones(len(mesh.faces), dtype=bool)
        for axis, interval in bounds.items():
            interval = np.asarray(interval, dtype=float)
            require(
                interval.shape == (2,)
                and np.isfinite(interval).all()
                and interval[0] < interval[1],
                "Invalid split interval",
            )
            values = mesh.triangles[:, :, "xyz".index(axis)]
            selected &= ((values >= interval[0]) & (values <= interval[1])).all(axis=1)
        require(selected.any() and not selected.all(), "Split must select a proper face subset")
        # Select complete existing faces, without changing their vertices or UVs.
        result[index] = mesh.submesh([np.flatnonzero(~selected)], append=True, repair=False)
        result.append(mesh.submesh([np.flatnonzero(selected)], append=True, repair=False))
    for backing in recipe.get("shell_backings", []):
        index = backing["component"]
        require(type(index) is int and 0 <= index < len(result), "Invalid backing component")
        require(bool(backing.get("review")), "Review inferred thickness and rear appearance")
        require(backing["axis"] in "xyz" and len(backing["axis"]) == 1, "Invalid backing axis")
        result[index] = close_shell(result[index], "xyz".index(backing["axis"]), backing["plane"])
    return result


def close_shell(mesh, axis, plane):
    """Back a single-valued relief with its planar projection and boundary walls."""
    points, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    require(
        np.isfinite(plane) and (plane > points[:, axis].max() or plane < points[:, axis].min()),
        "Backing plane must be outside the shell",
    )
    normals = mesh.face_normals[:, axis]
    require(
        normals.min() >= -1e-8 or normals.max() <= 1e-8,
        "Backing requires a relief without folded or opposing faces",
    )
    # Weld only for boundary discovery; preserve the actual texture-seam vertices.
    _, welded = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    keys, inverse, counts = np.unique(
        np.sort(welded[edges], axis=1), axis=0, return_inverse=True, return_counts=True
    )
    require(len(keys) > 0 and counts.max() <= 2, "Non-manifold source shell")
    boundary = edges[counts[inverse] == 1]
    require(len(boundary) > 0, "Backing requires an open source shell")
    rear = points.copy()
    rear[:, axis] = plane
    n = len(points)
    triangles = np.vstack(
        [
            faces,
            faces[:, ::-1] + n,
            np.column_stack([boundary[:, 1], boundary[:, 0], boundary[:, 0] + n]),
            np.column_stack([boundary[:, 1], boundary[:, 0] + n, boundary[:, 1] + n]),
        ]
    )
    visual = None
    if mesh.visual.kind == "texture":
        visual = trimesh.visual.TextureVisuals(
            uv=np.vstack([mesh.visual.uv, mesh.visual.uv]), material=mesh.visual.material
        )
    elif mesh.visual.kind == "vertex":
        visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.vstack([mesh.visual.vertex_colors, mesh.visual.vertex_colors])
        )
    elif mesh.visual.kind == "face":
        colors = mesh.visual.face_colors
        wall_colors = colors[np.repeat(np.arange(len(faces)), 3)[counts[inverse] == 1]]
        visual = trimesh.visual.ColorVisuals(
            face_colors=np.vstack([colors, colors, wall_colors, wall_colors])
        )
    result = trimesh.Trimesh(
        vertices=np.vstack([points, rear]), faces=triangles, visual=visual, process=False
    )
    result.update_faces(result.nondegenerate_faces())
    probe = result.copy()
    probe.merge_vertices(merge_tex=True, merge_norm=True)
    probe.fix_normals()
    require(probe.is_volume, "Backing did not produce a closed, consistently oriented solid")
    flipped = (result.face_normals * probe.face_normals).sum(axis=1) < 0
    result.faces[flipped] = result.faces[flipped, ::-1]
    return result
