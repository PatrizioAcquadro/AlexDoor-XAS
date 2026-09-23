"""Essential guarantees for separating and backing welded graphics surfaces."""

import numpy as np
import pytest
import trimesh

from alexdoor_xas.qualification.mesh_preparation import close_shell, prepare_components
from alexdoor_xas.qualification.preparation import PreparationError


def test_split_keeps_source_faces_and_uvs_and_backing_closes_only_the_selected_leaf():
    # Two adjacent quads represent a welded leaf and a neighboring frame strip.
    vertices = np.array([[x, y, 0] for y in [0, 1] for x in [0, 1, 2]], dtype=float)
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    material = trimesh.visual.material.PBRMaterial(baseColorFactor=[120, 80, 40, 255])
    source = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=vertices[:, :2], material=material),
        process=False,
    )
    recipe = {
        "component_splits": [
            {"component": 0, "bounds": {"x": [-0.01, 1.01]}, "review": "Existing seam at X=1."}
        ],
        "shell_backings": [
            {"component": 1, "axis": "z", "plane": -0.04, "review": "Inferred 4 cm backing."}
        ],
    }
    frame, leaf = prepare_components([source], recipe)
    assert np.array_equal(frame.triangles, source.triangles[2:])
    assert np.allclose(leaf.triangles[:2], source.triangles[:2])
    assert np.array_equal(leaf.visual.uv[leaf.faces[:2]], source.visual.uv[source.faces[:2]])
    assert np.array_equal(leaf.visual.material.baseColorFactor, material.baseColorFactor)
    probe = leaf.copy()
    probe.merge_vertices(merge_tex=True, merge_norm=True)
    assert probe.is_volume and probe.volume == pytest.approx(0.04)
    assert len(source.faces) == 4 and np.array_equal(source.vertices, vertices)
    with pytest.raises(PreparationError, match="outside"):
        close_shell(source, 2, 0)
    recipe["component_splits"][0]["bounds"] = {"x": [-1, 3]}
    with pytest.raises(PreparationError, match="proper face subset"):
        prepare_components([source], recipe)


def test_backing_rejects_closed_or_folded_geometry():
    with pytest.raises(PreparationError, match="folded"):
        close_shell(trimesh.creation.box(), 2, -1)
