"""Runtime-specific checks separated from portable behavioral tests."""

from pathlib import Path


def test_publication_localizes_external_usd_texture_paths(tmp_path):
    from pxr import Sdf, Usd, UsdUtils

    from alexdoor_xas.qualification.prepared import _copy_sources

    original = tmp_path / "original"
    original.mkdir()
    texture = tmp_path / "external.png"
    texture.write_bytes(b"texture")
    source = original / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    prim = stage.DefinePrim("/Material")
    prim.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    inspected = {
        "source": str(source),
        "files": [{"path": str(p), "snapshot": str(p)} for p in (source, texture)],
    }
    destination = tmp_path / "published"
    relative = _copy_sources(inspected, destination)[str(source)]
    assert source.read_bytes() == before
    _, dependencies, missing = UsdUtils.ComputeAllDependencies(str(destination / relative))
    assert not missing
    assert all(Path(p).is_relative_to(destination) for p in dependencies)
