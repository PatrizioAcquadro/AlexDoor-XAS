"""Frozen recipe for the shared observed-only perception session."""

import json
from dataclasses import dataclass
from pathlib import Path

from alexdoor_xas.perception.inspection import load_inspection


@dataclass(frozen=True)
class PrototypeRecipe:
    """Diagnostic recipe, deliberately not a qualified PerceptionBinding release."""

    recipe_json: str

    def __post_init__(self):
        if self.config.get("scan_fusion") != "object-v1":
            raise ValueError("Static scan requires the object-v1 recipe")

    @property
    def config(self):
        return json.loads(self.recipe_json)

    def to_dict(self):
        return dict(
            schema="b1.perception.prototype.v1",
            config=self.config,
            offline_passed=False,
            dynamic_passed=False,
            frozen=False,
        )


def load_recipe(path, root):
    config = json.loads(Path(path).read_text())
    config["inspection"] = load_inspection(Path(root) / config.pop("inspection_path"))
    selected = config.pop("point2pose_selected_path", "configs/point2pose_selected.json")
    config["point2pose_selected"] = json.loads((Path(root) / selected).read_text())
    return PrototypeRecipe(json.dumps(config, sort_keys=True, allow_nan=False))
