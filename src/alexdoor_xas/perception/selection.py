"""Explicit geometric reference requests shared by maintained diagnostic ingress."""

import json
from pathlib import Path


def load_selection(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    allowed = {"sam3_id", "geometric_object", "reference_pixel", "after_s", "criterion"}
    if set(value) - allowed or "sam3_id" not in value or not value.get("criterion"):
        raise ValueError("selection_requires_identity_and_observed_criterion")
    if not isinstance(value["sam3_id"], int):
        raise ValueError("invalid_selected_identity")
    return dict(
        identity=value["sam3_id"],
        geometric_object=value.get("geometric_object"),
        reference_pixel=value.get("reference_pixel"),
        after_s=value.get("after_s", 0.0),
        criterion=value["criterion"],
    )
