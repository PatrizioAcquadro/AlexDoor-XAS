"""Training scope and recipe boundaries without starting optimization."""

from types import SimpleNamespace

import pytest

from alexdoor_xas.perception.training import training_scope, validate_resume_recipe


def test_fit_check_cannot_use_development_or_one_handedness():
    train = SimpleNamespace(
        episodes=[
            dict(asset_id="a", split="train", handedness="left"),
            dict(asset_id="b", split="train", handedness="right"),
        ]
    )
    dev = SimpleNamespace(episodes=[dict(asset_id="c", split="development", handedness="left")])
    scope = training_scope(train, train, True)
    assert scope["evaluation_split"] == "train"
    assert scope["train_doors"] == ["a", "b"]
    with pytest.raises(ValueError, match="same train subset"):
        training_scope(train, dev, True)
    with pytest.raises(ValueError, match="development evaluation"):
        training_scope(train, train, False)
    train.episodes[1]["handedness"] = "left"
    with pytest.raises(ValueError, match="one door per handedness"):
        training_scope(train, train, True)


def test_resume_rejects_legacy_loss_and_fit_to_full_scope():
    loss = dict(recipe="tolerance-normalized-v2", auxiliary_weight=0.1)
    scope = dict(kind="train_fit_check", train_doors=["a", "b"], evaluation_split="train")
    state = dict(loss_config=loss, scope=scope)
    validate_resume_recipe(state, loss, scope)
    with pytest.raises(ValueError, match="loss recipe"):
        validate_resume_recipe({}, loss, scope)
    with pytest.raises(ValueError, match="scope"):
        validate_resume_recipe(state, loss, {**scope, "kind": "full_training"})
