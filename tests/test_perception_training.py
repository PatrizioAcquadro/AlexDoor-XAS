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
    with pytest.raises(ValueError, match="nonempty"):
        training_scope(train, SimpleNamespace(episodes=[]), False)
    train.episodes[1]["handedness"] = "left"
    with pytest.raises(ValueError, match="one door per handedness"):
        training_scope(train, train, True)


def test_resume_rejects_legacy_loss_and_fit_to_full_scope():
    loss = dict(recipe="articulated-state-v3")
    scope = dict(kind="train_fit_check", train_doors=["a", "b"], evaluation_split="train")
    state = dict(loss_config=loss, scope=scope)
    validate_resume_recipe(state, loss, scope)
    with pytest.raises(ValueError, match="loss recipe"):
        validate_resume_recipe(
            {**state, "loss_config": dict(recipe="tolerance-normalized-v2", auxiliary_weight=0.1)},
            loss,
            scope,
        )
    with pytest.raises(ValueError, match="loss recipe"):
        validate_resume_recipe({}, loss, scope)
    with pytest.raises(ValueError, match="scope"):
        validate_resume_recipe(state, loss, {**scope, "kind": "full_training"})


def test_legacy_confidence_cannot_be_used_as_observed_state(tmp_path, gpu_models):
    import torch

    from alexdoor_xas.perception.model import STATE_CONFIDENCE, DoorEstimator, ObservedEstimator
    from alexdoor_xas.perception.training import load_checkpoint

    config = dict(hidden_size=128, history=4)
    model = DoorEstimator()
    payload = dict(schema="b1.perception.checkpoint.v1", config=config, model=model.state_dict())
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    legacy, _ = load_checkpoint(path, config, "cuda:0")
    assert legacy.confidence_scope == "contact-only"
    with pytest.raises(ValueError, match="articulated-state confidence"):
        ObservedEstimator(None, legacy, config)
    payload.update(schema="b1.perception.checkpoint.v2", confidence_scope=STATE_CONFIDENCE)
    torch.save(payload, path)
    current, _ = load_checkpoint(path, config, "cuda:0")
    assert not current.confidence_qualified
    ObservedEstimator(None, current, config)
    payload["confidence_scope"] = "contact-only"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="confidence contract"):
        load_checkpoint(path, config, "cuda:0")


def test_evaluation_requires_joint_state_accuracy_and_state_confidence(gpu_models):
    import torch

    from alexdoor_xas.perception.model import STATE_CONFIDENCE, decode
    from alexdoor_xas.perception.training import evaluate

    class Windows:
        episodes = [dict(asset_id="door", split="train", handedness="left")]
        window_phases = {(0, i): 2 for i in range(100)}

        def __len__(self):
            return 100

        def __getitem__(self, index):
            raw = torch.zeros(24, device="cpu")
            raw[[3, 7, 10, 17, 21]] = 1
            raw[23] = 10
            return (
                dict(
                    features=torch.tensor(index, device="cpu"),
                    geometry=torch.ones(1, 4, 1, 1, device="cpu"),
                ),
                decode(raw),
                0,
                index,
            )

    class Head(torch.nn.Module):
        confidence_scope = STATE_CONFIDENCE
        mode = "correct"

        def forward(self, features, geometry):
            raw = torch.zeros(len(features), 24)
            raw[:, [3, 7, 10, 17, 21]] = 1
            raw[:, 23] = 10
            if self.mode == "compensating":
                raw[:, 0] = 1
                raw[:, 14] = -1
            pred = decode(raw)
            if self.mode == "disjoint":
                # Each primitive p95 passes, but 8% of samples have a bad state.
                pred["hinge_origin"][features < 4, 0] += 0.1
                pred["dimensions"][(features >= 4) & (features < 8), 0] += 0.1
            if self.mode == "imprecise_confidence":
                pred["contact_position"][features < 5, 0] += 0.1
                pred["confidence"][(features >= 5) & (features < 10)] = 0
            return pred

    config = dict(
        batch_size=32,
        confidence_threshold=0.5,
        gates=dict(contact_position_p95_m=0.01, contact_orientation_p95_deg=5, valid_coverage=0.95),
    )
    model = Head()
    assert evaluate(model, Windows(), config, "cuda:0")["offline_passed"]
    model.mode = "compensating"
    result = evaluate(model, Windows(), config, "cuda:0")
    assert result["contact_geometry_passed"] and not result["geometry_passed"]
    assert not result["offline_passed"]
    model.mode = "disjoint"
    result = evaluate(model, Windows(), config, "cuda:0")
    assert all(
        result["per_door"]["door"]["errors_p95"][k] <= v
        for k, v in result["geometry_tolerances"].items()
    )
    assert result["per_door"]["door"]["geometric_coverage"] == pytest.approx(0.92)
    assert not result["offline_passed"]
    model.mode = "imprecise_confidence"
    result = evaluate(model, Windows(), config, "cuda:0")
    assert result["geometry_passed"]
    assert result["per_door"]["door"]["valid_coverage"] == pytest.approx(0.95)
    assert result["per_door"]["door"]["valid_precision"] < 0.95
    assert not result["offline_passed"]
    model.mode = "correct"
    model.confidence_scope = "contact-only"
    result = evaluate(model, Windows(), config, "cuda:0")
    assert result["geometry_passed"] and not result["offline_passed"]
