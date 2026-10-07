"""Essential causal state and no-ID-substitution contracts, without model inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.sam3_frontend import append_frame, identity_status
from alexdoor_xas.perception.sam3_runtime import masks_for_seed


@pytest.mark.parametrize("multiplex", [False, True])
def test_append_preserves_prompt_tracker_memory_and_only_arrived_frames(multiplex):
    memory = {"num_frames": 1, "conditioned_memory": object()}
    image0, image1 = object(), object()
    batch = SimpleNamespace(
        img_batch=SimpleNamespace(tensors=[image0]) if multiplex else [image0],
        find_inputs=[
            SimpleNamespace(
                img_ids=np.array([0]),
                img_ids_np=np.array([0]) if multiplex else None,
                text_ids=np.array([0]),
            )
        ],
        find_targets=[None],
        find_metadatas=[None],
    )
    state = dict(
        num_frames=1,
        input_batch=batch,
        previous_stages_out=["seed"],
        per_frame_raw_point_input=[None],
        per_frame_raw_box_input=[None],
        per_frame_visual_prompt=[None],
        per_frame_geometric_prompt=[None],
        per_frame_cur_step=[0],
    )
    state["sam2_inference_states" if multiplex else "tracker_inference_states"] = [memory]
    previous = memory["conditioned_memory"]
    assert append_frame(state, image1) == 1
    images = batch.img_batch.tensors if multiplex else batch.img_batch
    assert images == [image0, image1]
    assert state["num_frames"] == memory["num_frames"] == 2
    assert memory["conditioned_memory"] is previous
    assert state["previous_stages_out"] == ["seed", None]
    assert batch.find_inputs[0].img_ids.tolist() == [0]
    assert batch.find_inputs[1].img_ids.tolist() == [1]
    if multiplex:
        assert batch.find_inputs[1].img_ids_np.tolist() == [1]
    else:
        assert batch.find_inputs[1].img_ids_np is None
    assert batch.find_inputs[1].text_ids.tolist() == [0]
    with pytest.raises(IndexError):
        _ = images[2]  # Future RGB does not exist.


def test_missing_seed_is_loss_even_when_a_new_detection_exists():
    result = identity_status((4,), (9,))
    assert result == dict(
        state="lost", lost={"4": True}, new_ids=[9], loss_reason="missing_seed_mask"
    )
    assert identity_status((4,), (4,))["state"] == "tracking"
    assert identity_status((), (9,))["state"] == "not_initialized"
    with pytest.raises(ValueError, match="duplicate_sam3_identity"):
        identity_status((4,), (4, 4))
    assert not masks_for_seed(np.ones((1, 4, 5), bool), [9], [4], (4, 5)).any()
    assert masks_for_seed(np.ones((1, 4, 5), bool), [4], [4], (4, 5)).all()


def test_rgbd_surface_geometry_does_not_require_appearance_tokens():
    from alexdoor_xas.perception.geometry import surfaces
    from perception_helpers import recipe
    from test_scan_fusion import measured_surface

    expected, sensor, cue = measured_surface()
    cue = {
        key: value
        for key, value in cue.items()
        if key not in ("tokens", "token_shape", "pixel_mapping")
    }
    actual = surfaces(cue, sensor, recipe().config, visual_features=False)[0]
    np.testing.assert_array_equal(actual.points, expected.points)
    np.testing.assert_array_equal(actual.normal, expected.normal)
    np.testing.assert_array_equal(actual.observations[0].mask(), expected.observations[0].mask())
    assert actual.features.shape == (0, 0)


def test_initialization_passes_one_unchanged_semantic_mask_to_p2p():
    from alexdoor_xas.perception.panel_tracking import PanelTracking
    from perception_helpers import recipe
    from test_point2pose_temporal import make_surface, sensor

    class Engine:
        def reset(self):
            pass

        def submit(self, observed, masks, **kwargs):
            self.masks = masks

    observed = sensor()
    engine = Engine()
    tracker = PanelTracking(engine, recipe().config)
    semantic_mask = np.ones(observed["rgb"].shape[:2], bool)
    assert tracker.initialize([make_surface(observed)], observed, whole_leaf_mask=semantic_mask)
    assert len(tracker.candidates) == len(engine.masks) == 1
    np.testing.assert_array_equal(engine.masks[0], semantic_mask)
