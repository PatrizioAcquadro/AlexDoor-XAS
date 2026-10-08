"""Essential causal state and no-ID-substitution contracts, without model inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.sam3.frontend import append_frame, identity_status
from alexdoor_xas.perception.sam3.runtime import masks_for_seed


def test_retired_frontend_fails_before_model_loading():
    from alexdoor_xas.perception.sam3.frontend import Sam3CausalFrontend

    with pytest.raises(ValueError, match="unsupported_sam3_version"):
        Sam3CausalFrontend(None, prompt="door surface", version="sam3.1")


def test_reconditioning_control_changes_only_period_and_trace_preserves_plan():
    from alexdoor_xas.perception.sam3.diagnostics import (
        ReconditioningTrace,
        set_detection_reconditioning,
    )

    plan = dict(
        new_det_obj_ids=[9], trk_id_to_max_iou_high_conf_det={4: 2}, reconditioned_obj_ids=set()
    )
    metadata = object()
    tracker = SimpleNamespace(add_new_mask=lambda **kwargs: "mask-result")
    model = SimpleNamespace(
        recondition_every_nth_frame=16,
        reconstruction_bbox_iou_thresh=-1,
        tracker=tracker,
        new_det_thresh=0.7,
    )

    def recondition():
        return tracker.add_new_mask(obj_id=4)

    def planning():
        tracker.add_new_mask(obj_id=9)  # New detection remains independent.
        model._recondition_masklets()
        return plan, metadata

    model._recondition_masklets = recondition
    model.run_tracker_update_planning_phase = planning
    trace = ReconditioningTrace(model)
    trace.begin()
    result = model.run_tracker_update_planning_phase()
    assert result[0] is plan and result[1] is metadata
    assert trace.current["applied_ids"] == [4]
    assert trace.current["new_detection_ids"] == [9]
    assert trace.current["called"]
    before = vars(model).copy()
    set_detection_reconditioning(model, False)
    assert {k for k in before if before[k] != vars(model)[k]} == {"recondition_every_nth_frame"}
    assert model.recondition_every_nth_frame == 0
    assert model.new_det_thresh == 0.7 and model.reconstruction_bbox_iou_thresh == -1
    model.reconstruction_bbox_iou_thresh = 0.8
    with pytest.raises(ValueError, match="unsupported_additional"):
        set_detection_reconditioning(model, False)


def test_causal_store_preserves_global_indices_and_rejects_released_reads():
    from alexdoor_xas.perception.sam3.memory import CausalFrameStore

    store = CausalFrameStore(["seed"])
    for i in range(1, 100):
        store.append(i)
        store.release_before(i)
    assert len(store) == 100
    assert store[0] == "seed" and store[-1] == 99
    assert store[99:] == [99]
    assert len(store.values) == 2
    with pytest.raises(RuntimeError, match="released_causal_frame_requested"):
        store[98]


def test_sam3_release_keeps_old_quality_memories_across_a_long_low_score_gap():
    from alexdoor_xas.perception.sam3.memory import release_sam3_forward_history

    cond = {0: object()}
    noncond = {i: {"eff_iou_score": 0.9 if i < 20 else 0.0} for i in range(1, 1000)}
    local = dict(
        output_dict=dict(cond_frame_outputs=cond, non_cond_frame_outputs=noncond.copy()),
        output_dict_per_obj={
            0: dict(cond_frame_outputs=cond, non_cond_frame_outputs=noncond.copy())
        },
        temp_output_dict_per_obj={0: dict(cond_frame_outputs={}, non_cond_frame_outputs={})},
        consolidated_frame_inds=dict(cond_frame_outputs={0}, non_cond_frame_outputs=set()),
        frames_already_tracked={i: {} for i in range(1000)},
        mask_inputs_per_obj={0: cond.copy()},
        point_inputs_per_obj={0: {}},
    )
    batch = SimpleNamespace(
        **{
            k: list(range(1000))
            for k in ("img_batch", "find_inputs", "find_targets", "find_metadatas")
        }
    )
    state = dict(
        tracker_inference_states=[local],
        input_batch=batch,
        cached_frame_outputs={i: object() for i in range(1000)},
        tracker_metadata=dict(
            obj_id_to_tracker_score_frame_wise={i: {} for i in range(1000)},
            rank0_metadata=dict(suppressed_obj_ids={i: set() for i in range(1000)}),
        ),
    )
    for name in (
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
        "per_frame_cur_step",
    ):
        state[name] = list(range(1000))
    tracker = SimpleNamespace(
        use_memory_selection=True,
        memory_temporal_stride_for_eval=1,
        num_maskmem=7,
        max_obj_ptrs_in_encoder=16,
        mf_threshold=0.01,
    )
    release_sam3_forward_history(state, tracker, 999)
    # The native consumer can reach these 980+ frame-old high-quality memories.
    expected = set(range(5, 20)) | {999}
    assert set(local["output_dict"]["non_cond_frame_outputs"]) == expected
    assert all(local["output_dict"]["non_cond_frame_outputs"][i] is noncond[i] for i in expected)
    assert local["output_dict"]["cond_frame_outputs"] is cond
    assert set(local["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == expected
    assert set(state["cached_frame_outputs"]) == {999}
    assert set(batch.img_batch.values) == {0, 999}
    assert len(batch.img_batch) == 1000


def test_append_preserves_prompt_tracker_memory_and_only_arrived_frames():
    memory = {"num_frames": 1, "conditioned_memory": object()}
    image0, image1 = object(), object()
    batch = SimpleNamespace(
        img_batch=[image0],
        find_inputs=[
            SimpleNamespace(
                img_ids=np.array([0]),
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
    state["tracker_inference_states"] = [memory]
    previous = memory["conditioned_memory"]
    assert append_frame(state, image1) == 1
    images = batch.img_batch
    assert images == [image0, image1]
    assert state["num_frames"] == memory["num_frames"] == 2
    assert memory["conditioned_memory"] is previous
    assert state["previous_stages_out"] == ["seed", None]
    assert batch.find_inputs[0].img_ids.tolist() == [0]
    assert batch.find_inputs[1].img_ids.tolist() == [1]
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
    from test_scan_fusion import measured_surface

    from alexdoor_xas.perception.geometry import surfaces
    from perception_helpers import recipe

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
    from test_point2pose_temporal import make_surface, sensor

    from alexdoor_xas.perception.point2pose.tracking import PanelTracking
    from perception_helpers import recipe

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
