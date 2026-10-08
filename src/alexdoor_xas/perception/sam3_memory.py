"""Payload lifetimes for the pinned SAM3 models in forward-only inference.

This is not the interactive/reverse video API. Keep every memory the model can
still select, including image features; release only unreachable past payloads.
"""


class CausalFrameStore:
    """Global frame indexing with bounded payload storage and a seed template."""

    def __init__(self, values):
        self.length = len(values)
        self.values = dict(enumerate(values))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        if index not in self.values:
            raise RuntimeError(f"released_causal_frame_requested: {index}")
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def append(self, value):
        self.values[self.length] = value
        self.length += 1

    def release_before(self, index):
        for key in list(self.values):
            if key != 0 and key < index:
                del self.values[key]


def _retain(mapping, keep):
    for key in list(mapping):
        if key not in keep:
            del mapping[key]


def release_sam3_forward_history(state, tracker, index):
    """Preserve SAM3's score-selected memories, rather than an age window.

    With stride one, frame_filter selects the latest max_obj_ptrs-1 qualifying
    nonconditioning outputs and always includes the immediately previous frame.
    Older qualifying outputs displaced from that set can never re-enter it in
    forward inference; low-score outputs cannot become qualifying later. Keep
    the current output even when its score is low. Conditioning outputs remain
    intact (the selected OFF recipe has only the seed).
    """
    if (
        not tracker.use_memory_selection
        or tracker.memory_temporal_stride_for_eval != 1
        or tracker.num_maskmem > tracker.max_obj_ptrs_in_encoder
    ):
        raise ValueError("unsupported_sam3_causal_memory_selection")
    for local in state["tracker_inference_states"]:
        outputs = local["output_dict"]
        noncond = outputs["non_cond_frame_outputs"]
        qualified = [
            i
            for i, out in noncond.items()
            if i > 0 and out.get("eff_iou_score", float("-inf")) > tracker.mf_threshold
        ]
        keep_noncond = set(sorted(qualified)[-(tracker.max_obj_ptrs_in_encoder - 1) :])
        keep_noncond.add(index)
        _retain(noncond, keep_noncond)
        for field in ("output_dict_per_obj", "temp_output_dict_per_obj"):
            for per_object in local[field].values():
                _retain(per_object["non_cond_frame_outputs"], keep_noncond)
        local["consolidated_frame_inds"]["non_cond_frame_outputs"].intersection_update(keep_noncond)
        keep = set(outputs["cond_frame_outputs"]) | keep_noncond
        _retain(local["frames_already_tracked"], keep)
        for field in ("mask_inputs_per_obj", "point_inputs_per_obj"):
            for per_object in local[field].values():
                _retain(per_object, keep)

    _retain(state["cached_frame_outputs"], {index})
    metadata = state["tracker_metadata"]
    _retain(metadata["obj_id_to_tracker_score_frame_wise"], {index})
    _retain(metadata["rank0_metadata"]["suppressed_obj_ids"], {index})
    # The detector already evicts previous backbone features. Retain its text,
    # chunk/communication caches and persistent identity/occlusion metadata.
    batch = state["input_batch"]
    containers = [(batch, "img_batch")]
    containers.extend((batch, name) for name in ("find_inputs", "find_targets", "find_metadatas"))
    containers.extend(
        (state, name)
        for name in (
            "previous_stages_out",
            "per_frame_raw_point_input",
            "per_frame_raw_box_input",
            "per_frame_visual_prompt",
            "per_frame_geometric_prompt",
            "per_frame_cur_step",
        )
    )
    for owner, name in containers:
        value = owner[name] if isinstance(owner, dict) else getattr(owner, name)
        if not isinstance(value, CausalFrameStore):
            value = CausalFrameStore(value)
            if isinstance(owner, dict):
                owner[name] = value
            else:
                setattr(owner, name, value)
        value.release_before(index)
