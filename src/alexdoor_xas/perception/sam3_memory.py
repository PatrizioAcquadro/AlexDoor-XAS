"""Lifetimes for the pinned multiplex model in forward-only causal inference.

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


def release_forward_history(state, tracker, index):
    """Run after current output/encoding, never before memory attention.

    Without memory selection, future attention reads only recent nonconditioning
    frames and the closest conditioning frames. Retain the original conditioning
    frame too, so its first-annotation metadata remains valid. The horizon covers
    both spatial memory at its native stride and all native object pointers.
    """
    if tracker.use_memory_selection or tracker.max_cond_frames_in_attn < 2:
        raise ValueError("unsupported_causal_memory_selection")
    horizon = max(
        tracker.max_obj_ptrs_in_encoder,
        tracker.memory_temporal_stride_for_eval * (tracker.num_maskmem - 1) + 1,
    )
    cutoff = index - horizon + 1
    for local in state["sam2_inference_states"]:
        outputs = local["output_dict"]
        cond = outputs["cond_frame_outputs"]
        keep_cond = set(sorted(cond)[-tracker.max_cond_frames_in_attn :])
        if cond:
            keep_cond.add(min(cond))
        keep_cond.update(i for i in cond if i >= cutoff)
        keep_noncond = {i for i in outputs["non_cond_frame_outputs"] if i >= cutoff}
        for name, keep in (
            ("cond_frame_outputs", keep_cond),
            ("non_cond_frame_outputs", keep_noncond),
        ):
            _retain(outputs[name], keep)
            for field in ("output_dict_per_obj", "temp_output_dict_per_obj"):
                for per_object in local[field].values():
                    _retain(per_object[name], keep)
            local["consolidated_frame_inds"][name].intersection_update(keep)
        keep = keep_cond | keep_noncond
        _retain(local["frames_already_tracked"], keep)
        for field in ("mask_inputs_per_obj", "point_inputs_per_obj"):
            for per_object in local[field].values():
                _retain(per_object, keep)

    # Past output masks and per-frame scores serve interactive replay/export;
    # forward association uses persistent object/confirmation/occlusion metadata.
    _retain(state["cached_frame_outputs"], {index})
    metadata = state["tracker_metadata"]
    _retain(metadata["obj_id_to_sam2_score_frame_wise"], {index})
    _retain(metadata["rank0_metadata"]["suppressed_obj_ids"], {index})
    batch = state["input_batch"]
    containers = [(batch.img_batch, "tensors")]
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
