"""Causal adapter for the pinned official SAM3 detector and video tracker.

Only the newly arrived RGB image is appended. Native thresholds, reconditioning
and memory are retained. A one-frame forward propagation cannot revise earlier
outputs using hotstart lookahead. No re-prompting, ID substitution or mask fallback.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from alexdoor_xas.perception.visual_worker import SAM3_REVISION


def identity_status(seed_ids, current_ids):
    """A new semantic detection never substitutes for a missing seed identity."""
    if len(set(current_ids)) != len(current_ids):
        raise ValueError("duplicate_sam3_identity")
    lost = {str(i): i not in current_ids for i in seed_ids}
    return dict(
        state="not_initialized" if not seed_ids else "lost" if any(lost.values()) else "tracking",
        lost=lost,
        new_ids=[i for i in current_ids if i not in seed_ids],
        loss_reason="no_initial_detection"
        if not seed_ids
        else ("missing_seed_mask" if any(lost.values()) else None),
    )


def append_frame(state, image):
    """Extend official input metadata without restarting tracker/feature memory."""
    index = state["num_frames"]
    batch = state["input_batch"]
    stage = deepcopy(batch.find_inputs[0])
    stage.img_ids[...] = index
    if getattr(stage, "img_ids_np", None) is not None:
        stage.img_ids_np[...] = index
    images = getattr(batch.img_batch, "tensors", batch.img_batch)
    images.append(image)
    batch.find_inputs.append(stage)
    batch.find_targets.append(None)
    batch.find_metadatas.append(None)
    for name in (
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
    ):
        state[name].append(None)
    state["per_frame_cur_step"].append(0)
    state["num_frames"] = index + 1
    trackers = state.get("tracker_inference_states", state.get("sam2_inference_states", []))
    for tracker in trackers:
        tracker["num_frames"] = index + 1
    return index


class Sam3CausalFrontend:
    def __init__(self, root, *, prompt, version="sam3"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 video requires CUDA; no CPU fallback")
        root = Path(root)
        if version == "sam3":
            from sam3.model.sam3_video_inference import Sam3VideoInference
            from sam3.model_builder import build_sam3_video_model

            self.model = build_sam3_video_model(
                checkpoint_path=str(root / "sam3/sam3.pt"),
                bpe_path=str(root / "sam3/bpe_simple_vocab_16e6.txt.gz"),
                device="cuda",
                load_from_HF=False,
                compile=False,
            )
            self.propagate = Sam3VideoInference.propagate_in_video
        elif version == "sam3.1":
            from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(root / "sam3.1/sam3.1_multiplex.pt"),
                bpe_path=str(root / "sam3/bpe_simple_vocab_16e6.txt.gz"),
                use_fa3=False,  # RTX 4090 uses the official PyTorch attention path.
                compile=False,
                warm_up=False,
            )
            self.model = predictor.model
            # Incoming frames require one-frame grounding chunks. With the video
            # demo's 16-frame default, (chunk_start, available_end) changes on
            # every arrival and retains duplicated prefix feature batches.
            self.model.batched_grounding_batch_size = 1
            self.propagate = Sam3MultiplexTracking.propagate_in_video
        else:
            raise ValueError("unsupported_sam3_version")
        self.model.eval().requires_grad_(False)
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("SAM3 parameters must all be on CUDA")
        self.torch, self.prompt = torch, prompt
        self.state = self.last_capture = self.seed_ids = None
        self.runtime = dict(
            model=f"official {version} video",
            source_revision=SAM3_REVISION,
            prompt=prompt,
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            future_frames_supplied=0,
            propagation="single newest available frame, forward only",
            grounding_frames_per_chunk=1,
            compile=False,
            native_thresholds=dict(
                detection=self.model.score_threshold_detection,
                new_detection=self.model.new_det_thresh,
                association_iou=self.model.assoc_iou_thresh,
                hotstart_delay=self.model.hotstart_delay,
                recondition_every_nth_frame=self.model.recondition_every_nth_frame,
            ),
        )

    def infer(self, rgb, frame, time_s):
        import time

        from PIL import Image
        from sam3.model.io_utils import load_resource_as_video_frames

        torch = self.torch
        capture = int(frame), float(time_s)
        if not np.isfinite(capture).all():
            raise ValueError("nonfinite_sam3_capture")
        if self.last_capture is not None and (
            capture[0] <= self.last_capture[0] or capture[1] <= self.last_capture[1]
        ):
            raise ValueError("nonmonotonic_sam3_capture")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("sam3_requires_uint8_rgb")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.state is None:
                self.state = self.model.init_state(
                    [Image.fromarray(rgb)], offload_video_to_cpu=True
                )
                self.state["is_image_only"] = False
                batch = self.state["input_batch"]
                if hasattr(batch.img_batch, "tensors"):
                    batch.img_batch.tensors = list(batch.img_batch.tensors.cpu().unbind(0))
                else:
                    batch.img_batch = list(batch.img_batch.cpu().unbind(0))
                index, output = self.model.add_prompt(self.state, frame_idx=0, text_str=self.prompt)
                self.seed_ids = tuple(int(i) for i in output["out_obj_ids"])
            else:
                if rgb.shape[:2] != (self.state["orig_height"], self.state["orig_width"]):
                    raise ValueError("sam3_capture_shape_changed")
                images, _, _ = load_resource_as_video_frames(
                    [Image.fromarray(rgb)],
                    self.model.image_size,
                    True,
                    self.model.image_mean,
                    self.model.image_std,
                )
                index = append_frame(self.state, images[0])
                # Use the official one-frame propagation, including current-frame
                # suppression. No interactive replay or buffered future outputs.
                outputs = list(
                    self.propagate(
                        self.model,
                        self.state,
                        start_frame_idx=index,
                        # The state ends at this frame, so no following input exists.
                        # None also avoids SAM3.1's explicit-bound chunk mismatch.
                        max_frame_num_to_track=None,
                        reverse=False,
                    )
                )
                if len(outputs) != 1 or outputs[0][0] != index:
                    raise RuntimeError("sam3_noncausal_or_missing_output")
                _, output = outputs[0]
        masks = np.asarray(output["out_binary_masks"], dtype=bool)
        ids = [int(i) for i in output["out_obj_ids"]]
        identity = identity_status(self.seed_ids, ids)
        tracker_scores = (
            self.state["tracker_metadata"]
            .get(
                "obj_id_to_tracker_score_frame_wise",
                self.state["tracker_metadata"].get("obj_id_to_sam2_score_frame_wise", {}),
            )
            .get(index, {})
        )
        torch.cuda.synchronize()
        self.last_capture = capture
        return dict(
            frame=capture[0],
            time_s=capture[1],
            index=index,
            ids=ids,
            seed_ids=self.seed_ids,
            **identity,
            masks=masks,
            scores=output["out_probs"].tolist(),
            tracker_scores={str(i): float(score) for i, score in tracker_scores.items()},
            latency_s=time.perf_counter() - started,
            torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_peak_bytes=torch.cuda.max_memory_allocated(),
            arrived_frames=self.state["num_frames"],
        )
