"""Frozen CUDA image models in a separate process with isolated native SAM3 dependencies."""

import pickle
import struct
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

SAM3_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"


def receive(stream):
    header = stream.read(8)
    if not header:
        return None
    size = struct.unpack("!Q", header)[0]
    payload = bytearray()
    while len(payload) < size:
        part = stream.read(size - len(payload))
        if not part:
            raise EOFError("Visual worker pipe closed")
        payload.extend(part)
    return pickle.loads(payload)


def send(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


class FrozenModels:
    def __init__(self, root, config):
        import numpy as np
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback")
        torch.set_num_threads(4)
        self.torch, self.np, self.config = torch, np, config
        self.ground_processor = AutoProcessor.from_pretrained(
            root / "grounding-dino", local_files_only=True
        )
        self.ground = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                root / "grounding-dino", local_files_only=True
            )
            .eval()
            .to("cuda:0")
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(
            root / "dinov3", local_files_only=True
        )
        self.dino = AutoModel.from_pretrained(root / "dinov3", local_files_only=True)
        self.dino.eval().to("cuda:0")
        self.sam = build_sam3_image_model(
            checkpoint_path=str(root / "sam3/sam3.pt"),
            bpe_path=str(root / "sam3/bpe_simple_vocab_16e6.txt.gz"),
            device="cuda",
            load_from_HF=False,
            compile=False,
        )
        self.sam.eval().to("cuda:0")
        self.sam_processor = Sam3Processor(
            self.sam, confidence_threshold=config["mask_threshold"], device="cuda:0"
        )
        for model in (self.ground, self.dino, self.sam):
            model.requires_grad_(False)
            if any(p.device != torch.device("cuda:0") for p in model.parameters()):
                raise RuntimeError("A visual model is not entirely on cuda:0")
        self.runtime = dict(
            device="cuda:0",
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            numpy=np.__version__,
            sam3_source_revision=SAM3_REVISION,
            frozen=all(
                not p.requires_grad
                for m in (self.ground, self.dino, self.sam)
                for p in m.parameters()
            ),
            training_started=False,
        )

    def infer(self, request):
        from PIL import Image

        torch, np = self.torch, self.np
        rgb = np.frombuffer(request["rgb"], np.uint8).reshape(request["shape"])
        image = Image.fromarray(rgb)
        height, width = rgb.shape[:2]
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.ground_processor(images=image, text="door.", return_tensors="pt")
            inputs = inputs.to("cuda:0")
            output = self.ground(**inputs)
            boxes = self.ground_processor.post_process_grounded_object_detection(
                output,
                inputs.input_ids,
                threshold=self.config["box_threshold"],
                text_threshold=self.config["text_threshold"],
                target_sizes=[(height, width)],
            )[0]
            # Preserve the whole image and its invertible pixel mapping for patch associations.
            scale = min(224 / width, 224 / height)
            size = (round(width * scale), round(height * scale))
            pad_x, pad_y = (224 - size[0]) // 2, (224 - size[1]) // 2
            letterbox = Image.new("RGB", (224, 224))
            letterbox.paste(image.resize(size, Image.Resampling.BILINEAR), (pad_x, pad_y))
            pixels = self.dino_processor(images=letterbox, return_tensors="pt").to("cuda:0")
            features = self.dino(**pixels).last_hidden_state
            features = features[:, 1 + self.dino.config.num_register_tokens :]
            features = torch.nn.functional.normalize(features.float(), dim=-1)[0]
            tokens = features.cpu().numpy().astype(np.float32)
            masks, scores, selected_boxes = [], [], []
            if len(boxes["scores"]):
                state = self.sam_processor.set_image(image)
                order = boxes["scores"].argsort(descending=True)[: self.config["max_boxes"]]
                for i in order:
                    box = boxes["boxes"][i].detach().float().cpu().numpy()
                    x0, y0, x1, y1 = box
                    prompt = [
                        (x0 + x1) / (2 * width),
                        (y0 + y1) / (2 * height),
                        (x1 - x0) / width,
                        (y1 - y0) / height,
                    ]
                    self.sam_processor.reset_all_prompts(state)
                    result = self.sam_processor.add_geometric_prompt(prompt, True, state)
                    for mask, score in zip(result["masks"], result["scores"], strict=True):
                        masks.append(np.packbits(mask.detach().cpu().numpy().reshape(-1)).tobytes())
                        scores.append(float(score))
                        selected_boxes.append(box.tolist())
        torch.cuda.synchronize()
        return dict(
            masks=masks,
            scores=scores,
            boxes=selected_boxes,
            shape=(height, width),
            tokens=tokens.tobytes(),
            token_shape=tokens.shape,
            pixel_mapping=dict(scale=scale, pad_x=pad_x, pad_y=pad_y),
            latency_s=time.perf_counter() - start,
        )


def main():
    root = Path(sys.argv[1]).resolve()
    isolated = root / "runtime/venv/lib/python3.12/site-packages"
    if not isolated.is_dir():
        raise RuntimeError("Missing isolated SAM3 environment; see models/perception/README.md")
    sys.path.insert(0, str(isolated))
    request = receive(sys.stdin.buffer)
    try:
        with redirect_stdout(sys.stderr):
            models = FrozenModels(root, request)
        send(sys.stdout.buffer, dict(ready=models.runtime))
        while (request := receive(sys.stdin.buffer)) is not None:
            try:
                with redirect_stdout(sys.stderr):
                    result = models.infer(request)
                send(sys.stdout.buffer, result)
            except Exception as error:
                send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
    except Exception as error:
        send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
