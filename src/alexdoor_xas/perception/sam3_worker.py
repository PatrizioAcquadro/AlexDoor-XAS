"""Isolated SAM3 video worker; receives only causal RGB/time packets."""

import resource
import sys
from contextlib import redirect_stdout
from pathlib import Path

from alexdoor_xas.perception.visual_worker import receive, send


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root / "runtime/venv/lib/python3.12/site-packages"))
    import numpy as np
    import torch

    from alexdoor_xas.perception.sam3_frontend import Sam3CausalFrontend

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    request = receive(sys.stdin.buffer)
    try:
        with redirect_stdout(sys.stderr):
            frontend = Sam3CausalFrontend(
                root, prompt=request["prompt"], version=request.get("version", "sam3")
            )
        send(sys.stdout.buffer, dict(ready=frontend.runtime))
        while (request := receive(sys.stdin.buffer)) is not None:
            try:
                rgb = np.frombuffer(request["rgb"], np.uint8).reshape(request["shape"])
                with redirect_stdout(sys.stderr):
                    result = frontend.infer(rgb, request["frame"], request["time_s"])
                masks = result.pop("masks")
                result["shape"] = masks.shape
                result["masks"] = np.packbits(masks.reshape(-1)).tobytes()
                result["cpu_peak_rss_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                send(sys.stdout.buffer, result)
            except Exception as error:
                send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
                raise  # A failed state is not reused or restarted.
    except Exception as error:
        send(sys.stdout.buffer, dict(error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
