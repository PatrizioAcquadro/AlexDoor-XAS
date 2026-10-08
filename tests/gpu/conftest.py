"""CUDA is required for model and GPU storage checks."""

import pytest


@pytest.fixture
def gpu_models():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("model checks require CUDA")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)
