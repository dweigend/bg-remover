"""Model loading and device detection for BiRefNet."""

import os
from functools import lru_cache

import torch
from transformers import AutoModelForImageSegmentation

os.environ["HF_HUB_DISABLE_XET"] = "1"

MODEL_ID = "ZhengPeng7/BiRefNet"


def get_device() -> str:
    """Auto-detect the best available device.

    Returns:
        Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def load_model(device: str | None = None) -> AutoModelForImageSegmentation:
    """Load BiRefNet model from HuggingFace.

    Args:
        device: Target device. Auto-detected if None.

    Returns:
        Loaded model in eval mode on the specified device.
    """
    if device is None:
        device = get_device()

    torch.set_float32_matmul_precision("high")

    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.to(device)
    model.train(False)  # Set to evaluation mode

    return model
