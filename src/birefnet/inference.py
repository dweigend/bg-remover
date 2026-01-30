"""Inference module for BiRefNet."""

import torch

from .types import ProcessedImage


def infer(model: torch.nn.Module, processed: ProcessedImage, device: str) -> torch.Tensor:
    """Run forward pass through BiRefNet.

    Args:
        model: Loaded BiRefNet model
        processed: Preprocessed image data
        device: Target device string

    Returns:
        Sigmoid-activated mask tensor with values in [0, 1]
    """
    tensor = processed.tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    # BiRefNet returns list of outputs, last one is the refined mask
    mask = output[-1]

    return torch.sigmoid(mask)
