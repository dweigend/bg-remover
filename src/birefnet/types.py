"""Shared data types for BiRefNet pipeline."""

from dataclasses import dataclass

import torch


@dataclass
class ProcessedImage:
    """Container for preprocessed image data.

    Attributes:
        tensor: Normalized image tensor with shape [1, 3, H, W]
        original_size: Original image dimensions as (width, height)
    """

    tensor: torch.Tensor
    original_size: tuple[int, int]
