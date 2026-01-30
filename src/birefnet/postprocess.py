"""Postprocessing for BiRefNet mask output."""

import torch
from PIL import Image


def mask_to_pil(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """Convert mask tensor to PIL Image at target size.

    Args:
        tensor: Mask tensor with shape [1, 1, H, W] and values in [0, 1]
        size: Target size as (width, height)

    Returns:
        Grayscale PIL Image resized to target dimensions
    """
    # Remove batch and channel dims, convert to numpy
    mask_np = tensor.squeeze().cpu().numpy()

    # Convert to 8-bit grayscale
    mask_uint8 = (mask_np * 255).astype("uint8")
    mask_pil = Image.fromarray(mask_uint8, mode="L")

    # Resize to original dimensions
    return mask_pil.resize(size, Image.Resampling.BILINEAR)


def remove_background(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply mask to image as alpha channel.

    Args:
        image: Original RGB image
        mask: Grayscale mask (white = foreground)

    Returns:
        RGBA image with transparent background
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgba = image.copy()
    rgba.putalpha(mask)

    return rgba
