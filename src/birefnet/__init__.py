"""BiRefNet Background Removal Pipeline.

Usage:
    from src.birefnet import process_image
    from PIL import Image

    img = Image.open("input.png")
    result = process_image(img)
    result.save("output.png")
"""

from PIL import Image

from .inference import infer
from .model import get_device, load_model
from .postprocess import mask_to_pil, remove_background
from .preprocess import preprocess
from .types import ProcessedImage

__all__ = [
    "ProcessedImage",
    "get_device",
    "infer",
    "load_model",
    "mask_to_pil",
    "preprocess",
    "process_image",
    "remove_background",
]


def process_image(image: Image.Image, size: int = 1024) -> Image.Image:
    """Full pipeline: remove background from image.

    Args:
        image: Input PIL Image
        size: Processing size (default 1024)

    Returns:
        RGBA image with transparent background
    """
    device = get_device()
    model = load_model(device)

    processed = preprocess(image, size)
    mask_tensor = infer(model, processed, device)
    mask_pil = mask_to_pil(mask_tensor, processed.original_size)

    return remove_background(image, mask_pil)
