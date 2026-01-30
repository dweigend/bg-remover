"""Image preprocessing for BiRefNet."""

from PIL import Image
from torchvision import transforms

from .types import ProcessedImage

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_SIZE = 1024


def preprocess(image: Image.Image, size: int = DEFAULT_SIZE) -> ProcessedImage:
    """Convert PIL Image to normalized tensor for BiRefNet.

    Args:
        image: Input PIL Image (any mode, converted to RGB)
        size: Target size for square resize (default 1024)

    Returns:
        ProcessedImage with normalized tensor [1, 3, H, W] and original size
    """
    original_size = image.size  # (width, height)

    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return ProcessedImage(tensor=tensor, original_size=original_size)
