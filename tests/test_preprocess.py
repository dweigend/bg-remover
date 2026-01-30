"""Tests for image preprocessing."""

import torch
from PIL import Image
from src.birefnet.preprocess import preprocess


class TestPreprocess:
    def test_output_tensor_shape(self, sample_image: Image.Image) -> None:
        result = preprocess(sample_image)
        assert result.tensor.shape == (1, 3, 1024, 1024)

    def test_custom_size(self, sample_image: Image.Image) -> None:
        result = preprocess(sample_image, size=512)
        assert result.tensor.shape == (1, 3, 512, 512)

    def test_preserves_original_size(self, sample_image: Image.Image) -> None:
        result = preprocess(sample_image)
        assert result.original_size == sample_image.size

    def test_tensor_dtype(self, sample_image: Image.Image) -> None:
        result = preprocess(sample_image)
        assert result.tensor.dtype == torch.float32

    def test_rgb_conversion(self, test_files_dir) -> None:
        # Create RGBA image
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = preprocess(rgba)
        assert result.tensor.shape[1] == 3  # 3 channels

    def test_normalization_applied(self, sample_image: Image.Image) -> None:
        result = preprocess(sample_image)
        # After ImageNet normalization, values should not be in [0, 1]
        # They can go negative or above 1
        tensor = result.tensor
        assert tensor.min() < 0 or tensor.max() > 1
