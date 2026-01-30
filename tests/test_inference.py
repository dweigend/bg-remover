"""Tests for inference module."""

import torch
from PIL import Image
from src.birefnet.inference import infer
from src.birefnet.model import get_device, load_model
from src.birefnet.preprocess import preprocess


class TestInfer:
    def test_output_shape(self, sample_image: Image.Image) -> None:
        device = get_device()
        model = load_model(device)
        processed = preprocess(sample_image)

        result = infer(model, processed, device)

        # Should be [1, 1, H, W] after sigmoid
        assert len(result.shape) == 4
        assert result.shape[0] == 1

    def test_output_value_range(self, sample_image: Image.Image) -> None:
        device = get_device()
        model = load_model(device)
        processed = preprocess(sample_image)

        result = infer(model, processed, device)

        # Sigmoid output should be in [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_deterministic_output(self, sample_image: Image.Image) -> None:
        device = get_device()
        model = load_model(device)
        processed = preprocess(sample_image)

        result1 = infer(model, processed, device)
        result2 = infer(model, processed, device)

        assert torch.allclose(result1, result2)
