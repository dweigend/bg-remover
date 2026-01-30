"""Tests for postprocessing module."""

import torch
from PIL import Image

from src.birefnet.postprocess import mask_to_pil, remove_background


class TestMaskToPil:
    def test_output_type(self) -> None:
        tensor = torch.rand(1, 1, 1024, 1024)
        result = mask_to_pil(tensor, (100, 100))
        assert isinstance(result, Image.Image)

    def test_output_size(self) -> None:
        tensor = torch.rand(1, 1, 1024, 1024)
        result = mask_to_pil(tensor, (200, 150))
        assert result.size == (200, 150)

    def test_output_mode(self) -> None:
        tensor = torch.rand(1, 1, 1024, 1024)
        result = mask_to_pil(tensor, (100, 100))
        assert result.mode == "L"  # Grayscale

    def test_value_range(self) -> None:
        # Test with known values
        tensor = torch.ones(1, 1, 10, 10)  # All white
        result = mask_to_pil(tensor, (10, 10))
        pixels = list(result.get_flattened_data())
        assert all(p == 255 for p in pixels)


class TestRemoveBackground:
    def test_output_mode(self) -> None:
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        mask = Image.new("L", (100, 100), 255)
        result = remove_background(image, mask)
        assert result.mode == "RGBA"

    def test_alpha_from_mask(self) -> None:
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        mask = Image.new("L", (100, 100), 128)  # 50% opacity
        result = remove_background(image, mask)

        r, g, b, a = result.split()
        alpha_pixels = list(a.get_flattened_data())
        assert all(p == 128 for p in alpha_pixels)

    def test_rgb_preserved(self) -> None:
        image = Image.new("RGB", (100, 100), (100, 150, 200))
        mask = Image.new("L", (100, 100), 255)
        result = remove_background(image, mask)

        r, g, b, a = result.split()
        assert list(r.get_flattened_data())[0] == 100
        assert list(g.get_flattened_data())[0] == 150
        assert list(b.get_flattened_data())[0] == 200

    def test_handles_rgba_input(self) -> None:
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
        mask = Image.new("L", (100, 100), 255)
        result = remove_background(image, mask)
        assert result.mode == "RGBA"
