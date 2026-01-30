"""Integration tests for full pipeline."""

from pathlib import Path

from PIL import Image
from src.birefnet import process_image


class TestFullPipeline:
    def test_process_single_image(self, sample_image: Image.Image) -> None:
        result = process_image(sample_image)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == sample_image.size

    def test_output_has_transparency(self, sample_image: Image.Image) -> None:
        result = process_image(sample_image)

        r, g, b, a = result.split()
        alpha_values = list(a.get_flattened_data())

        # Should have both transparent and opaque pixels
        has_transparent = any(v < 128 for v in alpha_values)
        has_opaque = any(v > 128 for v in alpha_values)

        assert has_transparent, "Expected some transparent pixels"
        assert has_opaque, "Expected some opaque pixels"

    def test_save_output(self, sample_image: Image.Image, output_dir: Path) -> None:
        result = process_image(sample_image)
        output_path = output_dir / "bread_nobg.png"
        result.save(output_path)

        assert output_path.exists()
        saved = Image.open(output_path)
        assert saved.mode == "RGBA"

    def test_multiple_images(self, test_files_dir: Path, output_dir: Path) -> None:
        # Process a few test images
        test_images = ["croissant.png", "cupcake.png", "donut_pink.png"]

        for filename in test_images:
            image_path = test_files_dir / filename
            if not image_path.exists():
                continue

            image = Image.open(image_path)
            result = process_image(image)

            output_path = output_dir / f"{image_path.stem}_nobg.png"
            result.save(output_path)

            assert output_path.exists()
            assert result.mode == "RGBA"
