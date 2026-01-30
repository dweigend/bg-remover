"""Pytest fixtures for BiRefNet tests."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def test_files_dir() -> Path:
    """Path to test_files directory."""
    return Path(__file__).parent.parent / "test_files"


@pytest.fixture
def sample_image(test_files_dir: Path) -> Image.Image:
    """Load bread.png as sample test image."""
    return Image.open(test_files_dir / "bread.png")


@pytest.fixture
def output_dir() -> Path:
    """Path to output directory, created if needed."""
    path = Path(__file__).parent.parent / "output"
    path.mkdir(exist_ok=True)
    return path
