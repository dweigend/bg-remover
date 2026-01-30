"""Tests for model loading and device detection."""

import torch
from src.birefnet.model import get_device, load_model


class TestGetDevice:
    def test_returns_valid_device_string(self) -> None:
        device = get_device()
        assert device in ("mps", "cuda", "cpu")

    def test_device_is_available(self) -> None:
        device = get_device()
        if device == "mps":
            assert torch.backends.mps.is_available()
        elif device == "cuda":
            assert torch.cuda.is_available()


class TestLoadModel:
    def test_returns_model_instance(self) -> None:
        model = load_model()
        assert model is not None

    def test_model_in_eval_mode(self) -> None:
        model = load_model()
        assert not model.training

    def test_model_is_cached(self) -> None:
        model1 = load_model()
        model2 = load_model()
        assert model1 is model2

    def test_explicit_device(self) -> None:
        model = load_model(device="cpu")
        # Model should work on CPU
        assert model is not None
