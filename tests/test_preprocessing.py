"""Tests for the composable preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gui_agent.config import PreprocessingConfig
from gui_agent.perception.preprocessing import (
    PreprocessingError,
    apply_preprocessing,
    crop_region,
    gaussian_blur,
    otsu_binarize,
    resize_image,
    sharpen_image,
    to_grayscale,
    to_numpy,
    to_pil,
)


def sample_image(width: int = 40, height: int = 20) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, width // 2 :] = 255
    return Image.fromarray(array, mode="RGB")


def test_to_numpy_accepts_pil_and_arrays() -> None:
    assert to_numpy(sample_image()).shape == (20, 40, 3)
    assert to_numpy(np.zeros((5, 5), dtype=np.uint8)).shape == (5, 5, 3)
    assert to_numpy(np.zeros((5, 5, 4), dtype=np.uint8)).shape == (5, 5, 3)


def test_to_numpy_rejects_unsupported_types() -> None:
    with pytest.raises(PreprocessingError):
        to_numpy("not an image")  # type: ignore[arg-type]


def test_to_numpy_rejects_unexpected_shapes() -> None:
    with pytest.raises(PreprocessingError):
        to_numpy(np.zeros((2, 2, 7), dtype=np.uint8))


def test_to_pil_rejects_unexpected_shapes() -> None:
    with pytest.raises(PreprocessingError):
        to_pil(np.zeros((2, 2, 5), dtype=np.uint8))


def test_to_pil_accepts_grayscale() -> None:
    assert to_pil(np.zeros((4, 4), dtype=np.uint8)).mode == "L"


def test_to_grayscale_returns_a_single_channel_image() -> None:
    gray = to_grayscale(sample_image())
    assert gray.mode == "L"
    assert gray.size == (40, 20)


def test_resize_by_scale_and_by_explicit_size() -> None:
    assert resize_image(sample_image(), scale=0.5).size == (20, 10)
    assert resize_image(sample_image(), size=(10, 5)).size == (10, 5)


def test_resize_rejects_a_non_positive_scale() -> None:
    with pytest.raises(PreprocessingError):
        resize_image(sample_image(), scale=0)


def test_resize_is_a_no_op_for_the_same_size() -> None:
    original = sample_image()
    assert np.array_equal(np.asarray(resize_image(original, scale=1.0)), np.asarray(original))


def test_gaussian_blur_keeps_the_size() -> None:
    assert gaussian_blur(sample_image(), 3).size == (40, 20)


@pytest.mark.parametrize("kernel", [0, 2, 4, -1])
def test_gaussian_blur_rejects_even_or_non_positive_kernels(kernel: int) -> None:
    with pytest.raises(PreprocessingError):
        gaussian_blur(sample_image(), kernel)


def test_otsu_binarize_produces_only_two_values() -> None:
    values = set(np.unique(np.asarray(otsu_binarize(sample_image()))))
    assert values.issubset({0, 255})


def test_sharpen_keeps_size_and_dtype() -> None:
    sharpened = sharpen_image(sample_image())
    assert sharpened.size == (40, 20)
    assert np.asarray(sharpened).dtype == np.uint8


def test_crop_region_returns_the_requested_area() -> None:
    assert crop_region(sample_image(), (0, 0, 10, 5)).size == (10, 5)


def test_crop_region_clamps_to_the_image_bounds() -> None:
    assert crop_region(sample_image(), (-5, -5, 100, 100)).size == (40, 20)


def test_crop_region_rejects_a_non_overlapping_box() -> None:
    with pytest.raises(PreprocessingError):
        crop_region(sample_image(), (100, 100, 120, 120))


def test_helpers_never_mutate_the_input_image() -> None:
    original = sample_image()
    before = np.asarray(original).copy()
    resize_image(original, scale=2.0)
    to_grayscale(original)
    gaussian_blur(original)
    otsu_binarize(original)
    sharpen_image(original)
    crop_region(original, (0, 0, 5, 5))
    apply_preprocessing(original, PreprocessingConfig(grayscale=True, resize_scale=0.5))
    assert np.array_equal(np.asarray(original), before)


def test_apply_preprocessing_respects_the_configuration() -> None:
    image = sample_image()
    assert apply_preprocessing(image, PreprocessingConfig()).size == image.size
    assert apply_preprocessing(image, PreprocessingConfig(grayscale=True)).mode == "L"
    assert apply_preprocessing(image, PreprocessingConfig(resize_scale=0.5)).size == (20, 10)


def test_apply_preprocessing_chains_steps() -> None:
    result = apply_preprocessing(
        sample_image(),
        PreprocessingConfig(grayscale=True, gaussian_blur=True, otsu_threshold=True),
    )
    assert set(np.unique(np.asarray(result))).issubset({0, 255})


def test_apply_preprocessing_accepts_numpy_input() -> None:
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    assert apply_preprocessing(array, PreprocessingConfig()).size == (10, 10)
