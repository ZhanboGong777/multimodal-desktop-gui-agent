"""Composable preprocessing steps that stabilise OCR.

Module contract: images are handled in **RGB** order. OpenCV helpers convert to
BGR internally and convert back before returning, so callers never have to think
about channel order. No function here writes to disk, and none of them mutate the
image they were given.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..config import PreprocessingConfig


class PreprocessingError(RuntimeError):
    """Raised when an image cannot be processed."""


def to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return an RGB ``uint8`` array without mutating the input."""
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise PreprocessingError(f"unsupported array shape: {array.shape}")
        return np.ascontiguousarray(array[:, :, :3])
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    raise PreprocessingError(f"unsupported image type: {type(image)!r}")


def to_pil(array: np.ndarray) -> Image.Image:
    """Wrap an RGB array as a Pillow image."""
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    raise PreprocessingError(f"unsupported array shape: {array.shape}")


def _bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_numpy(image), cv2.COLOR_RGB2BGR)


def _rgb(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def to_grayscale(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert to a single-channel grayscale image."""
    return to_pil(cv2.cvtColor(to_numpy(image), cv2.COLOR_RGB2GRAY))


def resize_image(
    image: Image.Image | np.ndarray,
    *,
    scale: float = 1.0,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """Resize by a scale factor or to an explicit ``(width, height)``."""
    array = to_numpy(image)
    if size is not None:
        width, height = size
    else:
        if scale <= 0:
            raise PreprocessingError(f"scale must be positive, got {scale}")
        width = max(1, round(array.shape[1] * scale))
        height = max(1, round(array.shape[0] * scale))
    if width <= 0 or height <= 0:
        raise PreprocessingError(f"target size must be positive, got {width}x{height}")
    if width == array.shape[1] and height == array.shape[0]:
        return to_pil(array)
    interpolation = cv2.INTER_AREA if width < array.shape[1] else cv2.INTER_LINEAR
    return to_pil(cv2.resize(_rgb(_bgr(image)), (width, height), interpolation=interpolation))


def gaussian_blur(image: Image.Image | np.ndarray, kernel_size: int = 3) -> Image.Image:
    """Gaussian blur; ``kernel_size`` must be a positive odd number."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise PreprocessingError(f"kernel_size must be a positive odd number, got {kernel_size}")
    return to_pil(_rgb(cv2.GaussianBlur(_bgr(image), (kernel_size, kernel_size), 0)))


def otsu_binarize(image: Image.Image | np.ndarray) -> Image.Image:
    """Otsu threshold, returning a grayscale image with values 0 and 255."""
    gray = cv2.cvtColor(to_numpy(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return to_pil(binary)


def sharpen_image(image: Image.Image | np.ndarray) -> Image.Image:
    """Unsharp mask; useful before OCR on low-contrast text."""
    source = to_numpy(image)
    blurred = cv2.GaussianBlur(source, (0, 0), 3)
    sharpened = cv2.addWeighted(source, 1.6, blurred, -0.6, 0)
    return to_pil(np.clip(sharpened, 0, 255).astype(np.uint8))


def crop_region(image: Image.Image | np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop with ``(left, top, right, bottom)`` bounds."""
    left, top, right, bottom = box
    array = to_numpy(image)
    height, width = array.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        raise PreprocessingError(f"crop box {box} does not overlap the image ({width}x{height})")
    return to_pil(array[top:bottom, left:right])


def apply_preprocessing(
    image: Image.Image | np.ndarray, config: PreprocessingConfig
) -> Image.Image:
    """Run the preprocessing steps enabled in the configuration, in order."""
    result: Image.Image | np.ndarray = image
    if config.resize_scale != 1.0:
        result = resize_image(result, scale=config.resize_scale)
    if config.grayscale:
        result = to_grayscale(result)
    if config.gaussian_blur:
        result = gaussian_blur(result)
    if config.otsu_threshold:
        result = otsu_binarize(result)
    return result if isinstance(result, Image.Image) else to_pil(result)
