"""Conservative tonal and detail adjustments for bird and nature photos."""

import cv2
import numpy as np

_CHUNK_ROWS = 256
_BLUR_OVERLAP = 8


def _validate_image(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("img must have non-empty shape (height, width, 3)")
    if img.dtype != np.uint8:
        raise ValueError("img must use uint8 pixels in the 0-255 range")


def _validate_strength(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number from 0 to 1")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be from 0 to 1")
    return float(value)


def apply_nature_adjustments(
    img: np.ndarray,
    *,
    shadows: float = 0.0,
    highlights: float = 0.0,
    vibrance: float = 0.0,
    detail: float = 0.0,
    denoise: float = 0.0,
) -> np.ndarray:
    """Apply restrained wildlife-oriented adjustments to a BGR uint8 image.

    Tonal changes operate on LAB luminance so plumage colors are not shifted.
    Vibrance favors less-saturated colors, while detail uses thresholded unsharp
    masking to avoid amplifying the smallest noise variations.
    """
    _validate_image(img)
    shadows = _validate_strength("shadows", shadows)
    highlights = _validate_strength("highlights", highlights)
    vibrance = _validate_strength("vibrance", vibrance)
    detail = _validate_strength("detail", detail)
    denoise = _validate_strength("denoise", denoise)

    result = img.copy()

    if denoise > 0:
        sigma = 0.55 + 1.35 * denoise
        source = result
        denoised = result.copy()
        for start in range(0, result.shape[0], _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, result.shape[0])
            source_start = max(0, start - _BLUR_OVERLAP)
            source_stop = min(result.shape[0], stop + _BLUR_OVERLAP)
            stripe = source[source_start:source_stop]
            softened = cv2.GaussianBlur(stripe, (0, 0), sigmaX=sigma, sigmaY=sigma)
            offset = start - source_start
            softened = softened[offset:offset + stop - start]
            denoised[start:stop] = cv2.addWeighted(
                source[start:stop],
                1.0 - 0.7 * denoise,
                softened,
                0.7 * denoise,
                0,
            )
        result = denoised

    if shadows > 0 or highlights > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0].astype(np.float32) / 255.0
        shadow_lift = 0.22 * shadows * (1.0 - luminance) ** 2
        highlight_recovery = 0.18 * highlights * luminance**3
        lab[:, :, 0] = np.clip(
            (luminance + shadow_lift - highlight_recovery) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if vibrance > 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]
        adaptive_gain = 1.0 + 0.55 * vibrance * (1.0 - saturation / 255.0)
        hsv[:, :, 1] = np.clip(saturation * adaptive_gain, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if detail > 0:
        source = result
        sharpened_result = result.copy()
        for start in range(0, result.shape[0], _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, result.shape[0])
            source_start = max(0, start - _BLUR_OVERLAP)
            source_stop = min(result.shape[0], stop + _BLUR_OVERLAP)
            stripe = source[source_start:source_stop].astype(np.float32)
            blurred = cv2.GaussianBlur(stripe, (0, 0), sigmaX=1.15, sigmaY=1.15)
            offset = start - source_start
            stripe = stripe[offset:offset + stop - start]
            blurred = blurred[offset:offset + stop - start]
            high_frequency = stripe - blurred
            threshold = np.max(np.abs(high_frequency), axis=2, keepdims=True) >= 2.0
            sharpened = stripe + 1.25 * detail * high_frequency
            sharpened_result[start:stop] = np.where(
                threshold,
                np.clip(sharpened, 0, 255),
                stripe,
            ).astype(np.uint8)
        result = sharpened_result

    return result
