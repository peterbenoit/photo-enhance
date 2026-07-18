"""Conservative tonal and detail adjustments for bird and nature photos."""

from dataclasses import dataclass

import cv2
import numpy as np

_CHUNK_ROWS = 256
_BLUR_OVERLAP = 8
_ANALYSIS_MAX_DIMENSION = 512


@dataclass(frozen=True)
class NatureSettings:
    """Reproducible strengths for wildlife-oriented adjustments."""

    shadows: float
    highlights: float
    vibrance: float
    detail: float
    denoise: float

    def __post_init__(self) -> None:
        for name in ("shadows", "highlights", "vibrance", "detail", "denoise"):
            _validate_strength(name, getattr(self, name))


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


def _analysis_sample(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    scale = min(1.0, _ANALYSIS_MAX_DIMENSION / max(height, width))
    if scale >= 1.0:
        return img
    return cv2.resize(
        img,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def analyze_nature(img: np.ndarray) -> NatureSettings:
    """Choose restrained tonal, color, detail, and denoise strengths."""
    _validate_image(img)
    sample = _analysis_sample(img)
    luminance = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p25, p95 = np.percentile(luminance, [25.0, 95.0])
    saturation = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    mean_saturation = float(saturation.mean() / 255.0)

    blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=1.0, sigmaY=1.0)
    noise_estimate = float(np.median(np.abs(luminance - blurred)))

    shadows = float(np.clip((90.0 - p25) / 90.0, 0.0, 1.0) * 0.45)
    highlights = float(np.clip((p95 - 210.0) / 45.0, 0.0, 1.0) * 0.40)
    vibrance = float(np.clip((0.32 - mean_saturation) / 0.32, 0.0, 1.0) * 0.25)
    denoise = float(np.clip((noise_estimate - 2.0) / 12.0, 0.0, 1.0) * 0.35)
    detail = float(np.clip(0.22 - 0.30 * denoise, 0.08, 0.22))

    return NatureSettings(
        shadows=round(shadows, 2),
        highlights=round(highlights, 2),
        vibrance=round(vibrance, 2),
        detail=round(detail, 2),
        denoise=round(denoise, 2),
    )


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

    preserve_grayscale = np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(
        img[:, :, 1], img[:, :, 2]
    )
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

    if preserve_grayscale:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.merge((gray, gray, gray))
    return result
