"""Classical CV auto-enhance: analysis, white balance, levels, and CLAHE."""

from dataclasses import dataclass

import cv2
import numpy as np

_ANALYSIS_MAX_DIMENSION = 512


@dataclass(frozen=True)
class AutoSettings:
    """Reproducible strengths for the three base Auto correction stages."""

    white_balance: float
    levels: float
    local_contrast: float

    def __post_init__(self) -> None:
        _validate_strength("white_balance", self.white_balance)
        _validate_strength("levels", self.levels)
        _validate_strength("local_contrast", self.local_contrast)


@dataclass(frozen=True)
class ImageMetrics:
    """Small, serializable measurements used to explain Auto's decisions."""

    low_percentile: float
    high_percentile: float
    luminance_std: float
    shadow_fraction: float
    highlight_fraction: float
    color_cast: float
    neutral_fraction: float
    mean_saturation: float


@dataclass(frozen=True)
class AutoAnalysis:
    settings: AutoSettings
    metrics: ImageMetrics


def _validate_image(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must have shape (height, width, 3) in BGR channel order")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("img must have non-zero height and width")
    if img.dtype != np.uint8:
        raise ValueError("img must use uint8 pixels in the 0-255 range")


def _validate_number(
    name: str, value: float, *, minimum: float, maximum: float | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value >= maximum):
        upper = f" and less than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be at least {minimum}{upper}")


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


def analyze_auto(img: np.ndarray) -> AutoAnalysis:
    """Measure an image and return bounded, reproducible Auto stage strengths."""
    _validate_image(img)
    sample = _analysis_sample(img)
    luminance = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low, high = np.percentile(luminance, [1.0, 99.0])
    luminance_std = float(luminance.std())
    dynamic_range = float(high - low)

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2]
    neutral_mask = (saturation <= 40) & (value >= 32) & (value <= 240)
    neutral_fraction = float(neutral_mask.mean())

    channel_means = sample.reshape(-1, 3).mean(axis=0)
    mean_channel = max(float(channel_means.mean()), 1.0)
    color_cast = float((channel_means.max() - channel_means.min()) / mean_channel)

    white_balance_signal = np.clip((color_cast - 0.015) / 0.22, 0.0, 1.0)
    white_balance_cap = 0.85 if neutral_fraction >= 0.05 else 0.30
    white_balance = float(white_balance_signal * white_balance_cap)
    levels = float(np.clip((220.0 - dynamic_range) / 150.0, 0.0, 1.0) * 0.90)
    local_contrast = float(np.clip((52.0 - luminance_std) / 42.0, 0.0, 1.0) * 0.65)

    return AutoAnalysis(
        settings=AutoSettings(
            white_balance=round(white_balance, 2),
            levels=round(levels, 2),
            local_contrast=round(local_contrast, 2),
        ),
        metrics=ImageMetrics(
            low_percentile=round(float(low), 2),
            high_percentile=round(float(high), 2),
            luminance_std=round(luminance_std, 2),
            shadow_fraction=round(float((luminance <= 42).mean()), 4),
            highlight_fraction=round(float((luminance >= 235).mean()), 4),
            color_cast=round(color_cast, 4),
            neutral_fraction=round(neutral_fraction, 4),
            mean_saturation=round(float(saturation.mean() / 255.0), 4),
        ),
    )


def gray_world_white_balance(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Scale each BGR channel so its mean matches the overall gray mean."""
    _validate_image(img)
    strength = _validate_strength("strength", strength)
    if strength == 0:
        return img.copy()
    result: np.ndarray = img.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    for c in range(3):
        if channel_means[c] > 0:
            result[:, :, c] *= gray_mean / channel_means[c]
    balanced = np.clip(result, 0, 255).astype(np.uint8)
    if strength == 1:
        return balanced
    return cv2.addWeighted(img, 1.0 - strength, balanced, strength, 0)


def auto_levels(img: np.ndarray, clip_percent: float = 0.5, strength: float = 1.0) -> np.ndarray:
    """Per-channel black/white point stretch, clipping the tail `clip_percent`% on each end."""
    _validate_image(img)
    _validate_number("clip_percent", clip_percent, minimum=0, maximum=50)
    strength = _validate_strength("strength", strength)
    if strength == 0:
        return img.copy()
    result: np.ndarray = img.astype(np.float32)
    for c in range(3):
        channel = result[:, :, c]
        low, high = np.percentile(channel, [clip_percent, 100 - clip_percent])
        if high <= low:
            continue
        channel = (channel - low) * (255.0 / (high - low))
        result[:, :, c] = np.clip(channel, 0, 255)
    leveled: np.ndarray = result.astype(np.uint8)
    if strength == 1:
        return leveled
    return cv2.addWeighted(img, 1.0 - strength, leveled, strength, 0)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    strength: float = 1.0,
) -> np.ndarray:
    """Adaptive histogram equalization on luminance only (LAB L channel), to avoid color shifts."""
    _validate_image(img)
    _validate_number("clip_limit", clip_limit, minimum=0)
    if clip_limit == 0:
        raise ValueError("clip_limit must be greater than 0")
    if (
        isinstance(tile_grid_size, bool)
        or not isinstance(tile_grid_size, int)
        or tile_grid_size <= 0
    ):
        raise ValueError("tile_grid_size must be a positive integer")
    strength = _validate_strength("strength", strength)
    if strength == 0:
        return img.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if strength == 1:
        return contrasted
    return cv2.addWeighted(img, 1.0 - strength, contrasted, strength, 0)


def auto_enhance(
    img: np.ndarray,
    clip_percent: float = 0.5,
    settings: AutoSettings | None = None,
) -> np.ndarray:
    """Full pipeline: white balance -> levels -> CLAHE. Expects/returns BGR uint8."""
    _validate_image(img)
    _validate_number("clip_percent", clip_percent, minimum=0, maximum=50)
    settings = settings or analyze_auto(img).settings
    result = gray_world_white_balance(img, strength=settings.white_balance)
    result = auto_levels(result, clip_percent=clip_percent, strength=settings.levels)
    result = apply_clahe(result, strength=settings.local_contrast)
    return result
