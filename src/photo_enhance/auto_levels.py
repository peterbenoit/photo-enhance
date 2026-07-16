"""Classical CV auto-enhance: gray-world white balance, percentile levels, CLAHE."""

import cv2
import numpy as np


def _validate_image(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must have shape (height, width, 3) in BGR channel order")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("img must have non-zero height and width")
    if img.dtype != np.uint8:
        raise ValueError("img must use uint8 pixels in the 0-255 range")


def _validate_number(name: str, value: float, *, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value >= maximum):
        upper = f" and less than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be at least {minimum}{upper}")


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Scale each BGR channel so its mean matches the overall gray mean."""
    _validate_image(img)
    result = img.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    for c in range(3):
        if channel_means[c] > 0:
            result[:, :, c] *= gray_mean / channel_means[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def auto_levels(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """Per-channel black/white point stretch, clipping the tail `clip_percent`% on each end."""
    _validate_image(img)
    _validate_number("clip_percent", clip_percent, minimum=0, maximum=50)
    result = img.astype(np.float32)
    for c in range(3):
        channel = result[:, :, c]
        low, high = np.percentile(channel, [clip_percent, 100 - clip_percent])
        if high <= low:
            continue
        channel = (channel - low) * (255.0 / (high - low))
        result[:, :, c] = np.clip(channel, 0, 255)
    return result.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Adaptive histogram equalization on luminance only (LAB L channel), to avoid color shifts."""
    _validate_image(img)
    _validate_number("clip_limit", clip_limit, minimum=0)
    if clip_limit == 0:
        raise ValueError("clip_limit must be greater than 0")
    if isinstance(tile_grid_size, bool) or not isinstance(tile_grid_size, int) or tile_grid_size <= 0:
        raise ValueError("tile_grid_size must be a positive integer")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_enhance(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """Full pipeline: white balance -> levels -> CLAHE. Expects/returns BGR uint8."""
    _validate_image(img)
    _validate_number("clip_percent", clip_percent, minimum=0, maximum=50)
    result = gray_world_white_balance(img)
    result = auto_levels(result, clip_percent=clip_percent)
    result = apply_clahe(result)
    return result
