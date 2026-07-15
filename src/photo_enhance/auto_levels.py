"""Classical CV auto-enhance: gray-world white balance, percentile levels, CLAHE."""

import cv2
import numpy as np


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Scale each BGR channel so its mean matches the overall gray mean."""
    result = img.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    for c in range(3):
        if channel_means[c] > 0:
            result[:, :, c] *= gray_mean / channel_means[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def auto_levels(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """Per-channel black/white point stretch, clipping the tail `clip_percent`% on each end."""
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
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_enhance(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """Full pipeline: white balance -> levels -> CLAHE. Expects/returns BGR uint8."""
    result = gray_world_white_balance(img)
    result = auto_levels(result, clip_percent=clip_percent)
    result = apply_clahe(result)
    return result
