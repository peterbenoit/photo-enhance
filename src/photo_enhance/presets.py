"""Load and apply creative filter presets (tone curves + saturation) after auto-enhance."""

import json
from importlib import resources

import cv2
import numpy as np

PRESETS_PACKAGE = "photo_enhance.preset_data"


def list_presets() -> list[str]:
    files = resources.files(PRESETS_PACKAGE).iterdir()
    return sorted(f.name.removesuffix(".json") for f in files if f.name.endswith(".json"))


def load_preset(name: str) -> dict:
    try:
        text = resources.files(PRESETS_PACKAGE).joinpath(f"{name}.json").read_text()
    except FileNotFoundError as e:
        available = ", ".join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}") from e
    return json.loads(text)


def _curve_to_lut(points: list[list[int]]) -> np.ndarray:
    """Build a 256-entry LUT from control points via linear interpolation."""
    points = sorted(points)
    xs, ys = zip(*points)
    full_range = np.arange(256)
    lut = np.interp(full_range, xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_preset(img: np.ndarray, preset: dict) -> np.ndarray:
    """Apply a preset's per-channel tone curves and saturation adjustment. Expects/returns BGR uint8."""
    result = img.copy()

    curves = preset.get("curve", {})
    for channel_name, channel_index in (("b", 0), ("g", 1), ("r", 2)):
        points = curves.get(channel_name)
        if points:
            lut = _curve_to_lut(points)
            result[:, :, channel_index] = cv2.LUT(result[:, :, channel_index], lut)

    saturation = preset.get("saturation")
    if saturation is not None and saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result
