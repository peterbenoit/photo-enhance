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


def list_preset_choices() -> list[dict]:
    """Presets with their display name/description, for populating UI controls."""
    choices = []
    for preset_id in list_presets():
        preset = load_preset(preset_id)
        choices.append({
            "id": preset_id,
            "name": preset.get("name", preset_id),
            "description": preset.get("description", ""),
        })
    return choices


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


def apply_preset_blended(img: np.ndarray, preset: dict, intensity: float) -> np.ndarray:
    """Blend between the un-presetted image (intensity=0) and the full preset effect (intensity=1)."""
    intensity = max(0.0, min(1.0, intensity))
    if intensity <= 0:
        return img.copy()
    full = apply_preset(img, preset)
    if intensity >= 1:
        return full
    return cv2.addWeighted(img, 1.0 - intensity, full, intensity, 0)
