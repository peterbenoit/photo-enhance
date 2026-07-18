"""Load and apply creative filter presets (tone curves + saturation) after auto-enhance."""

import json
import re
from copy import deepcopy
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Required, TypedDict, cast

import cv2
import numpy as np

from photo_enhance.finishing import apply_finishing
from photo_enhance.nature import apply_nature_adjustments

PRESETS_PACKAGE = "photo_enhance.preset_data"
PRESET_SCHEMA_VERSION = 1
_PRESET_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class PresetDefaults(TypedDict, total=False):
    intensity: int
    shadows: int
    highlights: int
    vibrance: int
    detail: int
    denoise: int
    temperature: int
    fade: int
    vignette: int
    grain: int


class Preset(TypedDict, total=False):
    schema_version: Required[int]
    name: Required[str]
    description: Required[str]
    curve: Required[dict[str, list[list[int]]]]
    saturation: Required[float]
    category: str
    swatch: list[str]
    defaults: PresetDefaults


def _list_builtin_presets() -> list[str]:
    files = resources.files(PRESETS_PACKAGE).iterdir()
    return sorted(f.name.removesuffix(".json") for f in files if f.name.endswith(".json"))


def _validated_user_dir(user_dir: Path | None) -> Path | None:
    if user_dir is None:
        return None
    directory = Path(user_dir).expanduser()
    if not directory.is_dir():
        raise ValueError(f"User preset directory does not exist or is not a folder: {directory}")
    return directory


def list_presets(user_dir: Path | None = None) -> list[str]:
    names = set(_list_builtin_presets())
    directory = _validated_user_dir(user_dir)
    if directory is not None:
        names.update(
            path.stem
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() == ".json"
            and _PRESET_NAME.fullmatch(path.stem)
        )
    return sorted(names)


def _validate_preset(data: object, *, name: str) -> Preset:
    if not isinstance(data, dict):
        raise ValueError(f"Preset '{name}' must contain a JSON object")
    if data.get("schema_version") != PRESET_SCHEMA_VERSION:
        raise ValueError(f"Preset '{name}' must use schema_version {PRESET_SCHEMA_VERSION}")
    for field in ("name", "description"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Preset '{name}' requires a non-empty {field}")

    curves = data.get("curve")
    if not isinstance(curves, dict):
        raise ValueError(f"Preset '{name}' requires a curve object")
    for channel in ("r", "g", "b"):
        points = curves.get(channel)
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"Preset '{name}' curve.{channel} requires at least two points")
        previous_x = -1
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
                or any(not 0 <= value <= 255 for value in point)
            ):
                raise ValueError(
                    f"Preset '{name}' curve.{channel} points must be [x, y] integers from 0 to 255"
                )
            if point[0] <= previous_x:
                raise ValueError(
                    f"Preset '{name}' curve.{channel} x values must be strictly increasing"
                )
            previous_x = point[0]

    saturation = data.get("saturation")
    if (
        isinstance(saturation, bool)
        or not isinstance(saturation, (int, float))
        or not np.isfinite(saturation)
        or not 0 <= saturation <= 2
    ):
        raise ValueError(f"Preset '{name}' saturation must be from 0 to 2")

    category = data.get("category", "creative")
    if category not in {"creative", "nature"}:
        raise ValueError(f"Preset '{name}' category must be creative or nature")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Preset '{name}' defaults must be an object")
    allowed_defaults = set(PresetDefaults.__annotations__)
    unknown_defaults = set(defaults) - allowed_defaults
    if unknown_defaults:
        raise ValueError(
            f"Preset '{name}' has unknown defaults: {', '.join(sorted(unknown_defaults))}"
        )
    for key, value in defaults.items():
        minimum = -100 if key == "temperature" else 0
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 100:
            raise ValueError(f"Preset '{name}' defaults.{key} must be from {minimum} to 100")
    return cast(Preset, data)


@cache
def _load_preset_cached(name: str) -> Preset:
    try:
        text = resources.files(PRESETS_PACKAGE).joinpath(f"{name}.json").read_text()
    except FileNotFoundError as e:
        available = ", ".join(_list_builtin_presets())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Preset '{name}' contains invalid JSON: {exc.msg}") from exc
    return _validate_preset(data, name=name)


def load_preset(name: str, user_dir: Path | None = None) -> Preset:
    """Load and validate a versioned preset without exposing the cached object to mutation."""
    if not isinstance(name, str) or not _PRESET_NAME.fullmatch(name):
        raise ValueError("Preset names may contain only letters, numbers, hyphens, and underscores")
    directory = _validated_user_dir(user_dir)
    if directory is not None:
        path = directory / f"{name}.json"
        if path.is_symlink():
            raise ValueError(f"User preset '{name}' must not be a symbolic link")
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Could not read user preset '{name}': {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"User preset '{name}' contains invalid JSON: {exc.msg}") from exc
            return _validate_preset(data, name=name)
    if name not in _list_builtin_presets():
        available = ", ".join(list_presets(directory))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return deepcopy(_load_preset_cached(name))


def list_preset_choices(user_dir: Path | None = None) -> list[dict]:
    """Presets with their display name/description, for populating UI controls."""
    choices = []
    for preset_id in list_presets(user_dir):
        preset = load_preset(preset_id, user_dir)
        choices.append(
            {
                "id": preset_id,
                "name": preset.get("name", preset_id),
                "description": preset.get("description", ""),
                "swatch": preset.get("swatch", ["#34383d", "#aeb4b8"]),
                "category": preset.get("category", "creative"),
                "defaults": preset.get("defaults", {}),
            }
        )
    return choices


@lru_cache(maxsize=128)
def _cached_curve_lut(points: tuple[tuple[int, int], ...]) -> np.ndarray:
    xs, ys = zip(*points, strict=True)
    lut = np.interp(np.arange(256), xs, ys)
    result = np.clip(lut, 0, 255).astype(np.uint8)
    result.flags.writeable = False
    return result


def _curve_to_lut(points: list[list[int]]) -> np.ndarray:
    """Build a 256-entry LUT from control points via linear interpolation."""
    key = tuple((point[0], point[1]) for point in points)
    return _cached_curve_lut(key)


def apply_preset(img: np.ndarray, preset: Preset) -> np.ndarray:
    """Apply per-channel tone curves and saturation. Expects and returns BGR uint8."""
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


def apply_preset_blended(img: np.ndarray, preset: Preset, intensity: float) -> np.ndarray:
    """Blend from the un-presetted image at zero to the full effect at one."""
    intensity = max(0.0, min(1.0, intensity))
    if intensity <= 0:
        return img.copy()
    full = apply_preset(img, preset)
    if intensity >= 1:
        return full
    return cv2.addWeighted(img, 1.0 - intensity, full, intensity, 0)


def apply_preset_with_defaults(img: np.ndarray, preset: Preset) -> np.ndarray:
    """Apply a preset at its authored intensity plus any bundled adjustments."""
    defaults = preset.get("defaults", {})
    result = apply_preset_blended(img, preset, defaults.get("intensity", 100) / 100.0)
    result = apply_nature_adjustments(
        result,
        shadows=defaults.get("shadows", 0) / 100.0,
        highlights=defaults.get("highlights", 0) / 100.0,
        vibrance=defaults.get("vibrance", 0) / 100.0,
        detail=defaults.get("detail", 0) / 100.0,
        denoise=defaults.get("denoise", 0) / 100.0,
    )
    return apply_finishing(
        result,
        temperature=defaults.get("temperature", 0) / 100.0,
        fade=defaults.get("fade", 0) / 100.0,
        vignette=defaults.get("vignette", 0) / 100.0,
        grain=defaults.get("grain", 0) / 100.0,
    )
