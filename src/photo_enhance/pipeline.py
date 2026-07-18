"""Shared, explainable enhancement pipeline used by the CLI and web UI."""

from dataclasses import dataclass

import cv2
import numpy as np

from photo_enhance.auto_levels import AutoAnalysis, AutoSettings, analyze_auto, auto_enhance
from photo_enhance.finishing import apply_finishing
from photo_enhance.nature import NatureSettings, analyze_nature, apply_nature_adjustments
from photo_enhance.presets import Preset, apply_preset_blended


@dataclass(frozen=True)
class EnhancementOptions:
    """Immutable recipe for every optional stage in one enhancement run."""

    auto_settings: AutoSettings | None = None
    nature_settings: NatureSettings | None = None
    preset: Preset | None = None
    preset_intensity: float = 1.0
    auto_scale: float = 1.0
    nature_scale: float = 1.0
    white_balance: bool = True
    levels: bool = True
    local_contrast: bool = True
    temperature: float = 0.0
    fade: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0
    grain_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("preset_intensity", "auto_scale", "nature_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.preset_intensity <= 1:
            raise ValueError("preset_intensity must be from 0 to 1")
        if not 0 <= self.auto_scale <= 2:
            raise ValueError("auto_scale must be from 0 to 2")
        if not 0 <= self.nature_scale <= 2:
            raise ValueError("nature_scale must be from 0 to 2")
        for name in ("white_balance", "levels", "local_contrast"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class EnhancementResult:
    """Rendered pixels plus the exact analysis and settings used to produce them."""

    image: np.ndarray
    auto_image: np.ndarray
    auto_analysis: AutoAnalysis
    auto_settings: AutoSettings
    nature_settings: NatureSettings


class EnhancementError(Exception):
    """Expected pipeline failure with a stable stage for user-facing error handling."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")


def _scaled(value: float, scale: float) -> float:
    return min(1.0, max(0.0, value * scale))


def enhance_image(
    image: np.ndarray,
    options: EnhancementOptions | None = None,
) -> EnhancementResult:
    """Analyze and render an image through Auto, preset, nature, and finishing stages."""
    options = options or EnhancementOptions()

    try:
        auto_analysis = analyze_auto(image)
        recommended = options.auto_settings or auto_analysis.settings
        auto_settings = AutoSettings(
            white_balance=(
                _scaled(recommended.white_balance, options.auto_scale)
                if options.white_balance
                else 0.0
            ),
            levels=_scaled(recommended.levels, options.auto_scale) if options.levels else 0.0,
            local_contrast=(
                _scaled(recommended.local_contrast, options.auto_scale)
                if options.local_contrast
                else 0.0
            ),
        )
        auto_image = auto_enhance(image, settings=auto_settings)
    except (TypeError, ValueError, cv2.error) as exc:
        raise EnhancementError("auto", exc) from exc

    try:
        recommended_nature = options.nature_settings or analyze_nature(auto_image)
        nature_settings = NatureSettings(
            shadows=_scaled(recommended_nature.shadows, options.nature_scale),
            highlights=_scaled(recommended_nature.highlights, options.nature_scale),
            vibrance=_scaled(recommended_nature.vibrance, options.nature_scale),
            detail=_scaled(recommended_nature.detail, options.nature_scale),
            denoise=_scaled(recommended_nature.denoise, options.nature_scale),
        )
    except (TypeError, ValueError, cv2.error) as exc:
        raise EnhancementError("nature", exc) from exc

    result = auto_image
    if options.preset is not None:
        try:
            result = apply_preset_blended(result, options.preset, options.preset_intensity)
        except (TypeError, ValueError, cv2.error) as exc:
            raise EnhancementError("preset", exc) from exc

    try:
        result = apply_nature_adjustments(
            result,
            shadows=nature_settings.shadows,
            highlights=nature_settings.highlights,
            vibrance=nature_settings.vibrance,
            detail=nature_settings.detail,
            denoise=nature_settings.denoise,
        )
    except (TypeError, ValueError, cv2.error) as exc:
        raise EnhancementError("nature", exc) from exc

    try:
        result = apply_finishing(
            result,
            temperature=options.temperature,
            fade=options.fade,
            vignette=options.vignette,
            grain=options.grain,
            grain_seed=options.grain_seed,
        )
    except (TypeError, ValueError, cv2.error) as exc:
        raise EnhancementError("finishing", exc) from exc

    return EnhancementResult(
        image=result,
        auto_image=auto_image,
        auto_analysis=auto_analysis,
        auto_settings=auto_settings,
        nature_settings=nature_settings,
    )
