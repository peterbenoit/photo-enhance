from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from photo_enhance.auto_levels import AutoSettings
from photo_enhance.nature import NatureSettings
from photo_enhance.pipeline import (
    EnhancementError,
    EnhancementOptions,
    enhance_image,
)


def _image() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.integers(60, 190, size=(40, 48, 3), dtype=np.uint8)


def test_options_are_immutable():
    options = EnhancementOptions()

    with pytest.raises(FrozenInstanceError):
        options.fade = 0.5


def test_shared_pipeline_returns_recipe_metrics_and_pixels_without_mutating_source():
    source = _image()
    original = source.copy()

    result = enhance_image(source)

    assert result.image.shape == source.shape
    assert result.image.dtype == np.uint8
    assert result.auto_analysis.settings == result.auto_settings
    assert 0 <= result.nature_settings.detail <= 1
    assert np.array_equal(source, original)


def test_explicit_zero_recipe_is_an_identity_render():
    source = _image()
    options = EnhancementOptions(
        auto_settings=AutoSettings(0, 0, 0),
        nature_settings=NatureSettings(0, 0, 0, 0, 0),
    )

    result = enhance_image(source, options)

    assert np.array_equal(result.image, source)


def test_pipeline_errors_identify_the_failed_stage():
    with pytest.raises(EnhancementError) as error:
        enhance_image(_image(), EnhancementOptions(fade=2.0))

    assert error.value.stage == "finishing"
