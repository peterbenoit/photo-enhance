import numpy as np
import pytest

from photo_enhance.auto_levels import (
    AutoSettings,
    analyze_auto,
    apply_clahe,
    auto_enhance,
    auto_levels,
    gray_world_white_balance,
)


def _flat_image(bgr_value: tuple[int, int, int], size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr_value
    return img


def test_gray_world_white_balance_corrects_color_cast():
    # Strong blue cast: B channel much brighter than G/R.
    img = _flat_image((220, 100, 100))
    result = gray_world_white_balance(img)
    means = result.reshape(-1, 3).mean(axis=0)
    # Channels should be pulled much closer together than the original 120-point spread.
    assert means.max() - means.min() < 5


def test_gray_world_white_balance_preserves_neutral_gray():
    img = _flat_image((128, 128, 128))
    result = gray_world_white_balance(img)
    assert np.allclose(result, 128, atol=1)


def test_auto_levels_stretches_to_full_range():
    # Low-contrast image confined to the 100-150 band.
    rng = np.random.default_rng(42)
    img = rng.integers(100, 151, size=(32, 32, 3), dtype=np.uint8)
    result = auto_levels(img, clip_percent=0.5)
    assert result.min() < 20
    assert result.max() > 235


def test_auto_levels_ignores_outliers_via_percentile_clip():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    img[0, 0] = [0, 0, 0]
    img[0, 1] = [255, 255, 255]
    result = auto_levels(img, clip_percent=5.0)
    # The dominant 128 value should stretch to mid-range, not be squashed by the two outlier pixels.
    assert 100 < np.median(result) < 160


def test_apply_clahe_increases_local_contrast_without_changing_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(80, 180, size=(64, 64, 3), dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result.std() >= img.std()


def test_auto_enhance_pipeline_runs_end_to_end():
    rng = np.random.default_rng(1)
    img = rng.integers(90, 160, size=(48, 48, 3), dtype=np.uint8)
    result = auto_enhance(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_auto_analysis_returns_bounded_reproducible_settings_and_metrics():
    image = _flat_image((80, 120, 180), size=64)
    first = analyze_auto(image)
    second = analyze_auto(image.copy())

    assert first == second
    assert 0 <= first.settings.white_balance <= 1
    assert 0 <= first.settings.levels <= 1
    assert 0 <= first.settings.local_contrast <= 1
    assert first.metrics.color_cast > 0
    assert 0 <= first.metrics.neutral_fraction <= 1


def test_auto_enhance_uses_the_provided_settings_exactly():
    rng = np.random.default_rng(3)
    image = rng.integers(70, 180, size=(48, 48, 3), dtype=np.uint8)
    disabled = auto_enhance(image, settings=AutoSettings(0, 0, 0))
    full = auto_enhance(image, settings=AutoSettings(1, 1, 1))

    assert np.array_equal(disabled, image)
    assert not np.array_equal(full, image)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((8, 8), dtype=np.uint8), "shape"),
        (np.zeros((8, 8, 4), dtype=np.uint8), "shape"),
        (np.zeros((0, 8, 3), dtype=np.uint8), "non-zero"),
        (np.zeros((8, 8, 3), dtype=np.float32), "uint8"),
    ],
)
def test_auto_enhance_rejects_invalid_image_contract(image, message):
    with pytest.raises(ValueError, match=message):
        auto_enhance(image)


@pytest.mark.parametrize("clip_percent", [-1, 50, float("nan"), "0.5"])
def test_auto_levels_rejects_invalid_clip_percent(clip_percent):
    with pytest.raises(ValueError, match="clip_percent"):
        auto_levels(_flat_image((100, 100, 100)), clip_percent=clip_percent)


@pytest.mark.parametrize("clip_limit", [0, -1, float("inf"), "2"])
def test_apply_clahe_rejects_invalid_clip_limit(clip_limit):
    with pytest.raises(ValueError, match="clip_limit"):
        apply_clahe(_flat_image((100, 100, 100)), clip_limit=clip_limit)


@pytest.mark.parametrize("tile_grid_size", [0, -1, 1.5, True])
def test_apply_clahe_rejects_invalid_tile_grid_size(tile_grid_size):
    with pytest.raises(ValueError, match="tile_grid_size"):
        apply_clahe(_flat_image((100, 100, 100)), tile_grid_size=tile_grid_size)
