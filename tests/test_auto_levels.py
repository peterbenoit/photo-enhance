import numpy as np

from photo_enhance.auto_levels import apply_clahe, auto_enhance, auto_levels, gray_world_white_balance


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
