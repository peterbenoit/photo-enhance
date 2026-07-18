import numpy as np
import pytest

from photo_enhance.nature import apply_nature_adjustments


def test_zero_nature_adjustments_are_a_noop_without_mutating_input():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)

    result = apply_nature_adjustments(image)

    assert np.array_equal(result, image)
    assert result is not image


def test_shadows_lift_dark_tones_and_highlights_recover_bright_tones():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, :16] = 30
    image[:, 16:] = 230

    result = apply_nature_adjustments(image, shadows=1.0, highlights=1.0)

    assert result[:, :16].mean() > image[:, :16].mean()
    assert result[:, 16:].mean() < image[:, 16:].mean()
    assert np.max(np.ptp(result.astype(int), axis=2)) <= 2


def test_vibrance_boosts_muted_color_more_than_saturated_color():
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :8] = (100, 125, 150)
    image[:, 8:] = (0, 0, 220)

    result = apply_nature_adjustments(image, vibrance=1.0)

    muted_change = np.abs(result[:, :8].astype(int) - image[:, :8]).mean()
    saturated_change = np.abs(result[:, 8:].astype(int) - image[:, 8:]).mean()
    assert muted_change > saturated_change


def test_detail_increases_edge_contrast_and_denoise_reduces_noise():
    edge = np.full((64, 64, 3), 80, dtype=np.uint8)
    edge[:, 32:] = 170
    detailed = apply_nature_adjustments(edge, detail=1.0)
    assert detailed[:, 31].mean() < edge[:, 31].mean()
    assert detailed[:, 32].mean() > edge[:, 32].mean()

    rng = np.random.default_rng(7)
    noisy = np.clip(128 + rng.normal(0, 14, size=(64, 64, 3)), 0, 255).astype(np.uint8)
    denoised = apply_nature_adjustments(noisy, denoise=1.0)
    assert denoised.std() < noisy.std()


@pytest.mark.parametrize("name", ["shadows", "highlights", "vibrance", "detail", "denoise"])
def test_nature_adjustments_reject_invalid_strengths(name):
    image = np.full((8, 8, 3), 128, dtype=np.uint8)

    with pytest.raises(ValueError, match=name):
        apply_nature_adjustments(image, **{name: 1.1})
