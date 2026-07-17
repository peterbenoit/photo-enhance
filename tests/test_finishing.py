import numpy as np

from photo_enhance.finishing import apply_finishing


def test_zero_finishing_is_a_noop_without_mutating_input():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)

    result = apply_finishing(image)

    assert np.array_equal(result, image)
    assert result is not image


def test_vignette_darkens_edges_while_preserving_center():
    image = np.full((101, 101, 3), 200, dtype=np.uint8)

    result = apply_finishing(image, vignette=1.0)

    assert result[0, 0].mean() < 80
    assert result[50, 50].mean() == 200


def test_grain_is_visible_and_deterministic_for_a_session_seed():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    first = apply_finishing(image, grain=0.6, grain_seed=42)
    second = apply_finishing(image, grain=0.6, grain_seed=42)

    assert first.std() > 3
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, :, 0], first[:, :, 1])
    assert first.dtype == np.uint8
