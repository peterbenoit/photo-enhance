import numpy as np

from photo_enhance.presets import _curve_to_lut, apply_preset, list_presets, load_preset


def test_list_presets_finds_all_four_shipped_presets():
    presets = list_presets()
    assert set(presets) == {"warm_film", "cool_moody", "high_contrast_bw", "faded_vintage"}


def test_curve_to_lut_identity_curve_is_a_noop():
    lut = _curve_to_lut([[0, 0], [255, 255]])
    assert np.array_equal(lut, np.arange(256, dtype=np.uint8))


def test_curve_to_lut_interpolates_between_control_points():
    lut = _curve_to_lut([[0, 0], [128, 200], [255, 255]])
    assert lut[128] == 200
    assert lut[64] < 200  # midpoint of the rising segment should be below the peak


def test_apply_preset_bw_zeroes_saturation():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # pure red in BGR
    preset = load_preset("high_contrast_bw")
    result = apply_preset(img, preset)
    b, g, r = result[0, 0]
    assert abs(int(b) - int(g)) <= 1
    assert abs(int(g) - int(r)) <= 1
