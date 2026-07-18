import numpy as np

from photo_enhance.presets import (
    _curve_to_lut,
    apply_preset,
    apply_preset_blended,
    apply_preset_with_defaults,
    list_preset_choices,
    list_presets,
    load_preset,
)


def test_list_presets_finds_all_shipped_presets():
    presets = list_presets()
    assert set(presets) == {
        "backlit_bird",
        "bird_natural",
        "cool_moody",
        "cross_process",
        "faded_vintage",
        "feather_detail",
        "golden_hour",
        "high_contrast_bw",
        "overcast",
        "soft_portrait",
        "teal_ember",
        "warm_film",
        "woodland",
    }


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


def test_list_preset_choices_includes_display_name_and_description():
    choices = list_preset_choices()
    ids = {c["id"] for c in choices}
    assert ids == {
        "backlit_bird",
        "bird_natural",
        "cool_moody",
        "cross_process",
        "faded_vintage",
        "feather_detail",
        "golden_hour",
        "high_contrast_bw",
        "overcast",
        "soft_portrait",
        "teal_ember",
        "warm_film",
        "woodland",
    }
    warm = next(c for c in choices if c["id"] == "warm_film")
    assert warm["name"] == "Warm Film"
    assert warm["description"]
    assert warm["swatch"] == ["#5e3023", "#efb267"]
    assert warm["category"] == "creative"
    bird = next(c for c in choices if c["id"] == "bird_natural")
    assert bird["category"] == "nature"
    assert bird["defaults"]["detail"] == 22


def test_apply_preset_blended_zero_intensity_is_a_noop():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    preset = load_preset("warm_film")
    result = apply_preset_blended(img, preset, 0.0)
    assert np.array_equal(result, img)


def test_apply_preset_blended_full_intensity_matches_apply_preset():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    preset = load_preset("warm_film")
    blended = apply_preset_blended(img, preset, 1.0)
    full = apply_preset(img, preset)
    assert np.array_equal(blended, full)


def test_apply_preset_blended_half_intensity_is_between_original_and_full():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    preset = load_preset("warm_film")
    full = apply_preset(img, preset).astype(int)
    half = apply_preset_blended(img, preset, 0.5).astype(int)
    original = img.astype(int)
    # Halfway result should sit strictly between original and full effect on channels that change.
    changed = full != original
    assert np.any(changed)
    assert np.all((half[changed] >= np.minimum(original[changed], full[changed])))
    assert np.all((half[changed] <= np.maximum(original[changed], full[changed])))


def test_apply_preset_with_defaults_uses_nature_adjustments():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    image[:, 16:] = 180
    preset = load_preset("feather_detail")

    with_defaults = apply_preset_with_defaults(image, preset)
    tone_only = apply_preset_blended(image, preset, preset["defaults"]["intensity"] / 100)

    assert not np.array_equal(with_defaults, tone_only)
