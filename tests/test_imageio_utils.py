from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from photo_enhance.imageio_utils import (
    ImageMetadata,
    UnsupportedImageError,
    load_bgr,
    save_bgr,
)


def _srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_save_bgr_replaces_destination_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "photo.jpg"
    destination.write_bytes(b"original destination")
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    replacements: list[tuple[Path, Path]] = []

    from photo_enhance import imageio_utils

    real_replace = imageio_utils.os.replace

    def record_replace(source, target):
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(imageio_utils.os, "replace", record_replace)

    save_bgr(destination, image)

    assert len(replacements) == 1
    assert replacements[0][0].parent == destination.parent
    assert replacements[0][1] == destination
    assert Image.open(destination).size == (8, 8)


def test_save_bgr_cleans_up_temp_file_when_encoding_fails(tmp_path, monkeypatch):
    destination = tmp_path / "photo.jpg"
    destination.write_bytes(b"original destination")
    image = np.full((8, 8, 3), 120, dtype=np.uint8)

    def fail_save(self, path, **kwargs):
        raise OSError("simulated encoding failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)

    with pytest.raises(OSError, match="simulated encoding failure"):
        save_bgr(destination, image)

    assert destination.read_bytes() == b"original destination"
    assert list(tmp_path.iterdir()) == [destination]


def test_load_applies_exif_orientation_and_normalizes_tag(tmp_path):
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (6, 4), (20, 80, 140))
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "Test Artist"
    image.save(source, exif=exif)

    decoded, metadata = load_bgr(source)

    assert decoded.shape == (6, 4, 3)
    normalized = Image.Exif()
    normalized.load(metadata.exif)
    assert 274 not in normalized
    assert normalized[315] == "Test Artist"


def test_exif_icc_and_dpi_survive_round_trip(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    image = Image.new("RGB", (12, 10), (60, 90, 120))
    exif = Image.Exif()
    exif[315] = "Test Artist"
    profile = _srgb_profile()
    image.save(source, exif=exif, icc_profile=profile, dpi=(144, 144))

    decoded, metadata = load_bgr(source)
    save_bgr(output, decoded, metadata)

    with Image.open(output) as saved:
        assert saved.getexif()[315] == "Test Artist"
        assert saved.info["icc_profile"] == profile
        assert saved.info["dpi"] == pytest.approx((144, 144), abs=1)


@pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif", ".bmp", ".webp"])
def test_advertised_formats_round_trip_as_bgr_uint8(tmp_path, suffix):
    path = tmp_path / f"photo{suffix}"
    source = np.full((10, 12, 3), (30, 100, 180), dtype=np.uint8)

    save_bgr(path, source)
    decoded, _metadata = load_bgr(path)

    assert decoded.shape == source.shape
    assert decoded.dtype == np.uint8


def test_transparent_input_is_rejected_instead_of_flattened(tmp_path):
    source = tmp_path / "transparent.png"
    Image.new("RGBA", (8, 8), (100, 120, 140, 100)).save(source)

    with pytest.raises(UnsupportedImageError, match="alpha channel"):
        load_bgr(source)


def test_16_bit_input_is_rejected_instead_of_downconverted(tmp_path):
    source = tmp_path / "high-depth.png"
    pixels = np.full((8, 8), 40000, dtype=np.uint16)
    Image.fromarray(pixels).save(source)

    with pytest.raises(UnsupportedImageError, match="above 8 bits"):
        load_bgr(source)


def test_16_bit_rgb_png_is_rejected_instead_of_downconverted(tmp_path):
    import cv2

    source = tmp_path / "high-depth-rgb.png"
    pixels = np.full((8, 8, 3), 40000, dtype=np.uint16)
    assert cv2.imwrite(str(source), pixels)

    with pytest.raises(UnsupportedImageError, match="above 8 bits"):
        load_bgr(source)


@pytest.mark.parametrize("mode", ["L", "CMYK"])
def test_supported_non_rgb_modes_convert_to_bgr(tmp_path, mode):
    source = tmp_path / f"{mode}.tif"
    color = 120 if mode == "L" else (10, 20, 30, 5)
    Image.new(mode, (8, 6), color).save(source)

    decoded, _metadata = load_bgr(source)

    assert decoded.shape == (6, 8, 3)
    assert decoded.dtype == np.uint8


def test_quality_is_only_valid_for_lossy_output(tmp_path):
    image = np.full((8, 8, 3), 120, dtype=np.uint8)

    with pytest.raises(ValueError, match="JPEG or WebP"):
        save_bgr(tmp_path / "photo.png", image, quality=90)


def test_metadata_can_be_omitted_explicitly(tmp_path):
    output = tmp_path / "photo.jpg"
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    exif = Image.Exif()
    exif[315] = "Test Artist"
    metadata = ImageMetadata(exif=exif.tobytes(), icc_profile=_srgb_profile(), dpi=(144, 144))

    save_bgr(output, image, metadata=None)

    with Image.open(output) as saved:
        assert not saved.getexif()
        assert "icc_profile" not in saved.info
