"""Pillow/OpenCV bridging with explicit fidelity and metadata behavior."""

from dataclasses import dataclass
import io
import os
from pathlib import Path
import tempfile
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ORIENTATION_TAG = 274
BITS_PER_SAMPLE_TAG = 258


class UnsupportedImageError(ValueError):
    """Raised when decoding would silently discard unsupported image data."""


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata retained across the 8-bit BGR enhancement pipeline."""

    exif: bytes | None = None
    icc_profile: bytes | None = None
    dpi: tuple[float, float] | None = None


def _has_transparency(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _is_high_bit_depth(image: Image.Image) -> bool:
    if image.mode == "I" or image.mode == "F" or image.mode.startswith("I;16"):
        return True
    tag_data = getattr(image, "tag_v2", None)
    bits_per_sample = tag_data.get(BITS_PER_SAMPLE_TAG) if tag_data is not None else None
    if bits_per_sample is None:
        # Pillow exposes 16-bit RGB PNGs as mode=RGB; the decoder raw mode is
        # the only place the source depth remains visible before loading.
        for tile in image.tile:
            raw_mode = tile.args[0] if isinstance(tile.args, tuple) else tile.args
            if isinstance(raw_mode, str) and ";16" in raw_mode:
                return True
        return False
    values = bits_per_sample if isinstance(bits_per_sample, tuple) else (bits_per_sample,)
    return any(value > 8 for value in values)


def _decode_bgr(
    source: Path | BinaryIO,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> tuple[np.ndarray, ImageMetadata]:
    with Image.open(source) as image:
        width, height = image.size
        if max_dimension is not None and (width > max_dimension or height > max_dimension):
            raise UnsupportedImageError(
                f"Image dimensions cannot exceed {max_dimension:,} pixels per side."
            )
        if max_pixels is not None and width * height > max_pixels:
            raise UnsupportedImageError(f"Image cannot exceed {max_pixels:,} decoded pixels.")
        if _has_transparency(image):
            raise UnsupportedImageError(
                "Transparent images are not supported yet because enhancement would "
                "discard the alpha channel."
            )
        if _is_high_bit_depth(image):
            raise UnsupportedImageError(
                "Images above 8 bits per channel are not supported yet because enhancement "
                "would reduce their bit depth."
            )

        oriented = ImageOps.exif_transpose(image)
        exif = oriented.getexif()
        exif.pop(ORIENTATION_TAG, None)
        exif_bytes = exif.tobytes() if exif else None
        icc_profile = image.info.get("icc_profile")
        dpi = image.info.get("dpi")
        rgb = oriented.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)

    metadata = ImageMetadata(exif=exif_bytes, icc_profile=icc_profile, dpi=dpi)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR), metadata


def load_bgr(path: Path) -> tuple[np.ndarray, ImageMetadata]:
    """Load a file as oriented, 8-bit, three-channel BGR plus preservable metadata."""
    return _decode_bgr(Path(path))


def load_bgr_bytes(
    encoded: bytes,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> tuple[np.ndarray, ImageMetadata]:
    """Decode uploaded bytes using the same fidelity contract as file loading."""
    return _decode_bgr(
        io.BytesIO(encoded),
        max_pixels=max_pixels,
        max_dimension=max_dimension,
    )


def _save_options(
    suffix: str,
    metadata: ImageMetadata | None,
    quality: int | None,
) -> dict:
    suffix = suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported output format '{suffix or '(none)'}'.")
    if quality is not None and not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    if quality is not None and suffix not in {".jpg", ".jpeg", ".webp"}:
        raise ValueError("quality can only be used with JPEG or WebP output")

    options: dict = {}
    if suffix in {".jpg", ".jpeg"}:
        options.update(quality=quality or 92, optimize=True)
    elif suffix == ".webp":
        options.update(quality=quality or 90, method=4)
    elif suffix == ".png":
        options.update(optimize=True, compress_level=6)
    elif suffix in {".tif", ".tiff"}:
        options.update(compression="tiff_lzw")

    if metadata is not None:
        metadata_formats = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        if metadata.exif and suffix in metadata_formats:
            options["exif"] = metadata.exif
        if metadata.icc_profile and suffix in metadata_formats:
            options["icc_profile"] = metadata.icc_profile
        if metadata.dpi and suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            options["dpi"] = metadata.dpi
    return options


def save_bgr(
    path: Path,
    img_bgr: np.ndarray,
    metadata: ImageMetadata | None = None,
    *,
    quality: int | None = None,
) -> None:
    """Atomically save a BGR image with format-aware options and optional metadata."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    path = Path(path)
    save_options = _save_options(path.suffix, metadata, quality)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        image.save(temp_path, **save_options)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
