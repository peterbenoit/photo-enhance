"""PIL <-> OpenCV bridging with EXIF preservation."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_bgr(path: Path) -> tuple[np.ndarray, bytes | None]:
    """Return (BGR uint8 array, raw exif bytes or None)."""
    with Image.open(path) as pil_img:
        exif = pil_img.info.get("exif")
        rgb = pil_img.convert("RGB")
        arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), exif


def save_bgr(path: Path, img_bgr: np.ndarray, exif: bytes | None = None) -> None:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    save_kwargs = {"exif": exif} if exif else {}
    pil_img.save(path, **save_kwargs)


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
