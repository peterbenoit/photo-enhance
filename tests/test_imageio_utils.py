from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_enhance.imageio_utils import save_bgr


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
