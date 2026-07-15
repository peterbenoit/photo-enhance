import cv2
import numpy as np
from click.testing import CliRunner

from photo_enhance.cli import main


def _write_test_image(path) -> None:
    img = np.full((16, 16, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_single_file_refuses_to_overwrite_input_by_default(tmp_path):
    photo = tmp_path / "photo.jpg"
    _write_test_image(photo)

    runner = CliRunner()
    result = runner.invoke(main, [str(photo), "-o", str(photo)])

    assert result.exit_code != 0
    assert "overwrite" in result.output.lower()


def test_single_file_overwrite_flag_allows_it(tmp_path):
    photo = tmp_path / "photo.jpg"
    _write_test_image(photo)

    runner = CliRunner()
    result = runner.invoke(main, [str(photo), "-o", str(photo), "--overwrite"])

    assert result.exit_code == 0


def test_batch_refuses_output_dir_matching_input_dir_by_default(tmp_path):
    _write_test_image(tmp_path / "photo.jpg")

    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path), "--batch", "-o", str(tmp_path)])

    assert result.exit_code != 0
    assert "overwrite" in result.output.lower()


def test_batch_overwrite_flag_allows_matching_output_dir(tmp_path):
    _write_test_image(tmp_path / "photo.jpg")

    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path), "--batch", "-o", str(tmp_path), "--overwrite"])

    assert result.exit_code == 0
