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


def test_single_file_refuses_to_replace_existing_output_by_default(tmp_path):
    photo = tmp_path / "photo.jpg"
    output = tmp_path / "existing.jpg"
    _write_test_image(photo)
    output.write_bytes(b"keep me")

    result = CliRunner().invoke(main, [str(photo), "-o", str(output)])

    assert result.exit_code != 0
    assert "already exists" in result.output.lower()
    assert output.read_bytes() == b"keep me"


def test_single_file_overwrite_flag_replaces_existing_output(tmp_path):
    photo = tmp_path / "photo.jpg"
    output = tmp_path / "existing.jpg"
    _write_test_image(photo)
    output.write_bytes(b"replace me")

    result = CliRunner().invoke(main, [str(photo), "-o", str(output), "--overwrite"])

    assert result.exit_code == 0
    assert output.read_bytes() != b"replace me"


def test_batch_skips_existing_outputs_and_prints_summary(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    _write_test_image(input_dir / "one.jpg")
    _write_test_image(input_dir / "two.jpg")
    (input_dir / "notes.txt").write_text("not an image")
    (output_dir / "one.jpg").write_bytes(b"keep me")

    result = CliRunner().invoke(main, [str(input_dir), "--batch", "-o", str(output_dir)])

    assert result.exit_code == 0
    assert "1 processed, 2 skipped, 0 failed" in result.output
    assert (output_dir / "one.jpg").read_bytes() == b"keep me"
    assert (output_dir / "two.jpg").exists()


def test_batch_failure_returns_nonzero_and_continues(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_test_image(input_dir / "bad.jpg")
    _write_test_image(input_dir / "good.jpg")

    from photo_enhance import cli

    real_process_one = cli._process_one

    def fail_one(input_path, output_path, preset):
        if input_path.name == "bad.jpg":
            raise OSError("simulated write failure")
        real_process_one(input_path, output_path, preset)

    monkeypatch.setattr(cli, "_process_one", fail_one)

    result = CliRunner().invoke(main, [str(input_dir), "--batch", "-o", str(output_dir)])

    assert result.exit_code == 1
    assert "FAIL bad.jpg" in result.output
    assert "1 processed, 0 skipped, 1 failed" in result.output
    assert (output_dir / "good.jpg").exists()


def test_unknown_preset_is_rejected_by_click(tmp_path):
    photo = tmp_path / "photo.jpg"
    _write_test_image(photo)

    result = CliRunner().invoke(main, [str(photo), "--preset", "not-real"])

    assert result.exit_code != 0
    assert "invalid value for '--preset'" in result.output.lower()
