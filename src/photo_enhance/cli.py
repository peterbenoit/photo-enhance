"""CLI entry point: enhance <input> [-o output] [--preset name] [--batch]"""

from pathlib import Path

import click
import cv2
from PIL import Image, UnidentifiedImageError

from photo_enhance.auto_levels import auto_enhance
from photo_enhance.imageio_utils import is_supported_image, load_bgr, save_bgr
from photo_enhance.presets import apply_preset, list_presets, load_preset

PROCESSING_ERRORS = (
    OSError,
    ValueError,
    cv2.error,
    UnidentifiedImageError,
    Image.DecompressionBombError,
)


def _output_path(input_path: Path, output: Path | None, is_batch: bool) -> Path:
    if output is None:
        return input_path.with_stem(input_path.stem + "_enhanced")
    if is_batch:
        output.mkdir(parents=True, exist_ok=True)
        return output / input_path.name
    return output


def _process_one(
    input_path: Path,
    output_path: Path,
    preset: dict | None,
    *,
    strip_metadata: bool = False,
    quality: int | None = None,
) -> None:
    img, metadata = load_bgr(input_path)
    result = auto_enhance(img)
    if preset is not None:
        result = apply_preset(result, preset)
    save_bgr(output_path, result, metadata=None if strip_metadata else metadata, quality=quality)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None,
              help="Output file (single mode) or output folder (--batch mode).")
@click.option("--preset", "preset_name", type=click.Choice(list_presets()), default=None,
              help=f"Optional creative preset. Available: {', '.join(list_presets())}")
@click.option("--batch", is_flag=True, default=False, help="Treat input as a folder of photos.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Allow replacing source photos or existing output files.")
@click.option("--strip-metadata", is_flag=True, default=False,
              help="Remove EXIF (including GPS), ICC profile, and DPI metadata from output.")
@click.option("--quality", type=click.IntRange(1, 100), default=None, metavar="1-100",
              help="JPEG/WebP output quality (defaults: JPEG 92, WebP 90).")
def main(
    input_path: Path,
    output: Path | None,
    preset_name: str | None,
    batch: bool,
    overwrite: bool,
    strip_metadata: bool,
    quality: int | None,
) -> None:
    """Auto-enhance a photo (or a folder of photos with --batch)."""
    preset = load_preset(preset_name) if preset_name else None

    if batch:
        if not input_path.is_dir():
            raise click.UsageError("--batch requires INPUT_PATH to be a folder.")
        if output is not None and output.resolve() == input_path.resolve() and not overwrite:
            raise click.UsageError(
                "Output folder is the same as the input folder, which would overwrite your "
                "source photos. Pass --overwrite to allow this, or choose a different -o."
            )
        entries = sorted(p for p in input_path.iterdir() if p.is_file())
        files = [p for p in entries if is_supported_image(p)]
        if not files:
            click.echo(f"No supported images found in {input_path}")
            return
        processed = 0
        skipped = len(entries) - len(files)
        failed = 0
        for file_path in files:
            out_path = _output_path(file_path, output, is_batch=True)
            if out_path.exists() and not overwrite:
                skipped += 1
                click.echo(f"SKIP {file_path.name} (output already exists: {out_path})", err=True)
                continue
            try:
                _process_one(
                    file_path,
                    out_path,
                    preset,
                    strip_metadata=strip_metadata,
                    quality=quality,
                )
                processed += 1
                click.echo(f"OK   {file_path.name} -> {out_path}")
            except PROCESSING_ERRORS as exc:
                failed += 1
                click.echo(f"FAIL {file_path.name} ({exc})", err=True)
        click.echo(f"Summary: {processed} processed, {skipped} skipped, {failed} failed")
        if failed:
            raise click.exceptions.Exit(1)
    else:
        if not input_path.is_file():
            raise click.UsageError("INPUT_PATH must be a file (use --batch for a folder).")
        out_path = _output_path(input_path, output, is_batch=False)
        if out_path.resolve() == input_path.resolve() and not overwrite:
            raise click.UsageError(
                "Output path is the same as the input file, which would overwrite the original. "
                "Pass --overwrite to allow this, or choose a different -o."
            )
        if out_path.exists() and not overwrite:
            raise click.UsageError(
                f"Output already exists: {out_path}. Pass --overwrite to replace it, "
                "or choose a different -o."
            )
        try:
            _process_one(
                input_path,
                out_path,
                preset,
                strip_metadata=strip_metadata,
                quality=quality,
            )
        except PROCESSING_ERRORS as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"OK   {input_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
