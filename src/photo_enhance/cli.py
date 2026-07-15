"""CLI entry point: enhance <input> [-o output] [--preset name] [--batch]"""

from pathlib import Path

import click

from photo_enhance.auto_levels import auto_enhance
from photo_enhance.imageio_utils import is_supported_image, load_bgr, save_bgr
from photo_enhance.presets import apply_preset, list_presets, load_preset


def _output_path(input_path: Path, output: Path | None, is_batch: bool) -> Path:
    if output is None:
        return input_path.with_stem(input_path.stem + "_enhanced")
    if is_batch:
        output.mkdir(parents=True, exist_ok=True)
        return output / input_path.name
    return output


def _process_one(input_path: Path, output_path: Path, preset: dict | None) -> None:
    img, exif = load_bgr(input_path)
    result = auto_enhance(img)
    if preset is not None:
        result = apply_preset(result, preset)
    save_bgr(output_path, result, exif=exif)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None,
              help="Output file (single mode) or output folder (--batch mode).")
@click.option("--preset", "preset_name", type=str, default=None,
              help=f"Optional creative preset. Available: {', '.join(list_presets())}")
@click.option("--batch", is_flag=True, default=False, help="Treat input as a folder of photos.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Allow output to overwrite the input file, or a batch output folder to be the input folder.")
def main(input_path: Path, output: Path | None, preset_name: str | None, batch: bool, overwrite: bool) -> None:
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
        files = sorted(p for p in input_path.iterdir() if is_supported_image(p))
        if not files:
            click.echo(f"No supported images found in {input_path}")
            return
        for file_path in files:
            out_path = _output_path(file_path, output, is_batch=True)
            try:
                _process_one(file_path, out_path, preset)
                click.echo(f"OK   {file_path.name} -> {out_path}")
            except Exception as exc:
                click.echo(f"SKIP {file_path.name} ({exc})", err=True)
    else:
        if not input_path.is_file():
            raise click.UsageError("INPUT_PATH must be a file (use --batch for a folder).")
        out_path = _output_path(input_path, output, is_batch=False)
        if out_path.resolve() == input_path.resolve() and not overwrite:
            raise click.UsageError(
                "Output path is the same as the input file, which would overwrite the original. "
                "Pass --overwrite to allow this, or choose a different -o."
            )
        _process_one(input_path, out_path, preset)
        click.echo(f"OK   {input_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
