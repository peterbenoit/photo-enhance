"""Dependency-friendly console entry point for the optional web interface."""

import importlib

import click


def main() -> None:
    try:
        web = importlib.import_module("photo_enhance.web")
    except ModuleNotFoundError as exc:
        if exc.name != "flask":
            raise
        click.echo(
            "The web interface requires Flask. Install it with: "
            "uv sync --extra web (or pip install 'photo-enhance[web]').",
            err=True,
        )
        raise SystemExit(2) from exc
    web.main()
