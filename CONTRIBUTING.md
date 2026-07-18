# Contributing

Photo Enhance supports Python 3.11 and 3.12 on macOS and Linux. Install the
locked development environment, including the optional web UI, with:

```bash
uv sync --extra web --locked
```

## Local checks

Run the same checks as CI before committing:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest --cov --cov-report=term-missing
uv run pip-audit
```

Use `uv run pytest tests/test_pipeline.py -q` (or another focused file) while
iterating, then run the complete suite before handoff. The coverage threshold is
an initial floor, not a target; new behavior should include useful outcome
coverage without testing private implementation details indiscriminately.

## Builds

Build both distribution formats and inspect the result with:

```bash
uv build
uv run python -m zipfile --list dist/photo_enhance-*.whl
```

The installed-wheel smoke test creates a clean environment and exercises both
console entry points:

```bash
uv run pytest tests/test_packaged_wheel.py -q
```

## Releases

The project uses semantic versioning. Before a release:

1. Move completed notes into a versioned `CHANGELOG.md` section that separates
   user-visible additions, behavior changes, fixes, compatibility, and known
   limitations.
2. Update `project.version` in `pyproject.toml` and refresh `uv.lock`.
3. Run all local checks and build both the wheel and source distribution.
4. Tag the reviewed commit as `vMAJOR.MINOR.PATCH`; publish only artifacts built
   from that tag.

Never commit private photos, metadata-bearing samples, credentials, local agent
configuration, or generated build artifacts.
