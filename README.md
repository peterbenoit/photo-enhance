# Photo Auto-Enhance

A local Python tool that auto-corrects photo levels/white balance/contrast using
classical image processing (no ML model, no cloud calls), with optional creative
filter presets on top.

Two ways to use it:

- **Web UI** (recommended if you don't use the CLI often) — upload a photo once, then switch between filters and adjust intensity live, no re-upload needed. Runs entirely on `localhost`; nothing leaves the machine.
- **CLI** — for batch-processing a folder of photos.

See [`photo-auto-enhance-spec.md`](photo-auto-enhance-spec.md) for the original design spec, [`PROJECT_REVIEW.md`](PROJECT_REVIEW.md) for the outstanding review backlog, and [`TASKS.md`](TASKS.md) / [`CHANGELOG.md`](CHANGELOG.md) for build progress.

## Requirements

- macOS (Intel or Apple Silicon)
- Python 3.12 (pinned via `.python-version`; installed automatically by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — if you don't have it: `brew install uv`

## Setup

```bash
uv sync --extra web
```

This creates `.venv/`, installs the pinned Python 3.12, and installs all dependencies
(including Flask for the web UI).

## Running the web UI

**Easiest: VS Code Run button.** Open this folder in VS Code, open the Run and Debug
panel (⇧⌘D), select **"Web UI"**, and press the green play button. It starts the
server and opens `http://127.0.0.1:5050` in your browser automatically.

**Or from the terminal:**

```bash
uv run photo-enhance-web
```

Then open http://127.0.0.1:5050. Upload a photo once — switching the filter
dropdown or dragging the intensity slider re-applies the filter against the
already-uploaded image (no re-upload). The server keeps the last 20 uploaded
sessions in memory; restarting the server clears them.

By default the Flask dev server runs with debug off. For local debugging,
set `PHOTO_ENHANCE_DEBUG=1` before starting it.

## Running the CLI

```bash
# Single photo, writes photo_enhanced.jpg next to the original
uv run photo-enhance path/to/photo.jpg

# With a creative preset
uv run photo-enhance path/to/photo.jpg --preset warm_film

# Batch process a folder, writing to an output folder
uv run photo-enhance path/to/folder --batch -o path/to/output_folder
```

Available presets: `warm_film`, `cool_moody`, `high_contrast_bw`, `faded_vintage`
(defined as JSON tone curves in `src/photo_enhance/preset_data/`).

EXIF metadata is preserved where the source format supports it (JPEG/TIFF).

By default, the CLI refuses to let an output path overwrite its input (single
file, or a batch output folder that resolves to the input folder) — pass
`--overwrite` to allow it explicitly.

## Running tests

```bash
uv run pytest
```

## macOS setup notes

- `opencv-python` and `Pillow` both ship prebuilt wheels for `arm64` and `x86_64`
  Macs — `uv sync` just works, no compiling.
- Project is pinned to Python 3.12 (not the newest 3.14) specifically because
  `opencv-python` wheel coverage for very new Python versions can lag behind —
  3.12 is the safe, well-supported choice.
- **RAW support (stretch goal, not yet implemented):** if `rawpy` gets added later,
  it wraps `libraw`, which on Apple Silicon sometimes needs to be installed via
  Homebrew first (`brew install libraw`) before `pip install rawpy` behaves
  correctly. Do this before adding the dependency, not after hitting a build error.

## Project structure

```
photo-enhance/
├── README.md
├── TASKS.md                 # checkbox progress list
├── CHANGELOG.md              # log of changes made
├── pyproject.toml
├── .vscode/launch.json       # VS Code Run button config
├── src/photo_enhance/
│   ├── cli.py                 # CLI entry point
│   ├── web.py                 # Flask web UI entry point
│   ├── templates/index.html   # web UI page
│   ├── auto_levels.py         # white balance + levels + CLAHE
│   ├── presets.py              # preset loading/application
│   ├── imageio_utils.py        # PIL/OpenCV bridge + EXIF handling
│   └── preset_data/*.json      # tone-curve preset definitions
├── tests/
└── examples/                  # drop before/after sample images here (gitignored)
```

## Non-goals for MVP

No cloud/API calls, no user accounts, no ML model dependency. See the spec file
for the full list and the stretch goals considered out of scope for now (RAW
support, Real-ESRGAN hook, `.cube` LUT files).
