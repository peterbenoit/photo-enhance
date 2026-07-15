# Changelog

All notable changes to this project are logged here, newest first.

## 2026-07-15 — Initial scaffold

- Set up repo structure per spec: `src/photo_enhance/`, `tests/`, `examples/`.
- Pinned project to Python 3.12 via `uv` (`.python-version`), chosen over the
  system default 3.14 for safer `opencv-python` wheel compatibility.
- `git init` — local repo only, no remote configured yet.
- Implemented core pipeline (`auto_levels.py`):
  - `gray_world_white_balance()` — per-channel mean scaling to a neutral gray.
  - `auto_levels()` — per-channel percentile black/white point stretch (default
    0.5% clip each end).
  - `apply_clahe()` — CLAHE on the LAB L channel only, to avoid per-channel
    color shifts.
  - `auto_enhance()` — combines the three in sequence.
- Implemented preset system (`presets.py`):
  - JSON-defined per-channel tone curves (linear interpolation between control
    points) plus a saturation multiplier.
  - Shipped 4 presets: `warm_film`, `cool_moody`, `high_contrast_bw`,
    `faded_vintage`.
- Implemented `imageio_utils.py` — PIL/OpenCV bridging with EXIF byte
  preservation on load/save.
- Implemented CLI (`cli.py`) — `enhance <input> [-o output] [--preset name]
  [--batch]`, single-file and batch/folder modes, skips unsupported files in
  batch mode instead of crashing.
- Added a local Flask web UI (`web.py` + `templates/index.html`) as an addition
  beyond the original spec's CLI-only design — decision made because the CLI
  alone was judged unlikely to get revisited without a visual entry point.
  Runs on `localhost:5050`, in-memory processing, no persistence, no network
  calls.
- Added `.vscode/launch.json` so the web UI can be started with VS Code's Run
  button instead of a typed command.
- Added tests: `test_auto_levels.py` (white balance, levels, CLAHE, full
  pipeline) and `test_presets.py` (LUT interpolation, preset application).
- **Bug found and fixed:** `preset_data/` was originally named `presets/`,
  colliding with the `presets.py` module in the same package. Python resolved
  `photo_enhance.presets` to the module, not the directory, so
  `importlib.resources.files()` silently looked in the wrong place and
  `list_presets()` returned nothing. Renamed the directory to `preset_data/`
  to remove the collision. Caught by the test suite (`test_presets.py`), not
  by chance.
- Verified end-to-end: `uv sync --extra web` installs clean; full pytest suite
  (10 tests) passes; CLI single-file, `--preset`, and `--batch` (with a mixed
  file-type folder, including one non-image file that's skipped without
  crashing) all confirmed against a synthetic test photo, with measurable
  contrast increase (std 44 → 75) and channel rebalancing; Flask web UI
  confirmed serving the form with all four presets listed and `/enhance`
  returning valid before/after JPEG data URIs for both no-preset and
  preset-selected submissions.
