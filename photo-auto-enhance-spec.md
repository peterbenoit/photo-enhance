# Photo Auto-Enhance CLI — Project Spec

## Goal

A local Python CLI tool that takes a photo (or folder of photos) and:
1. Auto-corrects levels/white balance/contrast using classical image processing (no ML model required for MVP)
2. Applies optional creative filter presets (LUTs / tone curves) on top

No cloud calls, no GPU requirement, runs on a laptop.

## Why classical CV instead of an ML model

"Auto levels" in tools like Lightroom/darktable's auto-exposure is mostly histogram-based math, not a neural net. A gray-world white balance + per-channel histogram stretch + CLAHE gets ~90% of the visual result AI enhancers give, with zero model weights to manage and instant runtime. Keep it this way for MVP. ML upscaling/denoising (Real-ESRGAN, GFPGAN) is a separate concern and out of scope here — call it out as a possible v2 integration, not a dependency.

## Core features (MVP)

1. **Auto white balance** — gray-world assumption (or Retinex-based as a stretch option)
2. **Auto levels** — per-channel black/white point clipping with a small percentile clip (e.g. clip bottom/top 0.5%) to avoid blowing out outliers
3. **Contrast** — CLAHE (adaptive histogram equalization) on luminance channel, not per RGB channel, to avoid color shifts
4. **CLI interface**:
   - `enhance <input> [-o output] [--preset name] [--batch]`
   - Single file or folder input
   - Preserves EXIF where possible
5. **Filter presets** — a `presets/` folder of simple LUT or curve definitions (start with 3-4: e.g. "warm film", "cool/moody", "high contrast B&W", "faded/vintage"). Presets apply after auto-correction, as a distinct optional step.

## Stretch goals (v2, not MVP)

- RAW file support (via `rawpy`) since source photos may come from a Lumix FZ80D
- Optional Real-ESRGAN hook for upscale/denoise as a separate flag, not a hard dependency
- Simple before/after preview (side-by-side output image) for quick QA
- `.cube` LUT file support so real film-emulation LUTs can be dropped in

## Tech stack

- Python 3.11+
- `opencv-python` for the CV operations (white balance, CLAHE, levels)
- `Pillow` for I/O and EXIF preservation
- `click` for the CLI
- No GPU, no torch/tensorflow dependency for MVP

## macOS setup notes

Target platform is a MacBook (Intel or Apple Silicon), no GPU/server required.

- `opencv-python` and `Pillow` both ship prebuilt wheels for `arm64` and `x86_64` Macs — plain `pip install` works, no compiling.
- If/when the RAW support stretch goal (`rawpy`) gets added: `rawpy` wraps `libraw`, which sometimes needs to be present via Homebrew first (`brew install libraw`) before `pip install rawpy` behaves correctly, especially on Apple Silicon. Document this in the README so it's not a fresh debug session later.
- Recommend a `venv` (or `uv`) per project rather than global installs, consistent with existing setup conventions.

## Suggested repo structure

```
photo-auto-enhance/
├── README.md
├── pyproject.toml
├── src/
│   └── photo_enhance/
│       ├── __init__.py
│       ├── cli.py
│       ├── auto_levels.py      # white balance + levels + CLAHE
│       ├── presets.py          # preset loading/application
│       └── presets/
│           ├── warm_film.json
│           ├── cool_moody.json
│           ├── high_contrast_bw.json
│           └── faded_vintage.json
├── tests/
│   └── test_auto_levels.py
└── examples/
    └── (before/after sample images)
```

## Non-goals for MVP

- No web UI
- No cloud/API calls
- No user accounts, no watermarking
- No ML model dependency (that's a clearly separated v2 flag, not baked in)

## Definition of done for MVP

- `photo-enhance path/to/photo.jpg` produces a visibly better-exposed, better-balanced output next to the original
- `--preset warm_film` applies a distinguishable creative look on top
- `--batch` processes a folder without crashing on mixed file types
- Basic test coverage on the `auto_levels` math (not the CLI plumbing)
