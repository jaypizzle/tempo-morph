# Setup

## Prerequisites

- Ubuntu 24.04 (other Linux likely fine, untested)
- Python 3.10+
- ffmpeg

```bash
sudo apt install ffmpeg python3-venv
```

## Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `mediapipe<0.10.20`. Newer versions removed the `solutions` API used by this script.

## Optional: ROCm acceleration for rembg

If you already have ROCm 6 installed and configured (including `HSA_OVERRIDE_GFX_VERSION` if your card is not officially supported), you can install the ROCm provider for ONNX Runtime instead of the CPU one:

```bash
pip uninstall onnxruntime
pip install onnxruntime-rocm
```

rembg will detect the ROCm provider automatically. No script change needed. If anything goes wrong it will fall back to CPU.

This is optional and unlikely to make a meaningful difference for this workload (segmentation runs once per photo and is cached). Skip if you don't already have ROCm working.

## Verify install

```bash
python -c "import cv2, mediapipe, rembg, PIL, numpy, tqdm; print('OK')"
ffmpeg -version | head -1
```

# Usage

## Basic

```bash
python tempo_morph.py /path/to/photos /path/to/output.mp4
```

The script will:
1. Scan the folder for images, sort by EXIF date (falls back to date parsed from filename)
2. Preprocess each photo (alignment + segmentation), caching results in `<input_folder>/.morph_cache/`
3. Render the video with progress bar, overlaying the month/year on each held frame (Italian by default, English with `--date-lang en`)

First run on 73 photos: expect 5-15 minutes for preprocessing on CPU, plus rendering time. Subsequent runs reuse the cache and skip straight to rendering.

## Interactive mode

The easiest way to configure a run — prompts for every option with defaults shown:

```bash
python tempo_morph.py --interactive
# or with folders already known:
python tempo_morph.py /path/to/photos output.mp4 --interactive
```

At the end of the prompts the script prints the equivalent CLI command so you
can rerun with the same settings next time without going through the prompts again.

## All flags

```
--interactive, -i      Guided prompt for all parameters before rendering.
                       Prints the equivalent CLI command at the end.
--hold S               Hold duration in seconds for each photo (default: 1.1).
--hold-ends S          Hold duration for first and last photo (default: 2.5).
--morph S              Morph transition duration in seconds (default: 0.7).
--orientation O        Output orientation: landscape (1920×1080, default),
                       portrait (1080×1920), or square (1080×1080).
                       Portrait/square are center-cropped from the 1920×1080
                       aligned frame — no fill. Does not invalidate the cache.
--zoom F               Face zoom level (default: 1.0). <1.0 zooms out (more
                       head visible), >1.0 zooms in. Invalidates cache.
--date-label           Overlay month/year at bottom-right (default: on).
--no-date-label        Disable the date overlay.
--date-lang LANG       Language for the date label: it (Italian, default) or en (English).
                       Also configurable via --interactive.
--music PATH           Audio file to mix in. Loops if shorter than the video;
                       fades out 2s before the end.
--music-style STYLE    Use a bundled track by mood: calm, relaxing, upbeat,
                       nostalgic, emotional. Shortcut for --music audio/<file>.
                       Ignored if --music is also given.
--music-volume F       Volume level 0.0–1.0 (default: 0.8).
--limit N              Use only the first N photos — for smoke tests.
--clear-cache          Force re-preprocessing of all photos.
```

## Smoke test

```bash
python tempo_morph.py /path/to/photos /tmp/smoke.mp4 --limit 5
```

## Adding music

Five royalty-free tracks by Kevin MacLeod (CC BY 4.0) are listed in `audio/README.md`.
The MP3 files are not in git — download them once:

```bash
cd audio/
curl -L -o "Gymnopedie No 1.mp3"     "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Gymnopedie%20No%201.mp3"
curl -L -o "Airport Lounge.mp3"      "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Airport%20Lounge.mp3"
curl -L -o "Carefree.mp3"            "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Carefree.mp3"
curl -L -o "Comfortable Mystery.mp3" "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Comfortable%20Mystery.mp3"
curl -L -o "Heartbreaking.mp3"       "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Heartbreaking.mp3"
cd ..
```

Then pass a mood to the script:

```bash
python tempo_morph.py /path/to/photos output.mp4 --music-style calm
```

Available styles: `calm`, `relaxing`, `upbeat`, `nostalgic`, `emotional`.
You can also pass any audio file directly with `--music /path/to/track.mp3`.

## Force re-preprocess

If you change preprocessing parameters at the top of the script (resolution, eye targets, rembg model), the cache invalidates automatically. If for any other reason you want a clean run:

```bash
python tempo_morph.py /path/to/photos /path/to/output.mp4 --clear-cache
```

## Tuning

All tunable parameters are at the top of `tempo_morph.py` in a clearly marked section. Edit and re-run. Render is fast (cache hit) so iteration on render-time parameters (`HOLD_SECONDS`, `MORPH_SECONDS`, `BACKGROUND_BLUR_KERNEL`, etc.) is quick.

# Troubleshooting

**"No face detected in <file>"**: The photo is skipped. If it's a photo you want included, check that it's reasonably frontal and well-lit, or replace it.

**Halos around hair**: Try `REMBG_MODEL = "isnet-general-use"` at the top of the script. Slower but better on fine details.

**Face too small/large in frame**: Adjust `EYE_LEFT_TARGET` and `EYE_RIGHT_TARGET`. Move them closer together for a smaller face, farther apart for larger. Move both up or down to shift vertical position.

**Morph looks weird**: First, shorten `MORPH_SECONDS` to 0.4-0.5 — faster morphs hide artifacts. If it's a specific pair of photos, one of them may be off-axis; consider excluding it.

**ffmpeg not found**: `sudo apt install ffmpeg`.

**MediaPipe install fails**: On Ubuntu 24.04 you may need a specific Python version. Try `python3.11 -m venv venv` if 3.12 has issues.
