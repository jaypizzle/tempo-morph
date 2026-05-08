# TempoMorph

Generate a time-lapse video from a sequence of portrait photos of the same person,
with facial morphing transitions that make the passage of time feel continuous and organic.

## What it does

Given a folder of photos with similar framing (frontal portraits, subject centered),
produces an mp4 where each photo is held briefly and then morphs into the next via
Delaunay-triangulated face warping. The subject is segmented from the background so
the face acts as the visual anchor while the background dissolves separately and softly.

Works best with roughly evenly-spaced photos of the same person — monthly, yearly,
or any interval where the framing is consistent.

## Quick start

```bash
sudo apt install ffmpeg python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python tempo_morph.py /path/to/photos /path/to/output.mp4
```

Or use interactive mode to configure everything step by step:

```bash
python tempo_morph.py --interactive
```

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — install steps, usage details, troubleshooting

## Key flags

| Flag | Default | Effect |
|---|---|---|
| `--interactive` / `-i` | — | Guided setup: prompts for every option, then prints the equivalent CLI command. |
| `--zoom F` | `1.0` | Face zoom. `<1.0` = more head visible, `>1.0` = zoom in. |
| `--orientation O` | `landscape` | `landscape` (1920×1080), `portrait` (1080×1920), `square` (1080×1080). Center-cropped, no fill. |
| `--date-label` / `--no-date-label` | on | Month/year overlay at bottom-right. |
| `--date-lang LANG` | `it` | Date label language: `it` (Italian) or `en` (English). |
| `--music-style STYLE` | — | Use a bundled track by mood: `calm`, `relaxing`, `upbeat`, `nostalgic`, `emotional`. See `audio/README.md`. |
| `--music PATH` | — | Use any audio file. Loops and fades out at end. |
| `--limit N` | — | Use only the first N photos (smoke test). |
| `--clear-cache` | — | Force re-preprocessing all photos. |
