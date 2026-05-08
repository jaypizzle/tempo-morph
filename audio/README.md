# Audio tracks

This folder holds the royalty-free music tracks used by `--music-style`.
The MP3 files are **not stored in git** — download them once with the commands below.

All tracks are by **Kevin MacLeod** (incompetech.com), licensed **CC BY 4.0**.
Attribution: "Track name" by Kevin MacLeod — https://incompetech.com

---

## Download

```bash
cd audio/

curl -L -o "Gymnopedie No 1.mp3"     "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Gymnopedie%20No%201.mp3"
curl -L -o "Airport Lounge.mp3"      "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Airport%20Lounge.mp3"
curl -L -o "Carefree.mp3"            "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Carefree.mp3"
curl -L -o "Comfortable Mystery.mp3" "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Comfortable%20Mystery.mp3"
curl -L -o "Heartbreaking.mp3"       "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Heartbreaking.mp3"
```

---

## Tracks

| Style | File | Title | Duration |
|---|---|---|---|
| `calm` | `Gymnopedie No 1.mp3` | Gymnopedie No. 1 | 3:07 |
| `relaxing` | `Airport Lounge.mp3` | Airport Lounge | 5:07 |
| `upbeat` | `Carefree.mp3` | Carefree | 3:25 |
| `nostalgic` | `Comfortable Mystery.mp3` | Comfortable Mystery | 3:56 |
| `emotional` | `Heartbreaking.mp3` | Heartbreaking | 1:36 |

---

## Usage

```bash
# Use a bundled track by style
python tempo_morph.py /path/to/photos output.mp4 --music-style calm

# Use any custom file
python tempo_morph.py /path/to/photos output.mp4 --music /path/to/your/track.mp3

# Adjust volume (0.0–1.0, default 0.8)
python tempo_morph.py /path/to/photos output.mp4 --music-style upbeat --music-volume 0.6
```
