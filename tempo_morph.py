#!/usr/bin/env python3
"""
Generate a time-lapse video with morphing transitions between portrait photos
of the same person taken over time.

Pipeline:
  1. Read photos from a folder, sort by EXIF date
  2. For each photo: detect facial landmarks, align, segment subject/background
     (results cached on disk)
  3. For each consecutive pair: static hold + triangulated face morph with
     separate background dissolve
  4. Encode to mp4 via ffmpeg

Usage:
    python tempo_morph.py /path/to/photos/folder /path/to/output.mp4
"""

import argparse
import hashlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
from rembg import remove, new_session
from tqdm import tqdm

# =============================================================================
# PARAMETERS — tweak these to tune the result
# =============================================================================

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 30

# Per-orientation config. The render loop always outputs 1920×1080 BGR frames;
# cropping/scaling to the final shape happens inside ffmpeg so preprocessing
# is orientation-independent and the cache is always reusable.
#
#   vf_filter:    ffmpeg -vf argument (None = no filter)
#   label_right:  rightmost visible pixel in the 1920×1080 frame
#   label_bottom: bottommost visible pixel  (always 1080 for all orientations)
#   label_font:   Pillow font size for the date overlay
#
#   square  — center-crop to 1080×1080:  x=[420,1500]
#   portrait — center-crop 608×1080 then scale to 1080×1920: x=[656,1264]
ORIENTATIONS = {
    "landscape": dict(vf_filter=None,                                        label_right=1920, label_bottom=1080, label_font=62),
    "square":    dict(vf_filter="crop=1080:1080:420:0",                      label_right=1500, label_bottom=1080, label_font=48),
    "portrait":  dict(vf_filter="crop=608:1080:656:0,scale=1080:1920:flags=lanczos", label_right=1264, label_bottom=1080, label_font=42),
}

HOLD_SECONDS = 1.1          # static pause on each photo
HOLD_SECONDS_ENDS = 2.5     # longer pause on first and last photo
MORPH_SECONDS = 0.7         # morph transition duration

# Target eye position in the aligned canvas (as fraction of width/height).
# Tweak EYE_Y to shift the face up/down; use --zoom at the command line to
# control how large the face appears (1.0 = default, <1.0 = zoom out).
EYE_SPREAD = 0.16   # baseline spread at zoom 1.0 (right_x - left_x)
EYE_Y      = 0.42   # eyes sit at 42% of canvas height
EYE_LEFT_TARGET  = (0.50 - EYE_SPREAD / 2, EYE_Y)
EYE_RIGHT_TARGET = (0.50 + EYE_SPREAD / 2, EYE_Y)

# Background blur during crossfade (0 = no blur, helps the face "pop")
BACKGROUND_BLUR_KERNEL = 15

# rembg model: "u2net" (general, good quality), "u2net_human_seg" (optimized
# for people, faster and more precise on portraits), "isnet-general-use"
# (often better on hair/fine details, slower)
REMBG_MODEL = "u2net_human_seg"

# Accepted image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# Bundled royalty-free tracks — stored in the audio/ folder next to this script.
# All by Kevin MacLeod (incompetech.com), licensed CC BY 4.0.
# Usage: --music-style <style>  →  automatically uses the matching bundled track.
AUDIO_DIR = Path(__file__).parent / "audio"
BUNDLED_TRACKS = {
    #  style       filename                    display title          duration
    "calm":      ("Gymnopedie No 1.mp3",       "Gymnopedie No. 1",    "3:07"),
    "relaxing":  ("Airport Lounge.mp3",        "Airport Lounge",      "5:07"),
    "upbeat":    ("Carefree.mp3",              "Carefree",            "3:25"),
    "nostalgic": ("Comfortable Mystery.mp3",   "Comfortable Mystery", "3:56"),
    "emotional": ("Heartbreaking.mp3",         "Heartbreaking",       "1:36"),
}

MONTH_NAMES = {
    "it": ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
           "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"],
    "en": ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"],
}

# =============================================================================


@dataclass
class PhotoData:
    """Preprocessed data for a single photo."""
    path: Path
    date_str: str                  # "YYYY:MM:DD ..." used for the date overlay
    aligned_bgr: np.ndarray        # aligned image, OUTPUT_WIDTH x OUTPUT_HEIGHT
    mask: np.ndarray               # subject mask, uint8 [0..255]
    landmarks: np.ndarray          # (N, 2) landmark coords on aligned canvas


# -----------------------------------------------------------------------------
# Reading and ordering
# -----------------------------------------------------------------------------

def get_exif_datetime(path: Path):
    """Extract DateTimeOriginal from EXIF. Returns 'YYYY:MM:DD HH:MM:SS' or None."""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        for tag_id, value in exif.items():
            if ExifTags.TAGS.get(tag_id) == "DateTimeOriginal":
                return value
    except Exception:
        return None
    return None


def get_filename_datetime(path: Path):
    """Parse a date from the filename as fallback. Returns 'YYYY:MM:DD 00:00:00' or None.

    Matches the first YYYYMMDD token with a valid month (01-12) and day (01-31).
    Covers common phone/Google Photos naming conventions (PXL_20210716_..., etc.).
    """
    m = re.search(
        r"(?<!\d)(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)",
        path.stem,
    )
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 00:00:00"
    return None


def get_photo_date(path: Path):
    """Return the best available date string for a photo, or None."""
    return get_exif_datetime(path) or get_filename_datetime(path)


def collect_photos(folder: Path):
    """Find all photos in folder and sort by date (EXIF, then filename fallback)."""
    photos = [p for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    dated = []
    undated = []
    for p in photos:
        dt = get_photo_date(p)
        if dt:
            dated.append((dt, p))
        else:
            undated.append(p)

    dated.sort(key=lambda x: x[0])

    if undated:
        print(f"⚠️  {len(undated)} photos with no date (EXIF or filename), skipped:")
        for p in undated:
            print(f"   - {p.name}")

    return dated   # list of (date_str, Path)


# -----------------------------------------------------------------------------
# Landmark detection and alignment
# -----------------------------------------------------------------------------

# MediaPipe Face Mesh indices for iris centers
# (478 landmarks total when refine_landmarks=True; iris points are 468-477)
LEFT_EYE_IDX = 468   # left iris center
RIGHT_EYE_IDX = 473  # right iris center


def detect_landmarks(image_bgr, face_mesh):
    """Detect 478 facial landmarks. Returns (N, 2) array or None if no face."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = image_bgr.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)


def align_image(image_bgr, landmarks):
    """Align image to standard canvas using eyes as reference points."""
    eye_l = landmarks[LEFT_EYE_IDX]
    eye_r = landmarks[RIGHT_EYE_IDX]

    target_l = np.array([EYE_LEFT_TARGET[0] * OUTPUT_WIDTH,
                         EYE_LEFT_TARGET[1] * OUTPUT_HEIGHT])
    target_r = np.array([EYE_RIGHT_TARGET[0] * OUTPUT_WIDTH,
                         EYE_RIGHT_TARGET[1] * OUTPUT_HEIGHT])

    # Compute similarity transform (rotation + scale + translation)
    src = np.array([eye_l, eye_r], dtype=np.float32)
    dst = np.array([target_l, target_r], dtype=np.float32)

    # estimateAffinePartial2D returns a 2x3 transform matrix
    M, _ = cv2.estimateAffinePartial2D(src, dst)

    aligned = cv2.warpAffine(
        image_bgr, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )

    # Apply same transform to landmarks
    landmarks_h = np.hstack([landmarks, np.ones((len(landmarks), 1))])
    aligned_landmarks = (M @ landmarks_h.T).T.astype(np.float32)

    return aligned, aligned_landmarks


# -----------------------------------------------------------------------------
# Subject segmentation
# -----------------------------------------------------------------------------

def segment_subject(image_bgr, session):
    """Segment subject from background. Returns uint8 mask [0..255]."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    result = remove(pil, session=session, only_mask=True)
    return np.array(result, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Preprocessing cache
# -----------------------------------------------------------------------------

def cache_key(path: Path):
    """Generate unique cache key based on path + mtime + relevant params."""
    stat = path.stat()
    key_str = f"{path.resolve()}|{stat.st_mtime}|{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
    key_str += f"|{EYE_LEFT_TARGET}|{EYE_RIGHT_TARGET}|{REMBG_MODEL}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def load_or_preprocess(path: Path, date_str: str, cache_dir: Path,
                        face_mesh, rembg_session, bar=None):
    """Load from cache if available, otherwise preprocess and save."""
    def status(msg):
        if bar is not None:
            bar.set_postfix_str(msg, refresh=True)

    key = cache_key(path)
    cache_file = cache_dir / f"{key}.npz"

    if cache_file.exists():
        status("cached ✓")
        data = np.load(cache_file)
        return PhotoData(
            path=path,
            date_str=date_str,
            aligned_bgr=data["aligned"],
            mask=data["mask"],
            landmarks=data["landmarks"],
        )

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read {path}")

    status("landmarks…")
    landmarks = detect_landmarks(image, face_mesh)
    if landmarks is None:
        raise RuntimeError(f"No face detected in {path.name}")

    aligned, aligned_landmarks = align_image(image, landmarks)

    status("segmenting…")
    mask = segment_subject(aligned, rembg_session)

    status("saving…")
    np.savez_compressed(
        cache_file,
        aligned=aligned,
        mask=mask,
        landmarks=aligned_landmarks,
    )

    status("done ✓")
    return PhotoData(
        path=path,
        date_str=date_str,
        aligned_bgr=aligned,
        mask=mask,
        landmarks=aligned_landmarks,
    )


# -----------------------------------------------------------------------------
# Morphing
# -----------------------------------------------------------------------------

def compute_delaunay_triangles(landmarks, width, height):
    """Compute Delaunay triangulation on landmarks + 4 canvas corners."""
    # Add 4 corners to cover the whole canvas during warp
    corners = np.array([
        [0, 0], [width - 1, 0],
        [0, height - 1], [width - 1, height - 1],
    ], dtype=np.float32)
    points = np.vstack([landmarks, corners])

    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        # Subdiv2D wants float tuples, points must lie inside rect
        x, y = float(p[0]), float(p[1])
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        subdiv.insert((x, y))

    triangles = subdiv.getTriangleList()
    # Convert triangle vertex coords to indices in the points array
    indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for px, py in pts:
            # Find nearest point index
            distances = np.sum((points - np.array([px, py])) ** 2, axis=1)
            idx.append(int(np.argmin(distances)))
        if len(set(idx)) == 3:  # discard degenerate triangles
            indices.append(idx)
    return np.array(indices, dtype=np.int32), points


def warp_triangle(src_img, src_tri, dst_tri, dst_img):
    """Affine-warp a single triangle from src to dst."""
    # Bounding boxes of the triangles
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))

    src_tri_offset = [(p[0] - r1[0], p[1] - r1[1]) for p in src_tri]
    dst_tri_offset = [(p[0] - r2[0], p[1] - r2[1]) for p in dst_tri]

    src_crop = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_crop.size == 0:
        return

    M = cv2.getAffineTransform(
        np.float32(src_tri_offset), np.float32(dst_tri_offset)
    )
    warped = cv2.warpAffine(
        src_crop, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), (1.0, 1.0, 1.0))

    # Clamp bounding rect to image bounds — numpy silently clips slices but
    # warped/mask are still full-rect size, causing shape mismatch at edges.
    h, w = dst_img.shape[:2]
    x1 = max(r2[0], 0)
    y1 = max(r2[1], 0)
    x2 = min(r2[0] + r2[2], w)
    y2 = min(r2[1] + r2[3], h)
    if x2 <= x1 or y2 <= y1:
        return
    ox, oy = x1 - r2[0], y1 - r2[1]
    ew, eh = x2 - x1, y2 - y1

    dst_crop = dst_img[y1:y2, x1:x2]
    dst_crop[:] = dst_crop * (1 - mask[oy:oy+eh, ox:ox+ew]) + warped[oy:oy+eh, ox:ox+ew] * mask[oy:oy+eh, ox:ox+ew]


def morph_frame(photo_a: PhotoData, photo_b: PhotoData, t: float,
                triangles, points_a, points_b):
    """Generate a single morph frame at time t in [0, 1]."""
    # Intermediate geometry
    points_mid = (1 - t) * points_a + t * points_b

    # Warp both images toward the intermediate geometry
    warp_a = np.zeros_like(photo_a.aligned_bgr, dtype=np.float32)
    warp_b = np.zeros_like(photo_b.aligned_bgr, dtype=np.float32)

    img_a_f = photo_a.aligned_bgr.astype(np.float32)
    img_b_f = photo_b.aligned_bgr.astype(np.float32)

    for tri_idx in triangles:
        tri_a = [points_a[i] for i in tri_idx]
        tri_b = [points_b[i] for i in tri_idx]
        tri_mid = [points_mid[i] for i in tri_idx]
        warp_triangle(img_a_f, tri_a, tri_mid, warp_a)
        warp_triangle(img_b_f, tri_b, tri_mid, warp_b)

    # Subject blend (real morph)
    subject_blend = (1 - t) * warp_a + t * warp_b

    # Background blend (softer dissolve, with optional blur)
    bg_a = photo_a.aligned_bgr.astype(np.float32)
    bg_b = photo_b.aligned_bgr.astype(np.float32)
    if BACKGROUND_BLUR_KERNEL > 1:
        k = BACKGROUND_BLUR_KERNEL
        bg_a = cv2.GaussianBlur(bg_a, (k, k), 0)
        bg_b = cv2.GaussianBlur(bg_b, (k, k), 0)
    bg_blend = (1 - t) * bg_a + t * bg_b

    # Interpolated subject mask
    mask_a = photo_a.mask.astype(np.float32) / 255.0
    mask_b = photo_b.mask.astype(np.float32) / 255.0
    mask_blend = (1 - t) * mask_a + t * mask_b
    mask_3ch = np.stack([mask_blend] * 3, axis=-1)

    # Final composition: morphed subject over dissolved background
    final = subject_blend * mask_3ch + bg_blend * (1 - mask_3ch)
    return np.clip(final, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

_DATE_FONT_CACHE: dict = {}

def _load_date_font(size: int):
    if size not in _DATE_FONT_CACHE:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        font = ImageFont.load_default()
        for path in candidates:
            try:
                font = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                pass
        _DATE_FONT_CACHE[size] = font
    return _DATE_FONT_CACHE[size]


def draw_date_label(frame: np.ndarray, date_str: str,
                    label_right: int, label_bottom: int,
                    font_size: int, lang: str = "it") -> np.ndarray:
    """Return a copy of frame with a month/year label at bottom-right
    of the visible crop area. lang: 'it' (Italian) or 'en' (English)."""
    year  = int(date_str[0:4])
    month = int(date_str[5:7])
    label = f"{MONTH_NAMES[lang][month - 1]} {year}"

    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_date_font(font_size)

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 32
    x = label_right  - tw - margin
    y = label_bottom - th - margin

    # Dark outline, then white text
    for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3),
                   (-3, -3), (3, -3), (-3, 3), (3, 3),
                   (-3, -1), (3, -1), (-3, 1), (3, 1),
                   (-1, -3), (1, -3), (-1, 3), (1, 3)]:
        draw.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0))
    draw.text((x, y), label, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def write_video(frames_iter, output_path: Path, total_frames: int,
                vf_filter: str = None):
    """Write frames to mp4 via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
    ]
    if vf_filter:
        cmd += ["-vf", vf_filter]
    cmd += [str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for frame in tqdm(frames_iter, total=total_frames, desc="Rendering", unit="frame"):
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()


def mix_audio(video_path: Path, audio_path: Path, output_path: Path,
              volume: float, video_duration: float):
    """Second ffmpeg pass: copy video stream, mix audio with fade-out."""
    fade_start = max(0.0, video_duration - 2.0)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-stream_loop", "-1",        # loop audio if shorter than video
        "-i", str(audio_path),
        "-c:v", "copy",              # no re-encode
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",                 # cut audio at video end
        "-af", f"volume={volume},afade=t=out:st={fade_start:.2f}:d=2",
        str(output_path),
    ]
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg audio mixing failed — check the audio file path")


def generate_frames(photos, date_label: bool = True,
                    label_right: int = 1920, label_bottom: int = 1080,
                    label_font: int = 62, lang: str = "it"):
    """Generator yielding all video frames in order."""
    n = len(photos)

    for i, photo in enumerate(photos):
        # Static hold
        hold_dur = HOLD_SECONDS_ENDS if (i == 0 or i == n - 1) else HOLD_SECONDS
        n_hold = int(hold_dur * FPS)
        hold_frame = (draw_date_label(photo.aligned_bgr, photo.date_str,
                                      label_right, label_bottom, label_font, lang)
                      if date_label else photo.aligned_bgr)
        for _ in range(n_hold):
            yield hold_frame

        # Morph to next photo (skip on last)
        if i < n - 1:
            next_photo = photos[i + 1]
            triangles, points_a = compute_delaunay_triangles(
                photo.landmarks, OUTPUT_WIDTH, OUTPUT_HEIGHT
            )
            # For B's points use same indices (corners added identically)
            corners = np.array([
                [0, 0], [OUTPUT_WIDTH - 1, 0],
                [0, OUTPUT_HEIGHT - 1], [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            ], dtype=np.float32)
            points_b = np.vstack([next_photo.landmarks, corners])

            n_morph = int(MORPH_SECONDS * FPS)
            for f in range(1, n_morph + 1):
                t = f / (n_morph + 1)
                # Smooth easing (ease-in-out cubic)
                t_eased = 3 * t**2 - 2 * t**3
                yield morph_frame(photo, next_photo, t_eased,
                                  triangles, points_a, points_b)


def estimate_total_frames(n_photos):
    """Estimate total frame count for the progress bar."""
    holds = (n_photos - 2) * int(HOLD_SECONDS * FPS) + 2 * int(HOLD_SECONDS_ENDS * FPS)
    morphs = (n_photos - 1) * int(MORPH_SECONDS * FPS)
    return holds + morphs


# -----------------------------------------------------------------------------
# Interactive setup
# -----------------------------------------------------------------------------

def interactive_setup(args):
    """Prompt the user for every render parameter. Modifies args in-place."""

    def ask_float(label, default, lo=None, hi=None):
        while True:
            raw = input(f"  {label} [{default}]: ").strip()
            try:
                val = float(raw) if raw else float(default)
            except ValueError:
                print("    ⚠️  Please enter a number")
                continue
            if lo is not None and val < lo:
                print(f"    ⚠️  Must be ≥ {lo}")
                continue
            if hi is not None and val > hi:
                print(f"    ⚠️  Must be ≤ {hi}")
                continue
            return val

    def ask_bool(label, default):
        hint = "Y/n" if default else "y/N"
        while True:
            raw = input(f"  {label} [{hint}]: ").strip().lower()
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("    ⚠️  Enter y or n")

    print("\n🎬  Morph Video — interactive setup")
    print("    Press Enter to keep the default shown in [brackets].\n")

    # --- Input folder ---
    if args.input_folder is None:
        while True:
            raw = input("  Input photos folder: ").strip()
            if not raw:
                print("    ⚠️  Required")
                continue
            p = Path(raw).expanduser()
            if p.is_dir():
                args.input_folder = p
                break
            print(f"    ⚠️  Not found: {raw}")
    else:
        print(f"  Input folder:  {args.input_folder}  (from command line)")

    # --- Output video ---
    if args.output_video is None:
        while True:
            raw = input("  Output video path [output.mp4]: ").strip()
            p = Path(raw or "output.mp4").expanduser()
            if not p.suffix:
                p = p.with_suffix(".mp4")
            args.output_video = p
            break
    else:
        print(f"  Output video:  {args.output_video}  (from command line)")

    print()

    # --- Orientation ---
    orient_choices = list(ORIENTATIONS)  # landscape, portrait, square
    print(f"  Orientation  ({' / '.join(orient_choices)})")
    while True:
        raw = input(f"  [{args.orientation}]: ").strip().lower()
        if not raw:
            break
        if raw in orient_choices:
            args.orientation = raw
            break
        print(f"    ⚠️  Choose one of: {', '.join(orient_choices)}")

    # --- Zoom ---
    args.zoom = ask_float(
        "Face zoom  (1.0 = default · <1 = more head · >1 = closer in)",
        args.zoom, lo=0.1, hi=5.0,
    )

    print()

    # --- Timing ---
    hold_default  = args.hold      if args.hold      is not None else HOLD_SECONDS
    ends_default  = args.hold_ends if args.hold_ends is not None else HOLD_SECONDS_ENDS
    morph_default = args.morph     if args.morph     is not None else MORPH_SECONDS
    args.hold      = ask_float("Hold per photo, seconds",           hold_default,  lo=0.1)
    args.hold_ends = ask_float("Hold for first/last photo, seconds", ends_default, lo=0.1)
    args.morph     = ask_float("Morph transition, seconds",         morph_default, lo=0.1)

    print()

    # --- Date label ---
    args.date_label = ask_bool("Show month/year label", args.date_label)
    if args.date_label:
        print("  Label language  (it = Italian · en = English)")
        while True:
            raw = input(f"  [it]: ").strip().lower()
            if not raw or raw == "it":
                args.date_lang = "it"
                break
            if raw == "en":
                args.date_lang = "en"
                break
            print("    ⚠️  Enter it or en")

    print()

    # --- Music ---
    print("  Music:")
    print("    0) No music")
    style_list = list(BUNDLED_TRACKS.items())   # [(style, (filename, title, dur)), ...]
    for i, (style, (_, title, duration)) in enumerate(style_list, 1):
        print(f"    {i}) {style:<10}  {title} ({duration})")
    print("    c) Custom file path")

    # Determine sensible default choice string
    if args.music_style and args.music_style in BUNDLED_TRACKS:
        music_default = str(list(BUNDLED_TRACKS).index(args.music_style) + 1)
    elif args.music:
        music_default = "c"
    else:
        music_default = "0"

    while True:
        raw = input(f"  Choice [{music_default}]: ").strip().lower()
        if not raw:
            raw = music_default
        if raw == "0":
            args.music = None
            args.music_style = None
            break
        if raw == "c":
            while True:
                path_raw = input("  Audio file path: ").strip()
                if not path_raw:
                    print("    ⚠️  Required (or enter 0 to skip music)")
                    continue
                p = Path(path_raw).expanduser()
                if p.is_file():
                    args.music = p
                    args.music_style = None
                    break
                print(f"    ⚠️  File not found: {path_raw}")
            break
        try:
            idx = int(raw) - 1
            style, (filename, title, duration) = style_list[idx]
            args.music = AUDIO_DIR / filename
            args.music_style = style
            print(f"    ✓  {title} ({duration}) — Kevin MacLeod, CC BY 4.0")
            break
        except (ValueError, IndexError):
            print(f"    ⚠️  Enter a number 0–{len(style_list)}, or c")

    if args.music:
        args.music_volume = ask_float(
            "Music volume (0.0 = silent · 1.0 = full)",
            args.music_volume, lo=0.0, hi=1.0,
        )

    print()

    # --- Photo limit ---
    limit_default = args.limit if args.limit is not None else "all"
    while True:
        raw = input(f"  Use only first N photos — for quick tests [{limit_default}]: ").strip()
        if not raw or raw.lower() == "all":
            args.limit = None
            break
        try:
            args.limit = int(raw)
            if args.limit < 2:
                print("    ⚠️  Must be at least 2")
                continue
            break
        except ValueError:
            print("    ⚠️  Enter a number or press Enter for all")

    # --- Summary + confirmation ---
    print()
    print("  " + "─" * 50)
    print("  Settings:")
    print(f"    Input:        {args.input_folder}")
    print(f"    Output:       {args.output_video}")
    print(f"    Orientation:  {args.orientation}")
    print(f"    Zoom:         {args.zoom}")
    print(f"    Hold:         {args.hold}s  (ends: {args.hold_ends}s)")
    print(f"    Morph:        {args.morph}s")
    lang_label = f"  ({getattr(args, 'date_lang', 'it')})" if args.date_label else ""
    print(f"    Date label:   {'on' if args.date_label else 'off'}{lang_label}")
    if args.limit:
        print(f"    Limit:        first {args.limit} photos")
    if args.music:
        label = args.music_style or args.music.name
        print(f"    Music:        {label}  (volume {args.music_volume})")
    else:
        print("    Music:        none")
    print("  " + "─" * 50)
    print()

    if not ask_bool("  Proceed with these settings?", True):
        sys.exit("Aborted.")

    # Build and print the equivalent CLI command for future reuse
    parts = ["python morph_video.py",
             shlex.quote(str(args.input_folder)),
             shlex.quote(str(args.output_video))]
    if args.orientation != "landscape":
        parts.append(f"--orientation {args.orientation}")
    if args.zoom != 1.0:
        parts.append(f"--zoom {args.zoom}")
    hold_val  = args.hold      if args.hold      is not None else HOLD_SECONDS
    ends_val  = args.hold_ends if args.hold_ends is not None else HOLD_SECONDS_ENDS
    morph_val = args.morph     if args.morph     is not None else MORPH_SECONDS
    if hold_val != HOLD_SECONDS:
        parts.append(f"--hold {hold_val}")
    if ends_val != HOLD_SECONDS_ENDS:
        parts.append(f"--hold-ends {ends_val}")
    if morph_val != MORPH_SECONDS:
        parts.append(f"--morph {morph_val}")
    if not args.date_label:
        parts.append("--no-date-label")
    if args.date_label and getattr(args, "date_lang", "it") != "it":
        parts.append(f"--date-lang {args.date_lang}")
    if args.music_style:
        parts.append(f"--music-style {args.music_style}")
    elif args.music:
        parts.append(f"--music {shlex.quote(str(args.music))}")
    if args.music and args.music_volume != 0.8:
        parts.append(f"--music-volume {args.music_volume}")
    if args.limit:
        parts.append(f"--limit {args.limit}")

    print("  Equivalent command for next time:")
    print()
    # Multi-line display for readability, single line for easy copy
    indent = "      "
    multiline = (" \\\n" + indent).join(parts)
    print(f"    {multiline}")
    print()


def main():
    global HOLD_SECONDS, HOLD_SECONDS_ENDS, MORPH_SECONDS, EYE_LEFT_TARGET, EYE_RIGHT_TARGET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path, nargs="?", help="Folder with photos")
    parser.add_argument("output_video", type=Path, nargs="?", help="Output mp4 path")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactively prompt for all parameters before rendering.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear preprocessing cache before running")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Use only the first N photos (smoke test)")
    parser.add_argument("--date-label", default=True, action=argparse.BooleanOptionalAction,
                        help="Overlay month/year on each held frame (default: on).")
    parser.add_argument("--date-lang", choices=["it", "en"], default="it",
                        help="Language for the date label: it (Italian, default) or en (English).")
    parser.add_argument("--zoom", type=float, default=1.0, metavar="F",
                        help="Face zoom level (default: 1.0). "
                             "<1.0 zooms out (more head visible), >1.0 zooms in.")
    parser.add_argument("--orientation", choices=["landscape", "portrait", "square"],
                        default="landscape",
                        help="Output orientation: landscape (1920×1080, default), "
                             "portrait (1080×1920), or square (1080×1080). "
                             "Portrait and square are center-cropped from the aligned "
                             "1920×1080 frame — no fill, no padding. "
                             "Does not invalidate the cache.")
    parser.add_argument("--hold", type=float, default=None, metavar="S",
                        help=f"Hold duration in seconds for each photo (default: {HOLD_SECONDS}).")
    parser.add_argument("--hold-ends", type=float, default=None, metavar="S",
                        help=f"Hold duration for first and last photo (default: {HOLD_SECONDS_ENDS}).")
    parser.add_argument("--morph", type=float, default=None, metavar="S",
                        help=f"Morph transition duration in seconds (default: {MORPH_SECONDS}).")
    parser.add_argument("--music", type=Path, default=None, metavar="PATH",
                        help="Audio file (mp3/m4a/wav/…) to mix into the video. "
                             "Loops if shorter than the video; fades out 2s before the end.")
    parser.add_argument("--music-volume", type=float, default=0.8, metavar="F",
                        help="Music volume level, 0.0–1.0 (default: 0.8).")
    parser.add_argument("--music-style", choices=list(BUNDLED_TRACKS), default=None,
                        metavar="STYLE",
                        help="Use a bundled royalty-free track for the given mood "
                             f"({', '.join(BUNDLED_TRACKS)}). "
                             "Shortcut for --music audio/<track>.mp3. "
                             "Ignored if --music is also given.")
    args = parser.parse_args()

    if args.interactive:
        interactive_setup(args)

    if args.input_folder is None or args.output_video is None:
        parser.error("input_folder and output_video are required")

    orientation = ORIENTATIONS[args.orientation]
    vf_filter    = orientation["vf_filter"]
    label_right  = orientation["label_right"]
    label_bottom = orientation["label_bottom"]
    label_font   = orientation["label_font"]
    if vf_filter:
        print(f"📐 Orientation: {args.orientation} (cropped from 1920×1080)")

    # Resolve --music-style to a bundled track (only if --music not explicitly set)
    if args.music_style and args.music is None:
        filename, title, duration = BUNDLED_TRACKS[args.music_style]
        args.music = AUDIO_DIR / filename
        print(f"🎵 Music style '{args.music_style}': {title} ({duration}) — Kevin MacLeod, CC BY 4.0")

    if args.hold is not None or args.hold_ends is not None or args.morph is not None:
        if args.hold is not None:
            HOLD_SECONDS = args.hold
        if args.hold_ends is not None:
            HOLD_SECONDS_ENDS = args.hold_ends
        if args.morph is not None:
            MORPH_SECONDS = args.morph

    if args.zoom != 1.0:
        spread = EYE_SPREAD * args.zoom
        EYE_LEFT_TARGET  = (0.50 - spread / 2, EYE_Y)
        EYE_RIGHT_TARGET = (0.50 + spread / 2, EYE_Y)
        print(f"👁  Zoom {args.zoom}: eye spread {spread:.3f}")

    if not args.input_folder.is_dir():
        sys.exit(f"❌ Folder not found: {args.input_folder}")

    if shutil.which("ffmpeg") is None:
        sys.exit("❌ ffmpeg not found. Install with: sudo apt install ffmpeg")

    cache_dir = args.input_folder / ".morph_cache"
    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print(f"📂 Scanning {args.input_folder}...")
    photo_entries = collect_photos(args.input_folder)  # [(date_str, Path), ...]
    if args.limit:
        photo_entries = photo_entries[:args.limit]
        print(f"⚠️  --limit {args.limit}: using first {len(photo_entries)} photos only")
    if len(photo_entries) < 2:
        sys.exit(f"❌ Need at least 2 photos, found {len(photo_entries)}")
    print(f"✅ {len(photo_entries)} photos in chronological order")

    # Init models (reused across all photos)
    print("🧠 Loading models...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # required for iris landmarks
        min_detection_confidence=0.5,
    )
    rembg_session = new_session(REMBG_MODEL)

    # Preprocessing with cache
    print("🔍 Preprocessing (alignment + segmentation)...")
    photos = []
    with tqdm(photo_entries, desc="Photos", unit="img") as bar:
        for date_str, path in bar:
            bar.set_description(path.name[:40])
            try:
                photo = load_or_preprocess(path, date_str, cache_dir,
                                           face_mesh, rembg_session, bar=bar)
                photos.append(photo)
            except RuntimeError as e:
                tqdm.write(f"⚠️  Skipping {path.name}: {e}")

    if len(photos) < 2:
        sys.exit("❌ Too many photos rejected, cannot generate video")

    # Video generation
    total    = estimate_total_frames(len(photos))
    duration = total / FPS
    frames   = generate_frames(photos, date_label=args.date_label,
                               label_right=label_right, label_bottom=label_bottom,
                               label_font=label_font, lang=args.date_lang)

    if args.music:
        if not args.music.is_file():
            sys.exit(f"❌ Music file not found: {args.music}")
        # Render video-only to a temp file, then mix audio in a second pass
        tmp_video = args.output_video.with_suffix(".tmp.mp4")
        print(f"🎬 Rendering video → {tmp_video} (audio mix follows)")
        write_video(frames, tmp_video, total, vf_filter=vf_filter)
        print(f"🎵 Mixing audio → {args.output_video}")
        try:
            mix_audio(tmp_video, args.music, args.output_video,
                      args.music_volume, duration)
        finally:
            tmp_video.unlink(missing_ok=True)
    else:
        print(f"🎬 Generating video → {args.output_video}")
        write_video(frames, args.output_video, total, vf_filter=vf_filter)

    print(f"✅ Video complete: {duration:.1f}s ({total} frames)")


if __name__ == "__main__":
    main()
