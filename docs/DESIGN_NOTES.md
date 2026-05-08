# Design Notes

This document captures the reasoning behind the project — the discussion that led to the current design. Useful for understanding *why* before you change *what*.

## The core problem

The user has ~73 photos of their daughter, taken roughly monthly, all with similar framing (16:9 horizontal, subject centered, frontal, comparable distance). They want a video that walks through these photos chronologically with transitions that exploit the fact that it's the same person at different ages — something like a "metamorphosis" rather than a slideshow.

The perceptual challenge: a plain crossfade fails because the background also changes between photos, creating a confusing ghosting effect where the eye can't lock onto the subject. The face needs to be the visual anchor while the rest of the world transforms around it.

## Approaches considered

Three approaches were on the table:

1. **Alignment + simple crossfade.** Detect face landmarks, align all photos so the face sits in the same spot, then crossfade. Works but feels basic.
2. **Full morphing.** Detect landmarks, Delaunay-triangulate, interpolate both geometry and color between photos. The classic "Black or White" effect. Background gets warped along with the face, which can be distracting.
3. **Hybrid: segment + morph subject + dissolve background separately.** Use `rembg` to separate the subject from the background, morph only the subject, treat the background as a separate layer with a softer dissolve. Most complex but cleanest result.

Decision: **option 3**. Worth the extra complexity for a project meant to be rewatched in 10 years.

## Parameters chosen and why

- **73 photos × (1.2s hold + 0.8s morph) ≈ 2:30 total.** User wants visible "stops" on each photo, not just waypoints. First and last get 2.5s hold for opening/closing breathing room.
- **30 fps, 1920x1080, h264 crf 18.** HD horizontal as requested. crf 18 is visually lossless for this content.
- **Eyes as alignment anchors.** Most reliably detected facial features across ages, most perceptually important for identity.
- **MediaPipe Face Mesh with iris refinement** (478 landmarks). More landmarks than dlib's 68, and iris centers are more stable than estimated eye centers.
- **Cubic ease-in-out on morph progress.** `t_eased = 3t² - 2t³`. Linear progress feels mechanical; this gives the morph a natural acceleration/deceleration.
- **Background gaussian blur kernel 15** during dissolve. Subtle — the background doesn't compete with the morphing face.

## Things rejected and why

- **Google Photos API as input.** Google deprecated the relevant endpoints in March 2025. Apps can only read what they uploaded or albums they created — not user albums created in the Photos app. Google Drive API would work but adds OAuth setup overhead. User accepted that downloading a zip from Google Photos manually is fine for occasional regenerations.
- **GPU acceleration via ROCm.** The user has a Radeon with working ROCm 6 (using `HSA_OVERRIDE_GFX_VERSION` to map an unsupported card onto a supported gfx target — likely 10.3.0 / gfx1030 / RDNA2). For this specific workload it's not worth it: rembg is one-shot and cached, and the render loop is OpenCV/numpy CPU code that ROCm wouldn't touch. Best case ~5-10 minutes saved once. Code stays simple. Side benefit: if the user installs `onnxruntime-rocm`, rembg picks it up automatically — passive opt-in with no code change.
- **Per-file naming for chronological order.** User specifically preferred EXIF-based ordering as more trustworthy than filenames.
- **Project-as-package structure.** Single file is easier to read, modify, and trust for a personal project of this size.
- **Audio in v1.** Deferred. Will likely be a second ffmpeg pass mixing in a track later.

## Risk areas

These are things that might not work perfectly on the first run and how to think about them.

- **Photos not perfectly frontal.** User described framing as "rough approximation" — some photos may have head tilt or slight off-axis angles. Triangulated morphing handles small variations gracefully but degrades on large ones. Mitigation: the alignment step (similarity transform from eyes) corrects rotation and scale automatically. If specific photos look bad, they can be excluded or replaced.
- **Hair segmentation.** rembg's default `u2net_human_seg` is good but not perfect on flyaways. If halos appear, switch to `isnet-general-use`.
- **Large change between consecutive months.** Children change shape a lot in the first year. Even with monthly photos, some adjacent pairs will be visually quite different. Cubic easing helps; shorter morph helps more if needed.
- **First-run preprocessing time.** ~5-15 minutes on CPU for 73 photos. Cache makes subsequent runs near-instant for preprocessing — only the render loop runs again, which is where most parameter tweaks take effect anyway.

## The conversation arc, briefly

The discussion went: identify the perceptual problem → enumerate three approaches → pick the hybrid → settle resolution/format/duration/source-of-truth for ordering → discover Google Photos API isn't viable → decide local folder + manual zip download is fine → discuss GPU and decide CPU is the right call → write the script. No major pivots, all decisions hold.
