"""Image-quality gate for the KisanOS Vision Agent.

Runs BEFORE any analysis. Uses only Pillow + numpy (no OpenCV needed).
Threshold values are starting points - tune them on real test photos.
"""
import io
import numpy as np
from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
MIN_SHORT_SIDE = 640
BLUR_MIN = 100.0
BRIGHT_MIN = 40.0
BRIGHT_MAX = 220.0
PLANT_MIN_FRACTION = 0.10

TIPS = {
    "too_large": "The photo file is too big. Please send a photo under 10 MB.",
    "bad_file": "This file could not be opened as a photo. Please send a JPG or PNG.",
    "low_resolution": "The photo is too small. Please take it with the normal camera at full quality.",
    "blurry": "The photo is blurry. Hold the phone steady and tap the leaf to focus.",
    "too_dark": "The photo is too dark. Please take it in daylight.",
    "too_bright": "The photo is too bright or has glare. Please avoid direct sun on the camera.",
    "no_plant": "No crop is visible. Please move closer so the leaves fill most of the photo.",
}


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _plant_fraction(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("HSV")).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # PIL hue is 0-255. Roughly 40-150 degrees (yellow-green to green) -> 28-106
    mask = (h >= 28) & (h <= 106) & (s >= 40) & (v >= 40)
    return float(mask.mean())


def check_quality(image_bytes: bytes) -> dict:
    result = {"passed": False, "blur_score": None, "brightness": None,
              "plant_fraction": None, "width": None, "height": None,
              "issues": [], "tips": []}

    def fail(code):
        result["issues"].append(code)
        result["tips"].append(TIPS[code])

    if len(image_bytes) > MAX_BYTES:
        fail("too_large")
        return result
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        img = img.convert("RGB")
    except Exception:
        fail("bad_file")
        return result

    result["width"], result["height"] = img.size
    if min(img.size) < MIN_SHORT_SIDE:
        fail("low_resolution")

    # Work on a downscaled copy so scores do not depend on phone megapixels
    work = img.copy()
    work.thumbnail((1280, 1280))
    gray = np.asarray(work.convert("L"))
    result["blur_score"] = round(_laplacian_variance(gray), 1)
    result["brightness"] = round(float(gray.mean()), 1)
    result["plant_fraction"] = round(_plant_fraction(work), 3)

    if result["blur_score"] < BLUR_MIN:
        fail("blurry")
    if result["brightness"] < BRIGHT_MIN:
        fail("too_dark")
    elif result["brightness"] > BRIGHT_MAX:
        fail("too_bright")
    if result["plant_fraction"] < PLANT_MIN_FRACTION:
        fail("no_plant")

    result["passed"] = not result["issues"]
    return result
