from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

from . import ai_segment
from . import postprocess


def _extract_lab(rgba: np.ndarray):
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3]
    fg = alpha > 0

    lab = rgb2lab(rgb)  # L [0,100], a,b ~ [-128,127]
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    return fg, L, A, B

def _apply_mask(rgba: np.ndarray, mask: np.ndarray, fill_holes=True):
    if fill_holes:
        mask = binary_fill_holes(mask) # type: ignore

    out = np.zeros_like(rgba)
    out[mask] = rgba[mask]
    return out

def mask_brown_necrosis(rgba: np.ndarray):
    fg, L, A, B = _extract_lab(rgba)
    mask = fg & (L < 55) & (A > 5) & (B > 10)
    return _apply_mask(rgba, mask)


def mask_black_rot_severe_necrosis(rgba: np.ndarray):
    fg, L, A, B = _extract_lab(rgba)
    mask = fg & (L < 25) & (A > 0) & (B > 0)
    return _apply_mask(rgba, mask)

def mask_dark_spot_core_lesion_center(rgba: np.ndarray):
    fg, L, A, B = _extract_lab(rgba)
    mask = fg & (L < 35) & (A >= 5) & (A <= 20) & (B >= 5) & (B <= 25)
    return _apply_mask(rgba, mask)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
def _as_rgba_uint8(x) -> np.ndarray:
    """Accept PIL.Image or ndarray; return HxWx4 uint8 RGBA."""
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGBA"), dtype=np.uint8)

    arr = np.asarray(x)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image array, got {arr.shape}")

    if arr.shape[2] == 3:
        h, w, _ = arr.shape
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)

    return arr
def _mask_bool_from_rgba(mask_rgba: np.ndarray) -> np.ndarray:
    # Your mask outputs are RGBA with alpha>0 where the symptom pixels exist
    return mask_rgba[:, :, 3] > 0

def _lesion_count(mask_bool: np.ndarray, leaf_fg: np.ndarray, min_pixels: int = 25) -> int:
    """
    Count connected components (lesions). We ignore tiny specks via min_pixels.
    This does NOT score by area; it only counts lesions.
    """
    m = mask_bool & leaf_fg
    lab = label(m.astype(np.uint8), connectivity=2)
    n = 0
    for r in regionprops(lab):
        if r.area >= min_pixels:
            n += 1
    return n


def _mean_darkness_L(reseg_rgba: np.ndarray, lesion_bool: np.ndarray, leaf_fg: np.ndarray) -> float:
    """
    Mean darkness of lesion pixels using LAB L channel (lower L = darker).
    Returned as 0..1 where 1 = very dark lesions, 0 = not dark.
    """
    fg, L, A, B = _extract_lab(reseg_rgba)
    m = lesion_bool & leaf_fg
    if not np.any(m):
        return 0.0

    mean_L = float(np.mean(L[m]))  # L in [0..100]
    darkness = (60.0 - mean_L) / 60.0  # mean_L 60->0, 0->1 (cap)
    return _clamp01(darkness)

def analyse_leaf(image: Image.Image, debug: bool = True) -> dict:
    """Analyse a leaf image and return a dictionary of results (no AI)."""

    segmented_image = ai_segment.segment_image(image, "models/leaf_segmentation.pt")
    resegmented_image = postprocess.resegment(segmented_image)

    reseg_rgba = _as_rgba_uint8(resegmented_image)
    leaf_fg = reseg_rgba[:, :, 3] > 0

    masks_rgba = {
        "dark_spot_core_lesion_center": mask_dark_spot_core_lesion_center(reseg_rgba),
        "black_rot": mask_black_rot_severe_necrosis(reseg_rgba),
        "brown_necrosis": mask_brown_necrosis(reseg_rgba),
    }
    Image.fromarray(segmented_image).save("debug_segmented.png")
    Image.fromarray(reseg_rgba).save("debug_resegmented.png")
    for name, mask_rgba in masks_rgba.items():
        Image.fromarray(mask_rgba).save(f"debug_{name}.png")
    masks_bool = {k: _mask_bool_from_rgba(v) for k, v in masks_rgba.items()}


    # Lesion-focused mask (spots/necrosis)
    lesion_bool = (
        masks_bool["dark_spot_core_lesion_center"]
        | masks_bool["brown_necrosis"]
        | masks_bool["black_rot"]
    )
    
    # Signals (NOT area-based)
    black_rot_bool = masks_bool["black_rot"] & leaf_fg
    black_rot_count = _lesion_count(black_rot_bool, leaf_fg, min_pixels=80)  # 80 is a good start
    has_black_rot = black_rot_count > 0
    spot_count = _lesion_count(lesion_bool, leaf_fg, min_pixels=25)
    darkness = _mean_darkness_L(reseg_rgba, lesion_bool, leaf_fg)

    # Confidence (heuristic severity score)
    black_rot_term = 0.45 if has_black_rot else 0.0
    lesion_count_term = 0.45 * min(1.0, spot_count / 20.0)  # cap at ~20 lesions
    darkness_term = 0.25 * darkness

    confidence = _clamp01(black_rot_term + lesion_count_term + darkness_term)
    disease = "Doente" if confidence >= 0.35 else "Saudável"
    if(disease == "Saudável"):
        confidence = 1.0 - confidence
    result = {
        "status": "ok",
        "disease": disease,
        "confidence": float(confidence),
    }

    if debug:
        # Raw pixels flagged per mask (helps catch gray_pale false positives etc.)
        mask_pixels = {k: int(np.sum(v & leaf_fg)) for k, v in masks_bool.items()}

        # Human-readable reason
        reasons = []
        if has_black_rot:
            reasons.append("black_rot detected")
        if spot_count > 0:
            reasons.append(f"{spot_count} lesions")
        if darkness > 0.0:
            reasons.append(f"lesion darkness={darkness:.2f}")
        if not reasons:
            reasons.append("no strong signals")

        result["debug"] = {
            "has_black_rot": has_black_rot,
            "spot_count": int(spot_count),
            "darkness": float(darkness),
            "confidence_breakdown": {
                "black_rot_term": float(black_rot_term),
                "lesion_count_term": float(lesion_count_term),
                "darkness_term": float(darkness_term),
            },
            "mask_pixels": mask_pixels,
            "reason": " | ".join(reasons),
        }

    return result