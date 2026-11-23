"""Shared RGB ROI normalization and square support bounding box."""
import math
from numbers import Real
import numpy as np
import torch
IMAGENET_MEAN=(.485,.456,.406)
IMAGENET_STD=(.229,.224,.225)
APPEARANCE_MODES=('bbox','masked')
def _margin(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("margin_ratio must be finite in [0,1]")
    return float(value)

def square_bbox_for_mask(mask, margin_ratio=.15):
    """Return integer (x0,y0,x1,y1), right/bottom exclusive and possibly outside.

    The side is ceil(max(original_bbox_width,height)*(1+2*margin)); center is
    unchanged before deterministic floor rounding. Margin .15 is per side, not
    a 15% total size increment. No clipping or semantic connected components.
    """
    margin = _margin(margin_ratio)
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape) or mask.dtype.kind not in "buif" or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("mask must be a binary 2-D image")
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("empty mask has no classification crop")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = math.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin))
    left = math.floor((x0 + x1 - side) / 2)
    top = math.floor((y0 + y1 - side) / 2)
    return left, top, left + side, top + side

def normalize_crop(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("crop must be uint8 RGB")
    values = rgb.astype(np.float32) / 255
    values = (values - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(values.transpose(2, 0, 1).copy())

from numbers import Integral
from PIL import Image
def input_shape(value):
    value = (value, value) if isinstance(value, Integral) and not isinstance(value, bool) else value
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in value)):
        raise ValueError("input_size must be positive integer or [height,width]")
    return tuple(int(x) for x in value)

def crop_with_intrinsics(rgb, mask, K, gravity, *, input_size=224,
                         margin_ratio=.15, appearance_mode="bbox"):
    """Square native crop, constant114 padding, resize/letterbox and exact K.

    Integer source pixel centers map as
    u'=(u-x0+.5)*sx-.5+left, v'=(v-y0+.5)*sy-.5+top.
    K_crop=A@K uses actual rounded resize dimensions, not nominal scale.
    Crop/resize do not rotate the camera frame: gravity is only normalized.
    Empty masks remain explicit invalid objects with no invented K_crop.
    """
    height, width = input_shape(input_size)
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    K, gravity = np.asarray(K, np.float64), np.asarray(gravity, np.float64)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or mask.shape != rgb.shape[:2] or mask.dtype.kind not in "buif"
            or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError("rgb must be uint8 HxWx3 with a matching binary mask")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-12)
            or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12):
        raise ValueError("K must be finite positive-focal zero-skew camera intrinsics")
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-10:
        raise ValueError("gravity must be a finite nonzero camera-frame vector")
    gravity = gravity / np.linalg.norm(gravity)
    if appearance_mode not in APPEARANCE_MODES:
        raise ValueError("appearance_mode must be bbox or masked")
    # The shared helper also validates the exact per-side margin contract.
    if not np.any(mask):
        if isinstance(margin_ratio, bool) or not np.isfinite(margin_ratio) or not 0 <= margin_ratio <= 1:
            raise ValueError("invalid margin_ratio")
        return {"valid": False, "reason": "empty_roi", "crop": np.full((height, width, 3), 114, np.uint8),
                "K_crop": None, "gravity": gravity, "aux": np.zeros(7, np.float32), "mapping": None}
    x0, y0, x1, y1 = square_bbox_for_mask(mask, margin_ratio)
    native = np.full((y1-y0, x1-x0, 3), 114, np.uint8)
    original = rgb
    if appearance_mode == "masked":
        original = np.full_like(rgb, 114)
        original[mask > 0] = rgb[mask > 0]
    a, b, c, d = max(x0, 0), min(x1, rgb.shape[1]), max(y0, 0), min(y1, rgb.shape[0])
    native[c-y0:d-y0, a-x0:b-x0] = original[c:d, a:b]
    scale = min(width / (x1-x0), height / (y1-y0))
    resized_w = max(1, min(width, round((x1-x0)*scale)))
    resized_h = max(1, min(height, round((y1-y0)*scale)))
    left, top = (width-resized_w)//2, (height-resized_h)//2
    crop = np.full((height, width, 3), 114, np.uint8)
    crop[top:top+resized_h, left:left+resized_w] = np.asarray(
        Image.fromarray(native).resize((resized_w, resized_h), Image.Resampling.BILINEAR))
    sx, sy = resized_w/(x1-x0), resized_h/(y1-y0)
    A = np.array([[sx, 0., (.5-x0)*sx-.5+left],
                  [0., sy, (.5-y0)*sy-.5+top], [0., 0., 1.]])
    K_crop = A @ K
    aux = np.array([K_crop[0, 0]/width, K_crop[1, 1]/height,
                    K_crop[0, 2]/width, K_crop[1, 2]/height, *gravity], np.float32)
    return {"valid": True, "reason": "ok", "crop": crop, "K_crop": K_crop, "gravity": gravity,
            "aux": aux, "mapping": {"bbox_xyxy": [x0, y0, x1, y1], "resized_hw": [resized_h, resized_w],
                                     "scale_xy": [sx, sy], "padding_left_top": [left, top],
                                     "affine_source_to_input": A.tolist()}}
