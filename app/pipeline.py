"""Image -> layers (background, subject, text) and background resizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import models
from .text_detect import detect_text_boxes, extract_text_region

log = logging.getLogger(__name__)


@dataclass
class Layer:
    id: str
    name: str
    kind: str  # "subject" | "text"
    x: int
    y: int
    rgba: np.ndarray


@dataclass
class Separation:
    width: int
    height: int
    background: np.ndarray  # RGB
    layers: list[Layer] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _crop_rgba(rgba: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    ys, xs = np.nonzero(rgba[..., 3] > 2)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return int(x0), int(y0), rgba[y0:y1, x0:x1].copy()


def _split_objects(alpha: np.ndarray, hard: np.ndarray) -> list[np.ndarray]:
    """Split the BiRefNet mask into separate objects (largest first).

    BiRefNet marks everything salient as foreground, e.g. a book cover and a
    logo next to it; each connected blob becomes its own movable layer.
    Specks below 0.1% of the image are ignored (left in the background).
    """
    H, W = alpha.shape
    # Join nearby pieces (letters of a logo, parts of an object) into one blob;
    # the dilation also covers the soft alpha edge around the hard mask
    k = max(5, int(round(min(H, W) / 40))) | 1
    joined = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    min_area = 0.001 * H * W
    # area of the real (undilated) mask inside each blob
    real_area = np.bincount(labels[hard > 0].ravel(), minlength=n)
    ids = [i for i in range(1, n) if real_area[i] >= min_area]
    ids.sort(key=lambda i: -real_area[i])
    objects = []
    for i in ids:
        objects.append(alpha * (labels == i))
    return objects


def _removal_mask(img_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pixels to repaint for the removed objects, including their drop shadows."""
    H, W = alpha.shape
    obj = (alpha > 0.1).astype(np.uint8) * 255
    if not obj.any():
        return obj
    unit = min(H, W)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
    near = cv2.dilate(obj, ell(max(4, unit // 25)))
    far = cv2.dilate(obj, ell(max(6, unit // 8)))
    # Shadow = ring pixels clearly darker than the background just beyond it
    L = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    outer = (far > 0) & (near == 0)
    ref = cv2.blur(L * outer, (unit // 6 | 1,) * 2) / np.maximum(cv2.blur(outer.astype(np.float32), (unit // 6 | 1,) * 2), 1e-3)
    # (only mildly darker: much darker pixels are content such as text, not shadow)
    dark = ((near > 0) & (obj == 0) & (L < ref - 5) & (L > ref - 45)).astype(np.uint8) * 255
    # keep only dark pixels connected to the object
    seed = cv2.dilate(obj, ell(2))
    n, labels = cv2.connectedComponents(cv2.bitwise_or(dark, seed))
    touching = np.unique(labels[seed > 0])
    shadow = np.isin(labels, touching[touching > 0]).astype(np.uint8) * 255
    mask = cv2.bitwise_or(obj, shadow)
    return cv2.dilate(mask, ell(max(3, unit // 120)))


def separate(
    img_rgb: np.ndarray,
    detect_text: bool = True,
    detect_subject: bool = True,
    merge_text_lines: bool = True,
    segmenter=None,
    inpainter=None,
) -> Separation:
    H, W = img_rgb.shape[:2]
    info: dict = {}
    remove_mask = np.zeros((H, W), np.uint8)
    layers: list[Layer] = []

    # 1. Subject (BiRefNet)
    subject_alpha = None
    subject_hard = None
    if detect_subject:
        seg = segmenter if segmenter is not None else models.get_segmenter()
        if seg is None:
            info["subject"] = "BiRefNet model not available"
        else:
            subject_alpha = seg.predict(img_rgb)
            subject_hard = (subject_alpha > 0.5).astype(np.uint8) * 255
            coverage = cv2.countNonZero(subject_hard) / float(H * W)
            info["subject_coverage"] = round(coverage, 4)
            if coverage < 0.003 or coverage > 0.97:
                info["subject"] = "no clear subject found"
                subject_alpha = subject_hard = None

    # 2. Text (OpenCV)
    text_regions = []
    if detect_text:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        boxes = detect_text_boxes(img_bgr, exclude_mask=subject_hard, merge_lines=merge_text_lines)
        for box in boxes:
            region = extract_text_region(img_rgb, box)
            if region is not None:
                text_regions.append(region)
        info["text_regions"] = len(text_regions)

    text_mask = np.zeros((H, W), np.uint8)
    for r in text_regions:
        text_mask |= r.mask

    if subject_alpha is not None:
        objects = _split_objects(subject_alpha, subject_hard)
        for alpha in objects:
            alpha = alpha.copy()
            # Text drawn on top of an object belongs to the text layer
            alpha[text_mask > 0] = 0
            if (alpha > 0.5).sum() < 0.001 * H * W:  # mostly text, already its own layer
                continue
            cropped = _crop_rgba(np.dstack([img_rgb, (alpha * 255).astype(np.uint8)]))
            if cropped:
                x, y, crop = cropped
                num = sum(1 for layer in layers if layer.kind == "subject") + 1
                name = "תמונה" if len(objects) == 1 else f"תמונה {num}"
                layers.append(Layer(f"subject-{num}", name, "subject", x, y, crop))
        kept = np.max(objects, axis=0) if objects else np.zeros((H, W), np.float32)
        remove_mask |= _removal_mask(img_rgb, kept)

    for i, r in enumerate(text_regions):
        layers.append(Layer(f"text-{i + 1}", f"טקסט {i + 1}", "text", r.x, r.y, r.rgba))
    remove_mask |= text_mask

    # 3. Background (LaMa)
    inp = inpainter if inpainter is not None else models.get_inpainter()
    background = inp.inpaint(img_rgb, remove_mask) if remove_mask.any() else img_rgb.copy()
    info["inpainter"] = getattr(inp, "name", type(inp).__name__)

    return Separation(W, H, background, layers, info)


def resize_background(
    background: np.ndarray, width: int, height: int, mode: str = "extend", inpainter=None
) -> tuple[np.ndarray, dict]:
    """Resize the background to (width, height).

    Modes:
      extend  - fit inside, then let LaMa paint the new margins (outpainting)
      cover   - scale to fill and crop the overflow
      stretch - non-uniform scale

    Returns the new background and the transform {sx, sy, ox, oy} that maps
    old coordinates to new ones (new = old * s + o), so layers can follow.
    """
    H, W = background.shape[:2]
    if mode == "stretch":
        out = cv2.resize(background, (width, height), interpolation=cv2.INTER_CUBIC)
        return out, {"sx": width / W, "sy": height / H, "ox": 0.0, "oy": 0.0}

    s = max(width / W, height / H) if mode == "cover" else min(width / W, height / H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    scaled = cv2.resize(background, (nw, nh), interpolation=interp)
    ox, oy = (width - nw) // 2, (height - nh) // 2

    if mode == "cover":
        out = scaled[-oy:-oy + height, -ox:-ox + width].copy()
        return out, {"sx": s, "sy": s, "ox": float(ox), "oy": float(oy)}

    if mode != "extend":
        raise ValueError(f"unknown mode: {mode}")

    inp = inpainter if inpainter is not None else models.get_inpainter()
    canvas = _outpaint(scaled, ox, oy, width - nw - ox, height - nh - oy, inp)
    return canvas, {"sx": s, "sy": s, "ox": float(ox), "oy": float(oy)}


def _outpaint(img: np.ndarray, left: int, top: int, right: int, bottom: int, inpainter) -> np.ndarray:
    """Extend `img` by the given margins, painting the new area step by step.

    LaMa handles narrow holes well but darkens wide areas and image borders.
    So the canvas grows by at most ~20% per step, and beyond every new strip
    a mirrored copy of the image is placed: the strip becomes an interior
    hole with context on both sides. The mirror is cropped away afterwards.
    """
    pads = {"left": left, "top": top, "right": right, "bottom": bottom}
    out = img
    while any(v > 0 for v in pads.values()):
        h, w = out.shape[:2]
        st = {
            k: min(v, max(32, int(0.2 * (w if k in ("left", "right") else h))))
            for k, v in pads.items()
        }
        canvas, mask, crop = _mirror_layout(out, st["top"], st["bottom"], st["left"], st["right"])
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # overlap the seam a little
        y0, y1, x0, x1 = crop
        out = inpainter.inpaint(canvas, mask)[y0:y1, x0:x1]
        for k in pads:
            pads[k] -= st[k]
    return out


def _mirror_layout(img: np.ndarray, top: int, bottom: int, left: int, right: int):
    """[mirror | hole | image | hole | mirror] along both axes.

    Returns the canvas, the hole mask and the (y0, y1, x0, x1) crop that keeps
    image + holes.
    """
    canvas = img
    mask = np.zeros(img.shape[:2], np.uint8)

    def grow(arr, n, axis, before, fill):
        if n <= 0:
            return arr, 0
        m = min(n, arr.shape[axis])  # mirror thickness
        edge = np.take(arr, [0] if before else [-1], axis=axis)
        hole = np.repeat(edge if fill is None else np.full_like(edge, fill), n, axis=axis)
        src = np.take(arr, range(m) if before else range(arr.shape[axis] - m, arr.shape[axis]), axis=axis)
        mirror = np.flip(src, axis=axis)
        parts = [mirror, hole, arr] if before else [arr, hole, mirror]
        return np.concatenate(parts, axis=axis), m

    canvas, mt = grow(canvas, top, 0, True, None)
    mask, _ = grow(mask, top, 0, True, 255)
    canvas, mb = grow(canvas, bottom, 0, False, None)
    mask, _ = grow(mask, bottom, 0, False, 255)
    canvas, ml = grow(canvas, left, 1, True, None)
    mask, _ = grow(mask, left, 1, True, 255)
    canvas, mr = grow(canvas, right, 1, False, None)
    mask, _ = grow(mask, right, 1, False, 255)
    H, W = canvas.shape[:2]
    return np.ascontiguousarray(canvas), np.ascontiguousarray(mask), (mt, H - mb, ml, W - mr)
