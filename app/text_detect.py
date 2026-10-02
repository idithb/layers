"""Text detection and extraction with OpenCV.

The detector finds letter-like edge contours (Canny), groups letters of similar
height into lines (and lines into paragraphs), and then
separates the text strokes from their local background by colour distance,
so each block can become a transparent RGBA layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    rgba: np.ndarray  # (h, w, 4) uint8, text strokes on transparent background
    mask: np.ndarray  # (h, w) uint8 0/255, pixels to remove from the background


def _merge_boxes(boxes: list[list[int]], line_gap: float, para_gap: float) -> list[list[int]]:
    """Merge boxes on the same line (and optionally stacked lines) into blocks."""
    boxes = [b[:] for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        while boxes:
            x, y, w, h = boxes.pop()
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]
                lh = min(h, bh)
                # vertical overlap -> same line, allow a horizontal gap
                v_overlap = min(y + h, by + bh) - max(y, by)
                h_gap = max(bx - (x + w), x - (bx + bw))
                same_line = v_overlap > 0.5 * lh and h_gap < line_gap * lh
                # horizontal overlap -> stacked lines, allow a vertical gap
                h_overlap = min(x + w, bx + bw) - max(x, bx)
                v_gap = max(by - (y + h), y - (by + bh))
                similar_height = max(h, bh) < 1.35 * lh
                stacked = (
                    para_gap > 0
                    and h_overlap > 0.3 * min(w, bw)
                    and v_gap < para_gap * lh
                    and similar_height
                )
                if same_line or stacked or (v_overlap > 0 and h_overlap > 0):
                    nx, ny = min(x, bx), min(y, by)
                    w, h = max(x + w, bx + bw) - nx, max(y + h, by + bh) - ny
                    x, y = nx, ny
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append([x, y, w, h])
        boxes = out
    return boxes


def _char_candidates(gray: np.ndarray) -> np.ndarray:
    """Letter-like edge contours as an (n, 4) array of x, y, w, h."""
    sh, sw = gray.shape[:2]
    # Letter outlines from edges: scale-invariant and polarity-free
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    keep = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or h > 0.35 * sh or w * h > 0.03 * sh * sw:
            continue
        if not (0.08 <= w / h <= 2.0):
            continue
        keep.append((x, y, w, h))
    if not keep:
        return np.zeros((0, 4), int)
    b = np.unique(np.array(keep, int), axis=0)
    # Letters produce nested contours (outer edge, inner edge, counters):
    # keep only the outermost letter-sized box.
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    order = np.argsort(-area)
    b, x0, y0, x1, y1 = b[order], x0[order], y0[order], x1[order], y1[order]
    n = len(b)
    contains = np.zeros((n, n), bool)
    for i in range(n):
        inside = (x0 >= x0[i] - 1) & (y0 >= y0[i] - 1) & (x1 <= x1[i] + 1) & (y1 <= y1[i] + 1)
        inside[i] = False
        contains[i] = inside
    # A box wrapping several candidates is a word/blob region, not a letter
    alive = contains.sum(1) < 3
    # Inside a real letter, smaller boxes are its counters (holes) -> drop them
    for i in range(n):
        if alive[i]:
            alive &= ~contains[i]
    return b[alive]


def _group_into_lines(chars: np.ndarray) -> list[list[int]]:
    """Link letters of similar height that sit side by side into text lines."""
    n = len(chars)
    if n == 0:
        return []
    x, y, w, h = chars.T.astype(float)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        hmin = np.minimum(h, h[i])
        hmax = np.maximum(h, h[i])
        v_overlap = np.minimum(y + h, y[i] + h[i]) - np.maximum(y, y[i])
        h_gap = np.maximum(x - (x[i] + w[i]), x[i] - (x + w))
        link = (hmax < 1.8 * hmin) & (v_overlap > 0.55 * hmin) & (h_gap < 0.9 * hmax)
        link[i] = False
        for j in np.nonzero(link)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    lines = []
    for idx in groups.values():
        if len(idx) < 3:  # a "line" of 1-2 shapes is more often decoration than text
            continue
        g = chars[idx]
        gx0, gy0 = g[:, 0].min(), g[:, 1].min()
        gx1, gy1 = (g[:, 0] + g[:, 2]).max(), (g[:, 1] + g[:, 3]).max()
        if (gx1 - gx0) < 1.8 * (gy1 - gy0):
            continue
        lines.append([int(gx0), int(gy0), int(gx1 - gx0), int(gy1 - gy0)])
    return lines


def detect_text_boxes(
    img_bgr: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    merge_lines: bool = True,
) -> list[list[int]]:
    """Return text blocks as [x, y, w, h] in full-resolution coordinates."""
    H, W = img_bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(H, W))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lines = [b for b in _group_into_lines(_char_candidates(gray)) if _looks_like_text(lab, *b)]
    boxes = _merge_boxes(lines, line_gap=1.0, para_gap=0.8 if merge_lines else 0)

    inv = 1.0 / scale
    result = []
    for x, y, w, h in boxes:
        x0, y0 = int(np.floor(x * inv)), int(np.floor(y * inv))
        x1, y1 = min(W, int(np.ceil((x + w) * inv))), min(H, int(np.ceil((y + h) * inv)))
        if exclude_mask is not None:
            overlap = cv2.countNonZero(exclude_mask[y0:y1, x0:x1]) / float(max(1, (x1 - x0) * (y1 - y0)))
            if overlap > 0.4:
                continue
        result.append([x0, y0, x1 - x0, y1 - y0])
    result.sort(key=lambda b: (b[1], b[0]))
    return result


def _looks_like_text(lab: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Text sits on a fairly uniform surround and covers a moderate share of its box."""
    H, W = lab.shape[:2]
    p = max(2, h // 5)
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    crop = lab[y0:y1, x0:x1]
    border = np.concatenate([crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
                             crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(crop - bg, axis=2)
    border_dist = np.linalg.norm(border - bg, axis=1)
    if np.percentile(border_dist, 75) > 25:  # busy surround -> probably texture
        return False
    fg = dist > max(20.0, np.percentile(border_dist, 95) + 8)
    ratio = fg.mean()
    return 0.04 < ratio < 0.7


def extract_text_region(img_rgb: np.ndarray, box: list[int]) -> TextRegion | None:
    """Separate text strokes inside `box` from their background."""
    H, W = img_rgb.shape[:2]
    x, y, w, h = box
    p = max(3, min(w, h) // 6)
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    crop = img_rgb[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)

    bw = 2
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    bg_lab = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_lab, axis=2)

    d8 = np.clip(dist * (255.0 / max(1.0, dist.max())), 0, 255).astype(np.uint8)
    t8, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = t8 * max(1.0, dist.max()) / 255.0
    t = max(t, 12.0)
    strong = dist > t
    if strong.sum() < 10:
        return None
    d_fg = float(np.median(dist[strong]))
    lo = 0.45 * t
    alpha = np.clip((dist - lo) / max(1e-3, d_fg - lo), 0.0, 1.0)

    # Drop specks not connected to a real stroke
    n, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.15).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    min_area = max(3, int(0.00005 * crop.shape[0] * crop.shape[1]))
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    alpha *= keep[labels]

    # Un-mix anti-aliased edge pixels from the background colour
    bg_rgb = np.median(np.concatenate([crop[:bw].reshape(-1, 3), crop[-bw:].reshape(-1, 3),
                                       crop[:, :bw].reshape(-1, 3), crop[:, -bw:].reshape(-1, 3)]), axis=0)
    a3 = alpha[..., None]
    rgb = (crop.astype(np.float32) - (1.0 - a3) * bg_rgb) / np.maximum(a3, 1e-3)
    rgb = np.clip(rgb, 0, 255)
    rgb[alpha <= 0] = crop[alpha <= 0]

    rgba = np.dstack([rgb, alpha * 255.0]).astype(np.uint8)

    k = max(3, int(round(min(H, W) / 300)))
    mask = cv2.dilate((alpha > 0.05).astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

    # Trim fully transparent margins of the layer
    ys, xs = np.nonzero(alpha > 0.02)
    if len(xs) == 0:
        return None
    cx0, cx1, cy0, cy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    full_mask = np.zeros((H, W), np.uint8)
    full_mask[y0:y1, x0:x1] = mask
    return TextRegion(
        x=x0 + int(cx0), y=y0 + int(cy0), w=int(cx1 - cx0), h=int(cy1 - cy0),
        rgba=rgba[cy0:cy1, cx0:cx1].copy(),
        mask=full_mask,
    )
