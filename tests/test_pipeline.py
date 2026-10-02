"""Fast tests: OpenCV parts run for real, the models are replaced by stubs."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.models import OpenCVInpainter
from app.pipeline import _mirror_layout, resize_background, separate
from app.text_detect import detect_text_boxes

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def poster() -> np.ndarray:
    im = Image.new("RGB", (800, 600), (240, 232, 220))
    d = ImageDraw.Draw(im)
    d.text((60, 60), "Summer Sale", font=ImageFont.truetype(FONT, 64), fill=(60, 30, 80))
    d.text((60, 480), "הזוכה בפרס", font=ImageFont.truetype(FONT, 48), fill=(200, 160, 30))
    d.rectangle((450, 200, 700, 450), fill=(40, 120, 90))  # the "subject"
    return np.asarray(im).copy()


class SquareSegmenter:
    def predict(self, img):
        m = np.zeros(img.shape[:2], np.float32)
        m[200:451, 450:701] = 1.0
        return m


def test_detects_latin_and_hebrew_lines():
    boxes = detect_text_boxes(poster()[..., ::-1].copy())
    assert len(boxes) == 2
    (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes
    assert y1 < 120 and w1 > 300
    assert y2 > 450 and w2 > 200


def test_separate_produces_layers_and_clean_background():
    img = poster()
    result = separate(img, segmenter=SquareSegmenter(), inpainter=OpenCVInpainter())
    kinds = sorted(layer.kind for layer in result.layers)
    assert kinds == ["subject", "text", "text"]
    subject = next(layer for layer in result.layers if layer.kind == "subject")
    assert (subject.x, subject.y) == (450, 200)
    # the subject and the text are gone from the background
    bg = result.background.astype(int)
    assert np.abs(bg[300, 575] - [240, 232, 220]).max() < 20
    assert np.abs(bg[60:130, 60:450] - [240, 232, 220]).max(axis=2).mean() < 10
    # background + layers recompose to the original
    comp = Image.fromarray(result.background).convert("RGBA")
    for layer in result.layers:
        comp.alpha_composite(Image.fromarray(layer.rgba), (layer.x, layer.y))
    diff = np.abs(np.asarray(comp.convert("RGB")).astype(int) - img.astype(int)).mean()
    assert diff < 2.0


@pytest.mark.parametrize("mode", ["extend", "cover", "stretch"])
def test_resize_background_modes(mode):
    bg = np.full((300, 400, 3), 128, np.uint8)
    out, t = resize_background(bg, 500, 700, mode, inpainter=OpenCVInpainter())
    assert out.shape == (700, 500, 3)
    if mode == "extend":
        assert t["sx"] == t["sy"] == pytest.approx(1.25)
        assert t["ox"] == 0 and t["oy"] == pytest.approx((700 - 375) // 2)
        assert np.abs(out.astype(int) - 128).max() < 10
    elif mode == "cover":
        assert t["sx"] == pytest.approx(700 / 300)
        assert t["ox"] < 0
    else:
        assert (t["sx"], t["sy"]) == (pytest.approx(1.25), pytest.approx(700 / 300))


def test_mirror_layout_crop_keeps_image_and_holes():
    img = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    canvas, mask, (y0, y1, x0, x1) = _mirror_layout(img, 2, 3, 4, 1)
    inner = canvas[y0:y1, x0:x1]
    assert inner.shape == (5 + 2 + 3, 7 + 4 + 1, 3)
    assert np.array_equal(inner[2:7, 4:11], img)
    assert mask[y0:y1, x0:x1][2:7, 4:11].max() == 0
    assert mask[y0:y1, x0:x1][:2].min() == 255
