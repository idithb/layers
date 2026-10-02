"""FastAPI server: serves the editor and the separation / resize API.

Run:  uvicorn app.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import models
from .pipeline import resize_background, separate

logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_SIDE = 4096
MAX_UPLOAD = 40 * 1024 * 1024

app = FastAPI(title="Layers")
# One job at a time: each model run needs several GB of RAM on CPU
_job_lock = asyncio.Lock()


async def _run_job(fn, *args, **kwargs):
    async with _job_lock:
        return await asyncio.to_thread(fn, *args, **kwargs)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _read_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot read image: {e}") from e
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(img).copy()


def _png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return models.status()


@app.post("/api/separate")
async def api_separate(
    image: UploadFile = File(...),
    detect_text: bool = Form(True),
    detect_subject: bool = Form(True),
    merge_text_lines: bool = Form(True),
):
    img = _read_image(await image.read())
    result = await _run_job(
        separate, img, detect_text=detect_text, detect_subject=detect_subject, merge_text_lines=merge_text_lines
    )
    return {
        "width": result.width,
        "height": result.height,
        "background": _png_data_url(result.background),
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "kind": layer.kind,
                "x": layer.x,
                "y": layer.y,
                "width": int(layer.rgba.shape[1]),
                "height": int(layer.rgba.shape[0]),
                "image": _png_data_url(layer.rgba),
            }
            for layer in result.layers
        ],
        "info": result.info,
    }


@app.post("/api/resize")
async def api_resize(
    background: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    mode: str = Form("extend"),
):
    if not (16 <= width <= MAX_SIDE and 16 <= height <= MAX_SIDE):
        raise HTTPException(400, f"size must be between 16 and {MAX_SIDE}")
    if mode not in ("extend", "cover", "stretch"):
        raise HTTPException(400, "mode must be extend, cover or stretch")
    bg = _read_image(await background.read())
    out, transform = await _run_job(resize_background, bg, width, height, mode)
    return {"width": width, "height": height, "background": _png_data_url(out), "transform": transform}
