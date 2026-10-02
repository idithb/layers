"""FastAPI server: serves the editor and the separation / resize API.

Run:  uvicorn app.server:app --port 8000

Processing takes from seconds to minutes on CPU, longer than proxies in front
of a hosted site allow for one request. So the POST endpoints only queue a
job and return its id; the page polls GET /api/jobs/{id} for the result.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import models
from .pipeline import resize_background, separate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_SIDE = int(os.environ.get("LAYERS_MAX_SIDE", 4096))
MAX_UPLOAD = 40 * 1024 * 1024
MAX_QUEUE = int(os.environ.get("LAYERS_MAX_QUEUE", 20))
JOB_TTL = 30 * 60  # finished results are kept for 30 minutes

app = FastAPI(title="Layers")
app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")

# One job at a time: each model run needs several GB of RAM on CPU
_job_lock = asyncio.Lock()
_jobs: dict[str, dict] = {}
_tasks: set[asyncio.Task] = set()


def _cleanup_jobs() -> None:
    now = time.time()
    for job_id in [k for k, j in _jobs.items() if j["status"] in ("done", "error") and now - j["finished"] > JOB_TTL]:
        del _jobs[job_id]


def _submit(fn, *args, **kwargs) -> dict:
    _cleanup_jobs()
    pending = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
    if pending >= MAX_QUEUE:
        raise HTTPException(503, "השרת עמוס כרגע, נסו שוב בעוד כמה דקות")
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "created": time.time(), "finished": 0.0, "result": None, "error": None}

    async def run():
        job = _jobs[job_id]
        try:
            async with _job_lock:
                job["status"] = "running"
                job["result"] = await asyncio.to_thread(fn, *args, **kwargs)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            job["status"], job["error"] = "error", str(e)
        job["finished"] = time.time()

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job": job_id}


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


def _separate_job(img: np.ndarray, **options) -> dict:
    result = separate(img, **options)
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


def _resize_job(bg: np.ndarray, width: int, height: int, mode: str) -> dict:
    out, transform = resize_background(bg, width, height, mode)
    return {"width": width, "height": height, "background": _png_data_url(out), "transform": transform}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    # "gradio": processing goes through the Gradio API (ZeroGPU Space), see space_app.py
    return {**models.status(), "backend": "gradio" if os.environ.get("LAYERS_GRADIO") else "jobs"}


@app.post("/api/separate")
async def api_separate(
    image: UploadFile = File(...),
    detect_text: bool = Form(True),
    detect_subject: bool = Form(True),
    merge_text_lines: bool = Form(True),
):
    img = _read_image(await image.read())
    return _submit(
        _separate_job, img, detect_text=detect_text, detect_subject=detect_subject, merge_text_lines=merge_text_lines
    )


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
    return _submit(_resize_job, bg, width, height, mode)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (it may have expired)")
    if job["status"] == "done":
        return {"status": "done", "result": job["result"]}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    ahead = sum(
        1 for j in _jobs.values() if j["status"] in ("queued", "running") and j["created"] < job["created"]
    )
    return {"status": job["status"], "ahead": ahead}
