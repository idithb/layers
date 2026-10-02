"""Entry point for a Hugging Face Space (Gradio SDK, ZeroGPU hardware).

ZeroGPU attaches a GPU only while a function decorated with @spaces.GPU runs,
and those functions must be called through Gradio. So:
  * the models are loaded on CPU at startup,
  * the two heavy operations are exposed as Gradio API endpoints that move the
    models to the GPU for the duration of the call,
  * the regular editor (static/) is served at "/" and calls those endpoints
    with the Gradio JS client (mounted at /gradio).
Off ZeroGPU (e.g. locally) @spaces.GPU is a no-op and everything runs on CPU.
"""

import os

os.environ.setdefault("LAYERS_BIREFNET_BACKEND", "torch")
os.environ["LAYERS_GRADIO"] = "1"

import spaces  # noqa: E402  (must be imported before torch initialises CUDA)

import base64  # noqa: E402
import io  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

subprocess.run([sys.executable, "scripts/download_models.py"], check=True)

from app import models  # noqa: E402
from app.server import _read_image, _resize_job, _separate_job, app  # noqa: E402

segmenter = models.get_segmenter()
inpainter = models.get_inpainter()


@contextmanager
def on_gpu():
    import torch

    use_gpu = torch.cuda.is_available()
    movable = [m for m in (segmenter, inpainter) if hasattr(m, "to")]
    if use_gpu:
        for m in movable:
            m.to("cuda")
    try:
        yield
    finally:
        if use_gpu:
            for m in movable:
                m.to("cpu")


def _decode(data_url: str):
    return _read_image(base64.b64decode(data_url.split(",", 1)[-1]))


@spaces.GPU(duration=60)
def separate(image: str, detect_text: bool, detect_subject: bool, merge_text_lines: bool) -> dict:
    """Split an image (data URL) into background, subject and text layers."""
    img = _decode(image)
    with on_gpu():
        return _separate_job(
            img, detect_text=detect_text, detect_subject=detect_subject, merge_text_lines=merge_text_lines
        )


@spaces.GPU(duration=60)
def resize(background: str, width: int, height: int, mode: str) -> dict:
    """Resize the background (data URL); 'extend' paints the new margins with LaMa."""
    width, height = int(width), int(height)
    if not (16 <= width <= 4096 and 16 <= height <= 4096) or mode not in ("extend", "cover", "stretch"):
        raise gr.Error("invalid size or mode")
    bg = _decode(background)
    with on_gpu():
        return _resize_job(bg, width, height, mode)


with gr.Blocks(title="Layers API") as demo:
    gr.Markdown("Layers API – the editor is at [/](/)")
    # one shared queue: both calls move the same models to the GPU
    gr.api(separate, api_name="separate", concurrency_id="gpu", concurrency_limit=1)
    gr.api(resize, api_name="resize", concurrency_id="gpu", concurrency_limit=1)

demo.queue(default_concurrency_limit=1, max_size=20)
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
