"""Model wrappers: BiRefNet (subject segmentation) and LaMa (inpainting).

Both models are loaded lazily and cached. If a model file or its runtime is
missing, a lighter fallback is used so the app keeps working:
  * no BiRefNet -> no subject layer
  * no LaMa     -> OpenCV Telea inpainting
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import types
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("LAYERS_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
BIREFNET_PATH = MODELS_DIR / "birefnet-general.onnx"
LAMA_PATH = MODELS_DIR / "big-lama.pt"
BIREFNET_PTH = MODELS_DIR / "birefnet-general.pth"
BIREFNET_SRC_DIR = MODELS_DIR / "BiRefNet-src"
BIREFNET_SRC_COMMIT = "ebcc0bc8ec7fe919cec829f2dea656b3078acddc"
BIREFNET_SRC_REPO = "https://github.com/ZhengPeng7/BiRefNet"

# "onnx" (default, CPU-friendly) or "torch" (PyTorch, for GPUs)
SEGMENTER_BACKEND = os.environ.get("LAYERS_BIREFNET_BACKEND", "onnx")

MODEL_URLS = {
    BIREFNET_PATH.name: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
    LAMA_PATH.name: "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
}


def release_memory() -> None:
    """Hand freed heap memory back to the OS (glibc keeps it otherwise)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class BiRefNetSegmenter:
    """BiRefNet exported to ONNX (1024x1024 input, ImageNet normalisation)."""

    size = 1024

    def __init__(self, path: Path = BIREFNET_PATH):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # BiRefNet peaks at ~8 GB on CPU; without the arena that memory is
        # released after each run instead of being held next to LaMa's.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return a soft foreground mask in [0, 1] with the image's size."""
        H, W = img_rgb.shape[:2]
        x = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0][0, 0]
        release_memory()
        pred = 1.0 / (1.0 + np.exp(-out))
        mi, ma = float(pred.min()), float(pred.max())
        pred = (pred - mi) / max(1e-6, ma - mi)
        return cv2.resize(pred.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)


class TorchBiRefNetSegmenter:
    """BiRefNet in PyTorch (for GPUs, e.g. ZeroGPU Spaces).

    Uses the original model code from github.com/ZhengPeng7/BiRefNet (pinned
    commit, MIT) with the same "general" weights as the ONNX file, both
    fetched by scripts/download_models.py. This avoids `transformers`, whose
    huggingface-hub pin conflicts with Gradio's.
    """

    size = 1024

    def __init__(self, src_dir: Path | None = None, weights: Path | None = None):
        import torch

        self.torch = torch
        src_dir = src_dir or BIREFNET_SRC_DIR
        weights = weights or BIREFNET_PTH
        # The repo uses top-level module names (config, dataset, models); import
        # them from its folder without clashing with anything already loaded.
        names = ("config", "dataset", "models")
        saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
        sys.path.insert(0, str(src_dir))
        # dataset.py is training-only (needs torchvision); inference only needs
        # the label list, used when auxiliary classification is on (it is off).
        sys.modules["dataset"] = types.SimpleNamespace(class_labels_TR_sorted=[])
        try:
            from models.birefnet import BiRefNet
        finally:
            sys.path.remove(str(src_dir))
            for n in names:
                sys.modules.pop(n, None)
            sys.modules.update(saved)

        torch.set_float32_matmul_precision("high")
        self.model = BiRefNet(bb_pretrained=False)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        for prefix in ("module.", "_orig_mod."):
            state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device("cpu")

    def to(self, device) -> None:
        self.device = self.torch.device(device)
        self.model.to(self.device)
        # half precision on GPU, as in the model card
        self.model.half() if self.device.type == "cuda" else self.model.float()

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = img_rgb.shape[:2]
        x = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
        t = t.half() if self.device.type == "cuda" else t
        with torch.inference_mode():
            pred = self.model(t)[-1].sigmoid().float()[0, 0].cpu().numpy()
        return cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)


class LamaInpainter:
    """Big-LaMa TorchScript model."""

    name = "lama"

    def __init__(self, path: Path = LAMA_PATH, max_size: int = 1024):
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        self.max_size = max_size

    def to(self, device) -> None:
        self.device = self.torch.device(device)
        self.model.to(self.device)

    def _run(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = img.shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        img_p = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask_p = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        t_img = torch.from_numpy(img_p).permute(2, 0, 1)[None].float().div(255).to(self.device)
        t_mask = torch.from_numpy((mask_p > 127).astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(t_img, t_mask)
        out = out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        release_memory()
        return out[:h, :w]

    def inpaint(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint region by region ("crop" strategy).

        Each group of nearby holes is processed in a crop with its own context.
        Big holes are processed at <=512 px, where LaMa was trained: at higher
        resolutions it fills large areas with a flat grey blur. Small holes
        (text) keep up to `max_size` so detail is preserved.
        """
        if not mask.any():
            return img_rgb.copy()
        H, W = img_rgb.shape[:2]
        out = img_rgb.copy()
        remaining = (mask > 127).astype(np.uint8)
        group = cv2.dilate(remaining, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(group, connectivity=8)
        for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
            x, y, w, h = stats[i, :4]
            pad = max(48, int(0.5 * min(w, h)) + 32)
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
            crop_mask = remaining[y0:y1, x0:x1] * 255
            if not crop_mask.any():
                continue
            limit = self.max_size if max(w, h) < 256 else 512
            out[y0:y1, x0:x1] = self._inpaint_crop(out[y0:y1, x0:x1], crop_mask, limit)
            remaining[y0:y1, x0:x1] = 0
        return out

    def _inpaint_crop(self, img_rgb: np.ndarray, mask: np.ndarray, limit: int) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        scale = min(1.0, limit / max(H, W))
        if scale < 1.0:
            small = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            small_mask = cv2.dilate(small_mask, np.ones((3, 3), np.uint8))
            out = cv2.resize(self._run(small, small_mask), (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            out = self._run(img_rgb, mask)
        out = self._fix_thin_regions(img_rgb, out, mask)
        # Keep original pixels outside the mask, feather the seam
        soft = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.5)[..., None]
        soft = np.maximum(soft, (mask > 127)[..., None])
        return (out * soft + img_rgb * (1 - soft)).astype(np.uint8)

    @staticmethod
    def _fix_thin_regions(img_rgb: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove the faint "ghost" LaMa leaves on thin masks (e.g. text strokes).

        On thin parts of the mask, LaMa's low-frequency colour is replaced by
        the one from OpenCV's diffusion inpainting (which has no colour shift);
        LaMa's fine texture is kept. Wide areas are left untouched.
        """
        m = (mask > 127).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        w = np.clip((14.0 - dist) / 8.0, 0.0, 1.0) * m
        if not w.any():
            return out
        diffused = cv2.inpaint(img_rgb, m * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
        out_f = out.astype(np.float32)
        corr = cv2.GaussianBlur(diffused, (0, 0), 3) - cv2.GaussianBlur(out_f, (0, 0), 3)
        return np.clip(out_f + w[..., None] * corr, 0, 255).astype(np.uint8)


class OpenCVInpainter:
    name = "opencv"

    def inpaint(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return img_rgb.copy()
        radius = max(3, int(round(min(img_rgb.shape[:2]) / 200)))
        return cv2.inpaint(img_rgb, (mask > 127).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


_lock = threading.Lock()
_segmenter: BiRefNetSegmenter | None | bool = False  # False = not tried yet
_inpainter = None


def get_segmenter():
    global _segmenter
    with _lock:
        if _segmenter is False:
            try:
                if SEGMENTER_BACKEND == "torch":
                    _segmenter = TorchBiRefNetSegmenter()
                    log.info("BiRefNet (PyTorch) loaded from %s", BIREFNET_PTH)
                else:
                    _segmenter = BiRefNetSegmenter()
                    log.info("BiRefNet loaded from %s", BIREFNET_PATH)
            except Exception as e:  # noqa: BLE001
                log.warning("BiRefNet unavailable (%s); subject layer disabled", e)
                _segmenter = None
        return _segmenter  # type: ignore[return-value]


def get_inpainter():
    global _inpainter
    with _lock:
        if _inpainter is None:
            try:
                _inpainter = LamaInpainter()
                log.info("LaMa loaded from %s", LAMA_PATH)
            except Exception as e:  # noqa: BLE001
                log.warning("LaMa unavailable (%s); falling back to OpenCV inpainting", e)
                _inpainter = OpenCVInpainter()
        return _inpainter


def status() -> dict:
    return {
        "birefnet": (BIREFNET_PTH if SEGMENTER_BACKEND == "torch" else BIREFNET_PATH).exists(),
        "lama": LAMA_PATH.exists(),
        "models_dir": str(MODELS_DIR),
    }
