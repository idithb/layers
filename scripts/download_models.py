"""Download the models into models/: Big-LaMa (TorchScript) and BiRefNet (ONNX, or
PyTorch weights + model code when LAYERS_BIREFNET_BACKEND=torch)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import (  # noqa: E402
    BIREFNET_PATH,
    BIREFNET_PTH,
    BIREFNET_SRC_COMMIT,
    BIREFNET_SRC_DIR,
    BIREFNET_SRC_REPO,
    MODEL_URLS,
    MODELS_DIR,
    SEGMENTER_BACKEND,
)

TORCH_BIREFNET_URL = "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"-> {dest.name}  ({url})")

    last = [-1]

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total) // 10 * 10
            if pct != last[0]:
                last[0] = pct
                print(f"   {pct:3d}%  of {total / 1e6:.1f} MB", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    tmp.rename(dest)


def download_source(repo: str, commit: str, dest: Path) -> None:
    """Fetch one commit of a GitHub repo: git if available, else the source archive."""
    print(f"-> {dest.name}  ({repo} @ {commit[:7]})")
    tmp_dest = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp_dest, ignore_errors=True)
    try:
        tmp_dest.mkdir(parents=True)
        for cmd in (["init", "-q"], ["fetch", "-q", "--depth", "1", repo, commit], ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(tmp_dest), *cmd], check=True)
        shutil.rmtree(tmp_dest / ".git")
        tmp_dest.rename(dest)
        return
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"   git failed ({e}); trying the source archive")
        shutil.rmtree(tmp_dest, ignore_errors=True)
    url = f"{repo}/archive/{commit}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "src.tar.gz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        (top,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
        shutil.move(str(top), dest)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        if name == BIREFNET_PATH.name and SEGMENTER_BACKEND == "torch":
            continue  # the torch backend uses the .pth weights below instead
        if dest.exists():
            print(f"✓ {name} already exists")
            continue
        download(url, dest)
    if SEGMENTER_BACKEND == "torch":
        if not BIREFNET_PTH.exists():
            download(TORCH_BIREFNET_URL, BIREFNET_PTH)
        if not BIREFNET_SRC_DIR.exists():
            download_source(BIREFNET_SRC_REPO, BIREFNET_SRC_COMMIT, BIREFNET_SRC_DIR)
    print("done:", MODELS_DIR)


if __name__ == "__main__":
    main()
