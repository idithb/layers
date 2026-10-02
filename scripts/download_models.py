"""Download BiRefNet (ONNX) and Big-LaMa (TorchScript) into models/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import BIREFNET_PATH, MODEL_URLS, MODELS_DIR, SEGMENTER_BACKEND  # noqa: E402


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


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        if name == BIREFNET_PATH.name and SEGMENTER_BACKEND == "torch":
            continue  # the torch backend loads BiRefNet from the Hugging Face Hub
        if dest.exists():
            print(f"✓ {name} already exists")
            continue
        download(url, dest)
    print("done:", MODELS_DIR)


if __name__ == "__main__":
    main()
