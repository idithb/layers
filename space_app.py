"""Entry point for a Hugging Face Space with the (free) Gradio SDK.

The Space only needs a web server on port 7860, so this runs the regular
FastAPI app; the models are downloaded on first start.
"""

import os
import subprocess
import sys

import uvicorn

if __name__ == "__main__":
    subprocess.run([sys.executable, "scripts/download_models.py"], check=True)

    from app.server import app

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
