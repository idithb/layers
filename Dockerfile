# Image for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Bake the models (~1.2GB) into the image so restarts don't re-download them
RUN python scripts/download_models.py && chown -R user /app/models

USER user
ENV PORT=7860 LAYERS_MAX_SIDE=2560
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
