FROM python:3.11-slim

# System ffmpeg is preferred in the container (faster, no first-run download);
# app/audio.py falls back to the imageio-ffmpeg bundled binary if this is ever
# absent (e.g. a slimmer base image), so the service still works either way.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Runs as a non-root user — the process only needs to read uploads it writes
# itself under /tmp, never system paths.
RUN useradd --create-home appuser
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN chown -R appuser:appuser /home/appuser /app
USER appuser

# Bake the model into the image at build time instead of downloading it from
# Hugging Face on first request: faster cold starts, and the running
# container has no live dependency on an external model hub. MODEL_SIZE here
# must match the MODEL_SIZE the app is run with (default "base").
ARG MODEL_SIZE=base
ENV MODEL_SIZE=${MODEL_SIZE}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8')"

# Cloud Run (and similar PaaS targets) inject the listen port via $PORT at
# runtime rather than using a fixed port, so this must be read at container
# start, not baked into a fixed --port flag.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
