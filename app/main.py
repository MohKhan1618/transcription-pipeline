import logging
import os
import tempfile
import threading
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.audio import (
    AudioProcessingError,
    UnsupportedAudioError,
    convert_to_wav,
    probe_duration_seconds,
    validate_extension,
    validate_signature,
)
from app.config import settings
from app.jobs import create_job, get_job, process_long_audio
from app.schemas import JobAcceptedResponse, JobStatusResponse, SyncTranscriptionResponse
from app.transcriber import transcribe_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription-api")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Transcription Pipeline API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
async def demo_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    No-op when API_KEY isn't configured (local/dev). In any real deployment,
    set API_KEY (or swap this for real OAuth2/JWT) — an unauthenticated
    endpoint that runs a model and shells out to ffmpeg on every request is
    an open cost/DoS surface.
    """
    if settings.API_KEY is None:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _read_upload_with_cap(file: UploadFile, dest_path: str) -> None:
    total = 0
    chunk_size = 1024 * 1024
    with open(dest_path, "wb") as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds max allowed size of {settings.MAX_UPLOAD_BYTES} bytes",
                )
            out.write(chunk)
    if total == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file upload")


@app.post(
    "/v1/transcribe",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.RATE_LIMIT)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    # 1. Extension allow-list check (cheap, rejects obviously wrong uploads fast)
    try:
        ext = validate_extension(file.filename or "")
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # 2. Stream to disk under a size cap — never buffer an unbounded upload in memory
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_")
    raw_path = os.path.join(tmp_dir, f"upload{ext}")
    wav_path = os.path.join(tmp_dir, "normalized.wav")

    try:
        await _read_upload_with_cap(file, raw_path)

        # 3. Magic-byte check — the extension is user-supplied and not trustworthy on its own
        try:
            validate_signature(raw_path, ext)
        except UnsupportedAudioError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        # 4. Normalize to 16kHz mono WAV (bounded by FFMPEG_TIMEOUT_SECONDS)
        try:
            await run_in_threadpool(convert_to_wav, raw_path, wav_path)
            duration = await run_in_threadpool(probe_duration_seconds, wav_path)
        except AudioProcessingError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

        # 5. Route by duration: short files transcribe inline, long files go async.
        # Either way, transcription runs in a threadpool — never inline in this
        # coroutine — so one long request can't stall the whole event loop.
        if duration <= settings.SYNC_DURATION_THRESHOLD_SECONDS:
            try:
                result = await run_in_threadpool(transcribe_file, wav_path)
                return SyncTranscriptionResponse(**result.model_dump())
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                os.rmdir(tmp_dir)

        # Long-file path: ownership of wav_path (and tmp_dir cleanup) passes to
        # the background thread from this point on.
        job_id = create_job()
        threading.Thread(
            target=process_long_audio, args=(job_id, wav_path, tmp_dir), daemon=True
        ).start()
        return JobAcceptedResponse(job_id=job_id, status_url=f"/v1/jobs/{job_id}")

    except HTTPException:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir) if not os.listdir(tmp_dir) else None
        raise


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_key)])
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobStatusResponse(job_id=job.id, status=job.status, result=job.result, error=job.error)
