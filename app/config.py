import os
from typing import Optional


class Settings:
    # Model
    MODEL_SIZE: str = os.getenv("MODEL_SIZE", "base")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "int8")

    # Upload limits (Security by Design: bound every untrusted input)
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 100 * 1024 * 1024))  # 100 MB
    ALLOWED_EXTENSIONS: frozenset = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"})

    # Long-audio routing: files longer than this go through the async job path
    # instead of blocking a request/response cycle.
    SYNC_DURATION_THRESHOLD_SECONDS: float = float(
        os.getenv("SYNC_DURATION_THRESHOLD_SECONDS", 120)
    )

    # Long-file chunking (async path)
    CHUNK_TARGET_SECONDS: float = float(os.getenv("CHUNK_TARGET_SECONDS", 300))
    CHUNK_SEARCH_WINDOW_SECONDS: float = float(os.getenv("CHUNK_SEARCH_WINDOW_SECONDS", 30))

    # Subprocess safety
    FFMPEG_TIMEOUT_SECONDS: int = int(os.getenv("FFMPEG_TIMEOUT_SECONDS", 120))

    # Job store
    JOB_TTL_SECONDS: int = int(os.getenv("JOB_TTL_SECONDS", 3600))

    # Optional simple API key gate (stand-in for real OAuth2/JWT in production).
    # Leave unset to disable — but leaving it unset means the API is open, so the
    # README calls this out explicitly rather than silently defaulting to "secure".
    API_KEY: Optional[str] = os.getenv("API_KEY") or None

    # Rate limiting (abuse / cost-control guardrail)
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")


settings = Settings()
