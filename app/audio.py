"""
Audio validation, format normalization and chunking.

All ffmpeg calls use an explicit argument list (never shell=True) so a crafted
filename or header can't be used for command injection, and every subprocess
call carries a timeout so a malformed/adversarial file can't hang a worker.
"""
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings

_FFMPEG_PATH: Optional[str] = None

# Signature bytes for the formats we claim to support. Extensions are trivially
# spoofable, so we confirm the container matches what the filename claims
# before ever handing the file to ffmpeg/the model.
_SIGNATURES = {
    ".wav": [(0, b"RIFF")],
    ".flac": [(0, b"fLaC")],
    ".ogg": [(0, b"OggS")],
    ".mp3": [(0, b"ID3"), (0, b"\xff\xfb"), (0, b"\xff\xf3"), (0, b"\xff\xf2")],
    ".m4a": [(4, b"ftyp")],
    # Produced by the browser's MediaRecorder API for mic recordings (EBML/Matroska container).
    ".webm": [(0, b"\x1a\x45\xdf\xa3")],
}


class UnsupportedAudioError(ValueError):
    pass


class AudioProcessingError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        # Prefer a system ffmpeg if present; otherwise fall back to the static
        # binary bundled by imageio-ffmpeg so the service works without
        # requiring a system-wide install (useful on Windows dev machines).
        _FFMPEG_PATH = shutil.which("ffmpeg")
        if not _FFMPEG_PATH:
            import imageio_ffmpeg

            _FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG_PATH


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedAudioError(
            f"Unsupported extension '{ext}'. Allowed: {sorted(settings.ALLOWED_EXTENSIONS)}"
        )
    return ext


def validate_signature(path: str, claimed_ext: str) -> None:
    """Reject files whose actual bytes don't match the claimed extension."""
    signatures = _SIGNATURES.get(claimed_ext)
    if not signatures:
        return
    with open(path, "rb") as f:
        head = f.read(16)
    for offset, magic in signatures:
        if head[offset : offset + len(magic)] == magic:
            return
    raise UnsupportedAudioError(
        f"File content does not match its '{claimed_ext}' extension "
        "(failed magic-byte check)."
    )


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Normalize any supported input to 16kHz mono PCM WAV."""
    cmd = [
        ffmpeg_path(),
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        output_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=settings.FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            f"Audio conversion timed out after {settings.FFMPEG_TIMEOUT_SECONDS}s"
        ) from exc
    if proc.returncode != 0:
        raise AudioProcessingError(
            "ffmpeg failed to decode the file — likely invalid or corrupted audio."
        )


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")


def probe_duration_seconds(path: str) -> float:
    cmd = [ffmpeg_path(), "-i", path]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=settings.FFMPEG_TIMEOUT_SECONDS,
    )
    stderr = proc.stderr.decode(errors="ignore")
    match = _DURATION_RE.search(stderr)
    if not match:
        raise AudioProcessingError("Could not determine audio duration.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


_SILENCE_RE = re.compile(r"silence_(start|end):\s*([\d.]+)")


def _detect_silences(wav_path: str) -> List[Tuple[float, float]]:
    """Return (start, end) of silent intervals using ffmpeg's silencedetect filter."""
    cmd = [
        ffmpeg_path(),
        "-i",
        wav_path,
        "-af",
        "silencedetect=noise=-35dB:d=0.5",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=settings.FFMPEG_TIMEOUT_SECONDS,
    )
    stderr = proc.stderr.decode(errors="ignore")
    events = _SILENCE_RE.findall(stderr)
    silences = []
    pending_start = None
    for kind, value in events:
        if kind == "start":
            pending_start = float(value)
        elif kind == "end" and pending_start is not None:
            silences.append((pending_start, float(value)))
            pending_start = None
    return silences


def compute_chunk_boundaries(wav_path: str, duration: float) -> List[Tuple[float, float]]:
    """
    Split a long recording into ~CHUNK_TARGET_SECONDS chunks, snapping each cut
    to the nearest detected silence within CHUNK_SEARCH_WINDOW_SECONDS so we
    don't cut a chunk mid-word. Falls back to a hard cut if no silence is
    found near the target point.
    """
    target = settings.CHUNK_TARGET_SECONDS
    window = settings.CHUNK_SEARCH_WINDOW_SECONDS
    if duration <= target:
        return [(0.0, duration)]

    silences = _detect_silences(wav_path)
    silence_midpoints = [(s + e) / 2 for s, e in silences]

    boundaries = [0.0]
    cursor = target
    while cursor < duration:
        candidates = [
            m for m in silence_midpoints if abs(m - cursor) <= window and m > boundaries[-1]
        ]
        cut = min(candidates, key=lambda m: abs(m - cursor)) if candidates else cursor
        if cut - boundaries[-1] < 1.0:
            cursor += target
            continue
        boundaries.append(cut)
        cursor = cut + target
    boundaries.append(duration)

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def extract_chunk(wav_path: str, start: float, end: float, out_path: str) -> None:
    cmd = [
        ffmpeg_path(),
        "-y",
        "-i",
        wav_path,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-c",
        "copy",
        out_path,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=settings.FFMPEG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise AudioProcessingError(f"Failed to extract chunk [{start}, {end}]")
