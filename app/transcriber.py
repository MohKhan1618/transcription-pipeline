"""
Model wrapper. A single WhisperModel instance is shared across requests —
faster-whisper/ctranslate2 releases the GIL during inference, and callers are
always routed through a threadpool (see main.py), so this is safe for the
"one process, several worker threads" deployment this service uses.

For true multi-process horizontal scaling, each worker process loads its own
model instance (that's the standard faster-whisper/uvicorn --workers pattern);
this module's lazy singleton works unchanged in that model too.
"""
import threading
from typing import List

from faster_whisper import WhisperModel

from app.config import settings
from app.schemas import Segment, TranscriptionResult

_model = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    settings.MODEL_SIZE,
                    device=settings.DEVICE,
                    compute_type=settings.COMPUTE_TYPE,
                )
    return _model


def transcribe_file(wav_path: str, time_offset: float = 0.0) -> TranscriptionResult:
    """
    Blocking, CPU-bound call — must be invoked via run_in_threadpool (sync path)
    or from a background-task thread (async path), never awaited directly in
    an async def, or it stalls the event loop for every other request.
    """
    model = get_model()
    try:
        segments_iter, info = model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True,
        )
    except ValueError:
        # faster-whisper's language-detection step raises when VAD strips the
        # entire clip as non-speech (silence, pure tone, noise-only audio) —
        # there's nothing left to detect a language from. That's a valid
        # input, not a server error, so we return an empty transcript instead
        # of surfacing a 500 to the caller.
        from app.audio import probe_duration_seconds

        duration = probe_duration_seconds(wav_path)
        return TranscriptionResult(
            language="unknown",
            language_probability=0.0,
            duration=round(duration, 2),
            full_transcript="",
            segments=[],
        )

    segments: List[Segment] = []
    full_text_parts = []
    for i, seg in enumerate(segments_iter):
        text = seg.text.strip()
        segments.append(
            Segment(
                id=i,
                start=round(seg.start + time_offset, 2),
                end=round(seg.end + time_offset, 2),
                text=text,
            )
        )
        full_text_parts.append(text)

    return TranscriptionResult(
        language=info.language,
        language_probability=round(info.language_probability, 2),
        duration=round(info.duration, 2),
        full_transcript=" ".join(full_text_parts),
        segments=segments,
    )


def merge_results(chunk_results: List[TranscriptionResult], total_duration: float) -> TranscriptionResult:
    """Merge already-offset chunk results (see transcribe_file's time_offset) into one timeline."""
    all_segments = []
    full_text_parts = []
    for result in chunk_results:
        all_segments.extend(result.segments)
        full_text_parts.append(result.full_transcript)

    for i, seg in enumerate(all_segments):
        seg.id = i

    best = max(chunk_results, key=lambda r: len(r.full_transcript)) if chunk_results else None

    return TranscriptionResult(
        language=best.language if best else "unknown",
        language_probability=best.language_probability if best else 0.0,
        duration=round(total_duration, 2),
        full_transcript=" ".join(full_text_parts),
        segments=all_segments,
    )
