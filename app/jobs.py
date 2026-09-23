"""
Lightweight in-memory job store for the async (long-file) transcription path.

This demonstrates the architecture — submit, background-process, poll — with
zero extra infrastructure. It is explicitly NOT what should back a real
multi-instance deployment: jobs live in one process's memory, so they vanish
on restart and aren't visible across replicas. In production this dict is
replaced by Celery/RQ + Redis (or SQS + worker fleet); every call site here
(`create_job`, `set_result`, `get_job`) maps 1:1 onto that swap.
"""
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.schemas import TranscriptionResult

_lock = threading.Lock()
_jobs: Dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | processing | completed | failed
    result: Optional[TranscriptionResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


def create_job() -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = Job(id=job_id)
    return job_id


def set_status(job_id: str, status: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].status = status


def set_result(job_id: str, result: TranscriptionResult) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].status = "completed"
            _jobs[job_id].result = result


def set_error(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].status = "failed"
            _jobs[job_id].error = error


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def process_long_audio(job_id: str, wav_path: str, tmp_dir: Optional[str] = None) -> None:
    """Runs in a background thread. Owns cleanup of wav_path, chunk files and tmp_dir."""
    from app.audio import compute_chunk_boundaries, extract_chunk, probe_duration_seconds
    from app.transcriber import transcribe_file, merge_results

    chunk_paths = []
    try:
        set_status(job_id, "processing")
        duration = probe_duration_seconds(wav_path)
        boundaries = compute_chunk_boundaries(wav_path, duration)

        chunk_results = []
        for start, end in boundaries:
            chunk_path = f"{wav_path}.chunk_{start:.2f}_{end:.2f}.wav"
            extract_chunk(wav_path, start, end, chunk_path)
            chunk_paths.append(chunk_path)
            chunk_results.append(transcribe_file(chunk_path, time_offset=start))

        merged = merge_results(chunk_results, duration)
        set_result(job_id, merged)
    except Exception as exc:  # noqa: BLE001 — this is a background task boundary
        set_error(job_id, str(exc))
    finally:
        for path in chunk_paths + [wav_path]:
            if os.path.exists(path):
                os.remove(path)
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass  # not empty — leave it rather than risk deleting something unexpected
