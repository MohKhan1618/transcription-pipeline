from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    language: str
    language_probability: float
    duration: float
    full_transcript: str
    segments: List[Segment]


class SyncTranscriptionResponse(TranscriptionResult):
    status: str = "completed"
    mode: str = "sync"


class JobAcceptedResponse(BaseModel):
    status: str = "accepted"
    mode: str = "async"
    job_id: str
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    result: Optional[TranscriptionResult] = None
    error: Optional[str] = None
