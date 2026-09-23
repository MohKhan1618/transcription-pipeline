import io
import os
import sys

os.environ.setdefault("MODEL_SIZE", "tiny")  # keep tests fast; must be set before app import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from tests.generate_sample_audio import make_tone_wav  # noqa: E402

client = TestClient(app)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TONE_WAV = os.path.join(FIXTURE_DIR, "tone.wav")


def setup_module():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    make_tone_wav(TONE_WAV, seconds=3.0)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_rejects_unsupported_extension():
    resp = client.post(
        "/v1/transcribe",
        files={"file": ("clip.xyz", io.BytesIO(b"not audio"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported extension" in resp.json()["detail"]


def test_rejects_extension_content_mismatch():
    # .wav extension, but the bytes don't have a RIFF header — should fail the
    # magic-byte check even though the extension looks fine.
    resp = client.post(
        "/v1/transcribe",
        files={"file": ("fake.wav", io.BytesIO(b"this is definitely not a wav file"), "audio/wav")},
    )
    assert resp.status_code == 400
    assert "magic-byte" in resp.json()["detail"]


def test_rejects_empty_file():
    resp = client.post(
        "/v1/transcribe",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 400


def test_sync_transcription_end_to_end():
    # The fixture is a synthetic tone, not speech, so VAD correctly strips it
    # to zero speech segments. This test's job is to prove the pipeline runs
    # end-to-end (upload -> validate -> convert -> transcribe -> respond)
    # without erroring, and that a no-speech input degrades gracefully
    # instead of raising — it is not asserting transcription *accuracy*,
    # which needs a real speech sample and a human/reference transcript.
    with open(TONE_WAV, "rb") as f:
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("tone.wav", f, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "sync"
    assert body["status"] == "completed"
    assert isinstance(body["segments"], list)
    assert body["full_transcript"] == ""  # no speech in a pure tone
    assert body["duration"] > 0


def test_job_not_found():
    resp = client.get("/v1/jobs/does-not-exist")
    assert resp.status_code == 404
