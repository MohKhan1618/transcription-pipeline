# Transcription Pipeline API

A FastAPI service that accepts an audio file, normalizes it, transcribes it
with word/segment-level timestamps using [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
and returns structured JSON. Long files are routed to an async job queue
instead of blocking the request.

## Live demo

**https://transcription-api-662735249234.us-central1.run.app**

Open it directly for a browser UI (drag-and-drop upload or mic recording), or
call the API directly with `curl` (see below). Deployed on Google Cloud Run,
auto-deployed by GitHub Actions on every push to `master` (see
[CI/CD](#cicd) below).

**Auth is intentionally disabled on this deployment.** The service supports
an `X-API-Key` gate plus Cloud Run IAM (both described under
[Security by design](#security-by-design) below), but both are turned off
here specifically so anyone can test the demo without a credential. That's a
disclosed trade-off, not an oversight — the compensating controls are a
tightened per-IP rate limit (6 requests/minute) and a GCP billing budget
alert, not "no protection at all." In a real deployment both auth layers
would be re-enabled; the code path for both already exists and is exercised
by the test suite.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI, or:

```bash
curl -X POST http://127.0.0.1:8000/v1/transcribe -F "file=@sample.wav"
```

A short file returns synchronously:

```json
{
  "status": "completed",
  "mode": "sync",
  "language": "en",
  "language_probability": 0.98,
  "duration": 12.4,
  "full_transcript": "hello, this is a test recording",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "text": "hello, this is a test recording"}
  ]
}
```

A file longer than `SYNC_DURATION_THRESHOLD_SECONDS` (default 120s) returns
a job to poll instead:

```json
{"status": "accepted", "mode": "async", "job_id": "…", "status_url": "/v1/jobs/…"}
```

```bash
curl http://127.0.0.1:8000/v1/jobs/<job_id>
```

### Docker

```bash
docker build -t transcription-api .
docker run -p 8000:8000 -e MODEL_SIZE=base transcription-api
```

The container reads its listen port from the `$PORT` environment variable
(defaults to 8000 locally) rather than a fixed value — required for Cloud
Run, which injects `PORT=8080` at runtime. This was a real bug on first
deploy: the Dockerfile originally hardcoded `--port 8000`, and Cloud Run
couldn't reach the container until it was fixed to read `$PORT` dynamically.

### Frontend / demo UI

`GET /` serves a self-contained HTML/JS page (`app/static/index.html`) that
calls the same `/v1/transcribe` and `/v1/jobs/{id}` endpoints a script or
`curl` would — it's a client of the API, not a separate code path. It
supports drag-and-drop file upload and microphone recording (via
`MediaRecorder`, which outputs `.webm` — added to the allowed-format list
for that reason). Note: this is *record-then-transcribe*, not live streaming
transcription — faster-whisper is a batch model, not a streaming one; true
word-by-word live captioning would need a fundamentally different
architecture (WebSocket + sliding-window ASR) and was out of scope.

### CI/CD

`.github/workflows/deploy.yml` runs the pytest suite on every push and pull
request; on push to `master`, if tests pass, it deploys straight to Cloud
Run via `gcloud run deploy --source .`. Authentication to GCP uses Workload
Identity Federation scoped to this exact repo — no long-lived service
account key is stored in GitHub secrets.

### Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

Uses the `tiny` model (set via `MODEL_SIZE=tiny` in the test file) and a
synthetically generated tone fixture — no external audio or network fixture
needed. See "Known limitations" below for what this does and doesn't prove.

## Architecture

```
Upload → extension allow-list → size-capped streaming read to disk
       → magic-byte check (content must match claimed extension)
       → ffmpeg normalize → 16kHz mono WAV (timeout-bounded)
       → duration probe
           ├─ short (<= threshold) → transcribe in threadpool → 200 response
           └─ long  (>  threshold) → background job:
                  silence-aware chunking → transcribe each chunk
                  → offset + merge timestamps → poll via /v1/jobs/{id}
```

Every stage that touches untrusted input (extension, byte content, ffmpeg
decode, file size) validates and rejects before the next stage runs — no
step assumes the previous one already made the data safe.

## Design questions

### How do you handle different audio formats?

The model isn't handed raw uploads directly. An **audio normalization
step** (`app/audio.py::convert_to_wav`) runs every accepted file through
`ffmpeg` first, converting `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wav`, or
`.webm` (browser mic recordings) into a single canonical format: 16kHz mono
PCM WAV — the input faster-whisper's
feature extractor expects. This means the transcription code only ever
handles one format, decoder edge cases are isolated to one subprocess call,
and accuracy is consistent across input sources.

Two things sit in front of that conversion, because "the client says it's a
`.wav`" is not something to trust on its own:

- **Extension allow-list** (`validate_extension`) — cheap, rejects obvious
  junk before any I/O.
- **Magic-byte check** (`validate_signature`) — confirms the file's actual
  header (`RIFF` for WAV, `fLaC`, `OggS`, MP3 frame sync bytes, `ftyp` for
  M4A) matches the claimed extension, so a renamed file can't reach `ffmpeg`
  under a false format label.

`ffmpeg` is invoked as an argument list (`subprocess.run([...])`), never
through a shell string, so nothing in the filename or file content can be
interpreted as a shell command — and the call has a hard timeout
(`FFMPEG_TIMEOUT_SECONDS`) so a malformed or adversarial file that makes
`ffmpeg` hang can't tie up a worker indefinitely.

### How do you deal with long audio files?

Doing everything in one request/response cycle breaks down for long
recordings — gateway timeouts, and a blocking transcription call that stalls
every other request on that worker. This service routes on duration:

1. **Short files** (`<= SYNC_DURATION_THRESHOLD_SECONDS`, default 120s)
   transcribe inline and return directly. Even here, the actual
   `model.transcribe()` call is dispatched via `run_in_threadpool` rather
   than awaited in the coroutine — faster-whisper is synchronous CPU work,
   and calling it directly inside `async def` would block the event loop for
   every concurrent request, not just the current one.

2. **Long files** get a `job_id` immediately (202-style response) and are
   processed in a background thread:
   - `probe_duration_seconds` gets the total length.
   - `compute_chunk_boundaries` splits the file into
     `CHUNK_TARGET_SECONDS` (default 5 min) pieces, but snaps each cut to
     the nearest silence detected by `ffmpeg`'s `silencedetect` filter
     within a search window — so a chunk boundary doesn't land mid-word.
   - Each chunk is transcribed independently, and its segment timestamps
     are offset by the chunk's start time before merging, so the client
     gets one continuous timeline regardless of how many chunks were used
     internally.
   - The client polls `GET /v1/jobs/{job_id}` for status/result.

**What's a demo simplification vs. what's production-real:** the job store
here is an in-memory dict (`app/jobs.py`) and chunks are processed
sequentially in a background thread — enough to prove the architecture
without extra infrastructure. In production this becomes Celery/RQ +
Redis (or SQS + a worker fleet), which buys three things this version
doesn't have: jobs surviving a process restart, jobs visible across
multiple API replicas, and true parallel chunk processing across
worker processes/machines (each with its own model instance — a single
`WhisperModel` shouldn't be pounded concurrently from many processes
against one CPU/GPU). Every call site that touches the job store
(`create_job`, `set_result`, `get_job`) maps 1:1 onto that swap — the
API and background-task code don't change, only what's backing them.

## Security by design

Answering the interview question directly, in order of where each control
sits in the pipeline rather than as a generic checklist:

1. **Input boundary** — extension allow-list, magic-byte content
   verification, a hard upload size cap enforced *while streaming to disk*
   (not after buffering the whole file in memory), and a timeout on every
   `ffmpeg` subprocess call. All subprocess calls use argument lists, never
   `shell=True`.
2. **AuthN/cost control** — an `X-API-Key` header gate (`require_api_key`,
   a stand-in for real OAuth2/JWT in production) plus Cloud Run IAM, and
   per-IP rate limiting (`slowapi`, `RATE_LIMIT` env var) on the
   transcription endpoint — this is the endpoint that costs CPU/GPU time
   per call, so it's the one that needs abuse protection, not just auth.
   **Both auth layers are currently switched off on the public demo URL**
   (see [Live demo](#live-demo) above) as a deliberate, disclosed choice
   to make testing frictionless — the rate limit and a billing alert are
   the compensating controls while it's in that state.
3. **Storage/privacy** — uploads are written to per-request temp
   directories and deleted as soon as they're no longer needed (raw
   upload right after conversion; the normalized WAV right after
   transcription, or by the background job on completion/failure). No
   audio or transcript is written to persistent storage in this demo;
   in production, voice recordings are frequently treated as
   biometric/PII data under GDPR/CCPA-style rules, so persisted
   transcripts should be encrypted at rest, access-controlled (RBAC),
   and covered by an explicit retention policy — not kept indefinitely
   by default.
4. **Model/supply chain** — faster-whisper loads CTranslate2-converted
   weights (not arbitrary pickle files), pulled from a named, versioned
   model repo rather than an unpinned "latest". `requirements.txt` pins
   exact versions.
5. **Failure handling** — a no-speech/VAD-empty input (silence, a pure
   tone, noise) is a real input a public endpoint will receive, and
   faster-whisper raises on it internally; the service catches that
   specific case and returns an empty transcript rather than a 500 (see
   `app/transcriber.py`). A raw stack trace on unexpected input is itself
   a minor information-disclosure surface, so failure modes need to be
   enumerated deliberately rather than left to whatever bubbles up.

The point across all five: none of this was bolted on after the fact —
the upload-size cap, magic-byte check, and threadpool dispatch were part
of the first version of `main.py`, not a patch applied once something
broke. That's the actual answer to "how do you secure it from the start
instead of on the fly": the input boundary and failure modes get
designed alongside the happy path, not after it.

## Known limitations / what I'd do with more time

- **Speech-accuracy testing exists, but isn't wired into CI.** The
  `pytest` suite proves the pipeline runs end-to-end and fails gracefully
  on non-speech input, but doesn't assert transcription *correctness*
  against a reference transcript. Separately, there's a manual validation
  set (not checked into git — audio binaries don't belong in the repo) of
  12 TTS-narrated recordings — 6 English, 6 Urdu, 30s to ~13min, across
  all 5 supported formats — each paired with its exact source text, plus
  a script (`Dataset/run_live_test.py`) that feeds every file to the live
  API and computes word error rate against the known ground truth. That
  proves real-world accuracy manually; the gap is that it isn't yet a
  CI-gated regression test with a pass/fail WER threshold.
- **In-memory job store**, as noted above — fine for a single-process
  demo, not for multi-replica production.
- **PII redaction is not implemented** — a real deployment handling
  voice data would want a redaction pass (names, phone numbers, card
  numbers) on the transcript before persisting it, or a policy decision
  to not persist raw transcripts at all.
- **No malware/AV scan on uploads** — acceptable for a take-home; in
  production, uploaded files that get shelled out to `ffmpeg` are worth
  scanning or running `ffmpeg` in a more tightly sandboxed
  (seccomp/gVisor) environment given `ffmpeg`'s CVE history.
