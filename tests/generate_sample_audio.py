"""Generates a small synthetic WAV file with no external dependencies (stdlib
`wave` + `array` only) so the pipeline can be smoke-tested without needing a
real speech recording checked into the repo."""
import array
import math
import wave


def make_tone_wav(path: str, seconds: float = 3.0, freq: float = 440.0, sample_rate: int = 16000) -> None:
    n_samples = int(seconds * sample_rate)
    amplitude = 12000
    samples = array.array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * freq * t / sample_rate)) for t in range(n_samples)),
    )
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


if __name__ == "__main__":
    make_tone_wav("tests/fixtures/tone.wav")
    print("wrote tests/fixtures/tone.wav")
