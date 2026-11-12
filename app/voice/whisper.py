from functools import lru_cache
import tempfile
from pathlib import Path


@lru_cache(maxsize=1)
def get_whisper_model():
    from faster_whisper import WhisperModel
    from app.config import settings
    return WhisperModel(settings.whisper_model, compute_type="int8", device="cpu")


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    model = get_whisper_model()
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, _ = model.transcribe(tmp.name)
        return " ".join(seg.text for seg in segments).strip()
