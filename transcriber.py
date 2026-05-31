"""Whisper-based audio transcription with timestamps."""

import functools
import os

import stable_whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_USE_FASTER = os.getenv("WHISPER_USE_FASTER", "1") != "0"


@functools.lru_cache(maxsize=4)
def get_whisper_model(model_name: str = WHISPER_MODEL, use_faster: bool = WHISPER_USE_FASTER):
    if use_faster:
        return stable_whisper.load_faster_whisper(model_name)
    return stable_whisper.load_model(model_name)


def transcribe_with_timestamps(
    audio_path: str,
    model_name: str = WHISPER_MODEL,
    use_faster: bool = WHISPER_USE_FASTER,
) -> list[dict]:
    """Transcribe audio and return segments with timestamps.

    Returns:
        List of {"start": float, "end": float, "text": str} dicts.
    """
    model = get_whisper_model(model_name, use_faster)
    result = model.transcribe(audio_path)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in result.segments
    ]
