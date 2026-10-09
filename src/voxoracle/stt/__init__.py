"""Cloud speech-to-text backends and their provider protocol."""

from __future__ import annotations

from voxoracle.stt.mistral import DEFAULT_LANGUAGE, DEFAULT_MODEL, MistralTranscriber, clip_to_wav
from voxoracle.stt.protocol import AudioClip, Transcriber, Transcript

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "AudioClip",
    "MistralTranscriber",
    "Transcript",
    "Transcriber",
    "clip_to_wav",
]
