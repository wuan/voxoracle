"""Cloud text-to-speech backends and their provider protocol."""

from __future__ import annotations

from voxoracle.tts.mistral import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    MistralSpeechSynthesizer,
    pcm_float32_to_int16,
    wav_to_int16,
)
from voxoracle.tts.player import SpeechPlayer
from voxoracle.tts.protocol import SpeechChunk, Synthesizer

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLE_RATE",
    "MistralSpeechSynthesizer",
    "SpeechChunk",
    "SpeechPlayer",
    "Synthesizer",
    "pcm_float32_to_int16",
    "wav_to_int16",
]
