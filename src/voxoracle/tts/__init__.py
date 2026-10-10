"""Cloud text-to-speech backends and their provider protocol."""

from __future__ import annotations

from voxoracle.tts.mistral import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    MistralSpeechSynthesizer,
    pcm_float32_to_int16,
    wav_to_int16,
)
from voxoracle.tts.player import SpeechPlayer
from voxoracle.tts.protocol import SpeechChunk, Synthesizer

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_VOICE",
    "MistralSpeechSynthesizer",
    "SpeechChunk",
    "SpeechPlayer",
    "Synthesizer",
    "pcm_float32_to_int16",
    "wav_to_int16",
]
