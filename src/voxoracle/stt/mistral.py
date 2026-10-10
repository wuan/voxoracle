"""Mistral speech-to-text backend (Voxtral transcription).

Uses Mistral's offline transcription endpoint
``POST {base_url}/audio/transcriptions`` with model ``voxtral-mini-latest``
(Voxtral Mini Transcribe 2). The clip is uploaded as a 16-bit mono WAV via a
multipart form; the response is the OpenAI-compatible ``{"text": "..."}``.
"""

from __future__ import annotations

import io
import wave
from typing import Any, cast

from voxoracle.mistral.client import MistralAudioClient
from voxoracle.mistral.errors import MistralResponseError
from voxoracle.stt.protocol import AudioClip, Transcript

TRANSCRIPTIONS_PATH = "/audio/transcriptions"
DEFAULT_MODEL = "voxtral-mini-latest"
DEFAULT_LANGUAGE = "de"


def clip_to_wav(clip: AudioClip) -> bytes:
    """Encode a clip as a 16-bit mono PCM WAV byte string."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(clip.sample_rate)
        handle.writeframes(clip.samples.astype("<i2").tobytes())
    return buffer.getvalue()


class MistralTranscriber:
    """Transcriber backed by Mistral's Voxtral transcription service."""

    def __init__(
        self,
        client: MistralAudioClient,
        *,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._client = client
        self._model = model
        self._language = language

    @property
    def model(self) -> str:
        return self._model

    @property
    def language(self) -> str:
        return self._language

    async def transcribe(self, clip: AudioClip, language: str | None = None) -> Transcript:
        effective_language = language or self._language
        payload = await self._client.post_multipart(
            TRANSCRIPTIONS_PATH,
            files={"file": ("audio.wav", clip_to_wav(clip), "audio/wav")},
            data={"model": self._model, "language": effective_language},
        )
        body = cast("dict[str, Any]", payload) if isinstance(payload, dict) else None
        text = body.get("text") if body is not None else None
        if not isinstance(text, str):
            raise MistralResponseError(f"expected a 'text' string, got {payload!r}")
        return Transcript(text=text, language=effective_language)
