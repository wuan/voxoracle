"""Mistral speech-synthesis backend (Voxtral TTS).

Uses Mistral's speech endpoint ``POST {base_url}/audio/speech`` with model
``voxtral-mini-tts-2603``. The request carries the text (``input``), a ``voice_id``,
a ``response_format`` and a ``stream`` flag. Mistral selects the voice by
``voice_id`` and has no separate language field, so the configured language
provides the default voice when no explicit voice is set (German by default).

Two response shapes are supported:

* ``stream=False``: a JSON body ``{"audio_data": "<base64>"}`` (one blob).
* ``stream=True``: a ``text/event-stream`` whose ``data:`` frames carry
  ``{"type": "speech.audio.delta", "audio_data": "<base64>"}`` deltas terminated
  by a ``speech.audio.done`` frame (per Mistral's OpenAPI schema).

Decoding turns the provider's wire format into mono 16-bit PCM
(:class:`~voxoracle.tts.protocol.SpeechChunk`), which the WP2 ``AudioOutput``
layer plays. ``pcm`` is raw little-endian float32 samples (Mistral's recommended
format for streaming) whose sample rate is not published by the provider, so it
is configurable (``sample_rate``, default 24 kHz); ``wav`` is a PCM WAV
container whose header carries the authoritative frame rate.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import wave
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from voxoracle.mistral.client import MistralAudioClient
from voxoracle.mistral.errors import MistralResponseError
from voxoracle.tts.protocol import SpeechChunk

SPEECH_PATH = "/audio/speech"
DEFAULT_MODEL = "voxtral-mini-tts-2603"
DEFAULT_LANGUAGE = "de"
#: The configured language doubles as the default voice id (Mistral selects the
#: voice by ``voice_id`` and has no separate language field).
DEFAULT_VOICE = "de"
DEFAULT_SAMPLE_RATE = 24000

#: Response formats this backend can decode into PCM. Mistral also offers mp3,
#: flac and opus, which need codecs VoxOracle does not carry.
DECODABLE_FORMATS = ("pcm", "wav")

#: SSE event terminator. An event block ends at a blank line; multi-line ``data:``
#: fields within one event are NOT joined. Mistral sends one JSON payload per
#: ``data:`` line, which is all this parser needs.
SSE_EVENT_SEPARATOR = b"\n\n"


def pcm_float32_to_int16(data: bytes) -> NDArray[np.int16]:
    """Decode raw little-endian float32 samples (Mistral ``pcm``) to int16."""
    if len(data) % 4 != 0:
        raise MistralResponseError(f"pcm chunk length {len(data)} is not a multiple of 4")
    floats = np.frombuffer(data, dtype="<f4")
    clipped = np.clip(floats, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def wav_to_int16(data: bytes) -> tuple[NDArray[np.int16], int]:
    """Decode a 16-bit mono PCM WAV byte string to int16 samples and its rate.

    The WAV header carries the authoritative sample rate, which is returned so
    the caller plays the audio at the right speed instead of trusting a
    configured guess.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise MistralResponseError(
                    f"expected a 16-bit WAV, got {handle.getsampwidth() * 8}-bit"
                )
            channels = handle.getnchannels()
            frame_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise MistralResponseError(f"invalid WAV audio: {exc}") from exc
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    return np.ascontiguousarray(samples), frame_rate


def _decode_base64(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MistralResponseError(f"invalid base64 audio: {exc}") from exc


class MistralSpeechSynthesizer:
    """Synthesizer backed by Mistral's Voxtral speech-synthesis service."""

    def __init__(
        self,
        client: MistralAudioClient,
        *,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        voice: str | None = None,
        response_format: str = "pcm",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        stream: bool = True,
    ) -> None:
        if response_format not in DECODABLE_FORMATS:
            raise ValueError(
                f"response_format must be one of {DECODABLE_FORMATS}, got {response_format!r}"
            )
        if stream and response_format != "pcm":
            raise ValueError("streaming is only supported with response_format='pcm'")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self._client = client
        self._model = model
        self._language = language
        # Mistral selects the voice by ``voice_id``; the configured language picks
        # the default voice when no explicit voice is given, so a German language
        # setting yields a German voice.
        self._voice = voice or language
        self._response_format = response_format
        self._sample_rate = sample_rate
        self._stream = stream

    @property
    def model(self) -> str:
        return self._model

    @property
    def language(self) -> str:
        return self._language

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _payload(self, text: str, voice: str | None) -> dict[str, Any]:
        return {
            "model": self._model,
            "input": text,
            "voice_id": voice or self._voice,
            "response_format": self._response_format,
            "stream": self._stream,
        }

    async def synthesize(
        self, text: str, *, voice: str | None = None
    ) -> AsyncGenerator[SpeechChunk]:
        self._validate_text(text)
        payload = self._payload(text, voice)
        if self._stream:
            async for chunk in self._stream_chunks(payload):
                yield chunk
        else:
            yield await self._single_chunk(payload)

    def _validate_text(self, text: str) -> None:
        if not text.strip():
            raise ValueError("text must not be empty")

    async def _single_chunk(self, payload: Mapping[str, Any]) -> SpeechChunk:
        body = await self._client.post_json(SPEECH_PATH, json=payload)
        return self._chunk(_decode_base64(self._audio_data(body)))

    async def _stream_chunks(self, payload: Mapping[str, Any]) -> AsyncGenerator[SpeechChunk]:
        buffer = b""
        async for frame in self._client.stream_post_json(SPEECH_PATH, json=payload):
            # Normalise CRLF so an event boundary is always the blank line "\n\n".
            buffer = (buffer + frame).replace(b"\r\n", b"\n")
            # An SSE event is terminated by a blank line; emit only complete ones.
            while SSE_EVENT_SEPARATOR in buffer:
                event, buffer = buffer.split(SSE_EVENT_SEPARATOR, 1)
                for chunk in self._chunks_from_event(event):
                    yield chunk
        # Flush a trailing event if the stream did not end with a blank line.
        if buffer.strip():
            for chunk in self._chunks_from_event(buffer):
                yield chunk

    def _chunks_from_event(self, event: bytes) -> list[SpeechChunk]:
        return [self._chunk(_decode_base64(data)) for data in self._event_audio_data(event)]

    @staticmethod
    def _audio_data(body: Any) -> str:
        if not isinstance(body, dict):
            raise MistralResponseError(f"expected a JSON object, got {body!r}")
        data = cast("dict[str, Any]", body).get("audio_data")
        if not isinstance(data, str) or not data:
            raise MistralResponseError(f"expected an 'audio_data' string, got {body!r}")
        return data

    def _event_audio_data(self, event: bytes) -> list[str]:
        """Return the base64 audio payloads of one SSE event block.

        A ``speech.audio.done`` event carries no audio and is skipped (detected
        either from the ``event:`` name or the JSON ``type`` field, since Mistral
        may express it either way). Each ``data:`` line is parsed as a complete
        JSON payload; multi-line ``data:`` fields are not joined, which matches
        Mistral's one-JSON-per-event stream.
        """
        payloads: list[str] = []
        event_name: str | None = None
        for raw_line in event.decode("utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                decoded = json.loads(payload)
            except ValueError as exc:
                raise MistralResponseError(f"invalid SSE JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise MistralResponseError(f"expected an SSE JSON object, got {decoded!r}")
            data = cast("dict[str, Any]", decoded)
            if event_name == "speech.audio.done" or data.get("type") == "speech.audio.done":
                continue
            audio = data.get("audio_data")
            if not isinstance(audio, str):
                raise MistralResponseError(f"expected an SSE 'audio_data' string, got {data!r}")
            payloads.append(audio)
        return payloads

    def _chunk(self, raw: bytes) -> SpeechChunk:
        if self._response_format == "wav":
            # The WAV header's frame rate is authoritative for this chunk.
            samples, frame_rate = wav_to_int16(raw)
            return SpeechChunk(samples=samples, sample_rate=frame_rate)
        return SpeechChunk(samples=pcm_float32_to_int16(raw), sample_rate=self._sample_rate)
