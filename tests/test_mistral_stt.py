"""Mistral STT backend tests using a fake transport (no key, no network, no audio)."""

from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest

from voxoracle.mistral import (
    MistralAudioClient,
    MistralAuthError,
    MistralConnectionError,
    MistralRateLimitError,
    MistralResponseError,
    MistralServerError,
    MistralStatusError,
    MistralTimeoutError,
)
from voxoracle.stt import MistralTranscriber
from voxoracle.stt.protocol import AudioClip

SAMPLE_RATE = 16000


def clip() -> AudioClip:
    return AudioClip(samples=np.zeros(SAMPLE_RATE, dtype=np.int16), sample_rate=SAMPLE_RATE)


def run(coro):
    return asyncio.run(coro)


def transcribe(handler, *, retries=0, sample_rate=SAMPLE_RATE, language=None):
    async def scenario():
        client = MistralAudioClient(
            "test-key",
            retries=retries,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        transcriber = MistralTranscriber(client)
        try:
            return await transcriber.transcribe(clip(), language=language)
        finally:
            await client.aclose()

    return run(scenario())


def test_success_returns_transcript() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"text": "Wie funktioniert das?"})

    result = transcribe(handler)
    assert result.text == "Wie funktioniert das?"
    assert result.language == "de"


def test_default_language_is_german() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        assert b'name="language"' in body
        assert b"\r\n\r\nde\r\n" in body
        assert b"voxtral-mini-latest" in body
        assert b'name="file"' in body and b"RIFF" in body  # WAV upload
        return httpx.Response(200, json={"text": "Hallo"})

    assert transcribe(handler).language == "de"


def test_language_override() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="language"' in request.content
        assert b"en" in request.content
        return httpx.Response(200, json={"text": "Hello"})

    assert transcribe(handler, language="en").language == "en"


def test_empty_transcript_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": ""})

    assert transcribe(handler).text == ""


def test_null_text_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": None})

    with pytest.raises(MistralResponseError):
        transcribe(handler)


def test_missing_text_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"})

    with pytest.raises(MistralResponseError):
        transcribe(handler)


def test_non_json_body_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(MistralResponseError):
        transcribe(handler)


def test_auth_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(MistralAuthError):
        transcribe(handler, retries=3)
    assert calls == 1


def test_rate_limit_is_retried_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(MistralRateLimitError):
        transcribe(handler, retries=2)
    assert calls == 3


def test_server_error_is_retried_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(MistralServerError):
        transcribe(handler, retries=2)
    assert calls == 3


def test_bad_request_fails_immediately() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(MistralStatusError):
        transcribe(handler, retries=3)
    assert calls == 1


def test_timeout_raises_after_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(MistralTimeoutError):
        transcribe(handler, retries=2)
    assert calls == 3


def test_connection_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(MistralConnectionError):
        transcribe(handler, retries=1)


def test_retry_then_success() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"text": "ok"})

    assert transcribe(handler, retries=2).text == "ok"
    assert calls == 3
