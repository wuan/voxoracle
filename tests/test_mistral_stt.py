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
from voxoracle.mistral import client as mistral_client
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
        body = request.content
        assert b'name="language"' in body
        # language part value is exactly "en" (not merely containing "en")
        assert b'name="language"\r\n\r\nen\r\n' in body
        return httpx.Response(200, json={"text": "Hello"})

    assert transcribe(handler, language="en").language == "en"


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError):
        MistralAudioClient("test-key", retries=-1)


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


def recording_sleep(monkeypatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(mistral_client, "asyncio", _FakeAsyncio(fake_sleep))
    return delays


class _FakeAsyncio:
    """Stand-in exposing only ``sleep`` for the client's retry loop."""

    def __init__(self, sleep) -> None:
        self.sleep = sleep


def failing_client(monkeypatch, handler, *, retries, jitter=0.0, **kwargs) -> list[float]:
    delays = recording_sleep(monkeypatch)

    async def scenario() -> None:
        client = MistralAudioClient(
            "test-key",
            retries=retries,
            jitter=jitter,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )
        try:
            with pytest.raises(Exception):  # noqa: B017 - any typed failure
                await client.post_json("/x", json={})
        finally:
            await client.aclose()

    asyncio.run(scenario())
    return delays


def test_exponential_backoff_schedule(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    delays = failing_client(
        monkeypatch, handler, retries=3, retry_delay=0.5, backoff_factor=2.0, max_retry_delay=10.0
    )
    # attempt 0 -> 0.5, attempt 1 -> 1.0, attempt 2 -> 2.0 (jitter disabled)
    assert delays == [0.5, 1.0, 2.0]


def test_backoff_is_capped(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    delays = failing_client(
        monkeypatch, handler, retries=3, retry_delay=1.0, backoff_factor=10.0, max_retry_delay=5.0
    )
    assert delays == [1.0, 5.0, 5.0]


def test_retry_after_header_is_honored_on_429(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "7"})

    delays = failing_client(monkeypatch, handler, retries=1, retry_delay=0.5, max_retry_delay=30.0)
    assert delays == [7.0]  # server hint wins over the smaller backoff


def test_jitter_never_undercuts_retry_after() -> None:
    # With jitter on (default), the Retry-After delay is a hard lower bound.
    client = MistralAudioClient("test-key", retry_delay=0.5, backoff_factor=2.0, jitter=0.5)
    for _ in range(200):
        assert client._delay_for(0, retry_after=7.0) >= 7.0


def test_jitter_still_applies_without_retry_after() -> None:
    client = MistralAudioClient("test-key", retry_delay=10.0, backoff_factor=1.0, jitter=0.5)
    delays = {client._delay_for(0, retry_after=None) for _ in range(50)}
    assert len(delays) > 1  # jitter varies the delay
    assert all(5.0 <= d <= 10.0 for d in delays)  # within [1-jitter, 1] * base


def test_http_408_is_retried(monkeypatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(408, text="timeout")

    delays = failing_client(monkeypatch, handler, retries=2, retry_delay=0.1)
    assert calls == 3
    assert len(delays) == 2


def test_clip_rejects_non_mono_and_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        AudioClip(samples=np.zeros((2, 100), dtype=np.int16), sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError):
        AudioClip(samples=np.zeros(100, dtype=np.float32), sample_rate=SAMPLE_RATE)
