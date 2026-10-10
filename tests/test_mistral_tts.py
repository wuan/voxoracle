"""Mistral TTS backend tests using a fake transport (no key, no network, no audio)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest

from voxoracle.audio.protocols import AudioOutput
from voxoracle.config import load_settings
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
from voxoracle.tts import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    MistralSpeechSynthesizer,
    SpeechChunk,
    SpeechPlayer,
    wav_to_int16,
)
from voxoracle.tts.mistral import pcm_float32_to_int16
from voxoracle.tts.protocol import Synthesizer

SAMPLE_RATE = DEFAULT_SAMPLE_RATE


def run(coro):
    return asyncio.run(coro)


def pcm_bytes(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def sse(*frames: dict) -> bytes:
    return b"".join(b"event: x\ndata: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)


def synthesize(handler, *, retries=0, **kwargs) -> list[SpeechChunk]:
    async def scenario():
        client = MistralAudioClient(
            "test-key",
            retries=retries,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        synth = MistralSpeechSynthesizer(client, **kwargs)
        try:
            return [chunk async for chunk in synth.synthesize("Wie funktioniert das?")]
        finally:
            await client.aclose()

    return run(scenario())


# --- protocol / value types ---------------------------------------------------


def test_synthesizer_protocol_is_structural() -> None:
    client = MistralAudioClient("k")
    assert isinstance(MistralSpeechSynthesizer(client), Synthesizer)


def test_speech_chunk_validates() -> None:
    with pytest.raises(ValueError):
        SpeechChunk(samples=np.zeros((2, 10), dtype=np.int16), sample_rate=16000)
    with pytest.raises(ValueError):
        SpeechChunk(samples=np.zeros(10, dtype=np.float32), sample_rate=16000)
    with pytest.raises(ValueError):
        SpeechChunk(samples=np.zeros(10, dtype=np.int16), sample_rate=0)


# --- decoding -----------------------------------------------------------------


def test_pcm_float32_to_int16_roundtrip() -> None:
    raw = pcm_bytes([0.0, 1.0, -1.0, 0.5])
    decoded = pcm_float32_to_int16(raw)
    assert decoded.dtype == np.int16
    assert list(decoded) == [0, 32767, -32767, 16384]


def test_pcm_clips_out_of_range() -> None:
    decoded = pcm_float32_to_int16(pcm_bytes([2.0, -3.0]))
    assert list(decoded) == [32767, -32767]


def test_pcm_length_must_be_multiple_of_four() -> None:
    with pytest.raises(MistralResponseError):
        pcm_float32_to_int16(b"\x00\x00\x00")


def test_wav_to_int16_reads_mono_pcm() -> None:
    samples = np.array([1, 2, 3, 4], dtype=np.int16)
    assert list(wav_to_int16(wav_bytes(samples))) == [1, 2, 3, 4]


def test_wav_to_int16_downmixes_stereo() -> None:
    frames = np.array([[10, 20], [30, 40]], dtype="<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    assert list(wav_to_int16(buffer.getvalue())) == [10, 30]


def test_non_wav_bytes_are_malformed() -> None:
    with pytest.raises(MistralResponseError):
        wav_to_int16(b"not-a-wav")


# --- streaming success --------------------------------------------------------


def test_stream_success_returns_audio_chunks() -> None:
    chunk_a = pcm_bytes([0.0, 0.5])
    chunk_b = pcm_bytes([-0.5, 1.0])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["response_format"] == "pcm"
        return httpx.Response(
            200,
            content=sse(
                {"type": "speech.audio.delta", "audio_data": base64.b64encode(chunk_a).decode()},
                {"type": "speech.audio.delta", "audio_data": base64.b64encode(chunk_b).decode()},
                {"type": "speech.audio.done", "usage": {}},
            ),
            headers={"content-type": "text/event-stream"},
        )

    chunks = synthesize(handler)
    assert [len(chunk.samples) for chunk in chunks] == [2, 2]
    assert all(chunk.sample_rate == SAMPLE_RATE for chunk in chunks)


def test_stream_chunks_split_across_frames_are_reassembled() -> None:
    payload = sse(
        {"type": "speech.audio.delta", "audio_data": base64.b64encode(pcm_bytes([0.1])).decode()}
    )
    split = len(payload) // 2

    class SplitStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield payload[:split]
            yield payload[split:]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=SplitStream(), headers={"content-type": "text/event-stream"}
        )

    chunks = synthesize(handler)
    assert len(chunks) == 1
    assert len(chunks[0].samples) == 1


def test_stream_handles_event_name_done_frame() -> None:
    delta = base64.b64encode(pcm_bytes([0.0])).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"event: speech.audio.delta\ndata: "
            + json.dumps({"audio_data": delta}).encode()
            + b"\n\nevent: speech.audio.done\ndata: {}\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = synthesize(handler)
    assert len(chunks) == 1


def test_stream_handles_crlf_event_boundaries() -> None:
    frame = {
        "type": "speech.audio.delta",
        "audio_data": base64.b64encode(pcm_bytes([0.0])).decode(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = b"data: " + json.dumps(frame).encode() + b"\r\n\r\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = synthesize(handler)
    assert len(chunks) == 1


def test_request_shape_uses_german_defaults() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse({"type": "speech.audio.done", "usage": {}}))

    synthesize(handler)
    assert seen["model"] == DEFAULT_MODEL
    assert seen["voice_id"] == "de"
    assert seen["input"] == "Wie funktioniert das?"


def test_voice_and_model_overrides() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse({"type": "speech.audio.done", "usage": {}}))

    synthesize(handler, model="custom-model", voice="custom-voice")
    assert seen["model"] == "custom-model"
    assert seen["voice_id"] == "custom-voice"


def test_per_call_voice_override() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse({"type": "speech.audio.done", "usage": {}}))

    async def scenario():
        client = MistralAudioClient("test-key", retries=0, transport=httpx.MockTransport(handler))
        synth = MistralSpeechSynthesizer(client)
        try:
            return [chunk async for chunk in synth.synthesize("Hallo", voice="anna")]
        finally:
            await client.aclose()

    run(scenario())
    assert seen["voice_id"] == "anna"


# --- non-streaming success ----------------------------------------------------


def test_non_streaming_wav_returns_single_chunk() -> None:
    samples = np.array([5, 6, 7], dtype=np.int16)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["response_format"] == "wav"
        encoded = base64.b64encode(wav_bytes(samples)).decode()
        return httpx.Response(200, json={"audio_data": encoded})

    chunks = synthesize(handler, stream=False, response_format="wav")
    assert len(chunks) == 1
    assert list(chunks[0].samples) == [5, 6, 7]


# --- malformed responses ------------------------------------------------------


def test_missing_audio_data_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"})

    with pytest.raises(MistralResponseError):
        synthesize(handler, stream=False)


def test_invalid_base64_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio_data": "not base64!!"})

    with pytest.raises(MistralResponseError):
        synthesize(handler, stream=False)


def test_invalid_sse_json_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: not-json\n\n",
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(MistralResponseError):
        synthesize(handler)


def test_sse_delta_without_audio_data_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse({"type": "speech.audio.delta"}),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(MistralResponseError):
        synthesize(handler)


# --- error paths --------------------------------------------------------------


def test_auth_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(MistralAuthError):
        synthesize(handler, retries=3)
    assert calls == 1


def test_rate_limit_is_retried_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(MistralRateLimitError) as excinfo:
        synthesize(handler, retries=2)
    assert calls == 3
    assert excinfo.value.status_code == 429


def test_server_error_is_retried_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(MistralServerError):
        synthesize(handler, retries=2)
    assert calls == 3


def test_bad_request_fails_immediately() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(MistralStatusError):
        synthesize(handler, retries=3)
    assert calls == 1


def test_content_moderation_403_maps_to_auth_error() -> None:
    # Mistral returns 403 when content moderation rejects the text.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="moderation")

    with pytest.raises(MistralAuthError):
        synthesize(handler, retries=2)


def test_stream_timeout_raises_after_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(MistralTimeoutError):
        synthesize(handler, retries=2)
    assert calls == 3


def test_stream_connection_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(MistralConnectionError):
        synthesize(handler, retries=1)


def test_stream_retry_then_success() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200,
            content=sse(
                {
                    "type": "speech.audio.delta",
                    "audio_data": base64.b64encode(pcm_bytes([0.0])).decode(),
                }
            ),
            headers={"content-type": "text/event-stream"},
        )

    chunks = synthesize(handler, retries=2)
    assert calls == 2
    assert len(chunks) == 1


# --- constructor validation ---------------------------------------------------


def test_undecodable_response_format_rejected() -> None:
    client = MistralAudioClient("k")
    with pytest.raises(ValueError):
        MistralSpeechSynthesizer(client, response_format="mp3")


def test_streaming_requires_pcm() -> None:
    client = MistralAudioClient("k")
    with pytest.raises(ValueError):
        MistralSpeechSynthesizer(client, response_format="wav", stream=True)


def test_empty_text_rejected() -> None:
    client = MistralAudioClient("k")
    synth = MistralSpeechSynthesizer(client)

    async def scenario():
        return [chunk async for chunk in synth.synthesize("   ")]

    with pytest.raises(ValueError):
        run(scenario())


# --- playback + barge-in ------------------------------------------------------


class FakeOutput(AudioOutput):
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self.written: list[np.ndarray] = []
        self.stopped = False
        self.closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def write(self, samples: np.ndarray) -> None:
        self.written.append(samples)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSynthesizer:
    """Yields controllable chunks and records how far the consumer got."""

    def __init__(self, chunks: list[np.ndarray], *, sample_rate: int = SAMPLE_RATE) -> None:
        self._chunks = chunks
        self._sample_rate = sample_rate
        self.yielded = 0
        self.closed = False
        self.on_yield = None

    async def _gen(self) -> AsyncIterator[SpeechChunk]:
        try:
            for samples in self._chunks:
                self.yielded += 1
                if self.on_yield is not None:
                    self.on_yield(self.yielded)
                yield SpeechChunk(samples=samples, sample_rate=self._sample_rate)
        finally:
            self.closed = True

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        return self._gen()


def test_playback_writes_each_chunk() -> None:
    synth = FakeSynthesizer([np.zeros(3, dtype=np.int16), np.ones(2, dtype=np.int16)])
    output = FakeOutput()
    player = SpeechPlayer(synth, output)

    run(player.speak("Hallo"))

    assert len(output.written) == 2
    assert synth.yielded == 2
    assert not output.stopped
    assert not player.interrupted


def test_playback_resamples_mismatched_chunk_rate() -> None:
    # A 24 kHz provider stream played through a 16 kHz output must be resampled,
    # not written straight through (which would play ~1.5x too fast).
    source = np.arange(2400, dtype=np.int16)  # 0.1 s at 24 kHz
    synth = FakeSynthesizer([source], sample_rate=24000)
    output = FakeOutput(sample_rate=16000)
    player = SpeechPlayer(synth, output)

    run(player.speak("Hallo"))

    assert len(output.written) == 1
    written = output.written[0]
    assert written.dtype == np.int16
    assert written.size == 1600  # 0.1 s at the output's 16 kHz
    assert not player.interrupted


def test_playback_with_matching_rate_is_unchanged() -> None:
    source = np.arange(500, dtype=np.int16)
    synth = FakeSynthesizer([source], sample_rate=SAMPLE_RATE)
    output = FakeOutput(sample_rate=SAMPLE_RATE)
    player = SpeechPlayer(synth, output)

    run(player.speak("Hallo"))

    assert output.written[0] is source  # resample is a no-op at equal rates


def test_barge_in_cancels_mid_stream() -> None:
    synth = FakeSynthesizer([np.zeros(3, dtype=np.int16) for _ in range(5)])
    output = FakeOutput()
    player = SpeechPlayer(synth, output)

    def interrupt_after_second(_index: int) -> None:
        if synth.yielded == 2:
            player.interrupt()

    synth.on_yield = interrupt_after_second

    run(player.speak("Ein langer Satz"))

    assert synth.yielded == 2  # stopped consuming after the interrupt
    assert synth.closed  # provider stream closed / request cancelled
    assert output.stopped
    assert player.interrupted
    # The second chunk was abandoned (not written) as soon as barge-in was set.
    assert len(output.written) == 1


def test_interrupted_flag_resets_between_utterances() -> None:
    synth = FakeSynthesizer([np.zeros(1, dtype=np.int16)])
    output = FakeOutput()
    player = SpeechPlayer(synth, output)
    player.interrupt()

    run(player.speak("Neue Frage"))

    assert not player.interrupted
    assert len(output.written) == 1


# --- config resolution --------------------------------------------------------


def test_tts_defaults_are_mistral_german(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.tts.provider == "mistral"
    assert settings.tts.model == "voxtral-mini-tts-2603"
    assert settings.tts.language == "de"
    assert settings.tts.voice == "de"


def test_tts_voice_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  voice: anna\n  model: custom-tts\n", encoding="utf-8")
    settings = load_settings(config)
    assert settings.tts.voice == "anna"
    assert settings.tts.model == "custom-tts"
