"""Mistral TTS backend tests using a fake transport (no key, no network, no audio)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import wave
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

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
from voxoracle.tts.player import PLAY_SLICE_SECONDS
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

    async def scenario() -> bool:
        try:
            return isinstance(MistralSpeechSynthesizer(client), Synthesizer)
        finally:
            await client.aclose()

    assert run(scenario())


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


def test_pcm_non_finite_samples_become_silence() -> None:
    decoded = pcm_float32_to_int16(pcm_bytes([float("nan"), float("inf"), float("-inf")]))
    assert list(decoded) == [0, 32767, -32767]


def test_pcm_length_must_be_multiple_of_four() -> None:
    with pytest.raises(MistralResponseError):
        pcm_float32_to_int16(b"\x00\x00\x00")


def test_wav_to_int16_reads_mono_pcm_and_rate() -> None:
    samples = np.array([1, 2, 3, 4], dtype=np.int16)
    decoded, frame_rate = wav_to_int16(wav_bytes(samples, sample_rate=22050))
    assert list(decoded) == [1, 2, 3, 4]
    assert frame_rate == 22050


def test_wav_to_int16_downmixes_stereo() -> None:
    frames = np.array([[10, 20], [30, 40]], dtype="<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    decoded, frame_rate = wav_to_int16(buffer.getvalue())
    assert list(decoded) == [10, 30]
    assert frame_rate == 16000


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


def test_stream_flushes_trailing_event_without_blank_line() -> None:
    delta = base64.b64encode(pcm_bytes([0.0])).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        # A final delta with no terminating blank line before EOF.
        body = b"data: " + json.dumps({"audio_data": delta}).encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = synthesize(handler)
    assert len(chunks) == 1
    assert len(chunks[0].samples) == 1


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


def test_language_selects_default_voice() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse({"type": "speech.audio.done", "usage": {}}))

    # With no explicit voice, the configured language provides the voice id.
    synthesize(handler, language="en")
    assert seen["voice_id"] == "en"


def test_explicit_voice_overrides_language() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse({"type": "speech.audio.done", "usage": {}}))

    synthesize(handler, language="de", voice="anna")
    assert seen["voice_id"] == "anna"


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
        encoded = base64.b64encode(wav_bytes(samples, sample_rate=16000)).decode()
        return httpx.Response(200, json={"audio_data": encoded})

    chunks = synthesize(handler, stream=False, response_format="wav")
    assert len(chunks) == 1
    assert list(chunks[0].samples) == [5, 6, 7]
    # The WAV header rate (16 kHz) is authoritative, not the configured 24 kHz.
    assert chunks[0].sample_rate == 16000


def test_wav_chunk_rate_low_rate_wav_not_the_configured_default() -> None:
    # A low-rate WAV header must win over the configured default (24 kHz), else
    # SpeechPlayer would resample from a wrong label and play it too fast.
    samples = np.arange(800, dtype=np.int16)

    def handler(request: httpx.Request) -> httpx.Response:
        encoded = base64.b64encode(wav_bytes(samples, sample_rate=8000)).decode()
        return httpx.Response(200, json={"audio_data": encoded})

    chunks = synthesize(handler, stream=False, response_format="wav")
    assert chunks[0].sample_rate == 8000
    assert chunks[0].sample_rate != DEFAULT_SAMPLE_RATE


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


def test_unknown_sse_event_type_is_skipped() -> None:
    # A new event kind must not break the stream (forward compatibility), and
    # its audio_data (if any) must be ignored.
    delta = base64.b64encode(pcm_bytes([0.0])).decode()
    bogus = base64.b64encode(pcm_bytes([1.0])).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                {"type": "speech.something.new", "detail": "ignored"},
                {"type": "speech.something.new", "audio_data": bogus},
                {"type": "speech.audio.delta", "audio_data": delta},
                {"type": "speech.audio.done", "usage": {}},
            ),
            headers={"content-type": "text/event-stream"},
        )

    chunks = synthesize(handler)
    assert len(chunks) == 1
    assert list(chunks[0].samples) == [0]  # only the real delta played


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


class _FailingStream(httpx.AsyncByteStream):
    """Yields one frame, then raises the given transport error mid-stream."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"data: {"  # incomplete frame: buffered, produces no event yet
        raise self._error


def test_mid_stream_timeout_is_a_typed_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=_FailingStream(httpx.ReadTimeout("late timeout", request=request)),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(MistralTimeoutError):
        synthesize(handler, retries=2)
    assert calls == 1  # a mid-stream failure is not retried


def test_mid_stream_transport_error_is_a_typed_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=_FailingStream(httpx.ReadError("connection lost", request=request)),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(MistralConnectionError):
        synthesize(handler, retries=2)
    assert calls == 1


def test_mid_stream_decoding_error_is_a_typed_error() -> None:
    # httpx.DecodingError is a RequestError but not a TransportError; it must
    # still surface as a typed Mistral error, not a raw httpx exception.
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=_FailingStream(httpx.DecodingError("corrupt body")),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(MistralResponseError):
        synthesize(handler, retries=2)
    assert calls == 1


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
        self.started = 0
        self.stopped = False
        self.aborted = False
        self.closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        self.started += 1

    def write(self, samples: np.ndarray) -> None:
        self.written.append(samples)

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

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

    # resample is a no-op at equal rates, so the samples are unchanged.
    assert np.array_equal(np.concatenate(output.written), source)


class BlockingOutput(FakeOutput):
    """A write that blocks like PortAudio until released, signalling progress."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        super().__init__(sample_rate)
        self.write_started = threading.Event()
        self.release = threading.Event()

    def write(self, samples: np.ndarray) -> None:
        self.write_started.set()
        # Block the (worker) thread until the test releases it, emulating a
        # device write that lasts for the chunk's playback duration.
        assert self.release.wait(timeout=5.0)
        self.written.append(samples)


def test_blocking_write_does_not_stall_the_event_loop() -> None:
    # The device write blocks for the chunk's playback duration; it must run off
    # the event loop so a barge-in detector (WP6) can run meanwhile.
    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16)])
    output = BlockingOutput()
    player = SpeechPlayer(synth, output)
    loop_ran_during_write = False

    async def scenario() -> None:
        nonlocal loop_ran_during_write
        speak_task = asyncio.create_task(player.speak("Hallo"))
        # Wait until the (off-loop) write has started, then prove the event loop
        # is still able to run this coroutine while the write blocks.
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        loop_ran_during_write = True
        output.release.set()
        await speak_task

    run(scenario())

    assert loop_ran_during_write
    assert len(output.written) == 1


def test_barge_in_aborts_output_while_write_is_still_blocked() -> None:
    # interrupt() must abort the output immediately, without waiting for the
    # in-flight blocking write to return. The write stays blocked here on
    # purpose: playback must stop within a bound regardless of the release.
    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16) for _ in range(2)])
    output = BlockingOutput()
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        # Barge-in lands while the write is still blocked.
        player.interrupt()
        assert output.aborted  # output cancelled synchronously, write not returned
        output.release.set()
        # speak() must return promptly once the write unblocks.
        await asyncio.wait_for(speak_task, timeout=1.0)

    run(scenario())

    assert player.interrupted
    assert output.aborted
    # The first chunk was played; barge-in landed during its blocking write, so
    # the second chunk is never fetched or written.
    assert synth.yielded == 1
    assert len(output.written) == 1


class SliceOutput(FakeOutput):
    """Records slice lengths; used to prove long chunks play in bounded slices."""

    def write(self, samples: np.ndarray) -> None:
        self.written.append(samples.copy())


def test_long_chunk_is_written_in_bounded_slices() -> None:
    # A whole WAV answer arrives as one long chunk; barge-in must stop within a
    # bounded fraction of a second, so the chunk must be sliced, not written whole.
    long_chunk = np.zeros(SAMPLE_RATE * 3, dtype=np.int16)  # 3 s at 24 kHz
    synth = FakeSynthesizer([long_chunk], sample_rate=SAMPLE_RATE)
    output = SliceOutput(sample_rate=SAMPLE_RATE)
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        # Let the first couple of slices play, then barge in.
        while len(output.written) < 2:
            await asyncio.sleep(0)
        player.interrupt()
        await asyncio.wait_for(speak_task, timeout=1.0)

    run(scenario())

    expected_slice = round(SAMPLE_RATE * PLAY_SLICE_SECONDS)
    assert all(len(part) <= expected_slice for part in output.written)
    # Stopped well before the whole 3 s chunk was written.
    assert sum(len(part) for part in output.written) < long_chunk.size
    assert output.aborted


class AbortedWriteOutput(FakeOutput):
    """A write that blocks like Pa_WriteStream and fails once the stream aborted."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        super().__init__(sample_rate)
        self.write_started = threading.Event()

    def write(self, samples: np.ndarray) -> None:
        self.write_started.set()
        # Block until abort() is called, then fail as Pa_WriteStream does when
        # the stream is aborted mid-write.
        for _ in range(500):
            if self.aborted:
                raise RuntimeError("paOutputUnderflowed")
            time.sleep(0.001)
        self.written.append(samples)

    def abort(self) -> None:
        # Also fail if aborted twice, like a stopped stream on some hosts.
        if self.aborted:
            raise RuntimeError("abort on an already-stopped stream")
        self.aborted = True


def test_barge_in_swallows_write_failure_caused_by_our_abort() -> None:
    # Real Pa_WriteStream fails when we abort the stream mid-write; that failure
    # is our own doing and must not escape speak().
    long_chunk = np.zeros(SAMPLE_RATE * 3, dtype=np.int16)
    synth = FakeSynthesizer([long_chunk], sample_rate=SAMPLE_RATE)
    output = AbortedWriteOutput(sample_rate=SAMPLE_RATE)
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        player.interrupt()  # aborts the output; the in-flight write then fails
        await asyncio.wait_for(speak_task, timeout=2.0)

    run(scenario())  # must not raise

    assert player.interrupted
    assert output.aborted


def test_write_failure_without_barge_in_propagates() -> None:
    class BoomOutput(FakeOutput):
        def write(self, samples: np.ndarray) -> None:
            raise RuntimeError("device broken")

    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16)])
    player = SpeechPlayer(synth, BoomOutput())

    with pytest.raises(RuntimeError, match="device broken"):
        run(player.speak("Hallo"))


class StallingSynthesizer:
    """Yields one chunk, then blocks (as a quiet provider would) until closed."""

    def __init__(self) -> None:
        self.closed = False
        self.stalled = threading.Event()

    async def _gen(self) -> AsyncIterator[SpeechChunk]:
        try:
            yield SpeechChunk(samples=np.zeros(4, dtype=np.int16), sample_rate=SAMPLE_RATE)
            # Provider goes quiet: wait forever unless the stream is cancelled.
            self.stalled.set()
            await asyncio.Event().wait()
        finally:
            self.closed = True

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        return self._gen()


def test_barge_in_cancels_a_stalled_stream_promptly() -> None:
    # A slow/quiet provider must not defer barge-in until the read timeout: the
    # interrupt event must cancel the pending chunk fetch and close the stream.
    synth = StallingSynthesizer()
    output = FakeOutput()
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        # First chunk plays; the stream then stalls waiting for the next delta.
        while not synth.stalled.is_set():
            await asyncio.sleep(0)
        player.interrupt()
        # speak() must return without a read timeout elapsing.
        await asyncio.wait_for(speak_task, timeout=1.0)

    run(scenario())

    assert player.interrupted
    assert synth.closed  # the in-flight provider stream was cancelled
    assert output.aborted
    assert len(output.written) == 1


class RaisingSynthesizer:
    """Yields one chunk, then raises, to exercise speak()'s cleanup path."""

    async def _gen(self) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(samples=np.zeros(2, dtype=np.int16), sample_rate=SAMPLE_RATE)
        raise MistralResponseError("boom")

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[SpeechChunk]:
        return self._gen()


def test_speak_clears_stop_event_after_failure() -> None:
    # After a mid-stream failure, interrupt() must not touch a stale stop event.
    synth = RaisingSynthesizer()
    player = SpeechPlayer(synth, FakeOutput())

    with pytest.raises(MistralResponseError):
        run(player.speak("Hallo"))

    assert player._stop is None
    player.interrupt()  # must not raise on the cleared state


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
    assert output.aborted
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


def test_next_utterance_restarts_output_after_barge_in_abort() -> None:
    # Pa_AbortStream stops the stream; the next speak() must restart it so the
    # following answer actually plays (no hardware needed to verify the call).
    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16) for _ in range(2)])
    output = BlockingOutput()
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        first = asyncio.create_task(player.speak("Erste Frage"))
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        player.interrupt()  # abort mid-write
        output.release.set()
        await asyncio.wait_for(first, timeout=1.0)

        # The second utterance must restart the aborted output and then play.
        before = output.started
        await player.speak("Zweite Frage")
        assert output.started == before + 1

    run(scenario())

    assert output.aborted
    assert output.started == 1


def test_speak_does_not_restart_output_without_prior_abort() -> None:
    synth = FakeSynthesizer([np.zeros(4, dtype=np.int16)])
    output = FakeOutput()
    player = SpeechPlayer(synth, output)

    run(player.speak("Hallo"))

    assert output.started == 0  # nothing to resume


def test_speak_is_not_reentrant() -> None:
    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16) for _ in range(2)])
    output = BlockingOutput()
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        first = asyncio.create_task(player.speak("Erste Frage"))
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="re-entrant"):
            await player.speak("Zweite Frage")
        player.interrupt()
        output.release.set()
        await asyncio.wait_for(first, timeout=1.0)

    run(scenario())


class DoubleAbortOutput(FakeOutput):
    """An abort() that raises if called twice, as a stopped stream might."""

    def abort(self) -> None:
        if self.aborted:
            raise RuntimeError("abort on an already-stopped stream")
        self.aborted = True


def test_repeat_interrupt_does_not_reabort_the_output() -> None:
    # The wake word can fire again while speak() winds down; a second interrupt
    # must not abort the already-aborted output (which may raise on real hosts).
    synth = FakeSynthesizer([np.zeros(10, dtype=np.int16) for _ in range(2)])
    output = DoubleAbortOutput()
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        while not output.written:
            await asyncio.sleep(0)
        player.interrupt()
        player.interrupt()  # second barge-in while still winding down
        await asyncio.wait_for(speak_task, timeout=1.0)

    run(scenario())  # must not raise from the second abort

    assert output.aborted


def test_play_does_not_reabort_after_interrupt() -> None:
    # interrupt() aborts the output; _play's slice loop must only stop, never
    # abort a second time. Here the in-flight write succeeds (a host where a
    # stopped stream still accepts a write), so the loop reaches the next slice
    # boundary with the output already aborted.
    class BlockingThenOkOutput(DoubleAbortOutput):
        def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
            super().__init__(sample_rate)
            self.write_started = threading.Event()
            self.release = threading.Event()

        def write(self, samples: np.ndarray) -> None:
            self.write_started.set()
            assert self.release.wait(timeout=5.0)
            self.written.append(samples)

    long_chunk = np.zeros(SAMPLE_RATE, dtype=np.int16)  # several 0.1 s slices
    synth = FakeSynthesizer([long_chunk], sample_rate=SAMPLE_RATE)
    output = BlockingThenOkOutput(sample_rate=SAMPLE_RATE)
    player = SpeechPlayer(synth, output)

    async def scenario() -> None:
        speak_task = asyncio.create_task(player.speak("Ein langer Satz"))
        while not output.write_started.is_set():
            await asyncio.sleep(0)
        player.interrupt()  # aborts; the slice write succeeds, so _play loops
        output.release.set()
        await asyncio.wait_for(speak_task, timeout=2.0)

    run(scenario())  # must not raise from a second abort

    assert output.aborted
    assert len(output.written) == 1  # stopped after the first slice


# --- config resolution --------------------------------------------------------


def test_tts_defaults_are_mistral_german(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.tts.provider == "mistral"
    assert settings.tts.model == "voxtral-mini-tts-2603"
    assert settings.tts.language == "de"
    # voice is unset by default; the synthesizer derives the German voice from
    # the language, so a config with voice: null remains valid.
    assert settings.tts.voice is None
    assert settings.tts.sample_rate == 24000


def test_tts_voice_null_is_accepted(tmp_path, monkeypatch) -> None:
    # An existing config.yaml copied from the shipped example has voice: null.
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  voice: null\n", encoding="utf-8")
    assert load_settings(config).tts.voice is None


def test_tts_voice_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  voice: anna\n  model: custom-tts\n", encoding="utf-8")
    settings = load_settings(config)
    assert settings.tts.voice == "anna"
    assert settings.tts.model == "custom-tts"


def test_tts_sample_rate_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  sample_rate: 16000\n", encoding="utf-8")
    assert load_settings(config).tts.sample_rate == 16000


def test_tts_sample_rate_must_be_positive(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  sample_rate: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config)
