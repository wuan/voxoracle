"""Audio device layer tests with synthetic buffers and fake devices (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

from voxoracle.audio import (
    AudioFormatError,
    DeviceInfo,
    DeviceNotFoundError,
    EndpointingSettings,
    WebRtcVad,
    record_utterance,
    resample,
)
from voxoracle.audio.sounddevice_backend import resolve_device

FRAME_SAMPLES = 480  # 30 ms at 16 kHz
SAMPLE_RATE = 16000


def tone_frame(amplitude: int = 8000) -> np.ndarray:
    return (np.sin(np.linspace(0, 2 * np.pi * 8, FRAME_SAMPLES)) * amplitude).astype(np.int16)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


class FakeInput:
    """AudioInput fake that replays a list of frames, then silence."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self._index = 0

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return FRAME_SAMPLES

    def read_frame(self) -> np.ndarray:
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return frame
        return silence_frame()

    def close(self) -> None:
        return None


class FakeVad:
    """VAD fake driven by a predicate over the frame contents."""

    def __init__(self, speech_frames: set[int] | None = None) -> None:
        self._speech_frames = speech_frames
        self._calls = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        index = self._calls
        self._calls += 1
        if self._speech_frames is not None:
            return index in self._speech_frames
        return any(frame)  # non-zero audio counts as speech


class TestResample:
    def test_identity_when_rates_match(self) -> None:
        samples = tone_frame()
        result = resample(samples, SAMPLE_RATE, SAMPLE_RATE)
        assert np.array_equal(result, samples)

    def test_upsamples_to_target_length(self) -> None:
        samples = tone_frame()
        result = resample(samples, SAMPLE_RATE, 48000)
        assert result.dtype == np.int16
        assert result.size == round(samples.size * 48000 / SAMPLE_RATE)

    def test_downsamples_to_target_length(self) -> None:
        samples = tone_frame()
        result = resample(samples, 48000, 16000)
        assert result.size == round(samples.size * 16000 / 48000)

    def test_empty_input(self) -> None:
        assert resample(np.empty(0, dtype=np.int16), SAMPLE_RATE, 48000).size == 0

    def test_invalid_rate_is_rejected(self) -> None:
        with pytest.raises(AudioFormatError):
            resample(tone_frame(), 0, 16000)


class TestEndpointing:
    def settings(self, max_seconds: float = 5.0) -> EndpointingSettings:
        return EndpointingSettings(
            sample_rate=SAMPLE_RATE,
            frame_samples=FRAME_SAMPLES,
            max_seconds=max_seconds,
            endpoint_silence_ms=90,  # 3 frames
            min_speech_ms=60,  # 2 frames
        )

    def test_leading_and_trailing_silence_trimmed(self) -> None:
        frames = [silence_frame(), silence_frame(), tone_frame(), tone_frame(), tone_frame()]
        frames += [silence_frame() for _ in range(4)]
        result = record_utterance(FakeInput(frames), FakeVad(), self.settings())
        assert result is not None
        # three speech frames survive; leading silence dropped and trailing
        # silence removed by the endpoint.
        assert result.size == 3 * FRAME_SAMPLES

    def test_no_speech_returns_none(self) -> None:
        frames = [silence_frame() for _ in range(10)]
        assert record_utterance(FakeInput(frames), FakeVad(), self.settings()) is None

    def test_short_speech_is_rejected(self) -> None:
        frames = [tone_frame()] + [silence_frame() for _ in range(10)]
        # min_speech is 2 frames, only one speech frame seen
        assert record_utterance(FakeInput(frames), FakeVad(), self.settings()) is None

    def test_max_duration_caps_capture(self) -> None:
        frames = [tone_frame() for _ in range(100)]
        result = record_utterance(FakeInput(frames), FakeVad(), self.settings(max_seconds=0.3))
        assert result is not None
        # 0.3 s at 30 ms/frame == 10 frames maximum
        assert result.size == 10 * FRAME_SAMPLES


class TestWebRtcVad:
    def test_silence_is_not_speech(self) -> None:
        vad = WebRtcVad(aggressiveness=2)
        assert vad.is_speech(silence_frame().tobytes(), SAMPLE_RATE) is False

    def test_invalid_aggressiveness_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebRtcVad(aggressiveness=5)


class TestDeviceSelection:
    def devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(index=1, name="USB Microphone", kind="input", max_input_channels=1),
            DeviceInfo(index=2, name="Built-in Mic", kind="input", max_input_channels=1),
        ]

    def test_default_device_resolves_to_none(self) -> None:
        assert resolve_device("default", "input", self.devices()) is None
        assert resolve_device(None, "input", self.devices()) is None

    def test_configured_device_resolves_by_name(self) -> None:
        assert resolve_device("USB Microphone", "input", self.devices()) == 1

    def test_name_match_is_case_insensitive(self) -> None:
        assert resolve_device("built-in mic", "input", self.devices()) == 2

    def test_missing_device_raises_typed_error(self) -> None:
        with pytest.raises(DeviceNotFoundError) as excinfo:
            resolve_device("Nonexistent", "input", self.devices())
        assert excinfo.value.name == "Nonexistent"
        assert excinfo.value.kind == "input"
