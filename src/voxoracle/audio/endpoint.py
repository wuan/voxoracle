"""Voice activity detection and utterance endpointing.

The endpointing logic works on any :class:`~voxoracle.audio.protocols.AudioInput`
and :class:`~voxoracle.audio.protocols.VoiceActivityDetector`, so it is exercised
in tests with synthetic frames and a fake detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import webrtcvad  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from voxoracle.audio.errors import AudioFormatError
from voxoracle.audio.protocols import AudioInput, VoiceActivityDetector

# webrtcvad accepts mono 16-bit frames of exactly these durations at these rates.
WEBRTC_VAD_RATES = (8000, 16000, 32000, 48000)
WEBRTC_VAD_FRAME_MS = (10, 20, 30)


def _validate_vad_frame(frame: bytes, sample_rate: int) -> int:
    """Return the frame duration in ms, or raise :class:`AudioFormatError`."""
    if sample_rate not in WEBRTC_VAD_RATES:
        raise AudioFormatError(
            f"WebRTC VAD requires a sample rate of "
            f"{', '.join(str(r) for r in WEBRTC_VAD_RATES)} Hz, got {sample_rate}"
        )
    samples = len(frame) // 2  # mono 16-bit PCM
    frame_ms = round(samples * 1000 / sample_rate)
    if frame_ms not in WEBRTC_VAD_FRAME_MS:
        raise AudioFormatError(
            f"WebRTC VAD requires 10, 20 or 30 ms frames, got ~{frame_ms} ms "
            f"({samples} samples at {sample_rate} Hz)"
        )
    return frame_ms


class WebRtcVad(VoiceActivityDetector):
    """Voice activity detector backed by ``webrtcvad``.

    ``webrtcvad`` requires 16-bit mono PCM frames of exactly 10, 20 or 30 ms at
    8/16/32/48 kHz; the audio layer's default 30 ms / 16 kHz satisfies this.
    Unsupported inputs raise :class:`~voxoracle.audio.errors.AudioFormatError`
    instead of the C extension's opaque error.
    """

    def __init__(self, aggressiveness: int = 2) -> None:
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0..3, got {aggressiveness}")
        self._vad: Any = webrtcvad.Vad(aggressiveness)  # pyright: ignore[reportAny]

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        _validate_vad_frame(frame, sample_rate)
        return bool(self._vad.is_speech(frame, sample_rate))


@dataclass(frozen=True)
class EndpointingSettings:
    """Tuning for :func:`record_utterance`."""

    sample_rate: int
    frame_samples: int
    max_seconds: float
    endpoint_silence_ms: int = 700
    min_speech_ms: int = 200


def record_utterance(
    audio_input: AudioInput,
    vad: VoiceActivityDetector,
    settings: EndpointingSettings,
    should_stop: Callable[[], bool] | None = None,
) -> NDArray[np.int16] | None:
    """Record one utterance, trimming leading/trailing silence.

    Reading stops after ``endpoint_silence_ms`` of continuous silence following
    detected speech, or at ``max_seconds``, whichever comes first. ``should_stop``
    is polled before each frame so a caller (the voice session) can abort the
    recording cooperatively, e.g. on Ctrl-C, without waiting out
    ``max_seconds``; when it returns ``True`` the recording ends. Returns the
    captured ``int16`` samples, or ``None`` when no speech was detected (or the
    recording was aborted).
    """
    frame_ms = round(settings.frame_samples * 1000 / settings.sample_rate)
    endpoint_frames = max(1, round(settings.endpoint_silence_ms / frame_ms))
    min_speech_frames = max(1, round(settings.min_speech_ms / frame_ms))
    max_frames = max(1, round(settings.max_seconds * 1000 / frame_ms))

    captured: list[NDArray[np.int16]] = []
    speech_frames = 0
    trailing_silence = 0
    reached_endpoint = False
    aborted = False

    for _ in range(max_frames):
        if should_stop is not None and should_stop():
            aborted = True
            break
        frame = audio_input.read_frame()
        is_speech = vad.is_speech(frame.tobytes(), settings.sample_rate)
        if not captured and not is_speech:
            continue  # trim leading silence
        captured.append(frame)
        if is_speech:
            speech_frames += 1
            trailing_silence = 0
        elif speech_frames >= min_speech_frames:
            trailing_silence += 1
            if trailing_silence >= endpoint_frames:
                reached_endpoint = True
                break

    if aborted:
        return None
    if not captured or speech_frames < min_speech_frames:
        return None
    if reached_endpoint and trailing_silence:
        captured = captured[:-trailing_silence] or captured
    return np.concatenate(captured)
