"""Audio device layer: capture, playback, resampling and voice activity detection.

The session-facing pieces are the protocols in :mod:`voxoracle.audio.protocols`
plus :func:`~voxoracle.audio.endpoint.record_utterance`; the PortAudio-backed
implementations live in :mod:`voxoracle.audio.sounddevice_backend` and are only
imported on demand.
"""

from __future__ import annotations

from voxoracle.audio.endpoint import EndpointingSettings, WebRtcVad, record_utterance
from voxoracle.audio.errors import AudioError, AudioFormatError, DeviceNotFoundError
from voxoracle.audio.protocols import (
    AudioBackend,
    AudioInput,
    AudioOutput,
    DeviceInfo,
    DeviceKind,
    VoiceActivityDetector,
)
from voxoracle.audio.resample import resample

__all__ = [
    "AudioBackend",
    "AudioError",
    "AudioFormatError",
    "AudioInput",
    "AudioOutput",
    "DeviceInfo",
    "DeviceKind",
    "DeviceNotFoundError",
    "EndpointingSettings",
    "VoiceActivityDetector",
    "WebRtcVad",
    "record_utterance",
    "resample",
]
