"""Protocols and value types for the audio device layer.

Everything the session depends on is expressed here as a protocol so tests can
inject fakes and run without audio hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

DeviceKind = Literal["input", "output"]


@dataclass(frozen=True)
class DeviceInfo:
    """A selectable audio device as reported by the backend."""

    index: int
    name: str
    kind: DeviceKind
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: float = 0.0
    is_default: bool = False


@runtime_checkable
class AudioInput(Protocol):
    """A source of fixed-size mono frames of 16-bit PCM."""

    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def read_frame(self) -> np.ndarray:
        """Return the next frame as an ``int16`` numpy array of ``frame_samples``."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class AudioOutput(Protocol):
    """A sink that plays mono ``int16`` samples at its own device rate."""

    @property
    def sample_rate(self) -> int: ...

    def write(self, samples: np.ndarray) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Classifies a single frame of 16-bit PCM as speech or non-speech."""

    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


@runtime_checkable
class AudioBackend(Protocol):
    """A backend that enumerates devices and opens streams."""

    def list_devices(self, kind: DeviceKind) -> list[DeviceInfo]: ...

    def open_input(
        self, device: int | None, sample_rate: int, frame_samples: int
    ) -> AudioInput: ...

    def open_output(self, device: int | None, sample_rate: int) -> AudioOutput: ...
