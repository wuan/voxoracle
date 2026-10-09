"""Typed errors for the audio device layer."""

from __future__ import annotations


class AudioError(Exception):
    """Base class for audio device layer errors."""


class DeviceNotFoundError(AudioError):
    """A configured audio device is not available."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        super().__init__(f"{kind} device not found: {name!r}")


class AudioFormatError(AudioError):
    """Audio data or a configured format is not supported."""
