"""Typed errors for the wake-word layer."""

from __future__ import annotations


class WakeWordError(Exception):
    """Base class for wake-word layer errors."""


class WakeWordModelNotFoundError(WakeWordError):
    """The configured wake-word model could not be found."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"wake-word model not found: {name!r}")
