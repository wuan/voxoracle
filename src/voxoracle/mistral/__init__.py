"""Shared client for the Mistral cloud audio APIs (STT in WP4, TTS in WP5)."""

from __future__ import annotations

from voxoracle.mistral.client import DEFAULT_BASE_URL, MistralAudioClient
from voxoracle.mistral.errors import (
    MistralAuthError,
    MistralConnectionError,
    MistralError,
    MistralRateLimitError,
    MistralResponseError,
    MistralServerError,
    MistralStatusError,
    MistralTimeoutError,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "MistralAudioClient",
    "MistralAuthError",
    "MistralConnectionError",
    "MistralError",
    "MistralRateLimitError",
    "MistralResponseError",
    "MistralServerError",
    "MistralStatusError",
    "MistralTimeoutError",
]
