"""Provider-agnostic text-to-speech protocol and value types.

The session loop and the playback layer depend only on :class:`Synthesizer`; a
concrete cloud backend implements it. Audio leaves the protocol as
:class:`SpeechChunk` values (mono 16-bit PCM), the same shape the WP2
``AudioOutput`` device layer consumes, so the provider's wire format (base64,
float32 PCM, WAV, ...) never leaks past the backend.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpeechChunk:
    """A piece of synthesized mono 16-bit PCM audio."""

    samples: NDArray[np.int16]
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be mono (1-D), got shape {self.samples.shape}")
        if self.samples.dtype != np.int16:
            raise ValueError(f"samples must be int16, got dtype {self.samples.dtype}")


@runtime_checkable
class Synthesizer(Protocol):
    """Turns answer text into an asynchronous stream of playable audio chunks."""

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncGenerator[SpeechChunk]:
        """Yield audio for ``text``; ``voice`` overrides the configured default."""
        ...
