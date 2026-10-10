"""Provider-agnostic speech-to-text protocol and value types.

The session loop depends only on :class:`Transcriber`; concrete cloud backends
implement it. Audio is passed as an :class:`AudioClip` (mono 16-bit PCM) so the
protocol does not leak any provider's upload format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioClip:
    """A mono 16-bit PCM audio clip."""

    samples: NDArray[np.int16]
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be mono (1-D), got shape {self.samples.shape}")
        if self.samples.dtype != np.int16:
            raise ValueError(f"samples must be int16, got dtype {self.samples.dtype}")


@dataclass(frozen=True)
class Transcript:
    """The result of transcribing a clip."""

    text: str
    language: str | None = None


@runtime_checkable
class Transcriber(Protocol):
    """Turns captured audio into text."""

    async def transcribe(self, clip: AudioClip, language: str | None = None) -> Transcript:
        """Transcribe ``clip``; ``language`` overrides the configured default."""
        ...
