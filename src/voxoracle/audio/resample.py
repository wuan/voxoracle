"""Sample-rate conversion for playback.

A small linear resampler keeps the dependency surface at numpy and is adequate
for speech playback. It is not a high-quality resampler; swap the implementation
behind :func:`resample` if playback quality ever matters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from voxoracle.audio.errors import AudioFormatError


def resample(samples: NDArray[np.int16], source_rate: int, target_rate: int) -> NDArray[np.int16]:
    """Resample mono ``int16`` samples from ``source_rate`` to ``target_rate``.

    Returns ``samples`` unchanged when the rates match.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise AudioFormatError(f"sample rates must be positive, got {source_rate} -> {target_rate}")
    if source_rate == target_rate or samples.size == 0:
        return samples
    target_length = round(samples.size * target_rate / source_rate)
    if target_length <= 0:
        return np.empty(0, dtype=np.int16)
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(target_length, dtype=np.float64) * (source_rate / target_rate)
    resampled = np.interp(target_positions, source_positions, samples.astype(np.float64))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)
