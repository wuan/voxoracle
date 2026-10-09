"""Wake-word detection ("Franz") using openWakeWord.

The detector sits behind the :class:`~voxoracle.wakeword.detector.WakeWordDetector`
protocol so it stays swappable and testable without audio hardware.
"""

from __future__ import annotations

from voxoracle.wakeword.detector import (
    BLOCK_SAMPLES,
    PLACEHOLDER_MODEL,
    OpenWakeWordDetector,
    OpenWakeWordScorer,
    WakeWordDetector,
    WakeWordScorer,
    resolve_model_path,
)
from voxoracle.wakeword.errors import WakeWordError, WakeWordModelNotFoundError

__all__ = [
    "BLOCK_SAMPLES",
    "PLACEHOLDER_MODEL",
    "OpenWakeWordDetector",
    "OpenWakeWordScorer",
    "WakeWordDetector",
    "WakeWordError",
    "WakeWordModelNotFoundError",
    "WakeWordScorer",
    "resolve_model_path",
]
