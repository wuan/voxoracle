"""Wake-word detection ("Hey Franz") using livekit-wakeword.

The detector sits behind the :class:`~voxoracle.wakeword.detector.WakeWordDetector`
protocol so it stays swappable and testable without audio hardware.
"""

from __future__ import annotations

from voxoracle.wakeword.detector import (
    DEFAULT_STRIDE_FRAMES,
    WINDOW_SAMPLES,
    LiveKitWakeWordScorer,
    WakeWordDetector,
    WakeWordScorer,
    WindowedWakeWordDetector,
    resolve_model_path,
)
from voxoracle.wakeword.errors import WakeWordError, WakeWordModelNotFoundError

__all__ = [
    "DEFAULT_STRIDE_FRAMES",
    "WINDOW_SAMPLES",
    "LiveKitWakeWordScorer",
    "WakeWordDetector",
    "WakeWordError",
    "WakeWordModelNotFoundError",
    "WakeWordScorer",
    "WindowedWakeWordDetector",
    "resolve_model_path",
]
