"""Wake-word detection ("Hey Franz") using livekit-wakeword.

The detector is a streaming adapter: it accepts the fixed-size 16 kHz mono
frames produced by the WP2 audio capture, maintains a rolling ~2 s window, and
scores it against the configured threshold once per configured stride of
frames. livekit-wakeword's ``predict`` is stateless - the caller owns the audio
window - so all streaming state lives here, not in the model.

Model acquisition
-----------------
A purpose-trained "Hey Franz" model (``hey-franz.onnx``, trained with
livekit-wakeword's German pipeline, see ``training/hey-franz.yaml``) is
REQUIRED: a missing model raises :class:`WakeWordModelNotFoundError` and
``doctor``/session startup fail fast. The model path is configurable
(:func:`resolve_model_path`) so a retrained model can be dropped into
``models/`` without code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from voxoracle.wakeword.errors import WakeWordError, WakeWordModelNotFoundError

# livekit-wakeword scores a stateless ~2 s window: 32,000 samples at 16 kHz
# yield the 16 x 80 ms embeddings the classifier expects.
WINDOW_SAMPLES = 32000

# Score the window every N capture frames (30 ms each). Chosen by the Pi 3
# hardware spike (design D2): stateless inference recomputes the whole window
# per call, so the stride trades detection latency for CPU.
DEFAULT_STRIDE_FRAMES = 4


class WakeWordScorer(Protocol):
    """Scores the rolling audio window, returning a confidence per model name."""

    def predict(self, window: NDArray[np.int16]) -> dict[str, float]: ...

    def reset(self) -> None:
        """Clear any streaming state held across :meth:`predict` calls."""
        ...


class WakeWordDetector(Protocol):
    """A streaming wake-word detector fed fixed-size audio frames."""

    def process(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return ``True`` when a trigger fires."""
        ...

    def reset(self) -> None: ...


def resolve_model_path(model: str | Path, models_dir: Path) -> Path:
    """Resolve a configured wake-word model to an ``.onnx`` file path.

    An explicit ``.onnx`` path (or an absolute path) is used as-is and must
    exist. A bare name resolves to ``<models_dir>/<name>.onnx``. A missing
    model raises :class:`WakeWordModelNotFoundError` naming the model: a
    trained "Hey Franz" model is required, there is no placeholder fallback.
    """
    candidate = Path(model)
    if candidate.suffix == ".onnx" or candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise WakeWordModelNotFoundError(str(candidate))

    named = models_dir / f"{model}.onnx"
    if named.is_file():
        return named
    raise WakeWordModelNotFoundError(str(named))


class LiveKitWakeWordScorer(WakeWordScorer):
    """livekit-wakeword ONNX inference behind the scorer protocol."""

    def __init__(self, model_path: Path) -> None:
        try:
            from livekit.wakeword import WakeWordModel  # pyright: ignore[reportMissingTypeStubs]
        except OSError as exc:
            raise WakeWordError(f"livekit-wakeword is not available: {exc}") from exc
        self._model: Any = WakeWordModel(models=[str(model_path)])  # pyright: ignore[reportAny]
        # livekit-wakeword keys its scores by the model file stem.
        self.model_name: str = model_path.stem

    def predict(self, window: NDArray[np.int16]) -> dict[str, float]:
        scores: dict[str, float] = self._model.predict(window)  # pyright: ignore[reportAny]
        return scores

    def reset(self) -> None:
        """No-op: livekit-wakeword's ``predict`` is stateless."""


class WindowedWakeWordDetector(WakeWordDetector):
    """Streaming detector over a rolling window, scored on a stride.

    Frames are buffered into a rolling window of at most ``window_samples``;
    once the window is full, it is scored every ``stride_frames`` frames.
    ``threshold`` applies to the scorer's confidence for the configured model;
    a score at or above the threshold emits a trigger.
    """

    def __init__(
        self,
        scorer: WakeWordScorer,
        model_name: str,
        threshold: float = 0.5,
        stride_frames: int = DEFAULT_STRIDE_FRAMES,
        window_samples: int = WINDOW_SAMPLES,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be 0..1, got {threshold}")
        if stride_frames < 1:
            raise ValueError(f"stride_frames must be >= 1, got {stride_frames}")
        if window_samples <= 0:
            raise ValueError(f"window_samples must be positive, got {window_samples}")
        self._scorer = scorer
        self._model_name = model_name
        self._threshold = threshold
        self._stride_frames = stride_frames
        self._window_samples = window_samples
        self._buffer: list[NDArray[np.int16]] = []
        self._buffered = 0
        self._frames_since_score = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_name(self) -> str:
        return self._model_name

    def process(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return ``True`` when the wake word is detected."""
        self._buffer.append(frame)
        self._buffered += frame.size
        self._trim()
        self._frames_since_score += 1
        if self._buffered < self._window_samples:
            return False
        if self._frames_since_score < self._stride_frames:
            return False
        self._frames_since_score = 0
        window = np.concatenate(self._buffer)[-self._window_samples :]
        scores = self._scorer.predict(window)
        return scores.get(self._model_name, 0.0) >= self._threshold

    def reset(self) -> None:
        self._buffer.clear()
        self._buffered = 0
        self._frames_since_score = 0
        self._scorer.reset()

    def _trim(self) -> None:
        """Drop oldest frames so the buffer never exceeds the window size."""
        while self._buffer and self._buffered - self._buffer[0].size >= self._window_samples:
            self._buffered -= self._buffer.pop(0).size
