"""Wake-word detection ("Franz") using openWakeWord.

The detector is a thin streaming adapter: it accepts the fixed-size 16 kHz mono
frames produced by the WP2 audio capture, buffers them into openWakeWord's 80 ms
inference blocks, and reports a trigger when a model score crosses the configured
threshold. Inference runs locally through openWakeWord's ONNX backend.

Model acquisition
-----------------
openWakeWord ships no pretrained "Franz" model, and training one requires the
upstream synthetic-data pipeline (piper TTS plus large negative corpora) that is
out of scope for CI. The model path is therefore fully configurable
(:func:`resolve_model_path`) so a real ``franz.onnx`` can be dropped in via
``voxoracle setup`` without code changes; until then a pretrained placeholder
model is used. See ``openspec/changes/add-voxoracle-core/design.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from voxoracle.wakeword.errors import WakeWordError, WakeWordModelNotFoundError

# openWakeWord consumes 80 ms blocks (1280 samples at 16 kHz).
BLOCK_SAMPLES = 1280

# Fallback used when no "franz" model is present. Clearly a placeholder: it
# detects the phrase "hey jarvis", not "Franz".
PLACEHOLDER_MODEL = "hey_jarvis"


class WakeWordScorer(Protocol):
    """Scores an 80 ms audio block, returning a confidence per model name."""

    def predict(self, block: NDArray[np.int16]) -> dict[str, float]: ...


class WakeWordDetector(Protocol):
    """A streaming wake-word detector fed fixed-size audio frames."""

    def process(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return ``True`` when a trigger fires."""
        ...

    def reset(self) -> None: ...


def resolve_model_path(model: str | Path, models_dir: Path) -> Path:
    """Resolve a configured wake-word model to an ``.onnx`` file path.

    An explicit ``.onnx`` path (or an absolute path) is used as-is and must
    exist. A bare name resolves to ``<models_dir>/<name>.onnx``; if that file is
    absent the bundled placeholder model is used, so the appliance starts even
    before a purpose-trained ``franz.onnx`` is installed. Raises
    :class:`WakeWordModelNotFoundError` only when an explicit path is missing or
    no placeholder is available.
    """
    candidate = Path(model)
    if candidate.suffix == ".onnx" or candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise WakeWordModelNotFoundError(str(candidate))

    named = models_dir / f"{model}.onnx"
    if named.is_file():
        return named

    return _bundled_placeholder()


def _bundled_placeholder() -> Path:
    """Return the path to openWakeWord's bundled placeholder model."""
    try:
        import openwakeword  # pyright: ignore[reportMissingTypeStubs]
    except OSError as exc:  # PortAudio-like native dependency failure
        raise WakeWordError(f"openWakeWord is not available: {exc}") from exc
    for path in openwakeword.get_pretrained_model_paths():
        if PLACEHOLDER_MODEL in path:
            return Path(path)
    raise WakeWordModelNotFoundError(PLACEHOLDER_MODEL)


class OpenWakeWordScorer(WakeWordScorer):
    """openWakeWord ONNX inference backend behind the scorer protocol."""

    def __init__(self, model_path: Path) -> None:
        try:
            from openwakeword.model import Model  # pyright: ignore[reportMissingTypeStubs]
        except OSError as exc:
            raise WakeWordError(f"openWakeWord is not available: {exc}") from exc
        self._model: Any = Model(wakeword_model_paths=[str(model_path)])  # pyright: ignore[reportAny]
        self.model_name: str = next(iter(self._model.models.keys()))  # pyright: ignore[reportAny]

    def predict(self, block: NDArray[np.int16]) -> dict[str, float]:
        scores: dict[str, float] = self._model.predict(block)  # pyright: ignore[reportAny]
        return scores


class OpenWakeWordDetector(WakeWordDetector):
    """Streaming detector that buffers frames into openWakeWord blocks.

    ``threshold`` is applied to the scorer's confidence for the configured
    model; a score at or above the threshold emits a trigger.
    """

    def __init__(
        self,
        scorer: WakeWordScorer,
        model_name: str,
        threshold: float = 0.5,
        block_samples: int = BLOCK_SAMPLES,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be 0..1, got {threshold}")
        if block_samples <= 0:
            raise ValueError(f"block_samples must be positive, got {block_samples}")
        self._scorer = scorer
        self._model_name = model_name
        self._threshold = threshold
        self._block_samples = block_samples
        self._buffer: list[NDArray[np.int16]] = []
        self._buffered = 0

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
        triggered = False
        while self._buffered >= self._block_samples:
            block = self._collect(self._block_samples)
            scores = self._scorer.predict(block)
            if scores.get(self._model_name, 0.0) >= self._threshold:
                triggered = True
        return triggered

    def reset(self) -> None:
        self._buffer.clear()
        self._buffered = 0

    def _collect(self, count: int) -> NDArray[np.int16]:
        """Pop exactly ``count`` samples from the front of the buffer."""
        parts: list[NDArray[np.int16]] = []
        remaining = count
        while remaining > 0:
            head = self._buffer[0]
            if head.size <= remaining:
                parts.append(head)
                remaining -= head.size
                self._buffer.pop(0)
            else:
                parts.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                remaining = 0
        self._buffered -= count
        return np.concatenate(parts)
