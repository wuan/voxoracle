"""Wake-word detector tests: fake scorer plus the real ONNX adapter on fixtures.

No audio hardware is required; fixture WAVs are synthetic (TTS-generated).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from voxoracle.wakeword import (
    WINDOW_SAMPLES,
    LiveKitWakeWordScorer,
    WakeWordModelNotFoundError,
    WindowedWakeWordDetector,
    resolve_model_path,
)

FIXTURES = Path(__file__).parent / "fixtures"
FRAME_SAMPLES = 480  # 30 ms at 16 kHz, as produced by WP2 capture


def read_fixture(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def negative_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


class FakeScorer:
    """Returns a fixed score for a configured model name; records the window."""

    def __init__(self, score: float, name: str = "hey-franz") -> None:
        self._score = score
        self._name = name
        self.predicts = 0
        self.resets = 0
        self.last_window: np.ndarray | None = None

    def predict(self, window: np.ndarray) -> dict[str, float]:
        self.predicts += 1
        self.last_window = window
        assert window.dtype == np.int16
        return {self._name: self._score}

    def reset(self) -> None:
        self.resets += 1


class TestWindowedWakeWordDetector:
    """Detector behaviour with a small window for fast tests."""

    # 2 frames of 480 samples fill the 960-sample test window.
    WINDOW = 960

    def detector(
        self, score: float, **kwargs: object
    ) -> tuple[FakeScorer, WindowedWakeWordDetector]:
        scorer = FakeScorer(score)
        kwargs.setdefault("window_samples", self.WINDOW)
        kwargs.setdefault("stride_frames", 1)
        return scorer, WindowedWakeWordDetector(scorer, "hey-franz", **kwargs)  # type: ignore[arg-type]

    def test_no_trigger_before_window_is_full(self) -> None:
        _scorer, detector = self.detector(0.9)
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_triggers_at_or_above_threshold(self) -> None:
        _scorer, detector = self.detector(0.9)
        detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is True

    def test_no_trigger_below_threshold(self) -> None:
        _scorer, detector = self.detector(0.3)
        detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_threshold_is_honored(self) -> None:
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        _, hit = self.detector(0.7)
        _, miss = self.detector(0.7, threshold=0.8)
        hit.process(frame)
        miss.process(frame)
        assert hit.process(frame) is True
        assert miss.process(frame) is False

    def test_missing_model_score_is_no_trigger(self) -> None:
        scorer = FakeScorer(0.99, name="other")
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=1, window_samples=self.WINDOW
        )
        detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False

    def test_window_is_capped_and_scored_exactly(self) -> None:
        scorer = FakeScorer(0.9)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=1, window_samples=self.WINDOW
        )
        for _ in range(10):
            detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        assert scorer.predicts == 9  # every frame after the first (window full)
        assert scorer.last_window is not None
        assert scorer.last_window.size == self.WINDOW

    def test_window_slides_forward(self) -> None:
        scorer = FakeScorer(0.0)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=1, window_samples=self.WINDOW
        )
        first = np.full(FRAME_SAMPLES, 100, dtype=np.int16)
        second = np.full(FRAME_SAMPLES, 200, dtype=np.int16)
        third = np.full(FRAME_SAMPLES, 300, dtype=np.int16)
        detector.process(first)
        detector.process(second)
        detector.process(third)
        assert scorer.last_window is not None
        # The oldest frame must have been trimmed: the window holds the two
        # most recent frames (200s and 300s), not the first (100s).
        assert int(scorer.last_window[0]) == 200
        assert int(scorer.last_window[-1]) == 300

    def test_stride_controls_scoring_cadence(self) -> None:
        scorer = FakeScorer(0.9)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=3, window_samples=self.WINDOW
        )
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        results = [detector.process(frame) for _ in range(6)]
        # Window is full after frame 2; scoring happens on frames 3 and 6.
        assert results == [False, False, True, False, False, True]
        assert scorer.predicts == 2

    def test_trigger_is_reported_at_the_scoring_frame(self) -> None:
        scorer = FakeScorer(0.9)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=2, window_samples=self.WINDOW
        )
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        assert detector.process(frame) is False  # window still filling
        # Window full and two frames fed: the stride elapses, so this frame scores.
        assert detector.process(frame) is True
        # Scoring now repeats every stride_frames frames: skip one, score again.
        assert detector.process(frame) is False
        assert detector.process(frame) is True

    def test_reset_clears_window_and_stride_counter(self) -> None:
        scorer = FakeScorer(0.9)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=1, window_samples=self.WINDOW
        )
        detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        detector.reset()
        # After the reset the window must refill before the next score.
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is True
        assert scorer.predicts == 1

    def test_reset_also_resets_the_scorer(self) -> None:
        scorer = FakeScorer(0.9)
        detector = WindowedWakeWordDetector(
            scorer, "hey-franz", stride_frames=1, window_samples=self.WINDOW
        )
        detector.reset()
        assert scorer.resets == 1

    def test_invalid_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            WindowedWakeWordDetector(FakeScorer(0.5), "hey-franz", threshold=2.0)

    def test_invalid_stride_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            WindowedWakeWordDetector(FakeScorer(0.5), "hey-franz", stride_frames=0)

    def test_invalid_window_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            WindowedWakeWordDetector(FakeScorer(0.5), "hey-franz", window_samples=0)


class TestResolveModelPath:
    def test_explicit_existing_path(self, tmp_path: Path) -> None:
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"x")
        assert resolve_model_path(model, tmp_path) == model

    def test_explicit_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(WakeWordModelNotFoundError):
            resolve_model_path(tmp_path / "missing.onnx", tmp_path)

    def test_named_model_under_models_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model = models_dir / "hey-franz.onnx"
        model.write_bytes(b"x")
        assert resolve_model_path("hey-franz", models_dir) == model

    def test_missing_named_model_raises_naming_the_model(self, tmp_path: Path) -> None:
        # No placeholder fallback: the trained model is required.
        with pytest.raises(WakeWordModelNotFoundError) as excinfo:
            resolve_model_path("hey-franz", tmp_path / "models")
        assert "hey-franz" in excinfo.value.name


class TestRealOnnxAdapter:
    """Exercises the livekit-wakeword ONNX backend on synthetic fixtures.

    Skipped unless the trained ``hey-franz.onnx`` is installed (it is not
    committed to the repository; see training/hey-franz.yaml).
    """

    pytestmark = pytest.mark.filterwarnings(
        "ignore:Specified provider 'CUDAExecutionProvider':UserWarning"
    )

    def detector(self, threshold: float = 0.68) -> WindowedWakeWordDetector:
        model_path = resolve_model_path("hey-franz", Path("models"))
        scorer = LiveKitWakeWordScorer(model_path)
        return WindowedWakeWordDetector(
            scorer, scorer.model_name, threshold=threshold, stride_frames=2
        )

    def feed(self, detector: WindowedWakeWordDetector, audio: np.ndarray) -> bool:
        triggered = False
        for start in range(0, audio.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            if detector.process(audio[start : start + FRAME_SAMPLES]):
                triggered = True
        return triggered

    @pytest.fixture(autouse=True)
    def _require_trained_model(self) -> None:
        if not resolve_or_none():
            pytest.skip("trained hey-franz.onnx not installed (not committed to the repo)")

    def test_no_trigger_on_silence(self) -> None:
        silence = np.zeros(16000, dtype=np.int16)
        assert self.feed(self.detector(), silence) is False

    def test_no_trigger_on_noise(self) -> None:
        rng = np.random.default_rng(0)
        noise = (rng.integers(-500, 500, size=16000)).astype(np.int16)
        assert self.feed(self.detector(), noise) is False


def resolve_or_none() -> Path | None:
    candidate = Path("models") / "hey-franz.onnx"
    return candidate if candidate.is_file() else None


def test_window_size_matches_livekit_contract() -> None:
    # 32,000 samples at 16 kHz = ~2 s = the 16-embedding window the
    # livekit-wakeword classifier expects.
    assert WINDOW_SAMPLES == 32000


def test_negative_frame_helper_matches_capture_frames() -> None:
    assert negative_frame().size == FRAME_SAMPLES
