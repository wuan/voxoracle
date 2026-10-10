"""Wake-word detector tests: fake scorer plus the real ONNX adapter on fixtures.

No audio hardware is required; fixture WAVs are synthetic (TTS-generated).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from voxoracle.wakeword import (
    BLOCK_SAMPLES,
    OpenWakeWordDetector,
    WakeWordModelNotFoundError,
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
    """Returns a fixed score for a configured model name."""

    def __init__(self, score: float, name: str = "hey-franz") -> None:
        self._score = score
        self._name = name
        self.blocks_seen = 0
        self.resets = 0

    def predict(self, block: np.ndarray) -> dict[str, float]:
        self.blocks_seen += 1
        assert block.size == BLOCK_SAMPLES
        return {self._name: self._score}

    def reset(self) -> None:
        self.resets += 1


class TestOpenWakeWordDetector:
    def test_triggers_at_or_above_threshold(self) -> None:
        detector = OpenWakeWordDetector(FakeScorer(0.9), "hey-franz", threshold=0.5)
        assert detector.process(np.zeros(BLOCK_SAMPLES, dtype=np.int16)) is True

    def test_no_trigger_below_threshold(self) -> None:
        detector = OpenWakeWordDetector(FakeScorer(0.3), "hey-franz", threshold=0.5)
        assert detector.process(np.zeros(BLOCK_SAMPLES, dtype=np.int16)) is False

    def test_threshold_is_honored(self) -> None:
        frame = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
        assert OpenWakeWordDetector(FakeScorer(0.6), "hey-franz", threshold=0.5).process(frame)
        assert not OpenWakeWordDetector(FakeScorer(0.6), "hey-franz", threshold=0.7).process(frame)

    def test_missing_model_score_is_no_trigger(self) -> None:
        detector = OpenWakeWordDetector(FakeScorer(0.99, name="other"), "hey-franz")
        assert detector.process(np.zeros(BLOCK_SAMPLES, dtype=np.int16)) is False

    def test_frames_are_buffered_into_blocks(self) -> None:
        scorer = FakeScorer(0.9)
        detector = OpenWakeWordDetector(scorer, "hey-franz", threshold=0.5)
        # 3 x 30 ms frames == 1440 samples > one 1280-sample block
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is False
        assert detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is True
        assert scorer.blocks_seen == 1

    def test_reset_clears_partial_buffer(self) -> None:
        scorer = FakeScorer(0.9)
        detector = OpenWakeWordDetector(scorer, "hey-franz", threshold=0.5)
        detector.process(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        detector.reset()
        assert detector.process(np.zeros(BLOCK_SAMPLES, dtype=np.int16)) is True
        assert scorer.blocks_seen == 1

    def test_reset_also_resets_the_scorer(self) -> None:
        scorer = FakeScorer(0.9)
        detector = OpenWakeWordDetector(scorer, "hey-franz", threshold=0.5)
        detector.reset()
        assert scorer.resets == 1

    def test_invalid_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            OpenWakeWordDetector(FakeScorer(0.5), "hey-franz", threshold=2.0)


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

    def test_missing_named_model_falls_back_to_placeholder(self, tmp_path: Path) -> None:
        # The default "hey-franz" name must not prevent the appliance from starting.
        path = resolve_model_path("hey-franz", tmp_path / "models")
        assert path.suffix == ".onnx"
        assert path.is_file()

    def test_placeholder_falls_back_to_bundled(self, tmp_path: Path) -> None:
        path = resolve_model_path("placeholder", tmp_path / "models")
        assert path.suffix == ".onnx"
        assert path.is_file()

    def test_explicit_missing_path_raises_despite_placeholder(self, tmp_path: Path) -> None:
        with pytest.raises(WakeWordModelNotFoundError):
            resolve_model_path(tmp_path / "missing.onnx", tmp_path / "models")


class TestRealOnnxAdapter:
    """Exercises the openWakeWord ONNX backend on synthetic fixtures."""

    pytestmark = pytest.mark.filterwarnings(
        "ignore:Specified provider 'CUDAExecutionProvider':UserWarning"
    )

    def detector(self, threshold: float = 0.5) -> OpenWakeWordDetector:
        from voxoracle.wakeword import OpenWakeWordScorer

        model_path = resolve_model_path("placeholder", Path("models"))
        scorer = OpenWakeWordScorer(model_path)
        return OpenWakeWordDetector(scorer, scorer.model_name, threshold=threshold)

    def feed(self, detector: OpenWakeWordDetector, audio: np.ndarray) -> bool:
        triggered = False
        for start in range(0, audio.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            if detector.process(audio[start : start + FRAME_SAMPLES]):
                triggered = True
        return triggered

    def test_placeholder_triggers_on_its_wake_phrase(self) -> None:
        assert self.feed(self.detector(), read_fixture("hey_jarvis.wav")) is True

    def test_no_trigger_on_silence(self) -> None:
        silence = np.zeros(16000, dtype=np.int16)
        assert self.feed(self.detector(), silence) is False

    def test_no_trigger_on_noise(self) -> None:
        rng = np.random.default_rng(0)
        noise = (rng.integers(-500, 500, size=16000)).astype(np.int16)
        assert self.feed(self.detector(), noise) is False

    def test_placeholder_does_not_trigger_on_hey_franz(self) -> None:
        # Documents the current gap: the placeholder detects "hey jarvis", not "Hey Franz".
        assert self.feed(self.detector(), read_fixture("hey-franz.wav")) is False
