# Proposal: migrate-wakeword-to-livekit

## Why

The appliance still does not detect its actual activation word: openWakeWord ships no
pretrained "Hey Franz" model, so the detector falls back to a `hey_jarvis` placeholder,
and openWakeWord's own training pipeline was judged out of scope. livekit-wakeword
(LiveKit, Apache-2.0) shares openWakeWord's audio front-end but adds a streamlined
German-capable training pipeline and a conv-attention classifier head that measures two
orders of magnitude fewer false positives (0.08 vs 8.5 FPPH in upstream benchmarks).
Training a real German "Hey Franz" model and moving the runtime to livekit-wakeword
retires both the placeholder and the weak classifier architecture.

## What Changes

- **BREAKING** — Replace openWakeWord with livekit-wakeword as the wake-word runtime:
  - dependency swap in `pyproject.toml` (`openwakeword` out; `livekit-wakeword` +
    explicit `onnxruntime` in);
  - new scorer implementation behind the existing `WakeWordScorer` protocol;
  - detector reworked for livekit-wakeword's **stateless window contract**: the caller
    maintains a rolling ~2 s (32,000-sample) window and calls `predict` on a configured
    cadence (stride), instead of streaming 80 ms blocks into model-held state;
  - the `wakeword.engine` config field is removed (single engine, no fallback).
- **BREAKING** — Remove the placeholder-model machinery: the bundled `hey_jarvis`
  fallback, `PLACEHOLDER_MODEL`, and `_bundled_placeholder()` are deleted. A
  `hey-franz.onnx` model is **required**; `doctor` and session startup fail fast with a
  clear, typed error when it is missing.
- Add the trained German "Hey Franz" conv-attention model (trained off-device with
  livekit-wakeword's pipeline: VoxCPM TTS backend, ~50k samples, 50–100 German voice
  prompts, `model_size: small`), evaluated against an acceptance gate (FPPH ≤ 0.5 on
  German background audio) and exported to ONNX (optionally INT8-quantized).
- Install `hey-franz.onnx` out-of-band into `models/`; the model is **not committed** to
  the repository. README and `voxoracle setup` document how to obtain and install it.
- Verify the stateless-window inference cost on the Raspberry Pi 3 (Cortex-A53) and pick
  the predict stride from the measured CPU/latency tradeoff; derive the default
  detection threshold from the trained model's eval output.
- Update docs, `config.example.yaml`, `doctor` checks, and tests to the new engine and
  model-required semantics.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `wake-word`: engine requirement changes from openWakeWord to livekit-wakeword with
  stateless window scoring; the placeholder-model requirement is removed in favour of a
  required, fail-fast model; the runtime-backend requirement changes to livekit-wakeword's
  ONNX inference with a documented predict stride verified on the Pi 3.

## Impact

- **Code**: `src/voxoracle/wakeword/detector.py` (scorer + detector rework, placeholder
  removal), `src/voxoracle/session/build.py` (construction path), `src/voxoracle/config`
  (drop `engine` field), `src/voxoracle/cli.py` (`doctor`/`run` checks),
  `config.example.yaml`, `README.md`, tests (`test_wakeword.py`, `test_diagnostics.py`,
  `test_smoke.py`).
- **Dependencies**: remove `openwakeword>=0.4.0`; add `livekit-wakeword>=0.3.0` and
  `onnxruntime` (both runtime deps of livekit-wakeword are only numpy + onnxruntime;
  feature-extraction ONNX models are bundled in the wheel — no runtime downloads).
- **Ops/hardware**: requires 64-bit Raspberry Pi OS (aarch64) as before; training happens
  off-device on a fast machine (torch + VoxCPM extras are not installed on the Pi);
  `models/hey-franz.onnx` is provisioned out-of-band.
- **Risk**: livekit-wakeword's stateless predict recomputes embeddings over the full
  window per call; the predict stride on the Pi 3 is an empirical decision backed by a
  hardware spike. Thresholds change (conv-attention optimal is near 0.68, not 0.5).
