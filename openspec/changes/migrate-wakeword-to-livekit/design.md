# Design: migrate-wakeword-to-livekit

## Context

The wake-word path lives in `src/voxoracle/wakeword/detector.py`: an
`OpenWakeWordScorer` (openWakeWord ONNX backend, streaming 80 ms blocks with
model-held state) behind a `WakeWordScorer` protocol, and an
`OpenWakeWordDetector` that buffers 30 ms capture frames into 80 ms blocks and
thresholds `scores[model_name]`. `resolve_model_path` falls back to the bundled
`hey_jarvis` placeholder when `hey-franz.onnx` is absent; `wakeword.engine`
exists in the config dataclass but is never read. The session builder wires
scorer + detector in `src/voxoracle/session/build.py`; `doctor` checks the
import in `src/voxoracle/cli.py`.

livekit-wakeword 0.3.0 (verified from its docs/pyproject): runtime deps are
numpy + onnxruntime only; the feature-extraction ONNX models are bundled in the
wheel; `WakeWordModel.predict(audio_chunk) -> dict[name, float]` is **stateless**
— the caller owns a ~2 s (32,000-sample) rolling window and there is no
`reset()`; INT8 quantization is available at export; German training runs via
the VoxCPM TTS backend with extra samples/voice prompts recommended.

See proposal.md for motivation, specs/wake-word/spec.md for the behavior
contract.

## Goals / Non-Goals

**Goals:**

- Real German "Hey Franz" detection with a low false-positive classifier.
- Wake-word runtime that fits the Pi 3's always-on CPU budget, verified on
  hardware.
- Smaller dependency tree (openWakeWord out entirely).

**Non-Goals:**

- Automating or scheduling the training run; training happens manually on the
  operator's fast machine. The change delivers the training config and a
  documented recipe, not CI for it.
- Keeping openWakeWord as a selectable engine; single runtime, no fallback.
- Committing model artifacts to the repository.
- Changing the capture/audio layer, session loop, or anything downstream of the
  trigger event.

## Decisions

### D1: Keep the `WakeWordScorer` protocol; new livekit scorer implementation

The scorer protocol (`predict` returning `dict[str, float]`) survives; a new
`LiveKitWakeWordScorer` wraps `livekit.wakeword.WakeWordModel`.
`WakeWordModel` is stateless, so `reset()` becomes a no-op for this scorer.

*Alternative considered*: drop the protocol and call the library directly —
rejected; the protocol keeps tests hardware-free (fakes) per project rules and
keeps the door open for future engines.

### D2: Window ownership moves into the detector; scoring on a stride

The detector keeps its frame-buffering role but now maintains a rolling
~32,000-sample window and calls the scorer once per configured stride of frames
(config key `wakeword.stride_frames`, evaluated in the hardware spike; expected
zone 4–8 frames of 30 ms, i.e. 320–640 ms cadence). Detection latency grows by
the stride; the CPU cost drops proportionally because each stateless predict
recomputes mel + embeddings over the whole window.

*Alternative considered*: use livekit's async `WakeWordListener` — rejected; it
captures its own microphone via PyAudio and VoxOracle owns the capture path
(shared with VAD/endpointing).

### D3: Replace, don't dual-run

`openwakeword` is removed from pyproject; the `wakeword.engine` config field is
deleted rather than implemented. The conv-attention runtime can also load
plain openWakeWord-trained classifiers, so no capability is lost.

*Alternative considered*: keep `engine` as a selection switch with two scorer
implementations — rejected as dead configurability; nothing would select
openWakeWord once `hey-franz.onnx` exists.

### D4: Model required, placeholder machinery deleted

`PLACEHOLDER_MODEL`, `_bundled_placeholder()`, and the fallback branch of
`resolve_model_path` are removed. A bare model name still resolves to
`<models_dir>/<name>.onnx`; missing model raises `WakeWordModelNotFoundError`.
`doctor` reports a `[fail]` with install instructions; session startup aborts.
`models/` is documented in README plus `.gitignore` stays as is (no artifacts
committed); `voxoracle setup` points the operator at the training recipe.

### D5: Training recipe ships as a checked-in config + doc

A `training/hey-franz.yaml` (livekit-wakeword config: `model_name: hey-franz`,
`target_phrases: ["hey franz"]`, `tts_backend: voxcpm`,
`model_type: conv_attention`, `model_size: small`, `n_samples: 50000`, 50–100
German `voice_design_prompts`, `target_fp_per_hour: 0.5`) plus a README section
documenting the off-device training session
(`uv tool install livekit-wakeword[train,eval,export,voxcpm]`), the eval gate
(FPPH ≤ 0.5, DET curve, documented recall), and ONNX export (optionally
`--quantize`).

*Alternative considered*: keep the training config out of the repo entirely —
rejected; reproducibility of the deployed model matters and the config is
small.

### D6: Threshold and stride are data-driven defaults

The config default threshold changes from 0.5 to the value the trained model's
eval selects (upstream conv-attention models land near 0.68); the example config
comments say so. The spike (D2) fixes `stride_frames`.

## Risks / Trade-offs

- [Stateless window scoring costs more CPU than streaming] → hardware spike on
  the Pi 3 measures predict latency per stride and INT8 vs FP32; stride is
  chosen from measurements, not guesswork. If no stride fits, the spike result
  forces a design revisit (e.g., smaller `model_size`, or quantization changes).
- [German conv-attention accuracy lags English (English-heavy frozen
  embeddings)] → acceptance gate is explicit (FPPH ≤ 0.5); recall is documented
  so a weak model is a known, visible trade-off rather than a surprise.
- ["Hey Franz" is a short phrase → inherently higher false-positive risk] →
  adversarial negatives and the FPPH gate address it; threshold can be raised at
  the cost of recall.
- [Detection latency increases with stride] → bounded and documented; the
  utterance that follows the trigger is unaffected because endpointing starts
  after the trigger.
- [Archive ordering] → this delta's MODIFIED/REMOVED target the `wake-word`
  spec created by `add-voxoracle-core`; archive that change first (it is nearly
  complete).

## Migration Plan

1. Train + eval `hey-franz.onnx` off-device; record metrics.
2. Spike on the Pi 3: measure window predict latency (FP32 and INT8), pick
   stride and quantization variant.
3. Land the code change: dependency swap, scorer/detector rework, placeholder
   removal, doctor/config/docs/tests.
4. Install the model into `models/` on the device; run `voxoracle doctor`;
   verify live detection with the documented threshold.
5. Rollback: revert the commit; the previous openWakeWord runtime and
   placeholder behavior return unchanged (config `engine` field removal is the
   only user-visible config diff).

## Open Questions

- Exact `stride_frames` and quantization variant — resolved by the Pi 3 spike
  (D2/D6), does not change specs or task breakdown.
- Exact default threshold — resolved by the trained model's eval output (D6).
