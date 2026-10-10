# Tasks

## 1. Training pipeline (off-device, fast machine)

- [x] 1.1 Add `training/hey-franz.yaml` per design D5 and verify it loads (`livekit-wakeword` parses the config on the training machine without error)
- [ ] 1.2 Run the training session (`generate`, `augment`, `train`) on the fast machine and verify a trained classifier checkpoint exists in the output directory
- [ ] 1.3 Export ONNX (FP32 + INT8 `--quantize`) and verify both files load with `livekit.wakeword.WakeWordModel`
- [ ] 1.4 Run `livekit-wakeword eval` and verify FPPH ≤ 0.5 on German background audio; record recall, AUT, and the selected optimal threshold in the change notes

## 2. Hardware spike (Raspberry Pi 3)

- [ ] 2.1 Measure windowed `predict` latency on the Pi 3 for the FP32 and INT8 exports at each candidate stride (4, 6, 8 frames of 30 ms) and verify a written measurement table exists in the change notes
- [ ] 2.2 Pick `stride_frames` and the quantization variant from the measurements; verify the chosen combination holds a comfortable CPU margin (documented value, not saturated) and record detection latency

## 3. Runtime swap (code)

- [x] 3.1 Swap dependencies in `pyproject.toml` (remove `openwakeword`, add `livekit-wakeword` + explicit `onnxruntime`) and verify `uv sync` resolves on aarch64 and x86
- [x] 3.2 Implement `LiveKitWakeWordScorer` behind the existing `WakeWordScorer` protocol (stateless `predict`, no-op `reset`) and verify unit tests with a fake scorer pass
- [x] 3.3 Rework the detector to maintain a rolling ~2 s window and score on the configured stride; verify unit tests cover window fill, stride cadence, threshold trigger, and trigger reporting between strides
- [x] 3.4 Delete the placeholder machinery (`PLACEHOLDER_MODEL`, `_bundled_placeholder()`, fallback branch of `resolve_model_path`) and verify a missing model raises `WakeWordModelNotFoundError` naming the model
- [x] 3.5 Update `session/build.py` and config (remove `wakeword.engine`, add `wakeword.stride_frames`, new default threshold) and verify the session builds against a fake scorer in tests
- [x] 3.6 Update `doctor`/`run` checks and `config.example.yaml`; verify `voxoracle doctor` reports `[fail]` with install instructions when `hey-franz.onnx` is missing and `[pass]` when present

## 4. Docs and live verification

- [x] 4.1 Update README (training recipe pointer, out-of-band model install, new threshold/stride semantics, dependency notes) and verify the documented commands run as written
- [ ] 4.2 Install the trained `hey-franz.onnx` into `models/` on the Pi 3 and verify `voxoracle doctor` passes end-to-end
- [ ] 4.3 Verify live detection on the Pi 3: speaking "Hey Franz" triggers reliably at the documented threshold, background audio does not trigger, and measured CPU stays within the documented budget

## Workflow follow-up

- Archive `add-voxoracle-core` first (its `wake-word` spec is the target of this change's delta; archive refuses the delta while the main spec is missing).
- Archive this change after review requirements are satisfied and the archived result is verified.
