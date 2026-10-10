# Tasks

## 1. WP0 — Project foundation, README fix and OpenSpec init

- [x] 1.1 Create `pyproject.toml` (`requires-python = ">=3.13"`, `typer` runtime dep, `dev` dependency group for `pytest`/`ruff`/`basedpyright`/`pre-commit`) and run `uv sync` to generate `uv.lock`
- [x] 1.2 Add the `src/voxoracle/` skeleton (`audio/`, `wakeword/`, `stt/`, `tts/`, `docoracle/`, `session/`, `config/`, `cli.py`, `__main__.py`) with a `typer` CLI exposing `run`, `ask`, `doctor`, `setup` stubs
- [x] 1.3 Configure `ruff`, `basedpyright`, and `pytest` in `pyproject.toml`; add `.pre-commit-config.yaml`
- [x] 1.4 Add repository hygiene files: `.gitignore`, `.python-version`, `config.example.yaml`, `LICENSE` (MIT), `AGENTS.md`
- [x] 1.5 Rewrite `README.md` for the headless appliance design (Pi 3 + Raspberry Pi OS, mic + speaker, wake word "Franz", cloud STT/TTS, German first, DocOracle `POST /ask`, uv + Python 3.13, OpenSpec); remove the browser/Web Speech API framing
- [x] 1.6 Initialise OpenSpec (`openspec/config.yaml`) and this change with `proposal.md`, `design.md`, `tasks.md`, and capability specs
- [x] 1.7 Add `.github/workflows/ci.yml` (uv sync, ruff, basedpyright, pytest, openspec validate) and `tests/test_smoke.py`
- [x] 1.8 Verify: `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run basedpyright`, `openspec validate --all` all pass
- [x] 1.9 Verify WP0 on the target (64-bit Raspberry Pi OS, aarch64): `uv sync`, `uv run pytest`, `uv run ruff check .`, and `uv run basedpyright` pass on the Pi 3

## 2. WP1 — DocOracle API client

- [x] 2.1 Define typed Pydantic request/response models for `POST /ask` (`question`, `module`, `component`, `version`, `k`, `retrieval`, `show_sources`, `show_context`; response `answer`, `confidence`, `citations`, `reasoning`, `sources`, `source_details[]`, `retrieved_count`, `retrieval_mode`)
- [x] 2.2 Implement the async `httpx` client for `/ask`, `/health`, and `/info` with configurable base URL and timeouts
- [x] 2.3 Add typed errors and retry behavior for unreachable/timeout/malformed responses
- [x] 2.4 Implement text-mode `voxoracle ask "…"` that prints DocOracle's answer and exit codes
- [x] 2.5 Verify with mocked transport: success, timeout, unreachable server, and invalid-response cases

## 3. WP2 — Audio device layer

- [x] 3.1 Implement microphone capture at 16 kHz mono via `sounddevice`/PortAudio
- [x] 3.2 Implement speaker playback with resampling to the device rate
- [x] 3.3 Add device enumeration/selection from configuration
- [x] 3.4 Add voice activity detection / endpointing (silero-vad or webrtcvad) with a maximum-duration cap
- [x] 3.5 Verify capture/playback/VAD with synthetic buffers and no audio hardware

## 4. WP3 — Wake-word detection ("Franz", openWakeWord)

- [x] 4.1 Integrate openWakeWord behind a `WakeWordDetector` protocol and load the configured wake-word model (configurable name/path; placeholder until a "franz" model exists)
- [x] 4.2 Make the detection threshold configurable
- [x] 4.3 Emit a trigger event to the session loop; keep the always-on path lightweight
- [ ] 4.4 Use openWakeWord's ONNX backend on 64-bit Raspberry Pi OS (aarch64) and verify detection on the Pi 3 — ONNX backend + offline fixture detection verified on the Pi 3; live "Franz" detection pends a purpose-trained model
- [x] 4.5 Verify detection against fixture audio and document false-activation tuning

## 5. WP4 — Cloud speech-to-text

- [x] 5.1 Define the provider-agnostic `STT` protocol (audio in, transcript out)
- [x] 5.2 Implement a configurable cloud STT backend (provider, endpoint, credentials, language)
- [x] 5.3 Default to German (`de`) and read credentials from config/env
- [x] 5.4 Add retries/timeouts and typed errors
- [x] 5.5 Verify a German fixture utterance with a mocked provider

## 6. WP5 — Cloud text-to-speech

- [x] 6.1 Define the provider-agnostic `TTS` protocol (text in, audio out)
- [x] 6.2 Implement a configurable cloud TTS backend with a German voice
- [x] 6.3 Add streaming playback and barge-in support
- [x] 6.4 Verify German answer text with a mocked provider and confirm barge-in stops playback
- [x] 6.5 Recover after barge-in: the output protocol has `start()` and `SpeechPlayer` restarts an aborted output at the next `speak()`, so the following utterance plays (fake-output regression test)
- [ ] 6.6 On-device (WP7) verify the real barge-in path and wire format: a write on a `Pa_AbortStream`-aborted stream restarts playback, and confirm Mistral's live `pcm` sample rate and SSE framing (including multi-line `data:` handling)

## 7. WP6 — Voice session state machine

- [ ] 7.1 Implement the wake → record → transcribe → ask → speak loop with injected collaborators
- [ ] 7.2 Add error paths (no speech, STT failure, DocOracle unreachable/timeout) that speak a prompt and return to IDLE
- [ ] 7.3 Add barge-in (wake word or speech during SPEAKING stops playback)
- [ ] 7.4 Add the optional follow-up listening window
- [ ] 7.5 Verify a simulated end-to-end conversation with fakes and the error paths

## 8. WP7 — Config, packaging and Pi deployment

- [x] 8.1 Implement the settings schema (Pydantic settings) and `config.yaml`/env loading matching `config.example.yaml`
- [ ] 8.2 Implement `voxoracle setup` (wake-word model bootstrap) and `voxoracle doctor` (devices, models, DocOracle status)
- [ ] 8.3 Finish `voxoracle run` for the target device
- [ ] 8.4 Add a systemd unit and an install script; start on boot
- [ ] 8.5 Verify `voxoracle run` on a Pi after install and document the hardware setup

## 9. WP8 — Quality, CI and release hardening

- [ ] 9.1 Expand CI to the supported Python/OS matrix
- [ ] 9.2 Add coverage for the core paths and enforce thresholds
- [ ] 9.3 Document and run a hardware-in-the-loop test plan
- [ ] 9.4 Archive the completed OpenSpec capabilities and tag a release with notes
