# Design

## Context

VoxOracle is a new repository. The only existing content is a README describing
a browser-based UI, which contradicts the agreed appliance design. The target is
a Raspberry Pi 3 running 64-bit (aarch64) Debian-based Raspberry Pi OS,
headless, with a microphone and a speaker. It listens for the wake word "Franz" locally, captures
a question, transcribes it with a cloud STT service, sends it to DocOracle
`POST /ask`, and speaks the grounded answer with a cloud TTS service. German
comes first. VoxOracle owns no retrieval or LLM logic.

The Pi 3 is constrained (1 GB RAM, Cortex-A53), so the always-on local path (the
wake word) must stay lightweight and the heavy speech work must be offloaded to
the cloud. Python 3.13+ is managed with `uv`, which supplies the interpreter
even when the OS ships an older Python.

This change (WP0) builds the foundation; the capabilities it defines are
implemented by WP1–WP8.

## Goals / Non-Goals

**Goals:**

- A `uv`-managed Python 3.13+ project with a `src/` layout and a testable
  package skeleton that mirrors the target architecture.
- A `typer` CLI exposing the operator surface: `run`, `ask`, `doctor`, `setup`.
- Quality gates (`ruff`, `basedpyright`, `pytest`, `pre-commit`) and CI.
- A README that accurately describes the headless appliance.
- OpenSpec initialisation with capability specs and the WP0–WP8 task breakdown.

**Non-Goals:**

- Implementing audio capture/playback, wake-word detection, STT, TTS, the
  DocOracle client, the session loop, or deployment (WP1–WP8).
- Selecting concrete cloud STT/TTS vendors and prices (settled in WP4/WP5).
- Any browser or Web Speech API support.

## Decisions

### `uv` with a `dependency-groups` dev group

Use `uv` as the package/dependency manager with `requires-python = ">=3.13"` and
a committed `uv.lock`. Development tooling lives in the standard `dev`
dependency group so a plain `uv sync` installs both runtime and development
dependencies, and `uv run <tool>` works without extra flags. CI uses
`uv sync --frozen` to guarantee the locked environment.

*Alternatives considered:* an optional `[project.optional-dependencies] dev`
extra (requires `uv sync --extra dev` on every invocation); a `requirements.txt`
workflow (loses the lockfile and interpreter management). Rejected in favour of
the uv-native dependency group.

### `typer` for the CLI

`typer` gives a typed, self-documenting CLI with minimal code and a clean
`--help`, which is the only operator surface on a screenless device. Commands
are stubs until their work package lands, so the operator surface and its help
text are stable from day one.

*Alternatives considered:* `click` (more boilerplate for the same result);
`argparse` (no shared typing ergonomics with the rest of the codebase).

### Capabilities behind protocols, I/O injected

`wakeword`, `stt`, and `tts` will sit behind protocols so providers are
swappable, and the `session` state machine will take all I/O as injected
collaborators. This keeps CI hardware-free and lets WP3/WP4/WP5 proceed in
parallel once WP2 freezes the interfaces.

### Wake-word model acquisition (WP3)

openWakeWord ships pretrained models for `alexa`, `hey_mycroft`, `hey_jarvis`,
`hey_marvin`, and command phrases such as `timer`/`weather` — but **no "Franz"**
model, and no such model exists in the public community collections. Training a
purpose-built model requires openWakeWord's separate synthetic-data pipeline
(piper TTS for positive clips plus large negative corpora, then the automated
training notebook); that is a GPU/large-dataset workflow that cannot run in CI
and is out of scope for WP3's code deliverable.

Decision: make the model path **fully configurable** (`wakeword.model` as a name
or an explicit `.onnx` path, plus `wakeword.models_dir`), and until a real
"franz.onnx" is installed use a clearly-labelled **pretrained placeholder**
(`hey_jarvis`). Everything else in WP3 — the detector protocol, the ONNX
streaming adapter with 80 ms buffering, the configurable threshold, the trigger
event, and the tests — is implemented and verified against both synthetic
fixtures and real hardware. Installing the real "Franz" model later is a file
drop via `voxoracle setup`, with no code changes.

*Alternatives considered:* (a) training a custom model off-device now — rejected
as infeasible in this environment (no synthetic-data pipeline, no large negative
corpus, no suitable compute); (b) shipping a placeholder but hard-coding it —
rejected because the model must be swappable by operators. A community model is
downloaded through `voxoracle setup` in WP7 if one becomes available.

False-activation tuning: `wakeword.threshold` is the primary control (default
0.5, matching openWakeWord's own recommendation). Raising it reduces false
activations at the cost of false rejects; `predict`'s `patience` option (not yet
exposed) can additionally require several consecutive frames above threshold.
On the Pi 3, openWakeWord's own guidance is that a single core handles many
models in real time, so only the configured model is loaded to keep the
always-on path light. Verified on the target: the ONNX backend loads and the
`hey_jarvis` fixture is detected offline; ~3 s of ambient audio scored 0.0.

### Cloud STT/TTS provider: Mistral (WP4/WP5)

The user selected **Mistral** for both cloud speech-to-text and cloud
text-to-speech. STT uses the Voxtral offline transcription endpoint
(`POST https://api.mistral.ai/v1/audio/transcriptions`, model
`voxtral-mini-latest`, i.e. Voxtral Mini Transcribe 2), uploading the recorded
utterance as a 16-bit mono WAV multipart form with a `language` field
(default `de`). The response is OpenAI-compatible (`{"text": "..."}`).

TTS (WP5) uses the Voxtral speech-synthesis endpoint
(`POST https://api.mistral.ai/v1/audio/speech`, model `voxtral-mini-tts-2603`),
sending `input`, `voice_id` (the configured voice, or the German default derived
from `tts.language` when `tts.voice` is null), `response_format` (`pcm`) and
`stream=true`. With streaming, Mistral answers with a `text/event-stream` whose
`data:` frames carry `speech.audio.delta` events (`{"audio_data": "<base64>"}`)
terminated by `speech.audio.done`; the vendor's `pcm` format is raw
little-endian float32 samples, which the backend converts to the mono 16-bit PCM
the WP2 output layer plays. Mistral does not publish the `pcm` sample rate, so it
is configurable (`tts.sample_rate`, default 24 kHz; the player resamples chunks
to the device rate) and confirmed on-device in WP7. A non-streaming
`wav` response (`{"audio_data": "<base64>"}`) is also decodable.

Speech playback lives in a provider-agnostic `SpeechPlayer` that writes each
chunk to an `AudioOutput` as it arrives and exposes an `interrupt()` stop hook.
The output protocol gained `abort()` (`Pa_AbortStream`) so barge-in discards
buffered audio rather than draining it, and `start()` so the player restarts an
aborted output at the next utterance; the player also writes in short slices and
swallows a slice-write failure caused by its own abort. WP6 wires the hook to
the wake-word detector so a barge-in closes the provider stream (cancelling the
request) and aborts the speaker. On-device (WP7) confirm the real barge-in path
and the live `pcm` sample rate and SSE framing.

Both STT and TTS share a single authenticated client
(`voxoracle.mistral.MistralAudioClient`) that owns the base URL, bearer
authentication, timeout, bounded exponential-backoff retries (capped, with
jitter, honoring a numeric `Retry-After` on 429/503), and a typed-error
surface (authentication, rate-limit, server, malformed, timeout, connection).
The backends implement their own protocols on top of it, so a different
provider later means a new backend, not a new transport stack.

Credentials resolve as `mistral.api_key` (config / `.env` /
`VXORACLE_MISTRAL__API_KEY`) then `MISTRAL_API_KEY`, then `LLM_API_KEY` — the
key DocOracle already authenticates against `api.mistral.ai` with — so a single
Mistral key can cover DocOracle and VoxOracle STT/TTS. The non-prefixed names are
read from a real environment variable or from `.env` (env wins). No key is
committed; CI injects a fake transport and never needs a real key. Key scope is
confirmed on the target during WP7.

### OpenSpec as the behaviour contract

Each capability has a spec with testable scenarios. Work packages start as (or
update) an OpenSpec change; specs are validated in CI with `openspec validate`
and archived on completion. The umbrella change `add-voxoracle-core` covers the
WP0–WP8 scope.

### Rewrite the README rather than amend it

The existing README is entirely browser-centric. Incremental edits would leave
the wrong mental model in place, so it is rewritten around the appliance.

## Risks / Trade-offs

- **Wake-word backend on 32-bit ARM** — `onnxruntime` publishes no `armv7`
  wheels and `tflite-runtime`'s armv7 wheels require a newer glibc, so a 32-bit
  OS cannot run WP3 with a modern Python. Decision: target **64-bit Raspberry Pi OS
  (aarch64)**, where openWakeWord's ONNX backend installs cleanly; verified on the
  Pi 3 during WP0.
- **Cloud STT/TTS vendors are unsettled** — a wrong early choice is costly.
  Resolved: **Mistral** (Voxtral) for both, per the user's decision; providers
  stay behind protocols so they remain swappable.
- **Cloud round-trips add latency and require network + API keys** — Mitigation:
  design for spoken "thinking" feedback and explicit timeouts in the session
  state machine.
- **Pi 3 resource limits** — Mitigation: keep only the wake word always-on
  locally; everything else is on demand and in the cloud.
- **Strict type checking on stubs** — Mitigation: keep placeholders typed and
  exclude tests from `basedpyright`; CI enforces the gates.

## Migration Plan

1. Land the foundation (this change): scaffolding, tooling, README, OpenSpec, CI.
2. Implement WP1 (DocOracle client) and WP2 (audio primitives).
3. Freeze the STT/TTS/wake-word protocols and implement WP3/WP4/WP5 in parallel.
4. Integrate in WP6, deploy in WP7, harden and release in WP8.
5. Archive the `add-voxoracle-core` change once all capabilities are accepted.

Rollback is trivial at this stage: the change adds files and rewrites one
README; no data or public interface exists yet.

## Open Questions

- Which cloud STT provider (WP4) and cloud TTS provider (WP5)? Resolved: Mistral
  (Voxtral) for both, per the user's decision. Exposed/pending in WP7: confirm
  the Mistral API key scope covers Voxtral transcription and TTS (the user
  intends one key for DocOracle and VoxOracle).
- Which wake-word model on the target Pi? Resolved to openWakeWord's ONNX
  backend on 64-bit Raspberry Pi OS (aarch64). No pretrained "Franz" model
  exists, so WP3 ships a configurable model path defaulting to the `hey_jarvis`
  placeholder; a purpose-trained "franz.onnx" is installed via `voxoracle setup`
  (WP7) once produced, with no code changes.
