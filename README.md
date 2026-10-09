# VoxOracle

The voice frontend for [DocOracle](https://github.com/wuan/docoracle) — ask your
documentation out loud.

VoxOracle puts a spoken conversational interface on top of DocOracle, an
experimental RAG/chatbot solution that provides natural-language access to
static Antora/AsciiDoc documentation. Speak your question; VoxOracle transcribes
it, sends it to DocOracle's `/ask` API, and reads the answer back — with the
citations available on request.

VoxOracle runs as a **headless, voice-only appliance** on a Raspberry Pi 3 with
a microphone and a speaker. There is no screen and no browser: you activate it
with the wake word **"Franz"**, and it listens, answers, and listens again.

Named in the spirit of its parent: if DocOracle is the oracle of your docs,
VoxOracle is its voice.

## How it works

```
You (speech) ─▶ Wake word "Franz" (local) ─▶ Speech-to-Text (cloud) ─▶ DocOracle /ask ─▶ LLM + RAG retrieval
                                                                                                   │
You (hearing) ◀─ Text-to-Speech (cloud) ◀─ Answer + citations ◀─────────────────────────────────────┘
```

1. Capture — openWakeWord detects the wake word "Franz" locally (no cloud
   round-trip), and the appliance records the question from its microphone.
2. Ask — the transcribed question is sent to a running DocOracle server via its
   HTTP API (`POST /ask`).
3. Retrieve & answer — DocOracle performs hybrid retrieval (semantic FAISS +
   German-aware BM25, fused via RRF) and generates a grounded, cited answer.
4. Speak — the answer is read back over the speaker via cloud text-to-speech,
   with barge-in to stop playback when you speak again.

VoxOracle owns **no retrieval or LLM logic**. It is a presentation and
interaction layer: audio in → text → DocOracle → text → audio out. All
answering, retrieval, citation, and filtering behavior comes from DocOracle,
which guarantees the two projects can evolve independently.

## Features

- **Hands-free activation** — the wake word "Franz", detected locally using
  [openWakeWord](https://github.com/dscripka/openWakeWord); no buttons, no
  screen, no browser.
- **Grounded answers** — every spoken answer carries DocOracle's citations
  (`module:pages:page#section`); the sources can be read back on request.
- **Filtering by voice** — restrict questions to a module, component, or
  version.
- **Conversation history** — local, per-device history of questions and
  answers.
- **Backend-agnostic** — works with either DocOracle answer backend (engine or
  agent).
- **Pluggable STT/TTS** — cloud speech providers behind protocols; **German
  first**.
- **Fallback text mode** — `voxoracle ask "…"` types the question instead of
  talking when speech isn't practical.
- **Testable without hardware** — audio I/O and providers are injected, so the
  session runs with fakes in CI.

## Hardware and OS

| Item | Target |
|---|---|
| Hardware | Raspberry Pi 3 (1 GB RAM, Cortex-A53) |
| Peripherals | USB microphone + speaker (ALSA/PipeWire) |
| OS | Debian-based Raspberry Pi OS, 64-bit (aarch64) |
| Display | None — headless, voice-only |
| Network | A reachable DocOracle server (localhost or LAN) |

### Notes for the Pi 3

- Run a **64-bit (aarch64) Debian-based Raspberry Pi OS**. The wake word uses
  openWakeWord's ONNX backend, and `onnxruntime` publishes no 32-bit `armv7`
  wheels; the Pi 3's Cortex-A53 supports 64-bit.
- Keep the always-on local path (the wake word) lightweight; STT and TTS are
  offloaded to the cloud.
- Verify audio and detection behavior on real hardware before relying on a
  backend.

The wake word, STT and TTS backends sit behind protocols, so providers are
swappable at runtime.

## Prerequisites

- A Raspberry Pi 3 running 64-bit (aarch64) Debian-based Raspberry Pi OS, with
  a USB microphone and a speaker.
- A running DocOracle server (Python 3.12+, with documentation ingested):

  ```bash
  docoracle serve --host 0.0.0.0 --port 8000
  ```

- [uv](https://docs.astral.sh/uv/) for installing the project; it provides the
  Python 3.13+ interpreter regardless of the version shipped by the OS.

## Installation

```bash
# Install uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and its development dependencies
uv sync

# Prepare the appliance (wake-word models, device checks)
uv run voxoracle setup
```

## Usage

The CLI is the operator surface of the headless device:

```bash
voxoracle run      # run the always-on voice session loop
voxoracle ask "…"  # ask DocOracle a question in text mode (no audio)
voxoracle doctor   # check devices, models, configuration and connectivity
voxoracle setup    # download wake-word models and prepare the device
```

> `ask` is implemented — it sends the question to DocOracle and prints the
> answer. `run`, `doctor` and `setup` are stubs being implemented work-package
> by work-package; see `openspec/changes/add-voxoracle-core/tasks.md` for the
> roadmap.

## Configuration

VoxOracle reads its settings from `config.yaml`. Start from the example:

```bash
cp config.example.yaml config.yaml
```

```yaml
docoracle:
  url: http://localhost:8000   # DocOracle server
  timeout: 60                  # Seconds to wait for POST /ask

audio:
  input_device: default        # ALSA/PipeWire device, or "default"
  output_device: default
  sample_rate: 16000           # Capture rate in Hz (16 kHz mono for speech)
  frame_ms: 30                 # Audio frame size in milliseconds

wakeword:
  engine: openwakeword
  model: franz
  threshold: 0.5               # Detection threshold in [0, 1]

mistral:
  base_url: https://api.mistral.ai/v1   # Shared by STT and TTS
  api_key: null                # Prefer the environment; never commit a key
  timeout: 30
  retries: 2

stt:
  provider: mistral            # Cloud speech-to-text provider
  model: voxtral-mini-latest   # Voxtral Mini Transcribe 2
  language: de                 # BCP-47 tag; German first
  timeout: 30

tts:
  provider: mistral            # Cloud text-to-speech provider
  language: de
  voice: null
  timeout: 30

session:
  max_record_seconds: 15       # Maximum length of a spoken question
  follow_up: false             # Keep listening briefly after an answer
  barge_in: true               # Stop playback when you speak again

logging:
  level: INFO                  # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

`config.yaml` is git-ignored. Keep provider API keys in the environment (or a
`.env` file) rather than in `config.yaml`. Environment overrides use the
`VXORACLE_` prefix with `__` between sections and keys, e.g.
`VXORACLE_DOCORACLE__URL`, and the environment wins over `config.yaml`.

Cloud speech uses **Mistral (Voxtral)**: set the key as
`VXORACLE_MISTRAL__API_KEY`, or reuse the `MISTRAL_API_KEY` / `LLM_API_KEY` that
DocOracle already authenticates with, so one Mistral key covers STT, TTS and
DocOracle.

## Architecture

```
src/voxoracle/
├── audio/        # device enumeration, capture, playback, resampling, VAD
├── wakeword/     # wake-word detector (openWakeWord, "Franz") + models
├── stt/          # cloud speech-to-text backends + protocol
├── tts/          # cloud text-to-speech backends + protocol
├── docoracle/    # typed HTTP client for /ask, /health, /info
├── session/      # conversation state machine, history, barge-in
├── config/       # settings schema + loading
├── cli.py        # `voxoracle run | ask | doctor | setup`
└── __main__.py
```

The `session` module is the orchestrator state machine; all I/O is injected so
it is testable with fakes and no real hardware.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run basedpyright
pre-commit install   # run ruff, basedpyright, pytest and openspec on commit
```

Tests must run **without audio hardware**; use synthetic fixtures and injected
fakes rather than real devices or user data.

Behaviour is specified with [OpenSpec](https://github.com/Fission-AI/OpenSpec)
under `openspec/`. The change `add-voxoracle-core` defines the capability specs
(`voice-session`, `wake-word`, `cloud-stt`, `cloud-tts`, `docoracle-client`,
`device-audio`, `service-ops`).

```bash
npm install -g @fission-ai/openspec
openspec validate --all
```

## Project status

Experimental — just like DocOracle. Expect breaking changes; the API surface is
not stable yet.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [DocOracle](https://github.com/wuan/docoracle) by
  [wuan](https://github.com/wuan) — the RAG engine this project gives a voice
  to.
- [openWakeWord](https://github.com/dscripka/openWakeWord) — local wake-word
  detection.
