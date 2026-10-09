# VoxOracle

The voice frontend for [DocOracle](https://github.com/wuan/docoracle) — a
**headless, voice-only appliance** that answers spoken questions about your
documentation.

VoxOracle runs on a small computer (a Raspberry Pi 3) with a microphone and a
speaker. It listens for its activation word, **"Franz"**, captures the spoken
question, sends it to DocOracle's `POST /ask` endpoint, and speaks the grounded
answer back through the speaker. There is no screen: it is a self-contained
voice appliance.

> Named in the spirit of its parent: if DocOracle is the oracle of your docs,
> VoxOracle is its voice.

## How it works

```
You speak "Franz ..."
        │
        ▼
   Wake word (openWakeWord, local)          "Franz"
        │
        ▼
   Record question (mic, 16 kHz mono)
        │
        ▼
   Speech-to-text (cloud, German)           question text
        │
        ▼
   DocOracle  POST /ask                     grounded answer + citations
        │
        ▼
   Text-to-speech (cloud, German)
        │
        ▼
   Speak answer (speaker)  ──▶  back to listening
```

VoxOracle owns **no retrieval or LLM logic**. It is a presentation and
interaction layer: audio in → text → DocOracle → text → audio out. All
answering, retrieval, citation and filtering behaviour comes from DocOracle, so
the two projects can evolve independently.

## Features

- **Hands-free activation** with the wake word **"Franz"**, detected locally
  using [openWakeWord](https://github.com/dscripka/openWakeWord).
- **Cloud STT and cloud TTS** with swappable providers behind protocols;
  **German first**.
- **Grounded answers** spoken directly from DocOracle's `POST /ask` response.
- **Barge-in**: the wake word during playback stops the speaker.
- **Headless**: no display or browser; the CLI is the operator surface.
- **Testable without hardware**: audio I/O and providers are injected, so the
  full session can be exercised with fakes in CI.

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

The `wakeword`, `stt` and `tts` backends sit behind protocols so providers are
swappable. The `session` module is the orchestrator state machine; all I/O is
injected so it is testable with fakes and no real hardware.

## Requirements

- Python **3.12 or higher** (managed by [uv](https://docs.astral.sh/uv/)).
- A running DocOracle server (Python 3.12+, with documentation ingested):

  ```bash
  docoracle serve --host 0.0.0.0 --port 8000
  ```

## Installation

This project is managed with `uv`, which also provides the Python 3.12+
interpreter regardless of the version shipped by the OS.

```bash
# Install uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and its development dependencies
uv sync
```

On the appliance, install the package and its CLI entry point:

```bash
uv run voxoracle --help
```

## Usage

```bash
voxoracle run      # run the always-on voice session loop
voxoracle ask "…"  # ask DocOracle a question in text mode (no audio)
voxoracle doctor   # check devices, models, configuration and connectivity
voxoracle setup    # download wake-word models and prepare the device
```

The CLI commands are the operator interface for a headless device. They are
stubs while the corresponding work packages are implemented; see
`openspec/changes/add-voxoracle-core/tasks.md` for the roadmap.

## Configuration

VoxOracle reads its settings from `config.yaml`. Start from the example:

```bash
cp config.example.yaml config.yaml
```

```yaml
docoracle:
  url: http://localhost:8000
  timeout: 60

audio:
  input_device: default
  output_device: default
  sample_rate: 16000
  frame_ms: 30

wakeword:
  engine: openwakeword
  model: franz
  threshold: 0.5

stt:
  provider: null
  language: de
  timeout: 30

tts:
  provider: null
  language: de
  voice: null
  timeout: 30

session:
  max_record_seconds: 15
  follow_up: false
  barge_in: true
```

`config.yaml` is git-ignored. Keep provider API keys in the environment (or a
`.env` file) rather than in `config.yaml`.

## DocOracle integration

VoxOracle speaks to DocOracle over HTTP. `POST {docoracle.url}/ask` takes a
`question` (plus optional filters such as `module`, `component` and `version`)
and returns the `answer`, `confidence`, `citations`, and retrieved sources.
VoxOracle speaks the `answer` and can read the citations. `GET /health` and
`GET /info` back the `voxoracle doctor` command.

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

## Specification (OpenSpec)

Behaviour is specified with [OpenSpec](https://github.com/Fission-AI/OpenSpec)
under `openspec/`. The first change, `add-voxoracle-core`, covers the WP0–WP8
roadmap and defines the capability specs (`voice-session`, `wake-word`,
`cloud-stt`, `cloud-tts`, `docoracle-client`, `device-audio`, `service-ops`).

```bash
npm install -g @fission-ai/openspec
openspec validate --all
```

## Project status

Experimental, just like DocOracle. Expect breaking changes; the API surface is
not stable yet.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [DocOracle](https://github.com/wuan/docoracle) by
  [wuan](https://github.com/wuan) — the RAG engine this project gives a voice to.
- [openWakeWord](https://github.com/dscripka/openWakeWord) — local wake-word
  detection.
