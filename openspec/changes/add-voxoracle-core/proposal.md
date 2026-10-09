# Proposal

## Why

DocOracle answers documentation questions over HTTP, but reaching it requires a
keyboard and a screen. A Raspberry Pi with a microphone and a speaker in the
room should be able to answer spoken questions hands-free. VoxOracle is that
appliance: a headless, voice-only frontend for DocOracle.

The repository is empty apart from an aspirational README describing a
browser-based web UI (Web Speech API). That framing does not match the agreed
design: an embedded device with a local wake word, cloud STT/TTS, and no
browser. This change establishes the foundation and corrects the description so
all later work builds on the real target.

## What Changes

- **BREAKING (project framing)**: rewrite the README and repository scaffolding
  around a headless, voice-only Raspberry Pi appliance instead of a browser UI.
- **Project foundation**: a `uv`-managed Python 3.13+ project (`pyproject.toml`,
  `uv.lock`, `src/voxoracle/`), a `typer` CLI (`run`, `ask`, `doctor`, `setup`),
  and the package skeleton mirroring the target architecture.
- **Quality tooling**: `ruff`, `basedpyright`, `pytest`, `pre-commit`, and a
  GitHub Actions CI workflow.
- **OpenSpec initialisation**: `openspec/config.yaml` and the first change
  (`add-voxoracle-core`) defining the capability specs below.
- **Repository hygiene**: `.gitignore`, `.python-version`, `config.example.yaml`,
  `LICENSE` (MIT), `AGENTS.md`.

This change covers the **WP0–WP8** roadmap. Only WP0 (foundation, README fix,
OpenSpec init, CI) is implemented here; the remaining work packages land
separately against the capabilities defined by this change.

## Capabilities

### New Capabilities

- `voice-session`: the wake → record → transcribe → ask → speak state machine,
  including error paths, barge-in, and optional follow-ups.
- `wake-word`: local detection of the activation word "Franz" with openWakeWord.
- `cloud-stt`: the provider-agnostic cloud speech-to-text capability.
- `cloud-tts`: the provider-agnostic cloud text-to-speech capability.
- `docoracle-client`: the typed HTTP client for DocOracle `/ask`, `/health`, and
  `/info`.
- `device-audio`: microphone capture, speaker playback, resampling, and voice
  activity detection.
- `service-ops`: the CLI, configuration schema, model bootstrap, and Pi service
  deployment.

### Modified Capabilities

<!-- None: this is the first change in the repository. -->

## Impact

- **New**: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`,
  `.pre-commit-config.yaml`, `config.example.yaml`, `LICENSE`, `AGENTS.md`,
  `src/voxoracle/`, `tests/`, `openspec/`, `.github/workflows/ci.yml`.
- **Rewritten**: `README.md` (removes the browser/Web Speech API framing).
- **Dependencies**: `typer` at runtime; `pytest`, `ruff`, `basedpyright`,
  `pre-commit` for development. Audio and cloud-provider dependencies are added
  by WP2–WP5.
- **Runtime target**: Raspberry Pi 3, Debian-based Raspberry Pi OS, headless.
