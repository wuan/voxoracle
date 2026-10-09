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
the cloud. Python 3.12+ is managed with `uv`, which supplies the interpreter
even when the OS ships an older Python.

This change (WP0) builds the foundation; the capabilities it defines are
implemented by WP1–WP8.

## Goals / Non-Goals

**Goals:**

- A `uv`-managed Python 3.12+ project with a `src/` layout and a testable
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

Use `uv` as the package/dependency manager with `requires-python = ">=3.12"` and
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
  OS cannot run WP3 with Python 3.12. Decision: target **64-bit Raspberry Pi OS
  (aarch64)**, where openWakeWord's ONNX backend installs cleanly; verified on the
  Pi 3 during WP0.
- **Cloud STT/TTS vendors are unsettled** — a wrong early choice is costly.
  Mitigation: keep providers behind protocols and defer the vendor decision to
  the WP4/WP5 design documents.
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

- Which cloud STT provider (WP4) and cloud TTS provider (WP5)? Deferred to their
  design documents.
- Which wake-word model on the target Pi? Resolved to openWakeWord's ONNX
  backend on 64-bit Raspberry Pi OS (aarch64); the "Franz" model is selected in
  WP3.
