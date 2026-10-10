# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks,
use judgment.

## Project context

VoxOracle is a **headless, voice-only appliance frontend** for DocOracle
(https://github.com/wuan/docoracle). It runs on a **Raspberry Pi 3** with
**Debian-based Raspberry Pi OS**, a microphone and a speaker. It owns **no
retrieval or LLM logic**: it is audio in -> text -> DocOracle `POST /ask` ->
text -> audio out.

- Activation word **"Hey Franz"**, detected locally with **openWakeWord**.
- **Cloud STT and cloud TTS**; **German first**.
- Python **3.13+**, managed with **uv** (`pyproject.toml` + `uv.lock`).
- Behaviour is specified with **OpenSpec** under `openspec/`.
- No screen: the CLI (`voxoracle run|ask|doctor|setup`) is the operator surface.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Use lowercase built-in types (`dict`, `list`, `set`, etc.) instead of typing module classes (`Dict`, `List`, `Set`, etc.) for type annotations.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work")
require constant clarification.

When writing tests:
- Never use private or real user data as test cases. Use synthetic, anonymized fixtures instead.
- Tests must run **without audio hardware**; inject fakes for I/O.

## 5. Hardware constraints

- The Pi 3 has 1 GB RAM and a Cortex-A53 CPU. Keep the always-on local path
  (wake word) lightweight; STT/TTS are offloaded to the cloud.
- Run 64-bit (aarch64) Raspberry Pi OS: `onnxruntime` ships no 32-bit `armv7`
  wheels, so openWakeWord's ONNX backend needs 64-bit. Verify on real hardware
  before relying on a backend.

## 6. Conventional Commits

**Follow the Conventional Commits format.**

- Format: `type(scope): description` (e.g. `feat: add export button`, `fix(auth): handle expired tokens`).
- Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- Keep the subject line imperative, lowercase, and without a trailing period.
