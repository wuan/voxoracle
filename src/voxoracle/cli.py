"""Command-line interface for VoxOracle.

The CLI is intentionally thin: it exposes the operator surface while the actual
voice-session behaviour lives in the ``session``/``audio``/``stt``/``tts``/
``wakeword``/``docoracle`` packages. ``run`` (WP6), ``ask`` (WP1), ``doctor`` and
``setup`` (WP7) are implemented.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable
from typing import Annotated

import typer

from voxoracle import diagnostics
from voxoracle.config import Settings, load_settings
from voxoracle.docoracle.client import DocOracleClient, DocOracleError
from voxoracle.docoracle.models import AskRequest
from voxoracle.session.build import SessionComponents, build_session

app = typer.Typer(
    name="voxoracle",
    help="Headless voice frontend for DocOracle (wake word -> STT -> /ask -> TTS).",
    no_args_is_help=True,
    add_completion=False,
)

_LOGGER = logging.getLogger("voxoracle")

_SEVERITY_MARK = {
    diagnostics.Severity.OK: "[ ok ]",
    diagnostics.Severity.WARN: "[warn]",
    diagnostics.Severity.FAIL: "[fail]",
}


def _configure_logging(level: str) -> None:
    """Configure stdlib logging at ``level`` (default INFO), to stderr.

    ``logging.level`` is an unvalidated string in the schema, so an unknown value
    falls back to INFO with a warning rather than being silently ignored.
    """
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        typer.echo(f"warning: unknown logging level {level!r}; using INFO", err=True)
        resolved = logging.INFO
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _serve(components: SessionComponents) -> None:
    """Run the voice session until stopped, then shut the components down.

    ``SIGINT``/``SIGTERM`` are wired to a cooperative stop (rather than relying
    on the default handler that cancels the main task at its current await and
    would cut playback mid-sentence): the session is asked to stop, it unwinds at
    the next step or frame boundary, and :meth:`SessionComponents.aclose` closes
    the device streams and HTTP clients.
    """
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, components.session.request_stop)
        except NotImplementedError:  # pragma: no cover - not on this platform
            continue
        installed.append(sig)
    try:
        await components.session.run()
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
        await components.aclose()


def _run_session(settings: Settings) -> None:
    """Build the device components and run the session loop under Ctrl-C."""
    components = build_session(settings)
    asyncio.run(_serve(components))


@app.command()
def run() -> None:
    """Run the always-on voice session loop."""
    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - surface any config error, like `setup`
        typer.echo(f"error: invalid configuration: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _configure_logging(settings.logging.level)
    _LOGGER.info("starting voice session (Ctrl-C to stop)")

    try:
        _run_session(settings)
    except Exception as exc:  # noqa: BLE001 - report startup failures clearly
        _LOGGER.error("voice session could not start: %s", exc)
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to send to DocOracle.")],
) -> None:
    """Ask DocOracle a question in text mode (no audio)."""

    async def _ask() -> str:
        settings = load_settings()
        client = DocOracleClient(
            base_url=settings.docoracle.url,
            timeout=settings.docoracle.timeout,
        )
        try:
            response = await client.ask(AskRequest(question=question))
        finally:
            await client.aclose()
        return response.answer

    try:
        answer = asyncio.run(_ask())
    except DocOracleError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(answer)


def _render(result: diagnostics.CheckResult) -> None:
    typer.echo(f"{_SEVERITY_MARK[result.severity]} {result.name}: {result.detail}")
    for hint in result.hints:
        typer.echo(f"         -> {hint}")


def _safe(name: str, check: Callable[[], diagnostics.CheckResult]) -> diagnostics.CheckResult:
    """Run a check, turning any unexpected error into a FAIL result.

    A broken dependency (e.g. an unimportable openWakeWord) must be reported by
    ``doctor`` as a failed check, not escape as a traceback that would skip the
    remaining checks.
    """
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 - a broken check becomes a FAIL result
        return diagnostics.CheckResult(
            name, diagnostics.Severity.FAIL, f"{type(exc).__name__}: {exc}"
        )


def _safe_all(
    name: str, check: Callable[[], list[diagnostics.CheckResult]]
) -> list[diagnostics.CheckResult]:
    """Run a multi-result check, turning any unexpected error into one FAIL result.

    Same intent as :func:`_safe`, for checks that return several results (e.g.
    the DocOracle ``/health`` + ``/info`` probe). ``check_docoracle`` only
    guards the request phase, so an invalid ``docoracle.url`` raises inside the
    client constructor; that must not escape and skip the remaining checks.
    """
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 - a broken check becomes a FAIL result
        return [
            diagnostics.CheckResult(name, diagnostics.Severity.FAIL, f"{type(exc).__name__}: {exc}")
        ]


def _audio_backend(settings: Settings) -> diagnostics.AudioBackend:
    """Build the real PortAudio backend, imported lazily so CI stays hardware-free."""
    from voxoracle.audio.sounddevice_backend import SoundDeviceBackend

    return SoundDeviceBackend(settings.audio)


@app.command()
def doctor() -> None:
    """Check configuration, devices, models, DocOracle connectivity and API keys."""
    results: list[diagnostics.CheckResult] = []
    try:
        settings = diagnostics.default_settings()
    except Exception as exc:  # noqa: BLE001 - surface any config error as a failed check
        _render(diagnostics.CheckResult("settings", diagnostics.Severity.FAIL, str(exc)))
        raise typer.Exit(code=1) from exc

    results.append(diagnostics.check_settings(settings))

    try:
        audio_report = diagnostics.enumerate_audio(settings, _audio_backend(settings))
    except Exception as exc:  # noqa: BLE001 - importing sounddevice may fail without PortAudio
        audio_report = diagnostics.AudioDeviceReport(error=f"{type(exc).__name__}: {exc}")
    results.extend(diagnostics.check_audio(settings, audio_report))

    results.append(_safe("wake word", lambda: diagnostics.check_wakeword(settings)))
    results.extend(
        _safe_all("docoracle", lambda: asyncio.run(diagnostics.check_docoracle(settings)))
    )
    results.append(_safe("mistral key", lambda: diagnostics.check_mistral_key(settings)))

    for result in results:
        _render(result)

    if not diagnostics.all_checks_passed(results):
        raise typer.Exit(code=1)


@app.command()
def setup() -> None:
    """Prepare the device: create runtime directories and report model status."""
    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - report config errors like doctor does
        typer.echo(f"error: invalid configuration: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Preparing VoxOracle directories...")
    for label, path, created in diagnostics.prepare_directories(settings):
        state = "created" if created else "already present"
        typer.echo(f"  {label}: {path} ({state})")

    # Report the wake-word status with the same logic `doctor` uses, so setup
    # and doctor agree for both bare names and explicit .onnx paths. Guarded the
    # same way as doctor: a broken openWakeWord install must render a [fail]
    # line, not crash setup with a traceback.
    model = _safe("wake word", lambda: diagnostics.check_wakeword(settings))
    typer.echo(f"  {_SEVERITY_MARK[model.severity]} wake word: {model.detail}")

    model_path = diagnostics.expected_model_path(settings)
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. copy config.example.yaml to config.yaml and set docoracle.url")
    typer.echo("  2. provide a Mistral key (MISTRAL_API_KEY or LLM_API_KEY)")
    typer.echo(f"  3. ensure a trained wake-word model at {model_path}")
    typer.echo("  4. run `voxoracle doctor` to verify")
