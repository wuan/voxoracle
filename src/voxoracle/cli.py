"""Command-line interface for VoxOracle.

The CLI is intentionally thin: it exposes the operator surface while the actual
voice-session behaviour lives in the ``session``/``audio``/``stt``/``tts``/
``wakeword``/``docoracle`` packages. ``ask`` (WP1), ``doctor`` and ``setup``
(WP7) are implemented; ``run`` is a placeholder until WP6 lands (see
``openspec/changes/add-voxoracle-core/tasks.md``).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from voxoracle import diagnostics
from voxoracle.config import Settings, load_settings
from voxoracle.docoracle.client import DocOracleClient, DocOracleError
from voxoracle.docoracle.models import AskRequest

app = typer.Typer(
    name="voxoracle",
    help="Headless voice frontend for DocOracle (wake word -> STT -> /ask -> TTS).",
    no_args_is_help=True,
    add_completion=False,
)

_SEVERITY_MARK = {
    diagnostics.Severity.OK: "[ ok ]",
    diagnostics.Severity.WARN: "[warn]",
    diagnostics.Severity.FAIL: "[fail]",
}


@app.command()
def run() -> None:
    """Run the always-on voice session loop."""
    typer.echo(
        "error: `voxoracle run` is not implemented yet (WP6); "
        "use `voxoracle ask \"…\"` for text mode and `voxoracle doctor` to check the device.",
        err=True,
    )
    raise typer.Exit(code=1)


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

    results.append(diagnostics.check_wakeword(settings))
    results.extend(asyncio.run(diagnostics.check_docoracle(settings)))
    results.append(diagnostics.check_mistral_key(settings))

    for result in results:
        _render(result)

    if not diagnostics.all_checks_passed(results):
        raise typer.Exit(code=1)


@app.command()
def setup() -> None:
    """Prepare the device: create runtime directories and report model status."""
    settings = load_settings()
    typer.echo("Preparing VoxOracle directories...")
    for label, path, created in diagnostics.prepare_directories(settings):
        state = "created" if created else "already present"
        typer.echo(f"  {label}: {path} ({state})")

    models_dir = diagnostics.resolve_models_dir(settings)
    configured = settings.wakeword.model
    model_path = models_dir / f"{configured}.onnx"
    if model_path.is_file():
        typer.echo(f"  wake word: {model_path} present")
    else:
        typer.echo(
            f"  wake word: {model_path} missing — a placeholder model is used until "
            f"a trained '{configured}.onnx' is added"
        )

    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. copy config.example.yaml to config.yaml and set docoracle.url")
    typer.echo("  2. provide a Mistral key (MISTRAL_API_KEY or LLM_API_KEY)")
    typer.echo(f"  3. drop a trained wake-word model at {model_path}")
    typer.echo("  4. run `voxoracle doctor` to verify")
