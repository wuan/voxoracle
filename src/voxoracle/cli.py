"""Command-line interface for VoxOracle.

The CLI is intentionally thin: it exposes the operator surface while the actual
voice-session behaviour lives in the ``session``/``audio``/``stt``/``tts``/
``wakeword``/``docoracle`` packages. ``ask`` is implemented (WP1); the remaining
commands are stubs until their work package lands (see
``openspec/changes/add-voxoracle-core/tasks.md``).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from voxoracle.config import load_settings
from voxoracle.docoracle.client import DocOracleClient, DocOracleError
from voxoracle.docoracle.models import AskRequest

app = typer.Typer(
    name="voxoracle",
    help="Headless voice frontend for DocOracle (wake word -> STT -> /ask -> TTS).",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run() -> None:
    """Run the always-on voice session loop."""
    typer.echo("voxoracle run is not implemented yet (WP6).")


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


@app.command()
def doctor() -> None:
    """Check devices, models, configuration and DocOracle connectivity."""
    typer.echo("voxoracle doctor is not implemented yet (WP7).")


@app.command()
def setup() -> None:
    """Download wake-word models and prepare the device."""
    typer.echo("voxoracle setup is not implemented yet (WP7).")
