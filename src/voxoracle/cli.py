"""Command-line interface for VoxOracle.

The CLI is intentionally thin: it exposes the operator surface while the actual
voice-session behaviour lives in the ``session``/``audio``/``stt``/``tts``/
``wakeword``/``docoracle`` packages. Commands are stubs until their work package
lands (see ``openspec/changes/add-voxoracle-core/tasks.md``).
"""

from __future__ import annotations

from typing import Annotated

import typer

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
    typer.echo(f"voxoracle ask is not implemented yet (WP1): {question!r}")


@app.command()
def doctor() -> None:
    """Check devices, models, configuration and DocOracle connectivity."""
    typer.echo("voxoracle doctor is not implemented yet (WP7).")


@app.command()
def setup() -> None:
    """Download wake-word models and prepare the device."""
    typer.echo("voxoracle setup is not implemented yet (WP7).")
