"""Smoke tests: the package imports and the CLI exposes its commands."""

from typer.testing import CliRunner

import voxoracle
from voxoracle.cli import app

runner = CliRunner()


def test_package_imports() -> None:
    assert voxoracle.__version__


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "ask", "doctor", "setup"):
        assert command in result.stdout


def test_cli_ask_stub_reports_question() -> None:
    result = runner.invoke(app, ["ask", "Wie funktioniert das?"])
    assert result.exit_code == 0
    assert "Wie funktioniert das?" in result.stdout
