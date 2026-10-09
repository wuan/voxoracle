"""Smoke tests: the package imports and the CLI exposes its commands."""

from typer.testing import CliRunner

import voxoracle
from voxoracle.cli import app
from voxoracle.docoracle.client import DocOracleConnectionError
from voxoracle.docoracle.models import AskResponse

runner = CliRunner()


def test_package_imports() -> None:
    assert voxoracle.__version__


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "ask", "doctor", "setup"):
        assert command in result.stdout


def test_cli_ask_prints_answer(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def ask(self, request: object) -> AskResponse:
            return AskResponse(question="q", answer="Die Antwort.", sources=[], source_details=[])

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("voxoracle.cli.DocOracleClient", FakeClient)
    result = runner.invoke(app, ["ask", "Wie funktioniert das?"])
    assert result.exit_code == 0
    assert "Die Antwort." in result.stdout


def test_cli_ask_reports_errors(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def ask(self, request: object) -> object:
            raise DocOracleConnectionError("cannot reach DocOracle")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("voxoracle.cli.DocOracleClient", FailingClient)
    result = runner.invoke(app, ["ask", "Wie funktioniert das?"])
    assert result.exit_code == 1
    assert "cannot reach DocOracle" in result.stderr
