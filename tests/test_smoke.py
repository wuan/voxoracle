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


def test_cli_run_is_explicit_placeholder() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "not implemented yet" in result.stderr
    assert "WP6" in result.stderr


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


def test_cli_setup_creates_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "models" in result.stdout
    assert (tmp_path / "models").is_dir()
    assert "Next steps" in result.stdout


def test_cli_setup_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["setup"]).exit_code == 0
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "already present" in result.stdout


def test_cli_setup_reports_bare_name_expected_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert str(tmp_path / "models" / "franz.onnx") in result.stdout
    assert "franz.onnx.onnx" not in result.stdout


def test_cli_setup_handles_explicit_model_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "custom.onnx"
    (tmp_path / "config.yaml").write_text(
        f"wakeword:\n  model: {missing}\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    # Expects the explicit path as-is, not a doubled .onnx suffix.
    assert str(missing) in result.stdout
    assert "custom.onnx.onnx" not in result.stdout


def test_cli_doctor_exits_nonzero_without_docoracle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # No DocOracle reachable and no key configured -> hard-fail exit code.
    monkeypatch.setattr("voxoracle.cli._audio_backend", lambda settings: None)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "docoracle /health" in result.stdout
    assert "mistral key" in result.stdout
