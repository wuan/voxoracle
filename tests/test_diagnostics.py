"""Doctor/setup diagnostics: settings, audio, wake word, DocOracle and keys.

All hardware, network and credentials are faked so these tests run in CI.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from voxoracle import diagnostics
from voxoracle.audio.protocols import DeviceInfo
from voxoracle.config import mistral_api_key_source, resolve_models_dir
from voxoracle.diagnostics import Severity
from voxoracle.docoracle.client import DocOracleClient


class FakeBackend:
    """Audio backend returning configured device lists."""

    def __init__(self, inputs=None, outputs=None, error: Exception | None = None) -> None:
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._error = error

    def list_devices(self, kind):
        if self._error is not None:
            raise self._error
        return self._inputs if kind == "input" else self._outputs


# --- settings ---------------------------------------------------------------


def test_check_settings_reports_urls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = diagnostics.check_settings(diagnostics.default_settings())
    assert result.severity is Severity.OK
    assert "localhost:8000" in result.detail


# --- audio ------------------------------------------------------------------


def test_enumerate_audio_captures_portaudio_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from voxoracle.audio.errors import AudioError

    settings = diagnostics.default_settings()
    report = diagnostics.enumerate_audio(settings, FakeBackend(error=AudioError("no PortAudio")))
    assert report.error is not None
    results = diagnostics.check_audio(settings, report)
    assert results[0].severity is Severity.FAIL
    assert "PortAudio" in results[0].detail
    assert any("libportaudio2" in hint for hint in results[0].hints)


def test_enumerate_audio_captures_native_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()
    report = diagnostics.enumerate_audio(settings, FakeBackend(error=RuntimeError("boom")))
    assert report.error is not None and "RuntimeError" in report.error


def test_check_audio_fails_when_no_devices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()
    report = diagnostics.enumerate_audio(settings, FakeBackend())
    results = diagnostics.check_audio(settings, report)
    assert all(r.severity is Severity.FAIL for r in results)


def test_check_audio_default_device_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()
    mic = DeviceInfo(1, "CD04 USB Audio", "input", 1, 0, 32000.0, is_default=True)
    spk = DeviceInfo(0, "bcm2835 Headphones", "output", 0, 1, 48000.0, is_default=True)
    report = diagnostics.enumerate_audio(settings, FakeBackend([mic], [spk]))
    results = diagnostics.check_audio(settings, report)
    assert [r.severity for r in results] == [Severity.OK, Severity.OK]
    assert "CD04" in results[0].detail and "bcm2835" in results[1].detail


def test_check_audio_configured_device_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "audio:\n  input_device: Missing Mic\n", encoding="utf-8"
    )
    settings = diagnostics.load_settings("config.yaml")
    mic = DeviceInfo(1, "CD04 USB Audio", "input", 1, 0, 32000.0, is_default=True)
    report = diagnostics.enumerate_audio(settings, FakeBackend([mic], [mic]))
    results = diagnostics.check_audio(settings, report)
    assert results[0].severity is Severity.FAIL
    assert "Missing Mic" in results[0].detail


def test_check_audio_configured_device_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "audio:\n  input_device: CD04\n  output_device: Headphones\n", encoding="utf-8"
    )
    settings = diagnostics.load_settings("config.yaml")
    mic = DeviceInfo(1, "CD04 USB Audio", "input", 1, 0, 32000.0, is_default=True)
    spk = DeviceInfo(0, "bcm2835 Headphones", "output", 0, 1, 48000.0, is_default=False)
    report = diagnostics.enumerate_audio(settings, FakeBackend([mic], [spk]))
    results = diagnostics.check_audio(settings, report)
    assert all(r.severity is Severity.OK for r in results)


# --- wake word --------------------------------------------------------------


def test_check_wakeword_falls_back_to_placeholder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()
    result = diagnostics.check_wakeword(settings)
    # No franz.onnx present and openWakeWord may not be importable: either the
    # placeholder warns, or the model is unresolvable (fail) with a setup hint.
    assert result.severity in (Severity.WARN, Severity.FAIL)
    if result.severity is Severity.WARN:
        assert "placeholder" in result.detail


def test_check_wakeword_finds_configured_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "franz.onnx").write_bytes(b"stub")
    settings = diagnostics.default_settings()
    result = diagnostics.check_wakeword(settings)
    assert result.severity is Severity.OK
    assert "franz.onnx" in result.detail


def test_resolve_models_dir_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()
    resolved = resolve_models_dir(settings)
    assert resolved.is_absolute()
    assert resolved.name == "models"


# --- DocOracle --------------------------------------------------------------


def test_check_docoracle_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={
                "total_chunks": 42,
                "semantic_chunks": 40,
                "bm25_chunks": 42,
                "retrieval_mode": "hybrid",
                "store_path": "/tmp/store",
                "modules": {"api": 1},
                "components": {},
                "modules_by_component": {},
            },
        )

    real_client = DocOracleClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("voxoracle.docoracle.client.DocOracleClient", factory)
    import asyncio

    results = asyncio.run(diagnostics.check_docoracle(settings))
    assert [r.severity for r in results] == [Severity.OK, Severity.OK]
    assert "ok" in results[0].detail and "42 chunks" in results[1].detail


def test_check_docoracle_health_unreachable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = diagnostics.default_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    real_client = DocOracleClient
    monkeypatch.setattr(
        "voxoracle.docoracle.client.DocOracleClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    import asyncio

    results = asyncio.run(diagnostics.check_docoracle(settings))
    assert results[0].severity is Severity.FAIL
    assert results[1].severity is Severity.WARN


# --- Mistral key ------------------------------------------------------------


def test_check_mistral_key_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = diagnostics.check_mistral_key(diagnostics.default_settings())
    assert result.severity is Severity.FAIL
    assert result.detail == "missing"


def test_check_mistral_key_found_never_prints_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MISTRAL_API_KEY", "super-secret-value")
    result = diagnostics.check_mistral_key(diagnostics.default_settings())
    assert result.severity is Severity.OK
    assert "super-secret-value" not in result.detail
    assert "MISTRAL_API_KEY" in result.detail


def test_mistral_api_key_source_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert mistral_api_key_source(diagnostics.default_settings()) is None
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert mistral_api_key_source(diagnostics.default_settings()) == "LLM_API_KEY (env)"


# --- exit code / setup ------------------------------------------------------


def test_all_checks_passed():
    ok = diagnostics.CheckResult("a", Severity.OK, "")
    warn = diagnostics.CheckResult("b", Severity.WARN, "")
    fail = diagnostics.CheckResult("c", Severity.FAIL, "")
    assert diagnostics.all_checks_passed([ok, warn])
    assert not diagnostics.all_checks_passed([ok, fail])


def test_prepare_directories_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = diagnostics.default_settings()
    first = diagnostics.prepare_directories(settings)
    assert all(created for _, _, created in first)
    assert all(path.is_dir() for _, path, _ in first)
    second = diagnostics.prepare_directories(settings)
    assert not any(created for _, _, created in second)
    assert {label for label, _, _ in first} == {"models_dir", "cache"}


def test_prepare_directories_uses_configured_models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"wakeword:\n  models_dir: {tmp_path / 'custom-models'}\n", encoding="utf-8"
    )
    settings = diagnostics.load_settings("config.yaml")
    outcomes = diagnostics.prepare_directories(settings)
    labels = {label: path for label, path, _ in outcomes}
    assert labels["models_dir"] == Path(tmp_path / "custom-models")
