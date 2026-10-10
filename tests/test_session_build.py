"""Component wiring tests for `voxoracle run` (no hardware, no network)."""

from __future__ import annotations

import asyncio

import pytest

from voxoracle import cli
from voxoracle.config import Settings
from voxoracle.session.build import ConfigurationError, SessionComponents, build_session


def test_build_session_without_key_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError, match="Mistral API key"):
        build_session(Settings())


class _Closable:
    def __init__(self) -> None:
        self.closed = False
        self.aclosed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed = True


def test_session_components_aclose_closes_streams_and_clients() -> None:
    audio_in = _Closable()
    audio_out = _Closable()
    docoracle = _Closable()
    components = SessionComponents(
        session=None,  # type: ignore[arg-type]
        audio_input=audio_in,  # type: ignore[arg-type]
        audio_output=audio_out,  # type: ignore[arg-type]
        docoracle_client=docoracle,  # type: ignore[arg-type]
    )

    asyncio.run(components.aclose())

    assert audio_in.closed and audio_out.closed  # device streams first
    assert docoracle.aclosed


class _FakeSession:
    def __init__(self, *, run_error: Exception | None = None) -> None:
        self.stopped = False
        self.ran = False
        self._run_error = run_error

    def request_stop(self) -> None:
        self.stopped = True

    async def run(self) -> None:
        self.ran = True
        if self._run_error is not None:
            raise self._run_error


class _FakeComponents:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.aclosed = False

    async def aclose(self) -> None:
        self.aclosed = True


def test_serve_builds_runs_and_closes(monkeypatch) -> None:
    session = _FakeSession()
    components = _FakeComponents(session)
    monkeypatch.setattr(cli, "build_session", lambda settings: components)

    asyncio.run(cli._serve(Settings()))

    assert session.ran
    assert components.aclosed


def test_serve_closes_components_when_the_session_raises(monkeypatch) -> None:
    session = _FakeSession(run_error=RuntimeError("boom"))
    components = _FakeComponents(session)
    monkeypatch.setattr(cli, "build_session", lambda settings: components)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(cli._serve(Settings()))

    # Shutdown still runs: device streams and clients are closed on failure too.
    assert components.aclosed
