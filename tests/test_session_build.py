"""Component wiring tests for `voxoracle run` (no hardware, no network)."""

from __future__ import annotations

import asyncio
import os
import signal

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
    def __init__(self, *, run_error: Exception | None = None, wait: bool = False) -> None:
        self.stopped = False
        self.ran = False
        self._run_error = run_error
        self._wait = wait
        self._release = asyncio.Event()

    def request_stop(self) -> None:
        self.stopped = True
        self._release.set()

    async def run(self) -> None:
        self.ran = True
        if self._run_error is not None:
            raise self._run_error
        if self._wait:
            await self._release.wait()


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


def test_serve_forwards_sigint_to_the_running_session(monkeypatch) -> None:
    # Regression: the handler must reach the session while it is running, so
    # Ctrl-C stops the loop rather than being observed by nothing (the default
    # SIGINT behaviour is replaced by add_signal_handler, so it cannot raise).
    session = _FakeSession(wait=True)
    components = _FakeComponents(session)
    monkeypatch.setattr(cli, "build_session", lambda settings: components)

    async def scenario() -> None:
        task = asyncio.create_task(cli._serve(Settings()))
        while not session.ran:
            await asyncio.sleep(0)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())

    assert session.stopped
    assert components.aclosed


def _patch_build_hardware(monkeypatch, *, input_stream, output_stream) -> None:
    """Swap the device backend and wake-word resolution for fakes."""

    class _FakeBackend:
        def __init__(self, settings: object) -> None:
            pass

        def open_configured_input(self) -> object:
            return input_stream

        def open_configured_output(self, sample_rate: int) -> object:
            return output_stream

    import voxoracle.audio.sounddevice_backend as backend_mod
    import voxoracle.wakeword.detector as detector_mod

    monkeypatch.setattr(backend_mod, "SoundDeviceBackend", _FakeBackend)
    monkeypatch.setattr(detector_mod, "resolve_model_path", lambda *a, **k: _raise_model())


def _raise_model() -> object:
    from voxoracle.wakeword.errors import WakeWordModelNotFoundError

    raise WakeWordModelNotFoundError("franz")


def test_build_session_closes_both_streams_when_a_later_step_fails(tmp_path, monkeypatch) -> None:
    # A failure after both streams are open (here: the wake-word model cannot be
    # resolved) must close the capture *and* playback streams, not just the input.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    audio_in = _Closable()
    audio_out = _Closable()
    _patch_build_hardware(monkeypatch, input_stream=audio_in, output_stream=audio_out)

    with pytest.raises(Exception, match="franz"):
        build_session(Settings())

    assert audio_in.closed and audio_out.closed
