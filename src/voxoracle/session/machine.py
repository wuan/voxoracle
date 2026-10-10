"""The voice-session state machine: wake -> record -> STT -> ask -> TTS.

All collaborators are injected (wake-word detector, audio input/VAD, transcriber,
DocOracle client, speech player), so a complete conversation can be exercised
with fakes and without audio hardware, network, or an API key.

Threading
---------
Every blocking device read (``AudioInput.read_frame``) runs in a worker thread via
``asyncio.to_thread`` so the event loop stays free to observe a stop request and
to run the wake-word watcher alongside playback. The barge-in path calls
``SpeechPlayer.interrupt`` from the event-loop thread only (``interrupt`` sets an
``asyncio.Event``, which is not thread-safe), scheduling the call with
``loop.call_soon_threadsafe``; this keeps ``interrupt`` safe even if a detector
implementation later feeds frames from its own thread.

The machine is deliberately a plain async class, not a task graph: ``run`` awaits
each state in order, and the only concurrency is the transient barge-in watcher
started around a single :meth:`SpeechPlayer.speak` call.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from voxoracle.audio.endpoint import EndpointingSettings, record_utterance
from voxoracle.audio.protocols import AudioInput, VoiceActivityDetector
from voxoracle.docoracle.client import DocOracleError
from voxoracle.docoracle.models import AskRequest, AskResponse
from voxoracle.mistral.errors import MistralError
from voxoracle.stt.protocol import AudioClip, Transcriber
from voxoracle.wakeword.detector import WakeWordDetector

_LOGGER = logging.getLogger(__name__)

#: Yield back to the event loop after a frame that produced no wake-word
#: trigger. A real device read already blocks for the frame's duration; this
#: keeps an instantaneous source (a test fake, or a device returning early) from
#: spinning the loop and spawning unbounded worker threads. Well below the 30 ms
#: frame time, so it does not affect detection latency.
_IDLE_POLL_SECONDS = 0.001


class SessionState(StrEnum):
    """The states the voice session transitions through."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ASKING = "asking"
    SPEAKING = "speaking"


@runtime_checkable
class QuestionAnswerer(Protocol):
    """Sends a question to DocOracle and returns the typed answer."""

    async def ask(self, request: AskRequest) -> AskResponse: ...


@runtime_checkable
class Speaker(Protocol):
    """Speaks text and can be interrupted mid-utterance (barge-in)."""

    async def speak(self, text: str, *, voice: str | None = None) -> None: ...

    def interrupt(self) -> None: ...

    @property
    def interrupted(self) -> bool: ...


@dataclass(frozen=True)
class Prompts:
    """Spoken messages for the non-answer paths (German first)."""

    no_speech: str = "Ich habe nichts gehört. Bitte versuche es noch einmal."
    stt_error: str = "Ich konnte die Frage leider nicht verstehen."
    docoracle_error: str = (
        "Die Wissensdatenbank ist gerade nicht erreichbar. Bitte versuche es später wieder."
    )
    tts_error: str = "Es tut mir leid, ich kann die Antwort gerade nicht vorlesen."


@dataclass(frozen=True)
class SessionConfig:
    """Tuning for :class:`VoiceSession`, independent of the settings schema."""

    sample_rate: int
    frame_samples: int
    max_record_seconds: float = 15.0
    endpoint_silence_ms: int = 700
    min_speech_ms: int = 200
    barge_in: bool = True
    follow_up: bool = False
    follow_up_seconds: float = 10.0
    language: str | None = None
    voice: str | None = None

    @property
    def follow_up_enabled(self) -> bool:
        """Follow-up listening is on only when requested and the window is positive."""
        return self.follow_up and self.follow_up_seconds > 0


class _After(StrEnum):
    """What the outer loop should do after one turn."""

    NEW_TURN = "new_turn"  # barge-in: start recording again immediately
    FOLLOW_UP = "follow_up"  # answer spoken: maybe open a follow-up window
    IDLE = "idle"  # error / no speech: watch for the wake word again


class VoiceSession:
    """Wires the wake word, audio, STT, DocOracle and TTS into one loop.

    Call :meth:`run` to start in IDLE. The loop alternates between watching for
    the wake word and running a turn (record -> transcribe -> ask -> speak).
    Known failures (no speech, STT error, DocOracle unreachable/timeout) speak a
    prompt and return to IDLE; an unexpected exception is logged and the loop
    continues. :meth:`request_stop` ends the loop after the current step.
    """

    def __init__(
        self,
        *,
        detector: WakeWordDetector,
        audio_input: AudioInput,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        client: QuestionAnswerer,
        player: Speaker,
        config: SessionConfig,
        prompts: Prompts | None = None,
        on_state: Callable[[SessionState], None] | None = None,
    ) -> None:
        self._detector = detector
        self._input = audio_input
        self._vad = vad
        self._transcriber = transcriber
        self._client = client
        self._player = player
        self._config = config
        self._prompts = prompts or Prompts()
        self._on_state = on_state
        self._state: SessionState | None = None
        self._stop = asyncio.Event()

    @property
    def state(self) -> SessionState | None:
        """The current state, or ``None`` before :meth:`run` starts."""
        return self._state

    def request_stop(self) -> None:
        """Ask :meth:`run` to stop after the current step (event-loop thread)."""
        self._stop.set()

    async def run(self) -> None:
        """Run the session until :meth:`request_stop` is called."""
        self._stop.clear()
        try:
            while not self._stop.is_set():
                self._set_state(SessionState.IDLE)
                self._detector.reset()
                if not await self._listen_for_wake():
                    break
                await self._run_turn_chain()
        finally:
            self._set_state(SessionState.IDLE)
        _LOGGER.info("voice session stopped")

    # -- turn handling --------------------------------------------------------

    async def _run_turn_chain(self) -> None:
        """Run a wake-triggered turn, then any barge-in/follow-up turns.

        An unexpected exception (a bug, or an untyped failure from a
        collaborator) is logged and ends the chain: the session returns to IDLE
        and keeps watching for the wake word rather than terminating.
        """
        max_seconds = self._config.max_record_seconds
        follow_up = False
        while not self._stop.is_set():
            try:
                after = await self._run_single_turn(max_seconds, follow_up=follow_up)
            except Exception:
                _LOGGER.exception("unexpected error during a voice turn; continuing")
                return
            if after is _After.NEW_TURN:
                max_seconds = self._config.max_record_seconds
                follow_up = False
                continue
            if after is _After.FOLLOW_UP and self._config.follow_up_enabled:
                max_seconds = self._config.follow_up_seconds
                follow_up = True
                continue
            break

    async def _run_single_turn(self, max_record_seconds: float, *, follow_up: bool) -> _After:
        """Record, transcribe, ask and speak one question.

        ``follow_up`` marks a window opened after a previous answer, where a
        silent window is normal and must not trigger a spoken prompt.
        """
        samples = await self._record(max_record_seconds)
        if samples is None:
            if not follow_up:
                await self._say(self._prompts.no_speech)
            return _After.IDLE

        try:
            question = await self._transcribe(samples)
        except MistralError as exc:
            _LOGGER.warning("speech-to-text failed: %s", exc)
            await self._say(self._prompts.stt_error)
            return _After.IDLE

        if not question.strip():
            await self._say(self._prompts.no_speech)
            return _After.IDLE

        try:
            answer = await self._ask(question)
        except DocOracleError as exc:
            _LOGGER.warning("DocOracle request failed: %s", exc)
            await self._say(self._prompts.docoracle_error)
            return _After.IDLE

        try:
            interrupted = await self._speak(answer)
        except MistralError as exc:
            _LOGGER.warning("text-to-speech failed: %s", exc)
            return _After.IDLE
        if interrupted:
            return _After.NEW_TURN
        return _After.FOLLOW_UP

    # -- states ---------------------------------------------------------------

    async def _listen_for_wake(self) -> bool:
        """Feed frames to the detector until it triggers or a stop is requested.

        Returns ``True`` when the wake word fired, ``False`` on a stop request.
        """
        while not self._stop.is_set():
            frame = await asyncio.to_thread(self._input.read_frame)
            if self._stop.is_set():
                return False
            if self._detector.process(frame):
                return True
            await asyncio.sleep(_IDLE_POLL_SECONDS)
        return False

    async def _record(self, max_seconds: float) -> NDArray[np.int16] | None:
        """Capture one utterance, or ``None`` when no speech was detected."""
        self._set_state(SessionState.RECORDING)
        settings = EndpointingSettings(
            sample_rate=self._config.sample_rate,
            frame_samples=self._config.frame_samples,
            max_seconds=max_seconds,
            endpoint_silence_ms=self._config.endpoint_silence_ms,
            min_speech_ms=self._config.min_speech_ms,
        )
        return await asyncio.to_thread(record_utterance, self._input, self._vad, settings)

    async def _transcribe(self, samples: NDArray[np.int16]) -> str:
        self._set_state(SessionState.TRANSCRIBING)
        clip = AudioClip(samples=samples, sample_rate=self._config.sample_rate)
        transcript = await self._transcriber.transcribe(clip, self._config.language)
        _LOGGER.info("heard: %s", transcript.text)
        return transcript.text

    async def _ask(self, question: str) -> str:
        self._set_state(SessionState.ASKING)
        response = await self._client.ask(AskRequest(question=question))
        return response.answer

    async def _say(self, text: str) -> None:
        """Speak a non-answer prompt, swallowing a TTS failure.

        An error prompt is best-effort: if the speaker itself fails, the loop
        must still return to IDLE rather than terminate.
        """
        try:
            await self._speak(text)
        except MistralError as exc:
            _LOGGER.warning("could not speak prompt: %s", exc)

    async def _speak(self, text: str) -> bool:
        """Speak ``text``; return ``True`` when interrupted by a wake word.

        A wake-word watcher runs alongside playback (when barge-in is enabled) so
        hearing the activation word cuts the answer short. The watcher signals
        from the event-loop thread, but still schedules ``interrupt`` with
        ``call_soon_threadsafe`` so the hook stays safe under any detector
        threading model.
        """
        if not text.strip():
            return False
        self._set_state(SessionState.SPEAKING)
        self._detector.reset()
        watcher: asyncio.Task[None] | None = None
        if self._config.barge_in:
            watcher = asyncio.create_task(self._watch_for_barge_in())
        speak = asyncio.create_task(self._player.speak(text, voice=self._config.voice))
        try:
            await speak
        finally:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
        return self._player.interrupted

    async def _watch_for_barge_in(self) -> None:
        """Feed frames to the detector until it fires, then interrupt playback."""
        while True:
            frame = await asyncio.to_thread(self._input.read_frame)
            if self._stop.is_set():
                return
            if self._detector.process(frame):
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self._player.interrupt)
                return
            await asyncio.sleep(_IDLE_POLL_SECONDS)

    def _set_state(self, state: SessionState) -> None:
        if state is self._state:
            return
        self._state = state
        _LOGGER.debug("session state -> %s", state)
        if self._on_state is not None:
            self._on_state(state)
