"""Voice-session state machine tests: full loop, errors, barge-in and follow-up.

Everything runs through injected fakes: no audio hardware, no network and no API
key. The fake audio input is a scripted queue of 16-bit mono frames; a frame
whose first sample is :data:`WAKE` is what the fake wake-word detector fires on.
"""

from __future__ import annotations

import asyncio

import numpy as np

from voxoracle.docoracle.client import DocOracleTimeoutError
from voxoracle.docoracle.models import AskRequest, AskResponse
from voxoracle.mistral.errors import MistralServerError
from voxoracle.session import Prompts, SessionConfig, SessionState, VoiceSession
from voxoracle.stt.protocol import AudioClip, Transcript

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30 ms at 16 kHz

#: First-sample marker the fake detector treats as a wake-word trigger.
WAKE = 999
#: First-sample marker for speech frames (the fake VAD classifies these as speech).
SPEECH = 100


def frame(value: int = 0) -> np.ndarray:
    data = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    data[0] = value
    return data


def speech_frames(count: int) -> list[np.ndarray]:
    return [frame(SPEECH) for _ in range(count)]


def silence_frames(count: int) -> list[np.ndarray]:
    return [frame(0) for _ in range(count)]


class FakeAudioInput:
    """A scripted frame queue; returns silence once the script is exhausted."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.reads = 0
        self.closed = False

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return FRAME_SAMPLES

    def read_frame(self) -> np.ndarray:
        self.reads += 1
        if not self._frames:
            return frame(0)
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    """Fires when it sees a :data:`WAKE`-marked frame."""

    def __init__(self) -> None:
        self.resets = 0
        self.processed = 0
        self.triggered = 0

    def process(self, frame_data: np.ndarray) -> bool:
        self.processed += 1
        if frame_data.size and int(frame_data[0]) == WAKE:
            self.triggered += 1
            return True
        return False

    def reset(self) -> None:
        self.resets += 1


class FakeVad:
    """Classifies a frame as speech when its first sample is non-zero."""

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        return bool(samples.size and int(samples[0]) != 0)


class FakeTranscriber:
    def __init__(
        self, text: str = "Wie funktioniert das?", *, error: Exception | None = None
    ) -> None:
        self._text = text
        self._error = error
        self.calls: list[AudioClip] = []

    async def transcribe(self, clip: AudioClip, language: str | None = None) -> Transcript:
        self.calls.append(clip)
        if self._error is not None:
            raise self._error
        return Transcript(text=self._text, language=language)


class FakeClient:
    def __init__(
        self, answer: str = "Es funktioniert so.", *, error: Exception | None = None
    ) -> None:
        self._answer = answer
        self._error = error
        self.questions: list[str] = []

    async def ask(self, request: AskRequest) -> AskResponse:
        self.questions.append(request.question)
        if self._error is not None:
            raise self._error
        return AskResponse(question=request.question, answer=self._answer)


class FakePlayer:
    """Records spoken text; can block until interrupted to emulate playback."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.interrupted = False
        self.speak_calls = 0
        self._block_next = False
        self._gate = asyncio.Event()

    def block_next_speak(self) -> None:
        self._block_next = True

    async def speak(self, text: str, *, voice: str | None = None) -> None:
        self.speak_calls += 1
        self.interrupted = False
        self.spoken.append(text)
        if self._block_next:
            self._block_next = False
            self._gate = asyncio.Event()
            await self._gate.wait()

    def interrupt(self) -> None:
        self.interrupted = True
        self._gate.set()


def build_session(
    audio: FakeAudioInput,
    *,
    detector: FakeDetector | None = None,
    transcriber: FakeTranscriber | None = None,
    client: FakeClient | None = None,
    player: FakePlayer | None = None,
    config: SessionConfig | None = None,
    barge_in: bool = True,
    on_state=None,
) -> tuple[VoiceSession, FakeDetector, FakeTranscriber, FakeClient, FakePlayer]:
    detector = detector or FakeDetector()
    transcriber = transcriber or FakeTranscriber()
    client = client or FakeClient()
    player = player or FakePlayer()
    session = VoiceSession(
        detector=detector,
        audio_input=audio,
        vad=FakeVad(),
        transcriber=transcriber,
        client=client,
        player=player,
        config=config
        or SessionConfig(sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES, barge_in=barge_in),
        on_state=on_state,
    )
    return session, detector, transcriber, client, player


def stop_after_first_idle(session: VoiceSession, states: list[SessionState]):
    """Return an on_state hook that stops the session after a full turn."""

    def hook(state: SessionState) -> None:
        states.append(state)
        if state is SessionState.IDLE and SessionState.SPEAKING in states:
            session.request_stop()

    return hook


def wake_then_question() -> list[np.ndarray]:
    """One wake frame followed by enough speech+silence to endpoint one turn.

    ``min_speech_ms=200`` needs 7 speech frames; ``endpoint_silence_ms=700`` then
    needs 23 silence frames, after which ``record_utterance`` stops reading.
    """
    return [frame(WAKE), *speech_frames(7), *silence_frames(23)]


def test_full_turn_transitions_and_speaks_answer() -> None:
    audio = FakeAudioInput(wake_then_question())
    states: list[SessionState] = []
    session, detector, transcriber, client, player = build_session(audio)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(session.run())

    assert states == [
        SessionState.IDLE,
        SessionState.RECORDING,
        SessionState.TRANSCRIBING,
        SessionState.ASKING,
        SessionState.SPEAKING,
        SessionState.IDLE,
    ]
    assert player.spoken == ["Es funktioniert so."]
    assert client.questions == ["Wie funktioniert das?"]
    assert len(transcriber.calls) == 1
    assert detector.triggered == 1
    # The detector is reset between utterances (at least once before the turn).
    assert detector.resets >= 1


def test_no_capture_before_activation() -> None:
    # No wake frame at all: the session must never transcribe or ask.
    audio = FakeAudioInput(silence_frames(5))
    session, _detector, transcriber, client, _player = build_session(audio)

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0.01)
        session.request_stop()
        await task

    asyncio.run(scenario())

    assert transcriber.calls == []
    assert client.questions == []
    assert session.state is SessionState.IDLE


def test_no_speech_speaks_prompt_and_returns_to_idle() -> None:
    # Wake, then only silence: record_utterance returns None.
    audio = FakeAudioInput([frame(WAKE), *silence_frames(60)])
    states: list[SessionState] = []
    session, _detector, transcriber, client, player = build_session(audio)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(session.run())

    assert player.spoken == [Prompts().no_speech]
    assert transcriber.calls == []
    assert client.questions == []
    assert states[-1] is SessionState.IDLE


def test_stt_failure_speaks_error_and_returns_to_idle() -> None:
    audio = FakeAudioInput(wake_then_question())
    transcriber = FakeTranscriber(error=MistralServerError(500, "boom"))
    states: list[SessionState] = []
    session, _detector, _t, client, player = build_session(audio, transcriber=transcriber)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(session.run())

    assert player.spoken == [Prompts().stt_error]
    assert client.questions == []
    assert states[-1] is SessionState.IDLE


def test_docoracle_timeout_speaks_error_and_returns_to_idle() -> None:
    audio = FakeAudioInput(wake_then_question())
    client = FakeClient(error=DocOracleTimeoutError("timed out"))
    states: list[SessionState] = []
    session, _detector, _t, _c, player = build_session(audio, client=client)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(session.run())

    assert player.spoken == [Prompts().docoracle_error]
    assert client.questions == ["Wie funktioniert das?"]
    assert states[-1] is SessionState.IDLE


def test_unexpected_exception_is_logged_and_loop_continues() -> None:
    # A non-typed failure must not crash the loop: the turn aborts, the session
    # returns to IDLE and keeps listening, so a later turn still completes.
    audio = FakeAudioInput([*wake_then_question(), *wake_then_question()])

    class FlakyClient:
        def __init__(self) -> None:
            self.questions: list[str] = []

        async def ask(self, request: AskRequest) -> AskResponse:
            self.questions.append(request.question)
            if len(self.questions) == 1:
                raise RuntimeError("unexpected")
            return AskResponse(question=request.question, answer="Antwort")

    client = FlakyClient()
    player = FakePlayer()
    session = VoiceSession(
        detector=FakeDetector(),
        audio_input=audio,
        vad=FakeVad(),
        transcriber=FakeTranscriber(),
        client=client,
        player=player,
        config=SessionConfig(sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES),
    )

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        while not player.spoken:
            await asyncio.sleep(0)
        session.request_stop()
        await asyncio.wait_for(task, timeout=2.0)

    # run() must not raise despite the unexpected RuntimeError.
    asyncio.run(scenario())

    assert player.spoken == ["Antwort"]
    assert client.questions == ["Wie funktioniert das?", "Wie funktioniert das?"]


def test_barge_in_during_speaking_starts_a_new_turn() -> None:
    # First turn: wake + question. During SPEAKING a second wake frame arrives,
    # which interrupts playback and begins a new turn (recorded right away).
    audio = FakeAudioInput(
        [
            *wake_then_question(),  # turn 1 (wake + question)
            frame(WAKE),  # barge-in during SPEAKING
            *speech_frames(7),
            *silence_frames(23),  # turn 2 question
        ]
    )
    player = FakePlayer()
    player.block_next_speak()  # turn 1 playback blocks until interrupted
    states: list[SessionState] = []
    session, detector, _t, client, player = build_session(audio, player=player)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert player.spoken == ["Es funktioniert so.", "Es funktioniert so."]
    assert client.questions == ["Wie funktioniert das?", "Wie funktioniert das?"]
    # Two wake triggers: the initial one and the barge-in one.
    assert detector.triggered == 2
    assert SessionState.RECORDING in states
    assert player.interrupted is False  # reset by the second, uninterrupted speak


def test_barge_in_disabled_ignores_wake_word_during_speaking() -> None:
    # With barge-in off the wake word during playback must not cut the answer.
    # Playback is instantaneous here; the extra wake frame and silence would be a
    # barge-in only if the flag were on.
    audio = FakeAudioInput([*wake_then_question(), frame(WAKE), *silence_frames(200)])
    player = FakePlayer()
    states: list[SessionState] = []
    session, _detector, _t, client, player = build_session(
        audio,
        player=player,
        config=SessionConfig(sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES, barge_in=False),
    )
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert player.spoken == ["Es funktioniert so."]
    assert player.speak_calls == 1
    assert client.questions == ["Wie funktioniert das?"]


def test_follow_up_enabled_answers_without_a_new_wake_word() -> None:
    audio = FakeAudioInput(
        [
            *wake_then_question(),  # turn 1 (wake + question)
            *speech_frames(7),
            *silence_frames(23),  # follow-up question, no wake frame
        ]
    )
    states: list[SessionState] = []
    session, detector, _t, client, _p = build_session(
        audio,
        # barge_in off keeps this test about follow-up only: the barge-in watcher
        # would otherwise race the follow-up's opening frames when playback ends
        # instantly (a real player lasts long enough that it does not).
        config=SessionConfig(
            sample_rate=SAMPLE_RATE,
            frame_samples=FRAME_SAMPLES,
            barge_in=False,
            follow_up=True,
            follow_up_seconds=10.0,
        ),
    )
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    # Both turns answered from a single wake-word activation.
    assert client.questions == ["Wie funktioniert das?", "Wie funktioniert das?"]
    assert detector.triggered == 1


def test_follow_up_disabled_requires_the_wake_word_again() -> None:
    # Same script as the enabled case, but follow-up is off: after the first
    # answer the session returns to IDLE and a follow-up is never recorded.
    audio = FakeAudioInput(
        [
            *wake_then_question(),
            *speech_frames(7),
            *silence_frames(23),
        ]
    )
    states: list[SessionState] = []
    session, detector, _t, client, _p = build_session(audio, barge_in=False)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert client.questions == ["Wie funktioniert das?"]
    assert detector.triggered == 1


def test_follow_up_seconds_zero_disables_the_window() -> None:
    # follow_up true but a zero window must behave as disabled (0=off).
    audio = FakeAudioInput([*wake_then_question(), *speech_frames(7), *silence_frames(23)])
    states: list[SessionState] = []
    session, _detector, _t, client, _p = build_session(
        audio,
        config=SessionConfig(
            sample_rate=SAMPLE_RATE,
            frame_samples=FRAME_SAMPLES,
            barge_in=False,
            follow_up=True,
            follow_up_seconds=0.0,
        ),
    )
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert client.questions == ["Wie funktioniert das?"]


def test_empty_transcript_speaks_no_speech_prompt() -> None:
    audio = FakeAudioInput(wake_then_question())
    transcriber = FakeTranscriber("   ")
    states: list[SessionState] = []
    session, _detector, _t, client, player = build_session(audio, transcriber=transcriber)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(session.run())

    assert player.spoken == [Prompts().no_speech]
    assert client.questions == []


class FailingPlayer(FakePlayer):
    """Raises a Mistral error on the first speak, to exercise the TTS guard."""

    def __init__(self, error: Exception, *, fail_on: int = 1) -> None:
        super().__init__()
        self._error = error
        self._fail_on = fail_on

    async def speak(self, text: str, *, voice: str | None = None) -> None:
        self.speak_calls += 1
        if self.speak_calls == self._fail_on:
            raise self._error
        await super().speak(text, voice=voice)


def test_tts_failure_on_answer_returns_to_idle() -> None:
    audio = FakeAudioInput(wake_then_question())
    # Fails only the first (answer) speak, so the apology prompt can play.
    player = FailingPlayer(MistralServerError(500, "tts down"), fail_on=1)
    states: list[SessionState] = []
    session, _detector, _t, client, _p = build_session(audio, player=player)
    session._on_state = stop_after_first_idle(session, states)

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert client.questions == ["Wie funktioniert das?"]
    # The answer failed, so the session speaks the TTS-error apology instead.
    assert player.spoken == [Prompts().tts_error]
    assert states[-1] is SessionState.IDLE


def test_say_swallows_tts_failure_on_error_prompt() -> None:
    # An error prompt is best-effort: if the speaker fails while saying it, the
    # session must still return to IDLE rather than terminate.
    audio = FakeAudioInput(wake_then_question())
    transcriber = FakeTranscriber(error=MistralServerError(500, "boom"))
    player = FailingPlayer(MistralServerError(500, "tts down"))
    session, _detector, _t, client, _p = build_session(
        audio, transcriber=transcriber, player=player
    )

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        while player.speak_calls < 1:
            await asyncio.sleep(0)
        session.request_stop()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())

    assert client.questions == []
    assert session.state is SessionState.IDLE


def test_speak_ignores_empty_text() -> None:
    audio = FakeAudioInput([])
    player = FakePlayer()
    session, _detector, _t, _c, _p = build_session(audio, player=player)

    async def scenario() -> None:
        await session._speak("   ")  # type: ignore[attr-defined]

    asyncio.run(scenario())

    assert player.speak_calls == 0


def test_barge_in_watcher_processes_frames_before_trigger() -> None:
    # Non-trigger frames must be fed (and the loop yielded) before the wake frame
    # so a barge-in lands only once the activation word is actually heard.
    audio = FakeAudioInput([*silence_frames(3), frame(WAKE)])
    player = FakePlayer()
    player.block_next_speak()
    session, detector, _t, _c, _p = build_session(audio, player=player)

    async def scenario() -> None:
        speak_task = asyncio.create_task(session._speak("Hallo"))  # type: ignore[attr-defined]
        await asyncio.wait_for(speak_task, timeout=2.0)
        assert player.spoken == ["Hallo"]

    asyncio.run(scenario())

    assert detector.triggered == 1
    assert player.interrupted is True


def test_request_stop_ends_the_watch_loop() -> None:
    audio = FakeAudioInput([])  # endless silence
    session, _detector, _t, _c, _p = build_session(audio)

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0.01)
        session.request_stop()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(scenario())

    assert session.state is SessionState.IDLE


def test_stop_during_recording_aborts_before_transcribing() -> None:
    # A stop requested while recording must end the turn at the next frame, not
    # run transcribe/ask/speak. The source yields the wake frame, then blocks so
    # the stop lands mid-record.
    import threading

    class BlockingInput(FakeAudioInput):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0
            self.release = threading.Event()

        def read_frame(self) -> np.ndarray:
            self.calls += 1
            if self.calls == 1:
                return frame(WAKE)  # trigger the wake word
            self.release.wait(timeout=5.0)  # block during recording
            return frame(SPEECH)

    audio = BlockingInput()
    session, _detector, transcriber, client, player = build_session(audio)

    def hook(state: SessionState) -> None:
        if state is SessionState.RECORDING:
            # Stop as soon as recording begins, then release the blocked read.
            session.request_stop()
            audio.release.set()

    session._on_state = hook

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert transcriber.calls == []  # never transcribed
    assert client.questions == []
    assert player.spoken == []
    assert session.state is SessionState.IDLE


def test_stop_between_steps_skips_the_answer() -> None:
    # A stop seen after transcription must skip the DocOracle request and speech.
    audio = FakeAudioInput(wake_then_question())
    session, _detector, transcriber, client, player = build_session(audio)

    def hook(state: SessionState) -> None:
        if state is SessionState.TRANSCRIBING:
            session.request_stop()

    session._on_state = hook

    asyncio.run(asyncio.wait_for(session.run(), timeout=2.0))

    assert len(transcriber.calls) == 1
    assert client.questions == []
    assert player.spoken == []


def test_stop_before_run_is_honored() -> None:
    # request_stop() before run() must not be lost: run() does not clear a
    # pre-set stop, so a session cancelled before it starts never records.
    audio = FakeAudioInput([frame(WAKE), *speech_frames(7), *silence_frames(23)])
    session, _detector, transcriber, _c, _p = build_session(audio)
    session.request_stop()

    asyncio.run(asyncio.wait_for(session.run(), timeout=1.0))

    assert transcriber.calls == []
    assert session.state is SessionState.IDLE


def test_session_can_be_rerun_after_a_stop() -> None:
    # A completed run resets the stop flag, so the same session runs again.
    audio = FakeAudioInput([*wake_then_question(), *wake_then_question()])
    session, _detector, _t, client, _p = build_session(audio)

    async def scenario() -> None:
        first = asyncio.create_task(session.run())
        await asyncio.sleep(0)
        session.request_stop()
        await asyncio.wait_for(first, timeout=1.0)
        assert not session._stop.is_set()  # reset for the next run  # type: ignore[attr-defined]
        second = asyncio.create_task(session.run())
        await asyncio.sleep(0)
        session.request_stop()
        await asyncio.wait_for(second, timeout=1.0)

    asyncio.run(scenario())

    assert session.state is SessionState.IDLE


def test_listen_for_wake_survives_a_device_read_failure() -> None:
    # A transient read failure during IDLE must not terminate the session; the
    # loop logs and retries, so the wake word is still heard afterwards.
    class FlakyInput(FakeAudioInput):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0

        def read_frame(self) -> np.ndarray:
            self.calls += 1
            if self.calls <= 2:
                raise OSError("usb-audio hiccup")
            return frame(WAKE)

    audio = FlakyInput()
    session, detector, _t, _c, _p = build_session(audio)

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        while detector.triggered == 0:
            await asyncio.sleep(0)
        session.request_stop()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())

    assert detector.triggered == 1
    assert session.state is SessionState.IDLE


def test_barge_in_awaits_the_in_flight_read_before_returning() -> None:
    # The barge-in watcher reads in a worker thread; when playback ends it must
    # finish that read before _speak returns, so no second reader touches the
    # capture stream (concurrent reads on one PortAudio stream are unsupported).
    import threading

    class SingleReaderInput(FakeAudioInput):
        def __init__(self) -> None:
            super().__init__([])
            self._lock = threading.Lock()
            self.concurrent = False
            self.in_read = 0
            self.first_started = threading.Event()
            self.release = threading.Event()

        def read_frame(self) -> np.ndarray:
            with self._lock:
                self.in_read += 1
                if self.in_read > 1:
                    self.concurrent = True
            try:
                first = not self.first_started.is_set()
                self.first_started.set()
                if first:
                    # Playback (and the watcher) start while this read blocks;
                    # it unblocks only when the test releases it.
                    self.release.wait(timeout=5.0)
                return frame(0)
            finally:
                with self._lock:
                    self.in_read -= 1

    audio = SingleReaderInput()
    player = FakePlayer()
    player.block_next_speak()
    session, _detector, _t, _c, _p = build_session(
        audio,
        player=player,
        config=SessionConfig(sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES),
    )
    in_flight_when_done: list[int] = []

    async def scenario() -> None:
        # The watcher's first read blocks; once it is in flight, finish playback
        # (via interrupt) so _speak's finally must await that read.
        speak = asyncio.create_task(session._speak("Hallo"))  # type: ignore[attr-defined]
        while not audio.first_started.is_set():
            await asyncio.sleep(0)
        player.interrupt()
        await asyncio.sleep(0)  # let speak wind down; the read is still blocked
        audio.release.set()
        await asyncio.wait_for(speak, timeout=2.0)
        # _speak must have waited for the in-flight read: no orphaned reader.
        in_flight_when_done.append(audio.in_read)

    asyncio.run(scenario())

    assert in_flight_when_done == [0]  # the read finished before _speak returned
    assert not audio.concurrent


def test_stop_during_speaking_cuts_playback_short() -> None:
    # A stop (Ctrl-C / systemd) while a long answer plays must interrupt playback
    # immediately instead of waiting for the whole answer, even with barge-in
    # disabled (no watcher running).
    audio = FakeAudioInput(wake_then_question())
    player = FakePlayer()
    player.block_next_speak()  # playback blocks until interrupted
    session, _detector, _t, _c, _p = build_session(audio, player=player, barge_in=False)

    async def scenario() -> None:
        task = asyncio.create_task(session.run())
        while not player.spoken:
            await asyncio.sleep(0)
        session.request_stop()
        # The stop must interrupt playback, so run() returns without the test
        # ever releasing the blocked speak.
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())

    assert player.speak_calls == 1
    assert player.interrupted  # playback was cut short
    assert session.state is SessionState.IDLE


def test_session_config_follow_up_enabled_flag() -> None:
    assert not SessionConfig(
        sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES, follow_up=False
    ).follow_up_enabled
    assert SessionConfig(
        sample_rate=SAMPLE_RATE, frame_samples=FRAME_SAMPLES, follow_up=True
    ).follow_up_enabled
    assert not SessionConfig(
        sample_rate=SAMPLE_RATE,
        frame_samples=FRAME_SAMPLES,
        follow_up=True,
        follow_up_seconds=0.0,
    ).follow_up_enabled
