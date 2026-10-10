## ADDED Requirements

### Requirement: Wake-to-answer session loop

The system SHALL run a continuous voice session that transitions through IDLE,
RECORDING, TRANSCRIBING, ASKING, and SPEAKING and returns to IDLE after each
answer. The loop MUST start in IDLE and MUST NOT record audio before the wake
word is detected.

#### Scenario: Full turn completes
- **WHEN** the wake word is detected and a question is spoken
- **THEN** the system records the question, transcribes it, asks DocOracle, speaks the answer, and returns to IDLE

#### Scenario: No capture before activation
- **WHEN** the session is in IDLE and no wake word has been detected
- **THEN** the system does not send audio to speech-to-text or DocOracle

#### Scenario: Recording is bounded
- **WHEN** recording is in progress
- **THEN** it ends at `audio.endpoint_silence_ms` of silence after speech, or at `session.max_record_seconds`, whichever comes first

### Requirement: Spoken errors return to IDLE

The session MUST handle no-speech, STT failure, and DocOracle unreachable/timeout
errors by speaking a short error prompt and returning to IDLE rather than
terminating. The typed Mistral errors raised by STT/TTS and the typed
`DocOracleError` raised by the client MUST both map to spoken messages.

#### Scenario: DocOracle is unreachable
- **WHEN** the ASKING step fails because DocOracle is unreachable or times out
- **THEN** the system speaks an error prompt and returns to IDLE

#### Scenario: Speech-to-text fails
- **WHEN** the TRANSCRIBING step fails with a typed Mistral error
- **THEN** the system speaks an error prompt and returns to IDLE

#### Scenario: Text-to-speech fails
- **WHEN** the SPEAKING step fails with a typed Mistral error while reading the answer
- **THEN** the system attempts a spoken error prompt (best effort) and returns to IDLE

#### Scenario: No speech captured
- **WHEN** recording ends without any detected speech
- **THEN** the system speaks a prompt and returns to IDLE without asking DocOracle

#### Scenario: Unexpected failure does not kill the loop
- **WHEN** a turn raises an unexpected (untyped) exception
- **THEN** the system logs it, returns to IDLE, and continues watching for the wake word

### Requirement: Barge-in stops playback

The session MUST stop speaking the current answer when the wake word is detected
during SPEAKING, and MUST then treat the input as a new turn (recording the next
question without requiring a further activation). Barge-in MUST be disabled when
`session.barge_in` is false.

#### Scenario: Wake word during playback
- **WHEN** the wake word is detected while the answer is being spoken
- **THEN** playback stops and the session begins a new turn

#### Scenario: Single reader on the capture stream
- **WHEN** a barge-in ends playback
- **THEN** the in-flight detector read is awaited before recording or listening resumes, so only one thread ever reads the capture stream (frames read during playback are discarded, so speech begun before the wake word completes may be partly lost)

#### Scenario: Interrupt is scheduled on the event loop
- **WHEN** the wake-word watcher fires during playback
- **THEN** `SpeechPlayer.interrupt` is invoked from the event-loop thread (scheduled with `call_soon_threadsafe`), since the interrupt hook is an `asyncio.Event` and is not thread-safe

#### Scenario: Barge-in disabled
- **WHEN** `session.barge_in` is false and the wake word is heard during playback
- **THEN** playback continues to the end of the answer and the session returns to IDLE

### Requirement: Optional follow-up window

The session MUST support a configurable follow-up mode in which it briefly keeps
listening for a follow-up question after an answer without requiring the wake
word again. It is enabled when `session.follow_up` is true and
`session.follow_up_seconds` is greater than zero; `follow_up_seconds` bounds how
long the window stays open, and a value of 0 disables it.

#### Scenario: Follow-up mode enabled
- **WHEN** follow-up mode is enabled and a follow-up question is spoken within the window
- **THEN** the session records and answers it without a new wake-word activation

#### Scenario: Follow-up mode disabled
- **WHEN** follow-up mode is disabled
- **THEN** the session returns to IDLE after speaking the answer and requires the wake word again

#### Scenario: Follow-up window elapses silently
- **WHEN** the follow-up window elapses without speech
- **THEN** the session returns to IDLE without speaking a prompt

### Requirement: Injected I/O for testability

The session MUST depend on injected audio, wake-word, STT, TTS, and DocOracle
collaborators so that a full conversation can be exercised with fakes and
without audio hardware, network, or an API key. Blocking device reads MUST run
off the event loop (a worker thread) so the wake-word watcher and stop requests
are not starved.

#### Scenario: Session runs with fakes
- **WHEN** the session is constructed with fake collaborators
- **THEN** a complete simulated turn runs without accessing real devices or the network

#### Scenario: Graceful shutdown
- **WHEN** a stop is requested (e.g. Ctrl-C, wired to `request_stop` via a signal handler)
- **THEN** the loop ends at the next step or frame boundary (recording is aborted promptly rather than waiting out `session.max_record_seconds`), the device streams are closed, and the HTTP clients are closed

#### Scenario: Stop before the loop starts
- **WHEN** a stop is requested before the loop is started
- **THEN** the loop exits immediately without recording, and the stop flag is reset when the run ends so the session can be run again
