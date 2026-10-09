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

### Requirement: Spoken errors return to IDLE

The session MUST handle no-speech, STT failure, and DocOracle unreachable/timeout
errors by speaking a short error prompt and returning to IDLE rather than
terminating.

#### Scenario: DocOracle is unreachable
- **WHEN** the ASKING step fails because DocOracle is unreachable or times out
- **THEN** the system speaks an error prompt and returns to IDLE

#### Scenario: Speech-to-text fails
- **WHEN** the TRANSCRIBING step fails
- **THEN** the system speaks an error prompt and returns to IDLE

#### Scenario: No speech captured
- **WHEN** recording ends without any detected speech
- **THEN** the system returns to IDLE without asking DocOracle

### Requirement: Barge-in stops playback

The session MUST stop speaking the current answer when the wake word or speech is
detected during SPEAKING, and MUST then treat the input as a new turn.

#### Scenario: Wake word during playback
- **WHEN** the wake word is detected while the answer is being spoken
- **THEN** playback stops and the session begins a new turn

### Requirement: Optional follow-up window

The session MUST support a configurable follow-up mode in which it briefly keeps
listening for a follow-up question after an answer without requiring the wake
word again.

#### Scenario: Follow-up mode enabled
- **WHEN** follow-up mode is enabled and a follow-up question is spoken within the window
- **THEN** the session records and answers it without a new wake-word activation

#### Scenario: Follow-up mode disabled
- **WHEN** follow-up mode is disabled
- **THEN** the session returns to IDLE after speaking the answer and requires the wake word again

### Requirement: Injected I/O for testability

The session MUST depend on injected audio, wake-word, STT, TTS, and DocOracle
collaborators so that a full conversation can be exercised with fakes and
without audio hardware.

#### Scenario: Session runs with fakes
- **WHEN** the session is constructed with fake collaborators
- **THEN** a complete simulated turn runs without accessing real devices or the network
