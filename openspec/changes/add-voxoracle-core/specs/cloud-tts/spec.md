## ADDED Requirements

### Requirement: Provider-agnostic text-to-speech protocol

The system SHALL define a text-to-speech protocol that accepts answer text and
produces mono 16-bit PCM audio for playback, so that cloud providers are
swappable without changing the session loop and the WP2 output-device layer can
consume the audio directly.

#### Scenario: Session depends on the protocol
- **WHEN** the session speaks an answer
- **THEN** it invokes the configured TTS implementation through the protocol

#### Scenario: Output is playable PCM
- **WHEN** the TTS implementation yields audio
- **THEN** it yields mono 16-bit PCM chunks with a sample rate the output layer can resample and write

### Requirement: Configurable cloud backend with a German voice

The TTS implementation MUST select its cloud provider, model, voice, and
language from configuration and MUST default to a German voice. The default
provider is Mistral, using the speech-synthesis endpoint
(``POST {base_url}/audio/speech``, model ``voxtral-mini-tts-2603``). Mistral
selects the voice by ``voice_id`` and has no separate language field, so the
configured language provides the default voice when no explicit voice is set.

#### Scenario: Provider, model and voice selected from configuration
- **WHEN** the configuration names a TTS provider, model and voice
- **THEN** synthesis requests are sent to that provider using that model and voice

#### Scenario: Default voice is German
- **WHEN** synthesis runs without an explicit voice
- **THEN** the configured language selects the German voice

#### Scenario: Language selects the default voice
- **WHEN** the configured language is set and no explicit voice is given
- **THEN** the request uses the language as the voice id

#### Scenario: Per-call voice override
- **WHEN** the caller passes a voice for a single utterance
- **THEN** that voice is used instead of the configured default

### Requirement: Shared Mistral authentication, timeout and errors

The Mistral TTS backend MUST reuse the shared Mistral HTTP client so
authentication, the configured timeout, bounded retries, and the typed-error
surface are identical to the STT backend. Credentials MUST resolve from
configuration or the environment (``mistral.api_key`` / ``VXORACLE_MISTRAL__API_KEY``,
then ``MISTRAL_API_KEY``, then ``LLM_API_KEY``); no key is committed.

#### Scenario: Credentials come from config or environment
- **WHEN** no credential is present in configuration
- **THEN** the implementation reads the credential from the environment

#### Scenario: One Mistral key covers STT and TTS
- **WHEN** only ``MISTRAL_API_KEY`` or the shared ``LLM_API_KEY`` is set
- **THEN** both the STT and TTS backends use it, so a single Mistral key serves VoxOracle and DocOracle

#### Scenario: Failures surface as typed errors
- **WHEN** synthesis fails (authentication, rate limit, timeout, server, or malformed response)
- **THEN** the backend raises the corresponding typed Mistral error

### Requirement: Streaming playback

The TTS implementation MUST stream synthesized audio to the speaker so that
playback begins before synthesis of the full answer finishes. The Mistral
backend requests a streaming response and consumes the provider's
``text/event-stream`` audio deltas as they arrive.

#### Scenario: Playback starts during synthesis
- **WHEN** the provider streams audio chunks
- **THEN** each chunk is written to the speaker before the next arrives

### Requirement: Barge-in stops playback

The playback layer MUST expose a stop hook that stops audio output promptly and
cancels the in-flight provider request when the session requests a barge-in,
without waiting for the provider to deliver another chunk, for the read timeout
to elapse, or for the currently playing audio chunk to finish.

#### Scenario: Playback is interrupted
- **WHEN** the session requests barge-in during playback
- **THEN** audio output stops, the provider stream is closed, and no further chunks are played

#### Scenario: A stalled provider does not delay barge-in
- **WHEN** barge-in is requested while the provider is quiet
- **THEN** playback stops and the in-flight request is cancelled promptly, without waiting for the read timeout

#### Scenario: A long chunk does not delay barge-in
- **WHEN** barge-in is requested while a long audio chunk (such as a whole WAV answer) is playing
- **THEN** the output stops within a bounded fraction of a second, not after the chunk finishes

#### Scenario: Interrupt hook is available to the session
- **WHEN** the wake-word detector fires during playback
- **THEN** it can call the playback layer's interrupt hook to stop the answer

### Requirement: Testable without a provider

The TTS implementation MUST be verifiable without network or audio hardware by
mocking the provider transport and playback sink.

#### Scenario: Mocked provider produces audio
- **WHEN** German answer text is synthesized with a mocked provider
- **THEN** audio is delivered to the playback sink and barge-in stops it
