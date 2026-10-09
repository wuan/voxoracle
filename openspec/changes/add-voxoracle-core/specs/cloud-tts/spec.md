## ADDED Requirements

### Requirement: Provider-agnostic text-to-speech protocol

The system SHALL define a text-to-speech protocol that accepts answer text and
produces audio for playback, so that cloud providers are swappable without
changing the session loop.

#### Scenario: Session depends on the protocol
- **WHEN** the session speaks an answer
- **THEN** it invokes the configured TTS implementation through the protocol

### Requirement: Configurable cloud backend with a German voice

The TTS implementation MUST select its cloud provider and voice from
configuration and MUST default to a German voice.

#### Scenario: Provider and voice selected from configuration
- **WHEN** the configuration names a TTS provider and voice
- **THEN** synthesis requests are sent to that provider using that voice

#### Scenario: Default voice is German
- **WHEN** synthesis runs without an explicit voice
- **THEN** a German voice with the configured language is used

### Requirement: Streaming playback

The TTS implementation MUST stream synthesized audio to the speaker so that
playback begins before synthesis of the full answer finishes.

#### Scenario: Playback starts during synthesis
- **WHEN** the provider streams audio chunks
- **THEN** the speaker begins playing before the last chunk arrives

### Requirement: Barge-in stops playback

The TTS implementation MUST stop playback promptly when the session requests a
barge-in.

#### Scenario: Playback is interrupted
- **WHEN** the session requests barge-in during playback
- **THEN** audio output stops and the provider request is cancelled

### Requirement: Testable without a provider

The TTS implementation MUST be verifiable without network or audio hardware by
mocking the provider transport and playback sink.

#### Scenario: Mocked provider produces audio
- **WHEN** German answer text is synthesized with a mocked provider
- **THEN** audio is delivered to the playback sink and barge-in stops it
