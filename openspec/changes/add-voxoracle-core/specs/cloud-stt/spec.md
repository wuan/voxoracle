## ADDED Requirements

### Requirement: Provider-agnostic speech-to-text protocol

The system SHALL define a speech-to-text protocol that accepts captured audio
and returns a transcript, so that cloud providers are swappable without changing
the session loop.

#### Scenario: Session depends on the protocol
- **WHEN** the session performs transcription
- **THEN** it invokes the configured STT implementation through the protocol

### Requirement: Configurable cloud backend

The STT implementation MUST select its cloud provider from configuration,
including the provider, endpoint, credentials, and language. The default
provider is Mistral, using the Voxtral offline transcription endpoint
(``POST {base_url}/audio/transcriptions``, model ``voxtral-mini-latest``).

#### Scenario: Provider selected from configuration
- **WHEN** the configuration names an STT provider and its credentials
- **THEN** transcription requests are sent to that provider

#### Scenario: Credentials come from config or environment
- **WHEN** no credential is present in configuration
- **THEN** the implementation reads the credential from the environment

#### Scenario: One Mistral key covers STT and TTS
- **WHEN** only ``MISTRAL_API_KEY`` or the shared ``LLM_API_KEY`` is set
- **THEN** the STT backend uses it, so a single Mistral key serves VoxOracle and DocOracle

### Requirement: German first

The STT implementation MUST default to German (`de`) and MUST send the
configured language to the provider.

#### Scenario: Default language is German
- **WHEN** transcription runs without an explicit language
- **THEN** the request specifies German

### Requirement: Timeouts, retries and typed errors

The STT implementation MUST enforce a configurable timeout and MUST retry
transient failures (timeouts, connection errors, HTTP 429 and 5xx) a bounded
number of times, surfacing typed errors (authentication, rate-limit,
server, malformed) to the session on final failure. The Mistral HTTP client is
shared with the TTS backend so both reuse the same authentication, retry and
error behaviour.

#### Scenario: Timeout is enforced
- **WHEN** the provider does not respond within the configured timeout
- **THEN** transcription fails with a typed timeout error

#### Scenario: Transient failure is retried
- **WHEN** the provider returns a transient error
- **THEN** the request is retried up to the configured limit before failing

#### Scenario: Authentication failure is reported
- **WHEN** the provider rejects the credentials
- **THEN** transcription fails with a typed authentication error and is not retried

### Requirement: Testable without a provider

The STT implementation MUST be verifiable without network access by mocking the
provider transport.

#### Scenario: Mocked provider returns a transcript
- **WHEN** a German fixture utterance is transcribed with a mocked provider
- **THEN** the expected transcript is returned
