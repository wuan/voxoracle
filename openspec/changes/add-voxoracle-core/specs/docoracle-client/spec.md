## ADDED Requirements

### Requirement: Typed client for DocOracle

The system SHALL provide a typed HTTP client for the DocOracle API exposing
`ask`, `health`, and `info`. The client MUST be the only component that knows
DocOracle's transport details, so the rest of VoxOracle depends on its typed
interface.

#### Scenario: Ask returns typed models
- **WHEN** `ask` is called with a question
- **THEN** it returns a typed response containing `answer`, `confidence`, `citations`, and retrieved sources

#### Scenario: Health and info are available
- **WHEN** `health` or `info` is called
- **THEN** the corresponding DocOracle endpoint is queried and its result returned typed

### Requirement: Ask request fields

The `ask` request MUST send `question` and MUST support the optional fields
`module`, `component`, `version`, `k`, `retrieval` (`hybrid` | `semantic` |
`bm25`), `show_sources`, and `show_context`, omitting those that are unset.

#### Scenario: Optional fields are omitted when unset
- **WHEN** `ask` is called with only a question
- **THEN** the request body contains the question and none of the optional fields

#### Scenario: Filters are forwarded
- **WHEN** `ask` is called with a `module` filter and a retrieval mode
- **THEN** the request body includes those values

### Requirement: Configurable endpoint and timeout

The client MUST read the DocOracle base URL and timeout from configuration and
MUST apply the timeout to every request.

#### Scenario: Base URL from configuration
- **WHEN** the configured base URL is `http://localhost:8000`
- **THEN** requests are sent to that base URL

#### Scenario: Timeout is enforced
- **WHEN** DocOracle does not answer within the configured timeout
- **THEN** the request fails with a typed timeout error

### Requirement: Bounded retries and typed errors

The client MUST retry transient failures a bounded number of times and MUST
surface typed errors for unreachable, timeout, and malformed-response failures
instead of returning untyped exceptions.

#### Scenario: Unreachable server
- **WHEN** DocOracle is unreachable
- **THEN** `ask` raises a typed connection error after bounded retries

#### Scenario: Malformed response
- **WHEN** DocOracle returns a body that does not match the response model
- **THEN** the client raises a typed error naming the validation failure

### Requirement: Text-mode CLI bridge

The `voxoracle ask "…"` command MUST ask DocOracle and print the answer as text,
proving the integration without audio.

#### Scenario: Answer is printed
- **WHEN** `voxoracle ask "Wie funktioniert das?"` is run against a reachable DocOracle
- **THEN** the grounded answer text is printed

#### Scenario: Failure is reported
- **WHEN** DocOracle is unreachable or times out
- **THEN** the command prints an error and exits with a non-zero status
