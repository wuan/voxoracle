## ADDED Requirements

### Requirement: Operator CLI

The system SHALL expose a `voxoracle` command with `run`, `ask`, `doctor`, and
`setup` subcommands. Because the device is headless, the CLI MUST be the operator
surface and each command MUST have self-documenting help.

#### Scenario: Help lists the commands
- **WHEN** `voxoracle --help` is run
- **THEN** `run`, `ask`, `doctor`, and `setup` are listed

#### Scenario: Run starts the session
- **WHEN** `voxoracle run` is invoked on the device
- **THEN** the always-on voice session loop starts

### Requirement: Configuration schema and loading

The system MUST load settings from `config.yaml` (and environment variables) into
a validated settings object, using `config.example.yaml` as the documented shape.

#### Scenario: Defaults are applied
- **WHEN** `config.yaml` omits a setting
- **THEN** the documented default is used

#### Scenario: Invalid configuration is rejected
- **WHEN** `config.yaml` contains an invalid value
- **THEN** loading fails with an error identifying the invalid setting

### Requirement: Doctor diagnostics

The `voxoracle doctor` command MUST report the status of audio devices, wake-word
models, configuration, and DocOracle connectivity without starting the session.

#### Scenario: Doctor reports status
- **WHEN** `voxoracle doctor` is run
- **THEN** it reports device, model, configuration, and DocOracle connectivity status

#### Scenario: Missing model is reported
- **WHEN** the wake-word model is not present
- **THEN** `voxoracle doctor` reports it as missing and suggests `voxoracle setup`

### Requirement: Setup and model bootstrap

The `voxoracle setup` command MUST download the wake-word model(s) and prepare the
device so that `voxoracle run` works without further manual steps.

#### Scenario: Setup prepares the device
- **WHEN** `voxoracle setup` completes successfully on a fresh device
- **THEN** the wake-word model is present and `voxoracle run` can start

### Requirement: Runs as a system service

The system MUST ship a systemd unit and an install script so the appliance starts
the voice session on boot.

#### Scenario: Service starts on boot
- **WHEN** the device reboots after installation
- **THEN** the VoxOracle service starts automatically and the session listens for the wake word

### Requirement: Continuous integration

The project MUST run `uv sync`, `ruff`, `basedpyright`, `pytest`, and
`openspec validate` in CI, and OpenSpec specs MUST be validated when the CLI is
available.

#### Scenario: CI gates a change
- **WHEN** a pull request is opened
- **THEN** dependency sync, lint, type checking, tests, and spec validation run and must pass

#### Scenario: Spec validation runs when available
- **WHEN** the OpenSpec CLI is installed in CI
- **THEN** `openspec validate --all` runs and must pass
