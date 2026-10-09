## ADDED Requirements

### Requirement: Microphone capture at 16 kHz mono

The audio layer SHALL capture microphone input as 16 kHz mono audio suitable for
speech processing, with a configurable frame size.

#### Scenario: Capture produces 16 kHz mono
- **WHEN** capture runs with the default settings
- **THEN** the produced audio is 16 kHz, single-channel

#### Scenario: Frame size is configurable
- **WHEN** a frame size in milliseconds is configured
- **THEN** the capture stream delivers frames of that duration

### Requirement: Speaker playback with resampling

The audio layer SHALL play audio through the configured speaker, resampling
audio to the device's supported rate when it differs from the source rate.

#### Scenario: Resampling on playback
- **WHEN** audio is played at a rate the device does not support natively
- **THEN** the audio is resampled to a supported rate before playback

#### Scenario: Playback can be stopped
- **WHEN** playback is cancelled
- **THEN** audio output stops promptly

### Requirement: Device selection

The audio layer MUST allow the input and output devices to be selected from
configuration, defaulting to the system default devices.

#### Scenario: Devices come from configuration
- **WHEN** input and output device names are configured
- **THEN** capture uses the configured input device and playback uses the configured output device

#### Scenario: Default devices are used
- **WHEN** no devices are configured
- **THEN** the system default input and output devices are used

#### Scenario: Unavailable device is reported
- **WHEN** a configured device is not available
- **THEN** the audio layer raises a typed error naming the missing device

### Requirement: Voice activity detection and endpointing

The audio layer SHALL detect speech and endpoint an utterance by trimming leading
and trailing silence, and MUST impose a configurable maximum recording duration.

#### Scenario: Silence is trimmed
- **WHEN** a recording contains leading and trailing silence
- **THEN** the returned utterance excludes that silence

#### Scenario: Long recording is capped
- **WHEN** speech continues beyond the configured maximum duration
- **THEN** recording stops at the maximum and the captured audio is returned

### Requirement: Hardware-free verification

The audio layer MUST be verifiable with synthetic audio buffers and injected
device fakes, without real audio hardware.

#### Scenario: Synthetic buffers
- **WHEN** tests run with synthetic buffers and a fake device
- **THEN** capture, resampling, playback, and VAD behavior are exercised without hardware
