## ADDED Requirements

### Requirement: Local wake-word detection with openWakeWord

The system SHALL detect the activation word "Franz" locally using openWakeWord,
without sending audio to any cloud service for wake-word detection. Detection
MUST emit a trigger event to the session loop.

#### Scenario: Wake word is detected
- **WHEN** the "Franz" wake-word model scores above the configured threshold on the incoming audio
- **THEN** the detector emits a trigger event

#### Scenario: Wake word runs locally
- **WHEN** the always-on detector is running
- **THEN** no audio leaves the device for wake-word detection

### Requirement: Configurable detection threshold

The detector MUST expose a configurable threshold so operators can trade
false activations against missed activations.

#### Scenario: Threshold is configurable
- **WHEN** the wake-word threshold is set in configuration
- **THEN** the detector uses that threshold when deciding whether to emit a trigger

#### Scenario: Higher threshold suppresses weak detections
- **WHEN** a low-confidence detection falls below a raised threshold
- **THEN** no trigger event is emitted

### Requirement: Lightweight always-on path

The always-on wake-word path MUST remain lightweight enough for the Raspberry Pi
3 and MUST NOT perform cloud calls or heavy model inference.

#### Scenario: Idle resource use
- **WHEN** the device is idle and only listening for the wake word
- **THEN** STT, TTS, and DocOracle are not invoked

### Requirement: Runtime backend

The wake-word runtime MUST use openWakeWord's ONNX inference backend on the
target 64-bit Raspberry Pi OS (aarch64), where `onnxruntime` is available; the
inference backend MUST sit behind the detector protocol so it stays swappable.

#### Scenario: ONNX backend on the target
- **WHEN** the wake-word detector runs on 64-bit Raspberry Pi OS
- **THEN** detection uses the `onnxruntime` ONNX backend

#### Scenario: Detection verified on target hardware
- **WHEN** the activation word "Franz" is spoken to the target device
- **THEN** detection triggers reliably with a documented threshold
