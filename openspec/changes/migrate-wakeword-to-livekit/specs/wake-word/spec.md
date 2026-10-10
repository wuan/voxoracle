# Spec Delta: wake-word

## ADDED Requirements

### Requirement: Local wake-word detection with livekit-wakeword

The system SHALL detect the activation word "Hey Franz" locally using
livekit-wakeword, without sending audio to any cloud service for wake-word
detection. Detection MUST emit a trigger event to the session loop. The detector
SHALL accept the 16 kHz mono frames produced by the audio layer and score them
against a rolling local audio window (approximately 2 seconds), invoking the
model on a configured stride rather than after every frame.

#### Scenario: Wake word is detected

- **WHEN** the "Hey Franz" wake-word model scores above the configured threshold on the rolling audio window
- **THEN** the detector emits a trigger event

#### Scenario: Wake word runs locally

- **WHEN** the always-on detector is running
- **THEN** no audio leaves the device for wake-word detection

#### Scenario: Stride controls scoring cadence

- **WHEN** the detector is configured with a scoring stride of N audio frames
- **THEN** the model is invoked once per N frames on the current window, and detections within that stride are reported at the next scoring step

### Requirement: Trained German wake-word model

The system SHALL use a purpose-trained "Hey Franz" model trained with
livekit-wakeword's pipeline (German synthetic speech, conv-attention classifier)
and exported as an ONNX classifier. The trained model MUST meet the acceptance
gate of at most 0.5 false positives per hour on German background audio before
it is deployed to the appliance.

#### Scenario: Model meets the acceptance gate

- **WHEN** the trained "Hey Franz" model is evaluated on held-out German background audio
- **THEN** its false-positive rate per hour is at most 0.5 and its recall is documented

## MODIFIED Requirements

### Requirement: Swappable wake-word model

The wake-word model MUST be selectable from configuration as a name or an
explicit ``.onnx`` path, so a retrained "Hey Franz" model can be installed
without code changes. A wake-word model is REQUIRED: when no model can be
resolved, the detector MUST raise a typed error naming the missing model, and
``doctor`` and session startup MUST fail fast with a clear operator-facing
message instead of falling back to any placeholder.

#### Scenario: Model comes from configuration

- **WHEN** a wake-word model name or path is configured
- **THEN** the detector loads that model

#### Scenario: Missing model is reported

- **WHEN** no configured wake-word model can be found
- **THEN** the detector raises a typed error naming the missing model and startup does not proceed

### Requirement: Runtime backend

The wake-word runtime MUST use livekit-wakeword's ONNX inference on the target
64-bit Raspberry Pi OS (aarch64), where ``onnxruntime`` is available. Because
the runtime scores a stateless window, the scoring stride and quantization
variant MUST be verified on the Raspberry Pi 3 so that the always-on path stays
within the device's CPU budget with documented detection latency.

#### Scenario: ONNX backend on the target

- **WHEN** the wake-word detector runs on 64-bit Raspberry Pi OS
- **THEN** detection uses the ``onnxruntime`` ONNX backend

#### Scenario: Window scoring fits the device budget

- **WHEN** the detector runs on the Raspberry Pi 3 with the configured stride
- **THEN** measured inference cost stays within the documented CPU budget and detection latency is documented

#### Scenario: Detection verified on target hardware

- **WHEN** the activation word "Hey Franz" is spoken to the target device
- **THEN** detection triggers reliably with a documented threshold

## REMOVED Requirements

### Requirement: Local wake-word detection with openWakeWord

**Reason**: openWakeWord is replaced entirely by livekit-wakeword, which shares
the same audio front-end but adds a conv-attention classifier with a far lower
false-positive rate and a German-capable training pipeline.

**Migration**: The new "Local wake-word detection with livekit-wakeword"
requirement replaces this behavior; the placeholder-model behavior it relied on
is removed per the updated "Swappable wake-word model" requirement.
