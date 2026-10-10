"""Pre-flight diagnostics and device bootstrap for the ``doctor``/``setup`` CLI.

The logic lives here, separate from :mod:`voxoracle.cli`, so it can be unit
tested with injected fakes: no real audio device, network or credentials are
needed. Every check returns a :class:`CheckResult`; the CLI only renders them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from voxoracle.audio.errors import AudioError
from voxoracle.audio.protocols import AudioBackend, DeviceInfo
from voxoracle.config import (
    Settings,
    load_settings,
    mistral_api_key_source,
    resolve_models_dir,
)


class Severity(StrEnum):
    """How a check outcome affects the process exit code."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """The outcome of one diagnostic check."""

    name: str
    severity: Severity
    detail: str
    #: Non-secret follow-up lines (e.g. "run `voxoracle setup`").
    hints: list[str] = field(default_factory=list)


@dataclass
class AudioDeviceReport:
    """Device enumeration for both directions, or the error that prevented it."""

    input_devices: list[DeviceInfo] = field(default_factory=list)
    output_devices: list[DeviceInfo] = field(default_factory=list)
    error: str | None = None


def check_settings(settings: Settings) -> CheckResult:
    """Report that the settings loaded; the CLI catches load errors itself."""
    return CheckResult(
        name="settings",
        severity=Severity.OK,
        detail=(
            f"docoracle.url={settings.docoracle.url} "
            f"audio.input_device={settings.audio.input_device!r} "
            f"audio.output_device={settings.audio.output_device!r}"
        ),
    )


def enumerate_audio(settings: Settings, backend: AudioBackend) -> AudioDeviceReport:
    """Enumerate input/output devices, capturing PortAudio errors as text.

    Returns a report whose ``error`` is set when enumeration failed (e.g. no
    PortAudio library or no hardware), instead of raising.
    """
    report = AudioDeviceReport()
    try:
        report.input_devices = backend.list_devices("input")
        report.output_devices = backend.list_devices("output")
    except AudioError as exc:
        report.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - PortAudio raises assorted native errors
        report.error = f"{type(exc).__name__}: {exc}"
    return report


def check_audio(settings: Settings, report: AudioDeviceReport) -> list[CheckResult]:
    """Turn an :class:`AudioDeviceReport` into input/output check results."""
    if report.error is not None:
        return [
            CheckResult(
                name="audio devices",
                severity=Severity.FAIL,
                detail=report.error,
                hints=[
                    "install PortAudio: `sudo apt install libportaudio2`",
                    "check the microphone/speaker are connected and in ALSA "
                    "(`arecord -l`, `aplay -l`)",
                ],
            )
        ]

    results: list[CheckResult] = []
    for kind, devices in (
        ("input", report.input_devices),
        ("output", report.output_devices),
    ):
        configured = (
            settings.audio.input_device if kind == "input" else settings.audio.output_device
        )
        named = ", ".join(_describe(device) for device in devices) or "none"
        if not devices:
            results.append(
                CheckResult(
                    name=f"audio {kind}",
                    severity=Severity.FAIL,
                    detail=f"no {kind} devices found; check ALSA (`arecord -l`/`aplay -l`)",
                )
            )
            continue
        matched = _match(configured, devices)
        if configured == "default":
            default = next((d for d in devices if d.is_default), devices[0])
            detail = f"configured 'default' -> {_describe(default)}; available: {named}"
        elif matched is None:
            results.append(
                CheckResult(
                    name=f"audio {kind}",
                    severity=Severity.FAIL,
                    detail=f"configured device {configured!r} not found; available: {named}",
                    hints=[f"set audio.{kind}_device in config.yaml to one of the above"],
                )
            )
            continue
        else:
            detail = f"configured {configured!r} -> {_describe(matched)}"
        results.append(CheckResult(name=f"audio {kind}", severity=Severity.OK, detail=detail))
    return results


def check_wakeword(settings: Settings) -> CheckResult:
    """Resolve the configured wake-word model and report its presence.

    Uses :func:`voxoracle.wakeword.resolve_model_path`, which falls back to the
    bundled placeholder model when a bare name has no ``.onnx`` yet.
    """
    from voxoracle.wakeword import PLACEHOLDER_MODEL, resolve_model_path
    from voxoracle.wakeword.errors import WakeWordError

    models_dir = resolve_models_dir(settings)
    configured = settings.wakeword.model
    try:
        path = resolve_model_path(configured, models_dir)
    except WakeWordError as exc:
        return CheckResult(
            name="wake word",
            severity=Severity.FAIL,
            detail=f"model {configured!r} could not be resolved: {exc}",
            hints=[f"provide {models_dir / (configured + '.onnx')} or run `voxoracle setup`"],
        )
    if PLACEHOLDER_MODEL in path.name:
        return CheckResult(
            name="wake word",
            severity=Severity.WARN,
            detail=(
                f"configured {configured!r} not found in {models_dir}; "
                f"using placeholder {PLACEHOLDER_MODEL!r} (detects the placeholder "
                "phrase, not 'Franz')"
            ),
            hints=[f"drop a trained {configured}.onnx into {models_dir}"],
        )
    return CheckResult(
        name="wake word",
        severity=Severity.OK,
        detail=f"configured {configured!r} -> {path}",
    )


async def check_docoracle(settings: Settings) -> list[CheckResult]:
    """Probe DocOracle ``/health`` (required) and ``/info`` (informational)."""
    from voxoracle.docoracle.client import DocOracleClient, DocOracleError

    client = DocOracleClient(base_url=settings.docoracle.url, timeout=settings.docoracle.timeout)
    results: list[CheckResult] = []
    try:
        health = await client.health()
        results.append(
            CheckResult(
                name="docoracle /health",
                severity=Severity.OK,
                detail=f"{settings.docoracle.url} -> status={health.status}",
            )
        )
    except DocOracleError as exc:
        results.append(
            CheckResult(
                name="docoracle /health",
                severity=Severity.FAIL,
                detail=f"{settings.docoracle.url} unreachable: {exc}",
                hints=["start DocOracle and set docoracle.url in config.yaml"],
            )
        )
    try:
        info = await client.info()
        results.append(
            CheckResult(
                name="docoracle /info",
                severity=Severity.OK,
                detail=(
                    f"{info.total_chunks} chunks, mode={info.retrieval_mode}, "
                    f"{len(info.modules)} modules"
                ),
            )
        )
    except DocOracleError as exc:
        results.append(
            CheckResult(
                name="docoracle /info",
                severity=Severity.WARN,
                detail=f"unavailable: {exc}",
            )
        )
    finally:
        await client.aclose()
    return results


def check_mistral_key(settings: Settings) -> CheckResult:
    """Report whether a Mistral key is resolvable, and from where — never the key."""
    source = mistral_api_key_source(settings)
    if source is None:
        return CheckResult(
            name="mistral key",
            severity=Severity.FAIL,
            detail="missing",
            hints=[
                "set MISTRAL_API_KEY (or LLM_API_KEY) in the environment or .env",
                "STT/TTS cannot work without it",
            ],
        )
    return CheckResult(name="mistral key", severity=Severity.OK, detail=f"found ({source})")


def all_checks_passed(results: list[CheckResult]) -> bool:
    """True when no check has :attr:`Severity.FAIL`."""
    return all(result.severity is not Severity.FAIL for result in results)


def prepare_directories(settings: Settings) -> list[tuple[str, Path, bool]]:
    """Create the appliance's runtime directories; idempotent.

    Returns ``(label, path, created)`` tuples where ``created`` is ``True`` when
    this call made the directory and ``False`` when it already existed.
    """
    models_dir = resolve_models_dir(settings)
    cache_dir = Path.home() / ".cache" / "voxoracle"
    outcomes: list[tuple[str, Path, bool]] = []
    for label, path in (("models_dir", models_dir), ("cache", cache_dir)):
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        outcomes.append((label, path, not existed))
    return outcomes


def default_settings() -> Settings:
    """Load settings via the standard config.yaml + environment path."""
    return load_settings()


def _describe(device: DeviceInfo) -> str:
    marker = " (default)" if device.is_default else ""
    return f"[{device.index}] {device.name}{marker}"


def _match(name: str, devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Mirror sounddevice_backend.resolve_device matching, for reporting."""
    if name == "default":
        return next((d for d in devices if d.is_default), None)
    needle = name.lower()
    for device in devices:
        if device.name.lower() == needle:
            return device
    for device in devices:
        if device.name.lower().startswith(needle):
            return device
    for device in devices:
        if needle in device.name.lower():
            return device
    return None
