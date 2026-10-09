"""PortAudio-backed audio device access (via ``sounddevice``).

``sounddevice`` imports PortAudio at import time, which is absent on CI and on
machines without the library. The import is therefore deferred to first use so
that importing this module (and the rest of the audio layer) never requires
PortAudio. Tests never construct these classes; they inject fakes instead.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from voxoracle.audio.errors import AudioError, DeviceNotFoundError
from voxoracle.audio.protocols import DeviceInfo, DeviceKind
from voxoracle.audio.resample import resample
from voxoracle.config import AudioSettings


def _sounddevice() -> Any:
    """Import ``sounddevice`` lazily, raising a typed error if PortAudio is absent."""
    try:
        import sounddevice  # pyright: ignore[reportMissingTypeStubs]
    except OSError as exc:  # PortAudio library not found
        raise AudioError(f"PortAudio is not available: {exc}") from exc
    return sounddevice


def list_devices(kind: DeviceKind) -> list[DeviceInfo]:
    """Enumerate input or output devices, marking the system default."""
    sd = _sounddevice()
    devices = cast(list[dict[str, Any]], list(sd.query_devices()))
    default_index = sd.default.device[0 if kind == "input" else 1]
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    results: list[DeviceInfo] = []
    for index, raw in enumerate(devices):
        if raw[channel_key] <= 0:
            continue
        results.append(
            DeviceInfo(
                index=index,
                name=str(raw["name"]),
                kind=kind,
                max_input_channels=int(raw["max_input_channels"]),
                max_output_channels=int(raw["max_output_channels"]),
                default_sample_rate=float(raw["default_samplerate"]),
                is_default=index == default_index,
            )
        )
    return results


def resolve_device(name: str | None, kind: DeviceKind, devices: list[DeviceInfo]) -> int | None:
    """Resolve a configured device name to an index within ``devices``.

    ``None`` or ``"default"`` selects the system default (``None`` passed to
    sounddevice). Any other value is matched case-insensitively against the
    device names and raises :class:`DeviceNotFoundError` if missing.
    """
    if name is None or name == "default":
        return None
    for device in devices:
        if device.name.lower() == name.lower():
            return device.index
    raise DeviceNotFoundError(name, kind)


class SoundDeviceInput:
    """Blocking 16-bit mono capture from a PortAudio input device."""
    def __init__(self, device: int | None, sample_rate: int, frame_samples: int) -> None:
        sd = _sounddevice()
        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            blocksize=frame_samples,
            dtype="int16",
        )
        self._stream.start()
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def read_frame(self) -> NDArray[np.int16]:
        data, _overflowed = self._stream.read(self._frame_samples)
        return np.asarray(data, dtype=np.int16).reshape(-1)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SoundDeviceOutput:
    """Playback that resamples to the device rate when it differs."""

    def __init__(self, device: int | None, sample_rate: int) -> None:
        sd = _sounddevice()
        info = sd.query_devices(device, "output")
        device_rate = int(round(float(info["default_samplerate"])))
        self._device_rate = device_rate
        self._stream = sd.OutputStream(
            device=device,
            channels=1,
            samplerate=device_rate,
            dtype="int16",
        )
        self._stream.start()
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        """The source rate accepted by :meth:`write`."""
        return self._sample_rate

    def write(self, samples: NDArray[np.int16]) -> None:
        prepared = resample(samples, self._sample_rate, self._device_rate)
        self._stream.write(prepared.reshape(-1, 1))

    def stop(self) -> None:
        self._stream.stop()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SoundDeviceBackend:
    """PortAudio backend that selects devices from :class:`AudioSettings`."""

    def __init__(self, settings: AudioSettings) -> None:
        self._settings = settings

    def list_devices(self, kind: DeviceKind) -> list[DeviceInfo]:
        return list_devices(kind)

    def open_input(self, device: int | None, sample_rate: int, frame_samples: int) -> SoundDeviceInput:
        return SoundDeviceInput(device, sample_rate, frame_samples)

    def open_output(self, device: int | None, sample_rate: int) -> SoundDeviceOutput:
        return SoundDeviceOutput(device, sample_rate)

    def _resolve(self, name: str, kind: DeviceKind) -> int | None:
        return resolve_device(name, kind, list_devices(kind))

    def open_configured_input(self) -> SoundDeviceInput:
        """Open capture at the configured sample rate and frame size."""
        frame_samples = round(self._settings.sample_rate * self._settings.frame_ms / 1000)
        device = self._resolve(self._settings.input_device, "input")
        return SoundDeviceInput(device, self._settings.sample_rate, frame_samples)

    def open_configured_output(self, sample_rate: int) -> SoundDeviceOutput:
        """Open playback for audio produced at ``sample_rate``."""
        device = self._resolve(self._settings.output_device, "output")
        return SoundDeviceOutput(device, sample_rate)
