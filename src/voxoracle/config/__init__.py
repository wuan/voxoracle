"""Settings schema and configuration loading.

The schema mirrors ``config.example.yaml``. ``load_settings`` reads the YAML
file and applies ``VXORACLE_*`` environment overrides; nested keys use the
``SECTION__KEY`` delimiter, e.g. ``VXORACLE_DOCORACLE__URL``, and the
environment wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config.yaml")


class LoggingSettings(BaseModel):
    level: str = "INFO"


class DocOracleSettings(BaseModel):
    url: str = "http://localhost:8000"
    timeout: float = 60.0


class AudioSettings(BaseModel):
    input_device: str = "default"
    output_device: str = "default"
    sample_rate: int = Field(default=16000, gt=0)
    frame_ms: int = Field(default=30, gt=0)
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    endpoint_silence_ms: int = Field(default=700, gt=0)


class WakeWordSettings(BaseModel):
    engine: str = "openwakeword"
    # Model name ("franz") or an explicit .onnx path; falls back to a placeholder
    # model until a "franz" model is provided (see design.md).
    model: str = "franz"
    models_dir: str = "models"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MistralSettings(BaseModel):
    """Shared Mistral cloud credentials/endpoint (STT in WP4, TTS in WP5)."""

    base_url: str = "https://api.mistral.ai/v1"
    # Prefer the environment (VXORACLE_MISTRAL__API_KEY, MISTRAL_API_KEY or
    # LLM_API_KEY); never commit a key to config.yaml.
    api_key: str | None = None
    timeout: float = Field(default=30.0, gt=0.0)
    retries: int = Field(default=2, ge=0)


class STTSettings(BaseModel):
    provider: str = "mistral"
    model: str = "voxtral-mini-latest"
    language: str = "de"


class TTSSettings(BaseModel):
    provider: str = "mistral"
    model: str = "voxtral-mini-tts-2603"
    language: str = "de"
    # Voice id sent to the provider. None selects the default voice from the
    # configured language (German), so a config with ``voice: null`` stays valid;
    # WP7 confirms the exact preset on-device.
    voice: str | None = None
    # Sample rate of the provider's PCM output in Hz. Mistral does not publish it,
    # so it is configurable and confirmed on-device in WP7; the output layer
    # resamples this to the device rate.
    sample_rate: int = Field(default=24000, gt=0)


class SessionSettings(BaseModel):
    max_record_seconds: float = 15.0
    follow_up: bool = False
    barge_in: bool = True


class Settings(BaseSettings):
    """VoxOracle settings: ``config.yaml`` plus ``VXORACLE_*`` env overrides."""

    model_config = SettingsConfigDict(
        env_prefix="VXORACLE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
    )

    docoracle: DocOracleSettings = Field(default_factory=DocOracleSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    wakeword: WakeWordSettings = Field(default_factory=WakeWordSettings)
    mistral: MistralSettings = Field(default_factory=MistralSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Lowest-priority source: an optional YAML settings file.

    The env source precedes this one in the customised source tuple, so
    ``VXORACLE_*`` environment variables override the file.
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, "", False

    def __call__(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        with self._path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from ``config.yaml`` (or ``path``) and the environment.

    A missing file is not an error: field defaults and ``VXORACLE_*``
    environment variables apply. ``path`` defaults to ``config.yaml`` in the
    current directory.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    class _Settings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
                YamlSettingsSource(settings_cls, config_path),
            )

    return _Settings()


def _dotenv_value(name: str) -> str | None:
    """Read ``name`` from the ``.env`` file in the current directory, if present."""
    from dotenv import dotenv_values  # python-dotenv ships with pydantic-settings

    value = dotenv_values(".env").get(name)
    return value or None


def resolve_mistral_api_key(settings: Settings) -> str | None:
    """Return the Mistral API key, preferring config/env over the shared env vars.

    Resolution order: ``mistral.api_key`` (config, ``.env`` or
    ``VXORACLE_MISTRAL__API_KEY``), then ``MISTRAL_API_KEY``, then
    ``LLM_API_KEY`` (the key DocOracle already uses, so one Mistral key can cover
    STT/TTS and DocOracle). For the last two, a real environment variable wins
    over a ``.env`` entry, mirroring the ``VXORACLE_*`` precedence. Returns
    ``None`` when no key is configured.
    """
    return _resolve_mistral_api_key(settings)[0]


def mistral_api_key_source(settings: Settings) -> str | None:
    """Return a short human-readable label for where the Mistral key came from.

    Never returns the key itself. Labels: ``"VXORACLE_MISTRAL__API_KEY (env)"``,
    ``"VXORACLE_MISTRAL__API_KEY (.env)"``, ``"config.yaml"``, the environment
    variables (``"MISTRAL_API_KEY (env)"`` / ``"LLM_API_KEY (env)"``) or their
    ``.env`` entries (``".env (MISTRAL_API_KEY)"`` / ``".env (LLM_API_KEY)"``).
    Returns ``None`` when no key is configured.
    """
    return _resolve_mistral_api_key(settings)[1]


def _resolve_mistral_api_key(settings: Settings) -> tuple[str | None, str | None]:
    """Resolve the Mistral key and its source label without exposing the key."""
    if settings.mistral.api_key:
        # ``mistral.api_key`` is fed by config.yaml or VXORACLE_MISTRAL__API_KEY
        # (env or .env); pydantic merges them, so attribute the env var when set.
        if os.environ.get("VXORACLE_MISTRAL__API_KEY"):
            return settings.mistral.api_key, "VXORACLE_MISTRAL__API_KEY (env)"
        if _dotenv_value("VXORACLE_MISTRAL__API_KEY"):
            return settings.mistral.api_key, "VXORACLE_MISTRAL__API_KEY (.env)"
        return settings.mistral.api_key, "config.yaml"
    for name, label in (
        ("MISTRAL_API_KEY", "MISTRAL_API_KEY (env)"),
        ("LLM_API_KEY", "LLM_API_KEY (env)"),
    ):
        if os.environ.get(name):
            return os.environ[name], label
    for name in ("MISTRAL_API_KEY", "LLM_API_KEY"):
        value = _dotenv_value(name)
        if value:
            return value, f".env ({name})"
    return None, None


def resolve_models_dir(settings: Settings) -> Path:
    """Return the wake-word ``models_dir`` as an absolute path.

    Relative paths are resolved against the current directory so callers
    (``doctor``/``setup``) can report and create a stable location.
    """
    return Path(settings.wakeword.models_dir).expanduser().resolve()
