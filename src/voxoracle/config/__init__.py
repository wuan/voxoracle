"""Settings schema and configuration loading.

The schema mirrors ``config.example.yaml``. ``load_settings`` reads the YAML
file and applies ``VXORACLE_*`` environment overrides; nested keys use the
``SECTION__KEY`` delimiter, e.g. ``VXORACLE_DOCORACLE__URL``, and the
environment wins over the file.
"""

from __future__ import annotations

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
    sample_rate: int = 16000
    frame_ms: int = 30


class WakeWordSettings(BaseModel):
    engine: str = "openwakeword"
    model: str = "franz"
    threshold: float = 0.5


class STTSettings(BaseModel):
    provider: str | None = None
    language: str = "de"
    timeout: float = 30.0


class TTSSettings(BaseModel):
    provider: str | None = None
    language: str = "de"
    voice: str | None = None
    timeout: float = 30.0


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
