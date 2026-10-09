"""Configuration loading: YAML file, environment overrides and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voxoracle.config import DEFAULT_CONFIG_PATH, load_settings, resolve_mistral_api_key


def test_defaults_without_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.docoracle.url == "http://localhost:8000"
    assert settings.docoracle.timeout == 60.0
    assert settings.wakeword.model == "franz"
    assert settings.session.barge_in is True
    assert settings.logging.level == "INFO"


def test_yaml_file_is_loaded(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "docoracle:\n  url: http://example.internal:8000\n  timeout: 7\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.docoracle.url == "http://example.internal:8000"
    assert settings.docoracle.timeout == 7.0


def test_environment_overrides_yaml(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "docoracle:\n  url: http://from-yaml:8000\n  timeout: 7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VXORACLE_DOCORACLE__URL", "http://from-env:9000")
    settings = load_settings(config)
    assert settings.docoracle.url == "http://from-env:9000"
    assert settings.docoracle.timeout == 7.0  # not overridden by env


def test_missing_file_falls_back_to_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VXORACLE_DOCORACLE__URL", "http://from-env:9000")
    settings = load_settings(tmp_path / "does-not-exist.yaml")
    assert settings.docoracle.url == "http://from-env:9000"


def test_unknown_keys_are_ignored(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("docoracle:\n  url: http://ok:8000\n  bogus: 1\n", encoding="utf-8")
    settings = load_settings(config)
    assert settings.docoracle.url == "http://ok:8000"


def test_dotenv_file_is_loaded(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "VXORACLE_DOCORACLE__URL=http://from-dotenv:7000\n", encoding="utf-8"
    )
    settings = load_settings()
    assert settings.docoracle.url == "http://from-dotenv:7000"


def test_environment_beats_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "VXORACLE_DOCORACLE__URL=http://from-dotenv:7000\n", encoding="utf-8"
    )
    monkeypatch.setenv("VXORACLE_DOCORACLE__URL", "http://from-env:8000")
    settings = load_settings()
    assert settings.docoracle.url == "http://from-env:8000"


def test_invalid_audio_settings_are_rejected(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("audio:\n  sample_rate: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config)


def test_wakeword_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.wakeword.model == "franz"
    assert settings.wakeword.models_dir == "models"
    assert settings.wakeword.threshold == 0.5


def test_invalid_wakeword_threshold_is_rejected(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("wakeword:\n  threshold: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config)


def test_default_path_is_config_yaml(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.logging.level == "DEBUG"
    assert Path("config.yaml") == DEFAULT_CONFIG_PATH


def test_stt_defaults_are_mistral_german(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.stt.provider == "mistral"
    assert settings.stt.model == "voxtral-mini-latest"
    assert settings.stt.language == "de"
    assert settings.mistral.base_url == "https://api.mistral.ai/v1"


def test_mistral_api_key_from_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("mistral:\n  api_key: from-config\n", encoding="utf-8")
    assert resolve_mistral_api_key(load_settings(config)) == "from-config"


def test_mistral_api_key_env_wins_over_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("mistral:\n  api_key: from-config\n", encoding="utf-8")
    monkeypatch.setenv("VXORACLE_MISTRAL__API_KEY", "from-env")
    assert resolve_mistral_api_key(load_settings(config)) == "from-env"


def test_mistral_api_key_falls_back_to_llm_key_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VXORACLE_MISTRAL__API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "docoracle-key")
    assert resolve_mistral_api_key(load_settings()) == "docoracle-key"


def test_mistral_api_key_falls_back_to_mistral_key_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VXORACLE_MISTRAL__API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    assert resolve_mistral_api_key(load_settings()) == "mistral-key"


def test_mistral_api_key_reads_mistral_key_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=dotenv-mistral-key\n", encoding="utf-8")
    assert resolve_mistral_api_key(load_settings()) == "dotenv-mistral-key"


def test_mistral_api_key_reads_llm_key_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("LLM_API_KEY=dotenv-llm-key\n", encoding="utf-8")
    assert resolve_mistral_api_key(load_settings()) == "dotenv-llm-key"


def test_mistral_api_key_env_wins_over_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VXORACLE_MISTRAL__API_KEY", raising=False)
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
    assert resolve_mistral_api_key(load_settings()) == "env-key"


def test_mistral_api_key_none_when_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("VXORACLE_MISTRAL__API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_mistral_api_key(load_settings()) is None
