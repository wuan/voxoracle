"""Configuration loading: YAML file, environment overrides and defaults."""

from __future__ import annotations

from pathlib import Path

from voxoracle.config import DEFAULT_CONFIG_PATH, load_settings


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


def test_default_path_is_config_yaml(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.logging.level == "DEBUG"
    assert Path("config.yaml") == DEFAULT_CONFIG_PATH
