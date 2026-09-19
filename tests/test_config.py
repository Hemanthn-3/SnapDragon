from backend.config import NexusSettings


def test_default_config():
    """Verifies that default configuration values are properly set."""
    cfg = NexusSettings()
    assert cfg.APP_NAME == "NEXUS"
    assert cfg.APP_SUBTITLE == "Offline Multimodal Work Agent"
    assert cfg.OFFLINE_MODE is True
    assert cfg.ALLOW_TELEMETRY is False
    assert cfg.HOST == "127.0.0.1"
    assert cfg.PORT == 8000
    assert "nexus.db" in cfg.DATABASE_URL
    assert cfg.TARGET_PLATFORM == "Snapdragon X Series"


def test_config_override_from_env(monkeypatch):
    """Verifies that environment variables override configuration defaults."""
    monkeypatch.setenv("NEXUS_PORT", "9090")
    monkeypatch.setenv("NEXUS_OFFLINE_MODE", "false")
    monkeypatch.setenv("NEXUS_APP_NAME", "NEXUS-TEST")

    cfg = NexusSettings()
    assert cfg.PORT == 9090
    assert cfg.OFFLINE_MODE is False
    assert cfg.APP_NAME == "NEXUS-TEST"
