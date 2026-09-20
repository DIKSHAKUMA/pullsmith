"""Configuration validation and token encryption.

Orchestrator behaviour moved to `test_orchestrator_integration.py` once the Phase 1 stub was
replaced by the real driver: it now needs a git repository, a scripted model and a sandbox,
which is more setup than belongs next to these unit tests.
"""

import pytest

from app.config.settings import AppEnv, SandboxBackend, Settings
from app.security.crypto import SessionCodec, TokenCipher


def test_production_config_requires_real_secrets() -> None:
    # _env_file=None keeps the developer's local .env out of the test. Without it this
    # test passes or fails depending on whose machine it runs on.
    unsafe = Settings(
        _env_file=None,
        app_env=AppEnv.production,
        database_url="sqlite+aiosqlite:///./local.db",
    )

    with pytest.raises(RuntimeError) as error:
        unsafe.validate_for_runtime()

    message = str(error.value)
    assert "SESSION_SECRET" in message
    assert "TOKEN_ENCRYPTION_KEY" in message
    assert "Postgres" in message


def test_unsafe_sandbox_is_blocked_in_production() -> None:
    settings = Settings(
        _env_file=None,
        app_env=AppEnv.production,
        database_url="postgresql+asyncpg://u:p@host/db",
        session_secret="a-real-secret",
        token_encryption_key="a-real-key",
        sandbox_backend=SandboxBackend.local_unsafe,
    )

    with pytest.raises(RuntimeError, match="no isolation"):
        settings.validate_for_runtime()


def test_development_config_is_permissive() -> None:
    Settings(_env_file=None, app_env=AppEnv.development).validate_for_runtime()


def test_token_cipher_round_trip() -> None:
    cipher = TokenCipher(None)
    token = "ghp_secret_value_0987654321"

    encrypted = cipher.encrypt(token)

    assert encrypted != token
    assert token not in encrypted
    assert cipher.decrypt(encrypted) == token


def test_token_cipher_rejects_foreign_ciphertext() -> None:
    with pytest.raises(ValueError, match="could not be decrypted"):
        TokenCipher(None).decrypt(TokenCipher(None).encrypt("x"))


def test_session_codec_round_trip_and_tamper_detection() -> None:
    codec = SessionCodec("secret-one", ttl_seconds=60)
    token = codec.issue("user-123")

    assert codec.verify(token) == "user-123"

    with pytest.raises(Exception):  # noqa: B017 - any jwt failure is acceptable
        SessionCodec("secret-two", ttl_seconds=60).verify(token)


def test_expired_session_is_rejected() -> None:
    codec = SessionCodec("secret", ttl_seconds=-10)
    token = codec.issue("user-123")

    with pytest.raises(Exception):  # noqa: B017
        codec.verify(token)
