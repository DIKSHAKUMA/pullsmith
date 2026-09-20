"""Redaction tests.

A single leaked token is a real incident, so the redaction rules are pinned by tests
rather than trusted to review.
"""

from app.observability.redaction import REDACTED, redact


def test_redacts_github_classic_token() -> None:
    text = "cloning with token ghp_abcdefghijklmnop1234567890 now"
    assert "ghp_abcdefghijklmnop1234567890" not in redact(text)
    assert REDACTED in redact(text)


def test_redacts_fine_grained_pat() -> None:
    text = "token github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"
    assert "github_pat_11ABCDEFG" not in redact(text)


def test_redacts_authorization_header() -> None:
    assert "secretvalue" not in redact("Authorization: Bearer secretvalue")


def test_redacts_provider_keys() -> None:
    assert "sk-abcdefghijklmnopqrst" not in redact("key sk-abcdefghijklmnopqrst")
    assert "AIzaSyA1234567890abcdefghij" not in redact("key AIzaSyA1234567890abcdefghij")


def test_redacts_key_value_secrets() -> None:
    assert "hunter2hunter2" not in redact("password=hunter2hunter2")
    assert "abc123def456" not in redact("api_key: abc123def456")


def test_redacts_database_url_credentials() -> None:
    redacted = redact("postgresql+asyncpg://neondb_owner:npg_secret@ep-x.neon.tech/db")
    assert "npg_secret" not in redacted
    assert "neondb_owner" not in redacted
    assert "ep-x.neon.tech" in redacted  # host is useful for debugging, not a secret


def test_leaves_ordinary_text_alone() -> None:
    text = "run 3f2a completed in 42s with 6 files changed"
    assert redact(text) == text


def test_redacts_google_ai_studio_aq_keys() -> None:
    """Google AI Studio issues keys prefixed "AQ." as well as "AIza".

    The AIza pattern does not match them, so a separate rule is needed. Found while
    wiring up the embedding provider: the key format was not what the existing filter
    expected.
    """
    # A synthetic key. The first version of this test used a fragment of a real one, which is
    # how a live credential ends up in a public repository: not in a config file, where everyone
    # looks, but pasted into a test for the very filter meant to protect it.
    text = "embedding request failed for key AQ.Fake000ExampleKeyForTestsOnly000000000"

    redacted = redact(text)

    assert "AQ.Fake000Example" not in redacted
    assert REDACTED in redacted


def test_aq_pattern_does_not_eat_ordinary_words() -> None:
    assert redact("status AQ ok") == "status AQ ok"
