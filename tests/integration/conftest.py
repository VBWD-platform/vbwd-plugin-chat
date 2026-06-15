"""Integration harness for the chat bot-consumer round-trip (S45.2).

Boots the full app so the live plugin set (chat + bot-base + bot-telegram) is
registered: the webhook route is mounted, bot-base's dispatcher can claim the
chat conversation, and the chat plugin is collected as a ``BotCommandProvider``.
Outbound transport uses bot-telegram's in-memory fake client (no network); the
LLM call is faked per-test (no network). Mirrors the bot_base / bot_telegram
integration conftest.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("TESTING", "true")


def _test_db_url() -> str:
    base = os.getenv("DATABASE_URL", "postgresql://vbwd:vbwd@postgres:5432/vbwd")
    prefix, _, dbname = base.rpartition("/")
    dbname = dbname.split("?")[0]
    return f"{prefix}/{dbname}_test"


def _ensure_test_db(url: str) -> None:
    from sqlalchemy import create_engine, text

    main_url = url.rsplit("/", 1)[0] + "/postgres"
    dbname = url.rsplit("/", 1)[1].split("?")[0]
    engine = create_engine(main_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": dbname}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{dbname}"'))
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def app():
    from vbwd.app import create_app
    from vbwd.extensions import db as _db

    test_url = _test_db_url()
    _ensure_test_db(test_url)
    application = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": test_url,
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_ENABLED": False,
            "RATELIMIT_STORAGE_URL": "memory://",
        }
    )
    with application.app_context():
        import plugins.bot_base.bot_base.models  # noqa: F401
        import plugins.bot_telegram.bot_telegram.models  # noqa: F401

        # Build the schema once per process (create_all, checkfirst — never
        # drops, so it cannot wipe data) and commit baseline reference rows
        # once. Each test then isolates itself via a rolled-back transaction
        # (no TRUNCATE, no DROP) — see vbwd/testing/integration_db.py.
        from vbwd.testing.integration_db import ensure_schema_and_baseline

        ensure_schema_and_baseline(_db)

    yield application

    with application.app_context():
        _db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _isolate_test(app, request):
    """Isolate every test in a rolled-back transaction (self-cleaning, no wipe).

    This suite self-seeds all its data; nothing a test writes persists past it
    (the rollback IS the cleanup), so each test is deterministic and
    order-independent. The schema + baseline reference rows are built once in the
    ``app`` fixture. See vbwd/testing/integration_db.py.

    A test marked ``no_db_isolation`` (e.g. a migration spec that opens its own
    connection and rolls back itself) runs WITHOUT the wrapper, keeping
    ``db.engine`` a real Engine.
    """
    from vbwd.extensions import db as _db

    if request.node.get_closest_marker("no_db_isolation") is not None:
        with app.app_context():
            yield
            _db.session.remove()
        return

    with app.app_context():
        from vbwd.testing.integration_db import rollback_isolation

        with rollback_isolation(_db):
            yield


@pytest.fixture(autouse=True)
def _inject_fake_telegram_client(app):
    """Route outbound through bot-telegram's in-memory client (no network)."""
    from plugins.bot_telegram.bot_telegram.services.telegram_client import (
        InMemoryTelegramClient,
    )

    fake = InMemoryTelegramClient()
    app.config["BOT_TELEGRAM_CLIENT"] = fake
    yield fake
    app.config.pop("BOT_TELEGRAM_CLIENT", None)
