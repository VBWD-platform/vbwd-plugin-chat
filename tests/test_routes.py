"""Tests for chat plugin API routes."""
import pytest
from uuid import UUID
from unittest.mock import MagicMock

from flask import Flask


@pytest.fixture
def mock_container(mocker):
    """Mock DI container with a token service + the CORE LLM client."""
    container = mocker.MagicMock()
    token_service = MagicMock()
    balance_obj = MagicMock()
    balance_obj.balance = 950
    token_service.get_balance.return_value = 1000
    token_service.debit_tokens.return_value = balance_obj
    container.token_service.return_value = token_service

    llm_client = MagicMock()
    llm_client.chat.return_value = "Hello!"
    container.llm_client.return_value = llm_client
    return container


@pytest.fixture
def app(mock_config_store, mock_container, mocker):
    """Create Flask app with chat blueprint registered."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True

    user_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    mock_auth_service = MagicMock()
    mock_auth_service.return_value.verify_token.return_value = str(user_id)
    mocker.patch("vbwd.middleware.auth.AuthService", mock_auth_service)

    mock_user = MagicMock()
    mock_user.id = user_id
    mock_user.status.value = "ACTIVE"

    mock_user_repo = MagicMock()
    mock_user_repo.return_value.find_by_id.return_value = mock_user
    mocker.patch("vbwd.middleware.auth.UserRepository", mock_user_repo)

    mock_db = MagicMock()
    mocker.patch("vbwd.middleware.auth.db", mock_db)

    from plugins.chat.src.routes import chat_bp

    flask_app.register_blueprint(chat_bp, url_prefix="/api/v1/plugins/chat")

    flask_app.config_store = mock_config_store
    flask_app.container = mock_container

    return flask_app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Auth headers with a dummy bearer token."""
    return {"Authorization": "Bearer test_token_123"}


class TestSendMessage:
    """Tests for POST /api/v1/plugins/chat/send."""

    def test_success(self, client, auth_headers, mock_container):
        mock_container.llm_client.return_value.chat.return_value = "Hello!"

        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hi there", "history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Hello!"
        assert "tokens_used" in data
        assert "balance" in data

    def test_returns_tokens_used(self, client, auth_headers, mock_container):
        mock_container.llm_client.return_value.chat.return_value = "Response"

        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hello world test", "history": []},
            headers=auth_headers,
        )
        data = resp.get_json()
        assert data["tokens_used"] >= 1

    def test_unauthorized(self, client):
        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hi"},
        )
        assert resp.status_code == 401

    def test_empty_body(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/send",
            headers=auth_headers,
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_missing_message_field(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "message" in resp.get_json()["error"].lower()

    def test_empty_message(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "   ", "history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_insufficient_balance(self, client, auth_headers, mock_container):
        mock_container.token_service.return_value.get_balance.return_value = 0

        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hello", "history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "insufficient" in resp.get_json()["error"].lower()

    def test_plugin_disabled(
        self, client, auth_headers, app, mock_config_store_disabled
    ):
        app.config_store = mock_config_store_disabled

        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hello", "history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_llm_error_returns_502(self, client, auth_headers, mock_container):
        from vbwd.llm.errors import LlmError

        mock_container.llm_client.return_value.chat.side_effect = LlmError(
            "upstream failed"
        )

        resp = client.post(
            "/api/v1/plugins/chat/send",
            json={"message": "Hello", "history": []},
            headers=auth_headers,
        )
        assert resp.status_code == 502


class TestGetConfig:
    """Tests for GET /api/v1/plugins/chat/config."""

    def test_returns_safe_fields(self, client, auth_headers):
        resp = client.get(
            "/api/v1/plugins/chat/config",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model" in data
        assert "max_message_length" in data
        assert "counting_mode" in data

    def test_never_returns_api_key(self, client, auth_headers):
        resp = client.get(
            "/api/v1/plugins/chat/config",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert "llm_api_key" not in data
        assert "api_key" not in data

    def test_never_returns_api_endpoint(self, client, auth_headers):
        resp = client.get(
            "/api/v1/plugins/chat/config",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert "llm_api_endpoint" not in data
        assert "api_endpoint" not in data

    def test_unauthorized(self, client):
        resp = client.get("/api/v1/plugins/chat/config")
        assert resp.status_code == 401

    def test_plugin_disabled(
        self, client, auth_headers, app, mock_config_store_disabled
    ):
        app.config_store = mock_config_store_disabled

        resp = client.get(
            "/api/v1/plugins/chat/config",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestEstimateCost:
    """Tests for POST /api/v1/plugins/chat/estimate."""

    def test_success(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/estimate",
            json={"message": "Hello world test"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "estimated_tokens" in data
        assert data["estimated_tokens"] >= 1

    def test_empty_message(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/estimate",
            json={"message": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["estimated_tokens"] >= 1

    def test_unauthorized(self, client):
        resp = client.post(
            "/api/v1/plugins/chat/estimate",
            json={"message": "Hi"},
        )
        assert resp.status_code == 401

    def test_missing_message(self, client, auth_headers):
        resp = client.post(
            "/api/v1/plugins/chat/estimate",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400
