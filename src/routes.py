"""Chat plugin API routes."""
import logging
from flask import Blueprint, jsonify, request, current_app, g

from src.middleware.auth import require_auth
from plugins.chat.src.llm_adapter import LLMAdapter, LLMError
from plugins.chat.src.chat_service import ChatService
from plugins.chat.src.token_counting import get_counting_strategy

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat_plugin", __name__)


def _check_chat_enabled():
    """Check chat plugin is enabled and return config.

    Returns:
        (config_dict, None) if enabled
        (None, (json_response, status_code)) if disabled
    """
    config_store = getattr(current_app, "config_store", None)
    if not config_store:
        return None, (jsonify({"error": "Plugin system not available"}), 503)

    entry = config_store.get_by_name("chat")
    if not entry or entry.status != "enabled":
        return None, (jsonify({"error": "Chat plugin not enabled"}), 404)

    config = config_store.get_config("chat")
    return config, None


def _build_chat_service(config):
    """Build ChatService from config and DI container."""
    container = current_app.container
    token_service = container.token_service()

    adapter = LLMAdapter(
        api_endpoint=config.get("llm_api_endpoint", ""),
        api_key=config.get("llm_api_key", ""),
        model=config.get("llm_model", "gpt-4o-mini"),
        system_prompt=config.get(
            "system_prompt", "You are a helpful assistant."
        ),
    )

    strategy = get_counting_strategy(config.get("counting_mode", "words"))

    return ChatService(token_service, adapter, strategy, config)


@chat_bp.route("/send", methods=["POST"])
@require_auth
def send_message():
    """POST /api/v1/plugins/chat/send

    Body: {"message": "...", "history": [...]}
    Response: {"response": "...", "tokens_used": 5, "balance": 1245}
    """
    config, err = _check_chat_enabled()
    if err:
        return err

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message must be a non-empty string"}), 400

    history = data.get("history", [])
    if not isinstance(history, list):
        return jsonify({"error": "History must be an array"}), 400

    if not config.get("llm_api_endpoint"):
        return jsonify({"error": "LLM API endpoint is not configured"}), 503

    service = _build_chat_service(config)

    try:
        result = service.send_message(
            user_id=g.user_id,
            message=message,
            history=history,
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except LLMError as e:
        logger.error("LLM error: %s", e)
        return jsonify({"error": "Chat service unavailable"}), 502


@chat_bp.route("/config", methods=["GET"])
@require_auth
def get_chat_config():
    """GET /api/v1/plugins/chat/config

    Returns safe config fields (never api_key or api_endpoint).
    """
    config, err = _check_chat_enabled()
    if err:
        return err

    return jsonify({
        "model": config.get("llm_model", "gpt-4o-mini"),
        "max_message_length": config.get("max_message_length", 4000),
        "counting_mode": config.get("counting_mode", "words"),
    }), 200


@chat_bp.route("/estimate", methods=["POST"])
@require_auth
def estimate_cost():
    """POST /api/v1/plugins/chat/estimate

    Body: {"message": "..."}
    Response: {"estimated_tokens": 3}
    """
    config, err = _check_chat_enabled()
    if err:
        return err

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]
    if not isinstance(message, str):
        return jsonify({"error": "Message must be a string"}), 400

    strategy = get_counting_strategy(config.get("counting_mode", "words"))
    estimated = strategy.calculate_tokens(message, config)

    return jsonify({"estimated_tokens": estimated}), 200
