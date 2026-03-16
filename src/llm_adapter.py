"""LLM adapter — HTTP calls to OpenAI-compatible chat completions API."""
import logging
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM API call fails."""
    pass


class LLMAdapter:
    """Adapter for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        timeout: int = 30,
        max_tokens: Optional[int] = None,
    ):
        self.api_endpoint = self._normalize_endpoint(api_endpoint)
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_tokens = max_tokens

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Append /chat/completions if the endpoint is a base URL.

        Accepts both full paths and base URLs:
          https://api.openai.com/v1/chat/completions -> as-is
          https://api.deepseek.com                   -> .../chat/completions
          https://api.deepseek.com/                  -> .../chat/completions
          https://api.deepseek.com/v1                -> .../v1/chat/completions
        """
        if not endpoint:
            return endpoint
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        return endpoint

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to LLM and return assistant response text.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            Assistant response text.

        Raises:
            LLMError: If the API call fails.
        """
        if not self.api_endpoint:
            raise LLMError("LLM API endpoint is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise LLMError("LLM API request timed out")
        except requests.RequestException as e:
            raise LLMError(f"LLM API request failed: {e}")

        if response.status_code != 200:
            raise LLMError(
                f"LLM API returned {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Invalid LLM API response: {e}")
