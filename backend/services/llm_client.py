"""Single place for LLM calls: per-request credentials, timeout, and bounded retries.

Uses the legacy openai==0.27.x SDK, but passes api_key/api_base per request instead of
mutating module-level globals, so several clients can coexist safely.
"""
import logging
import time
from typing import Callable, Dict, List

import openai

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails permanently (after retries) or returns nothing."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.error.Timeout, openai.error.RateLimitError,
                        openai.error.ServiceUnavailableError,
                        openai.error.APIConnectionError)):
        return True
    if isinstance(exc, openai.error.APIError):
        text = str(exc).lower()
        code = getattr(exc, "http_status", None)
        return code in (429, 500, 502, 503, 504) or "429" in text or "quota" in text
    return False


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str,
                 timeout: int = 90, max_retries: int = 5,
                 base_backoff: float = 5.0, max_backoff: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._sleep = sleep

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    request_timeout=self.timeout,
                    api_key=self.api_key,
                    api_base=self.base_url,
                )
                content = response["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise LLMError("LLM returned an empty response.")
                return content
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                if not _is_retryable(exc):
                    raise LLMError(f"LLM request failed: {exc}") from exc
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                wait = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                logger.warning("LLM call failed (%s); retry %d/%d in %.0fs",
                               type(exc).__name__, attempt + 1, self.max_retries, wait)
                self._sleep(wait)
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def ask(self, prompt: str, temperature: float = 0.0) -> str:
        return self.chat([{"role": "user", "content": prompt}], temperature)
