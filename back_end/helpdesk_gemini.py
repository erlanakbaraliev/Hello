"""Gemini client helpers for HelpDesk chat."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised for Gemini setup and runtime failures."""


def get_gemini_api_key(streamlit_secrets: Any | None = None) -> str | None:
    """Resolve API key from Streamlit secrets then environment."""
    if streamlit_secrets is not None:
        try:
            key = streamlit_secrets.get("GEMINI_API_KEY")
            if key:
                return str(key)
        except Exception:
            pass
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return None


def sanitize_user_text(text: str, max_length: int = 2000) -> str:
    """Trim, remove control chars, and enforce max length."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", (text or "")).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def generate_helpdesk_reply(
    messages: list[dict[str, str]],
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    timeout_seconds: int = 20,
    max_retries: int = 2,
) -> str:
    """Generate assistant response from Gemini based on chat messages."""
    if not api_key:
        raise GeminiClientError("Missing GEMINI_API_KEY.")
    if not messages:
        raise GeminiClientError("No messages provided.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise GeminiClientError(
            "Gemini SDK is not installed. Install dependency: google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    prompt_lines = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().lower()
        content = msg.get("content") or ""
        prompt_lines.append(f"{role.upper()}: {content}")
    prompt = "\n".join(prompt_lines)

    for attempt in range(max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(model.generate_content, prompt)
                response = future.result(timeout=timeout_seconds)
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            raise GeminiClientError("Gemini returned an empty response.")
        except TimeoutError:
            logger.warning("Gemini timeout (attempt %s)", attempt + 1)
            err = GeminiClientError("Gemini request timed out.")
        except GeminiClientError as exc:
            err = exc
        except Exception as exc:
            logger.exception("Gemini request failed")
            err = GeminiClientError(f"Gemini request failed: {exc}")

        if attempt < max_retries:
            time.sleep(0.4 * (attempt + 1))
            continue
        raise err
