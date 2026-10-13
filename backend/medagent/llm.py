"""Provider-agnostic LLM caller with JSON schema validation and retry logic."""

import json
import logging
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
from medagent.config import settings

logger = logging.getLogger("medagent.llm")


def call_llm_json(prompt: str, schema_cls: Type[BaseModel], max_retries: int = 2) -> Optional[BaseModel]:
    """
    Call configured LLM provider and validate output against Pydantic schema.
    Returns parsed instance or None if offline/failed.
    """
    provider = settings.LLM_PROVIDER.lower()
    if provider == "offline":
        logger.info("LLM_PROVIDER is offline; skipping remote LLM call.")
        return None

    if provider == "gemini":
        return _call_gemini(prompt, schema_cls, max_retries)
    elif provider == "openai":
        return _call_openai(prompt, schema_cls, max_retries)
    elif provider == "anthropic":
        return _call_anthropic(prompt, schema_cls, max_retries)
    else:
        logger.warning("Unknown LLM provider '%s'; falling back to offline rule engine.", provider)
        return None


def _call_gemini(prompt: str, schema_cls: Type[BaseModel], max_retries: int) -> Optional[BaseModel]:
    """Call Google Gemini model."""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.info("No GEMINI_API_KEY provided; falling back.")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = settings.LLM_MODEL or "gemini-2.5-flash"
        
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                    }
                )
                text = response.text or "{}"
                data = json.loads(text)
                return schema_cls.model_validate(data)
            except Exception as e:
                logger.warning("Gemini attempt %d failed: %s", attempt + 1, str(e))
                if attempt == max_retries:
                    return None
    except ImportError:
        logger.warning("google-genai package not installed.")
        return None


def _call_openai(prompt: str, schema_cls: Type[BaseModel], max_retries: int) -> Optional[BaseModel]:
    """Call OpenAI API."""
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.info("No OPENAI_API_KEY provided; falling back.")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model_name = settings.LLM_MODEL or "gpt-4o"

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a clinical decision support assistant. You must output valid JSON only adhering strictly to the requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                raw = response.choices[0].message.content or "{}"
                data = json.loads(raw)
                return schema_cls.model_validate(data)
            except Exception as e:
                logger.warning("OpenAI attempt %d failed: %s", attempt + 1, str(e))
                if attempt == max_retries:
                    return None
    except ImportError:
        logger.warning("openai package not installed.")
        return None


def _call_anthropic(prompt: str, schema_cls: Type[BaseModel], max_retries: int) -> Optional[BaseModel]:
    """Call Anthropic API."""
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.info("No ANTHROPIC_API_KEY provided; falling back.")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        model_name = settings.LLM_MODEL or "claude-3-5-sonnet-20241022"

        for attempt in range(max_retries + 1):
            try:
                message = client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system="You are a clinical decision support assistant. Return strictly raw JSON matching the required schema.",
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = message.content[0].text
                data = json.loads(raw)
                return schema_cls.model_validate(data)
            except Exception as e:
                logger.warning("Anthropic attempt %d failed: %s", attempt + 1, str(e))
                if attempt == max_retries:
                    return None
    except ImportError:
        logger.warning("anthropic package not installed.")
        return None
