"""
Unified LLM client that supports OpenAI, Anthropic, DeepSeek, and Zhipu.
Reference: step2_syn_query_extension.py, step10_llm_for_exp.py, step13.py
"""

import json
import re
from typing import Optional

from .config import LLMConfig

# Default API endpoints for providers that use OpenAI-compatible SDKs
_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}


def _call_openai(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    task: str = "generation",
    temperature: float = 0.0,
) -> str:
    """Call OpenAI-compatible API (OpenAI, DeepSeek, or proxy endpoints)."""
    from openai import OpenAI

    model = config.get_model(task)
    client_kwargs = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    elif config.provider in _DEFAULT_BASE_URLS:
        client_kwargs["base_url"] = _DEFAULT_BASE_URLS[config.provider]

    client = OpenAI(**client_kwargs)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


def _call_anthropic(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    task: str = "generation",
    temperature: float = 0.0,
) -> str:
    """Call Anthropic API."""
    import anthropic

    model = config.get_model(task)
    client_kwargs = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = anthropic.Anthropic(**client_kwargs)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return message.content[0].text.strip()


def _call_zhipu(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    task: str = "generation",
    temperature: float = 0.0,
) -> str:
    """Call Zhipu (GLM) API via OpenAI-compatible endpoint."""
    # Zhipu provides an OpenAI-compatible endpoint
    return _call_openai(config, system_prompt, user_prompt, task, temperature)


def call_llm(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    task: str = "generation",
    temperature: float = 0.0,
) -> str:
    """Route LLM call to the appropriate provider."""
    dispatcher = {
        "openai": _call_openai,
        "deepseek": _call_openai,  # DeepSeek uses OpenAI-compatible API
        "anthropic": _call_anthropic,
        "zhipu": _call_zhipu,
    }
    handler = dispatcher.get(config.provider)
    if handler is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    return handler(config, system_prompt, user_prompt, task, temperature)


def extract_json_from_response(raw: str) -> dict:
    """Robust JSON extraction from LLM output (handles markdown fences)."""
    content = raw.strip()

    # Remove markdown code fences
    if content.startswith("```"):
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0)
        else:
            content = content.replace("```json", "").replace("```", "").strip()

    # Remove trailing commas before closing brackets (common LLM mistake)
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to fix common issues
        # Sometimes LLM includes text before/after JSON
        bracket_start = content.find("{")
        bracket_end = content.rfind("}")
        bracket_start_arr = content.find("[")
        bracket_end_arr = content.rfind("]")

        if bracket_start_arr != -1 and (
            bracket_start == -1 or bracket_start_arr < bracket_start
        ):
            content = content[bracket_start_arr : bracket_end_arr + 1]
        elif bracket_start != -1:
            content = content[bracket_start : bracket_end + 1]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_output": raw, "error": "Failed to parse JSON"}
